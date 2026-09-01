import importlib.util
import os
import sys

import numpy as np
import torch
from loguru import logger

from syndrilla.utils import call_func_from_cfg, check_yaml_header, get_path, read_yaml

DEFAULT_CANDIDATES = tuple(range(100))  


def _projected_speedup(hists, pct, batch_size):
    """Scale-invariant (large-N) iteration-count speedup of stopping each batch at
    its ``pct``-th iteration percentile, offloading the slower tail. This is the
    metric to calibrate on; the raw prefix speedup mis-selects on short warm-ups."""
    bases, regs, tails, maxs = [], [], [], []
    for h in hists:
        total = int(h.sum())
        if total <= 0:
            continue
        cum = np.cumsum(h)
        cap = int(np.searchsorted(cum, (pct / 100.0) * total))  # cap iteration index
        last = int(np.max(np.nonzero(h)))  # slowest sample index
        tail = int(h[cap + 1 :].sum()) if cap + 1 < h.size else 0
        bases.append(last + 1)
        regs.append(cap + 1)
        tails.append(tail)
        maxs.append(last + 1)
    if not bases or batch_size <= 0:
        return float("nan")
    denom = np.mean(regs) + (np.mean(tails) / batch_size) * max(maxs)
    return np.mean(bases) / denom if denom else float("nan")


# Shared cross-decoder CUDA kernels (decoder/cuda/decoder.cu), JIT-compiled on first
# use and cached per process; `False` marks a failed build, so we fall back to
# torch.bincount without retrying the compile.
_DECODER_EXT = None


def _load_decoder_ext():
    """Compile decoder.cu on first use and cache it; return None if CUDA/nvcc is
    unavailable so callers fall back to torch.bincount."""
    global _DECODER_EXT
    if _DECODER_EXT is not None:
        return _DECODER_EXT or None  # False -> None (build previously failed)
    try:
        from torch.utils.cpp_extension import load

        src = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "cuda", "decoder.cu"
        )
        # Build only for the local GPU arch (faster first build, silences a warning).
        if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cap[0]}.{cap[1]}"
        _DECODER_EXT = load(
            name="decoder_cuda_ext",
            sources=[src],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as e:  # nvcc missing, no GPU, compile error -> fall back
        logger.warning(
            f"[rebatch_speedup] shared decoder CUDA kernel unavailable ({e}); "
            f"using torch.bincount."
        )
        _DECODER_EXT = False
        return None
    return _DECODER_EXT


class RebatchSpeedup:
    """Per-decoder iteration-speedup state: paces ``k`` by KL divergence then picks the cap."""

    def __init__(
        self, kl_eps=1e-4, kl_window=3, kl_min=3, candidates=DEFAULT_CANDIDATES
    ):
        self.kl_eps = float(kl_eps)
        self.kl_window = int(kl_window)
        self.kl_min = int(kl_min)
        self.candidates = tuple(candidates) if candidates else DEFAULT_CANDIDATES
        self.hists = []  # per-warm-up-batch iteration histograms
        self.pooled = None  # running pooled histogram over warm-up batches
        self.streak = 0  # consecutive settled (low-KL) batches
        self.batch_size = 0
        self.frac = None  # P/100 once chosen; None while warming up
        self.pct = None

    @classmethod
    def from_cfg(cls, cfg):
        """Build from the optional decoder-config ``rebatch_speedup`` block (unknown keys
        ignored). Returns None if the block is missing/empty so the feature is opt-in.
        """
        if not cfg:
            return None
        keys = ("kl_eps", "kl_window", "kl_min", "candidates")
        return cls(**{k: cfg[k] for k in keys if k in cfg})

    @property
    def done(self):
        return self.frac is not None

    def observe(self, iter_tensor, max_iter, batch_size):
        """Record one warm-up batch's stop-iteration histogram and, once the pooled
        distribution has settled (KL test), choose the cap. Call only during warm-up.
        """
        t = iter_tensor.detach()
        ext = _load_decoder_ext() if t.is_cuda else None
        if ext is not None:
            h = (
                ext.iter_histogram(t.flatten(), max_iter + 1)
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        else:
            h = (
                torch.bincount(t.clamp(min=0).long().flatten(), minlength=max_iter + 1)
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        self.hists.append(h)
        self.batch_size = max(self.batch_size, int(batch_size))
        prev, self.pooled = self.pooled, (h if self.pooled is None else self.pooled + h)
        if prev is None:  # first batch: no predecessor for KL
            return
        pa = self.pooled + 1.0
        pa /= pa.sum()  # Laplace-smoothed prefix dists
        pb = prev + 1.0
        pb /= pb.sum()
        kl = float(np.sum(pa * np.log(pa / pb)))
        settled = len(self.hists) >= self.kl_min and kl < self.kl_eps
        self.streak = self.streak + 1 if settled else 0
        logger.info(
            f"[rebatch_speedup] batch {len(self.hists)}: KL={kl:.2e} "
            f"streak={self.streak}/{self.kl_window}"
        )
        if self.streak >= self.kl_window:
            self._choose()

    def _choose(self):
        best_p, best_sp = self.candidates[0], -1.0
        for p in sorted(self.candidates):  # ascending: ties keep the higher cap
            sp = _projected_speedup(self.hists, p, self.batch_size)
            if sp == sp and sp >= best_sp:  # sp == sp filters NaN
                best_sp, best_p = sp, p
        self.pct, self.frac = best_p, best_p / 100.0
        logger.success(
            f"[rebatch_speedup] warm-up done after {len(self.hists)} batches: "
            f"cap p{best_p} (stop each batch at {best_p}% converged), "
            f"projected speedup {best_sp:.2f}x"
        )


class RoundFlattenWrapper(torch.nn.Module):
    """Wraps a decoder to transparently handle a rounds dimension (always dim=1)."""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.decoder, name)

    def _needs_flatten(self, synd):
        """1ch decoder expects 2D, 2ch decoder expects 3D. Extra dim = rounds."""
        base_ndim = getattr(self.decoder, "_base_synd_ndim", 2)
        return synd.ndim > base_ndim

    def forward(self, io_dict):
        synd = io_dict["synd"]
        if not self._needs_flatten(synd):
            return self.decoder(io_dict)

        B, d = synd.shape[0], synd.shape[1]
        Bd = B * d
        rest = synd.shape[2:]

        io_dict["synd"] = synd.reshape(Bd, *rest)

        llr0 = io_dict["llr0"]
        if llr0.ndim == synd.ndim and llr0.shape[:2] == (B, d):
            # the prior already carries the rounds dimension, so it is flattened the
            # way the syndrome is; expanding it instead hands the inner decoder a
            # [B*d, d, N] prior against a [B*d, M] syndrome
            io_dict["llr0"] = llr0.reshape(Bd, *llr0.shape[2:])
        else:
            # one prior per shot, shared by every round of that shot
            llr0_rest = llr0.shape[1:]
            io_dict["llr0"] = (
                llr0.unsqueeze(1).expand(B, d, *llr0_rest).reshape(Bd, *llr0_rest)
            )

        for key in ("llr", "converge", "iter", "e_v"):
            if key in io_dict and io_dict[key].shape[0] == B and io_dict[key].ndim >= 2:
                v = io_dict[key]
                io_dict[key] = v.reshape(Bd, *v.shape[2:])

        io_dict = self.decoder(io_dict)

        for key in ("e_v", "synd", "llr"):
            if key in io_dict and io_dict[key].shape[0] == Bd:
                v = io_dict[key]
                io_dict[key] = v.reshape(B, d, *v.shape[1:])
        for key in ("converge", "iter"):
            if key in io_dict and io_dict[key].shape[0] == Bd:
                io_dict[key] = io_dict[key].reshape(B, d)
        io_dict["llr0"] = llr0

        return io_dict


# Keys that belong under `config`, not at the top of the block. Rejected there rather
# than ignored: a silent fallback to a default still produces numbers.
MOVED_TO_CONFIG = (
    "max_iter",
    "damping_factor",
    "max_b_iter",
    "random_machine",
    "sign_flip_policy",
    "int_width",
    "frac_width",
    "sf",
    "mp_min_batch",
    "num_workers",
    # saq's blocks, and the weights only a learned decoder loads
    "model",
    "cpnd",
    "optimizer",
    "train",
    "checkpoint",
    # relay_bp's schedule
    "legs",
    "iteration_initial",
    "iteration_count",
    "solution",
    "init_mem_strength",
    "center",
    "width",
    "alpha",
    "alpha_scaling",
    "type",
)

# The other half of the same rule: these stay at the top of the block, so a `config`
# entry that carries one is rejected too. Each key keeps exactly one home.
SHARED_KEYS = (
    "algorithm",
    "check_type",
    "dtype",
    "device",
    "force_pytorch",
    "rebatch_speedup",
    "config",
)


def _split_config(dec_cfg: dict, algorithms: list, source: str):
    """Return one algorithm-specific config per algorithm, validating placement."""
    misplaced = [key for key in MOVED_TO_CONFIG if key in dec_cfg]
    if misplaced:
        per_algo = " (one entry per algorithm)" if len(algorithms) > 1 else ""
        raise ValueError(
            f'Decoder config {source} has <{", ".join(misplaced)}> at the top level; '
            f"algorithm-specific keys moved into `decoder.config`{per_algo}."
        )

    blocks = dec_cfg.get("config")
    n = len(algorithms)
    if blocks is None:
        blocks = [{}] * n
    elif isinstance(blocks, dict):
        # a mapping is the first stage's settings; the rest of the chain defaults
        blocks = [blocks] + [{}] * (n - 1)
    elif isinstance(blocks, list):
        if len(blocks) > n:
            raise ValueError(
                f"Decoder config {source} has <{len(blocks)}> entries under `decoder.config` "
                f"but only <{n}> under `decoder.algorithm`; they match by position."
            )
        # `- ` with nothing after it parses as None; read it as "no settings". A
        # short list leaves the stages past its end on their defaults, but only
        # trailing entries can be dropped: position binds an entry to an algorithm.
        blocks = [{} if b is None else b for b in blocks]
        blocks = blocks + [{}] * (n - len(blocks))
    else:
        raise ValueError(
            f"Decoder config {source} needs `decoder.config` to be a mapping or a "
            f"list of mappings, got <{type(blocks).__name__}>."
        )

    for algo, block in zip(algorithms, blocks):
        if not isinstance(block, dict):
            raise ValueError(
                f"Decoder config {source} needs every `decoder.config` entry to be a "
                f"mapping, got <{type(block).__name__}> for <{algo}>."
            )
        shared = [key for key in SHARED_KEYS if key in block]
        if shared:
            raise ValueError(
                f'Decoder config {source} has <{", ".join(shared)}> under `decoder.config` '
                f"for <{algo}>; those belong at the top of `decoder`."
            )
    return blocks


# every call a training run makes on its decoder, in order. The per-batch step is not
# one of them: the loop steps `self.optimizer`, which `configure_optimizer` installs
TRAINABLE_STAGES = (
    "train_fingerprint",
    "configure_optimizer",
    "train_state",
)


def is_trainable(decoder):
    """True if `decoder` implements the whole training protocol."""
    return all(hasattr(decoder, stage) for stage in TRAINABLE_STAGES)


def assert_trainable(decoders):
    """Raise unless the chain's last decoder implements the whole training protocol."""
    tail = decoders[-1]
    inner = getattr(tail, "decoder", tail)
    if is_trainable(inner):
        return
    missing = [stage for stage in TRAINABLE_STAGES if not hasattr(inner, stage)]
    raise ValueError(
        f'Decoder <{getattr(tail, "algo", type(inner).__name__)}> cannot train: it '
        f'implements none of <{", ".join(missing)}>. `-t` trains the last algorithm.'
    )


def resolve_configs(dec_cfg: dict, source: str = "dict"):
    """Return one flat config per algorithm, in `decoder.algorithm` order."""
    algorithms = dec_cfg["algorithm"]
    if isinstance(algorithms, str):
        algorithms = [algorithms]
    shared = {k: v for k, v in dec_cfg.items() if k != "config"}
    return [{**shared, **block} for block in _split_config(dec_cfg, algorithms, source)]


def create_decoder(yaml_path: str = None, cfg: dict = None, **kwargs):
    """Create decoder(s) from a '.decoder.yaml' file or a config dict."""
    header = "decoder"
    func_name = "algorithm"

    if cfg is not None:
        dec_cfg = cfg
        source = "dict"
        logger.info("Creating decoder class from config dict.")
    else:
        logger.info(f"Creating decoder class from <{get_path(yaml_path)}>.")
        full_path = get_path(yaml_path)
        load_cfg = read_yaml(full_path)
        check_yaml_header(load_cfg, header, full_path)
        dec_cfg = load_cfg[header]
        source = f"<{full_path}>"

    # Read algorithm(s)
    algorithms = dec_cfg[func_name]
    if isinstance(algorithms, str):
        algorithms = [algorithms]  # wrap single decoder into a list

    # Each decoder still receives one flat config, so the decoders themselves keep
    # reading `decoder_cfg.get(...)` without knowing the block is split.
    algo_cfgs = resolve_configs(dec_cfg, source)

    MULTI_CHANNEL_DECODERS = {"bp4"}

    decoders = []
    for algo, algo_cfg in zip(algorithms, algo_cfgs):
        dec_cfg_copy = dict(algo_cfg)
        dec_cfg_copy[func_name] = algo
        decoder = _create_one_decoder(dec_cfg_copy, header, func_name, **kwargs)
        decoder._base_synd_ndim = 3 if algo.lower() in MULTI_CHANNEL_DECODERS else 2
        decoders.append(RoundFlattenWrapper(decoder))

    return decoders


def _create_one_decoder(dec_cfg: dict, header: str, func_name: str, **kwargs):
    """Instantiate one decoder, picking the CUDA kernel implementation when the config
    asks for a CUDA device.
    """
    algo = dec_cfg[func_name].lower()
    device_type = (dec_cfg.get("device") or {}).get("device_type", "cpu")
    force_pytorch = dec_cfg.get("force_pytorch", False)

    if device_type == "cuda" and not force_pytorch:
        decoder_dir = os.path.dirname(__file__)
        cuda_file = os.path.join(decoder_dir, algo, f"{algo}_cuda.py")
        if os.path.isfile(cuda_file):
            spec = importlib.util.spec_from_file_location(
                f"create_{header}_with_{algo}_cuda", cuda_file
            )
            module_py = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module_py
            spec.loader.exec_module(module_py)
            return module_py.create(dec_cfg, **kwargs)

    # CPU device, or no CUDA kernel port for this algorithm: the torch module runs
    # on whatever device the config selects (it is device-agnostic).
    return call_func_from_cfg(
        dec_cfg, header, func_name, os.path.dirname(__file__), **kwargs
    )
