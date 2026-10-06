import importlib.util
import os
import sys

import numpy as np
import torch
from loguru import logger

from syndrilla.decoder.knobs import knob
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
            f"[rebatch_opt] shared decoder CUDA kernel unavailable ({e}); "
            f"using torch.bincount."
        )
        _DECODER_EXT = False
        return None
    return _DECODER_EXT


class RebatchSpeedup:
    """Per-decoder iteration-speedup state: paces ``k`` by KL divergence then picks the cap."""

    def __init__(
        self,
        kl_eps=1e-4,
        kl_window=3,
        kl_min=3,
        candidates=DEFAULT_CANDIDATES,
        min_pct=50,
        min_speedup=1.1,
    ):
        self.kl_eps = float(kl_eps)
        self.kl_window = int(kl_window)
        self.kl_min = int(kl_min)
        self.candidates = tuple(candidates) if candidates else DEFAULT_CANDIDATES
        self.min_pct = int(min_pct)  # lowest percentile the chooser considers
        self.min_speedup = float(min_speedup)  # below this projected gain, decline
        self.hists = []  # per-warm-up-batch iteration histograms
        self.pooled = None  # running pooled histogram over warm-up batches
        self.streak = 0  # consecutive settled (low-KL) batches
        self.batch_size = 0
        self.max_iter = None  # stop iteration of a shot that never converged
        self.frac = None  # P/100 once chosen; None while warming up or declined
        self.pct = None
        self.declined = None  # reason string once warm-up ends without a cap

    @classmethod
    def from_cfg(cls, decoding_cfg):
        """Build from the decoder config: None when ``rebatch_opt`` is false, else
        the cap from the optional ``rebatch_opt_params`` block (unknown keys ignored),
        or the class defaults when the block is absent. The old key
        ``rebatch_speedup`` raises ValueError."""
        if "rebatch_speedup" in decoding_cfg:
            raise ValueError(
                "decoder config key 'rebatch_speedup' is not supported; "
                "use 'rebatch_opt' (bool) and 'rebatch_opt_params' (dict) instead."
            )
        if not knob(decoding_cfg, "rebatch_opt", True):
            return None
        cfg = decoding_cfg.get("rebatch_opt_params") or {}
        keys = ("kl_eps", "kl_window", "kl_min", "candidates", "min_pct", "min_speedup")
        return cls(**{k: cfg[k] for k in keys if k in cfg})

    @property
    def done(self):
        """Warm-up is over: a cap was chosen (``frac`` set) or declined (``frac`` None)."""
        return self.frac is not None or self.declined is not None

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
        self.max_iter = int(max_iter)
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
            f"[rebatch_opt] batch {len(self.hists)}: KL={kl:.2e} "
            f"streak={self.streak}/{self.kl_window}"
        )
        if self.streak >= self.kl_window:
            self._choose()

    def _choose(self):
        """Pick the candidate percentile with the best projected speedup among
        ``min_pct <= p <= floor(100 * (1 - f)) - 1``, where ``f`` is the pooled
        fraction of shots that stopped at ``max_iter`` (never converged), so the
        cap always lands below the converged share. With no candidate in range, or
        a best projected speedup under ``min_speedup``, decline: warm-up ends
        (``done``) with ``frac`` None, so BP runs uncapped."""
        total = float(self.pooled.sum())
        f = float(self.pooled[self.max_iter]) / total if total else 0.0
        hi = int(np.floor(100.0 * (1.0 - f))) - 1
        best_p, best_sp = None, -1.0
        for p in sorted(self.candidates):  # ascending: ties keep the higher cap
            if not self.min_pct <= p <= hi:
                continue
            sp = _projected_speedup(self.hists, p, self.batch_size)
            if sp == sp and sp >= best_sp:  # sp == sp filters NaN
                best_sp, best_p = sp, p
        if best_p is None:
            self.declined = (
                f"no candidate in p{self.min_pct}..p{hi} "
                f"({100.0 * f:.1f}% of shots reach max_iter)"
            )
        elif best_sp < self.min_speedup:
            self.declined = (
                f"best projected speedup {best_sp:.2f}x (p{best_p}) "
                f"under min_speedup {self.min_speedup:g}x"
            )
        if self.declined is not None:
            logger.warning(
                f"[rebatch_opt] warm-up done after {len(self.hists)} batches: "
                f"cap off, {self.declined}"
            )
            return
        self.pct, self.frac = best_p, best_p / 100.0
        logger.success(
            f"[rebatch_opt] warm-up done after {len(self.hists)} batches: "
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
    "weights",
    "skip_converged",
    # saq's blocks, and the weights only a learned decoder loads
    "model",
    "cpnd",
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
    # bp_lottery, bp_lottery_parallel, bp_ots
    "flip_start_iter",
    "flip_interval",
    "num_parallel",
    "seeds",
    "select",
    "interval",
    "strength",
)

# The other half of the same rule: these stay at the top of the block, so a `config`
# entry that carries one is rejected too. Each key keeps exactly one home.
SHARED_KEYS = (
    "algorithm",
    "check_type",
    "dtype",
    "device",
    "force_pytorch",
    "rebatch_opt_params",
    "config",
)

# Settings of the training run rather than of the model it fits, so they left the
# decoding yaml entirely for the training yaml `-tr` names, mapped here to where each
# one went. Rejected wherever they still appear, for the same reason as above: a
# training run silently falling back to a default still produces a checkpoint.
MOVED_TO_TRAINING = {
    "optimizer": "training.optimizer",
    "train": "training.budget",
}


def _reject_training_keys(cfg: dict, source: str, placement: str):
    """Refuse a decoder config still carrying the run's own settings."""
    moved = [key for key in MOVED_TO_TRAINING if key in cfg]
    if moved:
        went = ", ".join(f"<{key}> as `{MOVED_TO_TRAINING[key]}`" for key in moved)
        raise ValueError(
            f'Decoder config {source} has <{", ".join(moved)}> {placement}; training '
            f"settings moved to the training yaml passed with `-tr`: {went}."
        )


def _split_config(dec_cfg: dict, algorithms: list, source: str):
    """Return one algorithm-specific config per algorithm, validating placement."""
    _reject_training_keys(dec_cfg, source, "at the top level")
    misplaced = [key for key in MOVED_TO_CONFIG if key in dec_cfg]
    if misplaced:
        per_algo = " (one entry per algorithm)" if len(algorithms) > 1 else ""
        raise ValueError(
            f'Decoder config {source} has <{", ".join(misplaced)}> at the top level; '
            f"algorithm-specific keys moved into `decoding.config`{per_algo}."
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
                f"Decoder config {source} has <{len(blocks)}> entries under `decoding.config` "
                f"but only <{n}> under `decoding.algorithm`; they match by position."
            )
        # `- ` with nothing after it parses as None; read it as "no settings". A
        # short list leaves the stages past its end on their defaults, but only
        # trailing entries can be dropped: position binds an entry to an algorithm.
        blocks = [{} if b is None else b for b in blocks]
        blocks = blocks + [{}] * (n - len(blocks))
    else:
        raise ValueError(
            f"Decoder config {source} needs `decoding.config` to be a mapping or a "
            f"list of mappings, got <{type(blocks).__name__}>."
        )

    for algo, block in zip(algorithms, blocks):
        if not isinstance(block, dict):
            raise ValueError(
                f"Decoder config {source} needs every `decoding.config` entry to be a "
                f"mapping, got <{type(block).__name__}> for <{algo}>."
            )
        _reject_training_keys(block, source, f"under `decoding.config` for <{algo}>")
        shared = [key for key in SHARED_KEYS if key in block]
        if shared:
            raise ValueError(
                f'Decoder config {source} has <{", ".join(shared)}> under `decoding.config` '
                f"for <{algo}>; those belong at the top of `decoding`."
            )
    return blocks


def is_trainable(decoder):
    """True if `decoding` is a learned decoder, by its own `TRAINABLE` declaration."""
    # declared rather than detected: the bp family holds `nn.Parameter` index tensors it
    # never learns, so anything based on having parameters would call them trainable
    return bool(getattr(decoder, "TRAINABLE", False))


def assert_trainable(decoders):
    """Raise unless the chain's last decoder is one a training run can fit."""
    tail = decoders[-1]
    inner = getattr(tail, "decoder", tail)
    if is_trainable(inner):
        return
    raise ValueError(
        f'Decoder <{getattr(tail, "algo", type(inner).__name__)}> cannot train: it is '
        f"not a learned decoder. `-t` trains the last algorithm."
    )


def assert_trained(decoders):
    """Raise unless every learned decoder in the chain carries the weights it decodes with.

    A decode run reports a logical error rate, and a learned decoder still at its random
    initialization reports one for random weights: a number that reads like a result and
    measures nothing. `-t` is the mode that starts from there, so a run that is not `-t`
    has to be given the weights.
    """
    untrained = []
    for decoder in decoders:
        inner = getattr(decoder, "decoder", decoder)
        if is_trainable(inner) and getattr(inner, "checkpoint", None) is None:
            untrained.append(getattr(decoder, "algo", type(inner).__name__))
    if not untrained:
        return
    raise ValueError(
        f'Decoder <{", ".join(untrained)}> is learned and was given no weights, so it '
        f"would decode at chance. Point the decoding yaml's `config.checkpoint` at a "
        f"trained checkpoint, or train one first with `-t`."
    )


def resolve_configs(dec_cfg: dict, source: str = "dict"):
    """Return one flat config per algorithm, in `decoding.algorithm` order."""
    algorithms = dec_cfg["algorithm"]
    if isinstance(algorithms, str):
        algorithms = [algorithms]
    shared = {k: v for k, v in dec_cfg.items() if k != "config"}
    return [{**shared, **block} for block in _split_config(dec_cfg, algorithms, source)]


def create_decoder(yaml_path: str = None, cfg: dict = None, **kwargs):
    """Create decoder(s) from a '.decoding.yaml' file or a config dict."""
    header = "decoding"
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
    # reading `decoding_cfg.get(...)` without knowing the block is split.
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
