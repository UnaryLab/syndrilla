import os
import sys
import importlib.util
import numpy as np
import torch
from loguru import logger

from syndrilla.utils import call_func_from_cfg, get_path, read_yaml, check_yaml_header


# --------------------------------------------------------------------------- #
# Adaptive iteration speedup for iterative BP decoders (`rebatch_speedup`, KL-paced).
#
# How it works
#   1. The first batches run UNCAPPED (warm-up). After each one the decoder hands
#      its per-sample stop-iteration histogram to RebatchSpeedup.observe.
#   2. Warm-up ends when the pooled iteration-count distribution stops moving:
#      KL(p_k || p_{k-1}) < kl_eps for kl_window consecutive batches (and at least
#      kl_min batches seen). This decides k automatically.
#   3. From the warm-up batches it picks the converged-fraction stop P that
#      maximizes the projected iteration-count speedup, and exposes frac = P/100.
#
# Afterwards the decode loop stops a batch as soon as >= P% of its samples have
# converged; the unconverged remainder is left with converge == 0 so main can
# offload it to the extra queue and decode all the hard cases together later.
# --------------------------------------------------------------------------- #

# converged-fraction stops to try when picking the cap (percent of the batch).
# The cap is chosen once from warm-up histograms, so sweeping every integer
# percentile is cheap and finds a finer optimum than a sparse hand-picked set.
DEFAULT_CANDIDATES = tuple(range(100))  # 0..99


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


# Shared cross-decoder CUDA kernels (decoder/cuda/decoder.cu), JIT-compiled on first use.
# RebatchSpeedup.observe uses its `iter_histogram` to bin warm-up stop-iteration counts
# on-device for any CUDA decoder. Cached per process; `False` marks a failed build so
# we fall back to torch.bincount (CPU decoders / no nvcc) without retrying the compile.
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

        For CUDA tensors the histogram is binned by the shared decoder kernel
        (``iter_histogram`` in decoder/decoder.cu); CPU decoders -- and any case where
        the kernel can't build -- fall back to ``torch.bincount``."""
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
    """
    Wraps a decoder to transparently handle a rounds dimension (always dim=1).

    Flattens [B, d, ...] → [B*d, ...] before the inner decoder and reshapes
    all outputs back to [B, d, ...] afterwards.  No-op when syndrome is 2D
    (1-channel, 1 round) or 3D with no rounds (2-channel, 1 round).

    Rounds dim convention: [B, rounds, (C), M]
    """

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


# --------------------------------------------------------------------------- #
# `decoder.config`: the algorithm-specific half of a decoder block.
#
# Everything at the top of the block is framework-wide -- `algorithm`, `check_type`,
# `dtype`, `device`, `force_pytorch`, `checkpoint` and `rebatch_speedup` are read by
# main.py, this loader, or every decoder alike. What only one algorithm understands
# (`max_iter`, `sf`, relay_bp's schedule, the quantization widths) lives under
# `config`, one entry per entry of `algorithm`, so a chain can tune each stage
# separately instead of sharing one flat namespace.
# --------------------------------------------------------------------------- #

# Keys that used to sit at the top of the block. Rejected there rather than ignored:
# a decoder silently falling back to max_iter=50 instead of the configured 181 still
# produces numbers, and they look like results.
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
    """Return one algorithm-specific config per algorithm, validating placement.

    `config` is positional: entry i belongs to `algorithm[i]`, which is what lets the
    same algorithm appear twice in a chain with different settings. A single-algorithm
    block may write the mapping directly instead of a one-element list.
    """
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
        if n != 1:
            raise ValueError(
                f"Decoder config {source} chains <{n}> algorithms, so `decoder.config` "
                f"is matched to them by position and has to be a list. Write "
                f"`- {{...}}` for the stage these settings belong to; a stage with no "
                f"settings of its own needs no entry."
            )
        blocks = [blocks]
    elif isinstance(blocks, list):
        if len(blocks) > n:
            raise ValueError(
                f"Decoder config {source} has <{len(blocks)}> entries under "
                f"`decoder.config` but only <{n}> under `decoder.algorithm`; they are "
                f"matched by position, so there is no algorithm for the last "
                f"<{len(blocks) - n}>."
            )
        # `- ` with nothing after it parses as None; read it as "no settings". A short
        # list leaves the stages past its end on their defaults, so a chain ending in
        # a decoder that configures nothing (osd_0 after BP) writes no entry for it.
        # Only trailing entries can be dropped: position is what binds an entry to an
        # algorithm, so a stage that takes no settings before one that does still
        # needs its `- {}` slot.
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
                f'Decoder config {source} has <{", ".join(shared)}> under '
                f"`decoder.config` for <{algo}>; those are read by the framework for "
                f"the whole block and belong at the top of `decoder`."
            )
    return blocks


# every call a training run makes on its decoder, in the order the run makes them
TRAINABLE_STAGES = (
    "check_train_batch",
    "train_fingerprint",
    "configure_optimizer",
    "backward",
    "update",
)


def is_trainable(decoder):
    """True if `decoder` implements the whole training protocol."""
    return all(hasattr(decoder, stage) for stage in TRAINABLE_STAGES)


def assert_trainable(decoders):
    """Raise unless the chain's last decoder implements the whole training protocol.

    `-t` drives the chain's last stage, so that is the decoder a run updates: a stage
    after it would leave the run learning from output its gradient never passed
    through. Checked once, before any batch, so an untrainable chain stops the run
    rather than half-running it.
    """
    tail = decoders[-1]
    inner = getattr(tail, "decoder", tail)
    if is_trainable(inner):
        return
    missing = [stage for stage in TRAINABLE_STAGES if not hasattr(inner, stage)]
    raise ValueError(
        f'Decoder <{getattr(tail, "algo", type(inner).__name__)}> cannot train: it '
        f'implements none of <{", ".join(missing)}>. `-t` trains the last entry of '
        f"`decoder.algorithm`, so that entry has to be a trainable decoder."
    )


def resolve_configs(dec_cfg: dict, source: str = "dict"):
    """Return one flat config per algorithm, in `decoder.algorithm` order.

    Each is the shared keys with that algorithm's `config` entry merged on top, which
    is what a decoder is actually built from. `main.py` needs one of these before
    `create_decoder` runs, so the positional rule that binds a `config` entry to an
    algorithm lives here alone rather than being re-derived at the call site.
    """
    algorithms = dec_cfg["algorithm"]
    if isinstance(algorithms, str):
        algorithms = [algorithms]
    shared = {k: v for k, v in dec_cfg.items() if k != "config"}
    return [{**shared, **block} for block in _split_config(dec_cfg, algorithms, source)]


def create_decoder(yaml_path: str = None, cfg: dict = None, **kwargs):
    """
    Create decoder(s) from a '.decoder.yaml' file or a config dict.

        create_decoder(yaml_path='bp_hx.decoder.yaml')
        create_decoder(cfg={'algorithm': 'bp_norm_min_sum', ...})
    """
    header = "decoder"
    func_name = "algorithm"

    if cfg is not None:
        dec_cfg = cfg
        source = "dict"
        logger.info(f"Creating decoder class from config dict.")
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

    There is no separate ``*_cuda`` algorithm: each algorithm ``<algo>`` lives in
    ``<algo>/<algo>.py`` (PyTorch, CPU or GPU) and, when a fused/per-step CUDA kernel
    port exists, in ``<algo>/<algo>_cuda.py`` alongside it. If the decoder's
    ``device.device_type`` is ``cuda`` and that ``<algo>_cuda.py`` exists, it is used;
    otherwise the standard ``<algo>/<algo>.py`` module handles the run. Whether the
    adaptive cap is active is decided purely by the presence of an ``rebatch_speedup``
    block in the config — not by the algorithm name.

    Set ``force_pytorch: true`` in the decoder config to skip the CUDA kernel port and
    run the plain ``<algo>/<algo>.py`` PyTorch module even on a ``cuda`` device (eager
    tensor ops on the GPU). This is the eager-vs-kernel baseline: same device, no custom
    kernel, so a speedup against it isolates the kernel rather than CPU-vs-GPU.
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
