import os
import yaml, copy
import numpy as np
import torch
from loguru import logger

from syndrilla.utils import call_func_from_cfg, get_path, read_yaml, check_yaml_header


# --------------------------------------------------------------------------- #
# Adaptive iteration speedup for iterative BP decoders (`iter_speedup`, KL-paced).
#
# How it works
#   1. The first batches run UNCAPPED (warm-up). After each one the decoder hands
#      its per-sample stop-iteration histogram to IterSpeedup.observe.
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
DEFAULT_CANDIDATES = tuple(range(100))      # 0..99


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
        cap = int(np.searchsorted(cum, (pct / 100.0) * total))   # cap iteration index
        last = int(np.max(np.nonzero(h)))                        # slowest sample index
        tail = int(h[cap + 1:].sum()) if cap + 1 < h.size else 0
        bases.append(last + 1)
        regs.append(cap + 1)
        tails.append(tail)
        maxs.append(last + 1)
    if not bases or batch_size <= 0:
        return float("nan")
    denom = np.mean(regs) + (np.mean(tails) / batch_size) * max(maxs)
    return np.mean(bases) / denom if denom else float("nan")


class IterSpeedup:
    """Per-decoder iteration-speedup state: paces ``k`` by KL divergence then picks the cap."""

    def __init__(self, kl_eps=1e-4, kl_window=3, kl_min=3, candidates=DEFAULT_CANDIDATES):
        self.kl_eps = float(kl_eps)
        self.kl_window = int(kl_window)
        self.kl_min = int(kl_min)
        self.candidates = tuple(candidates) if candidates else DEFAULT_CANDIDATES
        self.hists = []          # per-warm-up-batch iteration histograms
        self.pooled = None       # running pooled histogram over warm-up batches
        self.streak = 0          # consecutive settled (low-KL) batches
        self.batch_size = 0
        self.frac = None         # P/100 once chosen; None while warming up
        self.pct = None

    @classmethod
    def from_cfg(cls, cfg):
        """Build from the optional decoder-config ``iter_speedup`` block (unknown keys
        ignored). Returns None if the block is missing/empty so the feature is opt-in."""
        if not cfg:
            return None
        keys = ("kl_eps", "kl_window", "kl_min", "candidates")
        return cls(**{k: cfg[k] for k in keys if k in cfg})

    @property
    def done(self):
        return self.frac is not None

    def observe(self, iter_tensor, max_iter, batch_size):
        """Record one warm-up batch's stop-iteration histogram and, once the pooled
        distribution has settled (KL test), choose the cap. Call only during warm-up."""
        h = torch.bincount(iter_tensor.detach().clamp(min=0).long().flatten(),
                           minlength=max_iter + 1).cpu().numpy().astype(np.float64)
        self.hists.append(h)
        self.batch_size = max(self.batch_size, int(batch_size))
        prev, self.pooled = self.pooled, (h if self.pooled is None else self.pooled + h)
        if prev is None:                          # first batch: no predecessor for KL
            return
        pa = self.pooled + 1.0; pa /= pa.sum()    # Laplace-smoothed prefix dists
        pb = prev + 1.0; pb /= pb.sum()
        kl = float(np.sum(pa * np.log(pa / pb)))
        settled = len(self.hists) >= self.kl_min and kl < self.kl_eps
        self.streak = self.streak + 1 if settled else 0
        logger.info(f"[iter_speedup] batch {len(self.hists)}: KL={kl:.2e} "
                    f"streak={self.streak}/{self.kl_window}")
        if self.streak >= self.kl_window:
            self._choose()

    def _choose(self):
        best_p, best_sp = self.candidates[0], -1.0
        for p in sorted(self.candidates):         # ascending: ties keep the higher cap
            sp = _projected_speedup(self.hists, p, self.batch_size)
            if sp == sp and sp >= best_sp:        # sp == sp filters NaN
                best_sp, best_p = sp, p
        self.pct, self.frac = best_p, best_p / 100.0
        logger.success(f"[iter_speedup] warm-up done after {len(self.hists)} batches: "
                       f"cap p{best_p} (stop each batch at {best_p}% converged), "
                       f"projected speedup {best_sp:.2f}x")


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
        base_ndim = getattr(self.decoder, '_base_synd_ndim', 2)
        return synd.ndim > base_ndim

    def forward(self, io_dict):
        synd = io_dict['synd']
        if not self._needs_flatten(synd):
            return self.decoder(io_dict)

        B, d = synd.shape[0], synd.shape[1]
        Bd = B * d
        rest = synd.shape[2:]

        io_dict['synd'] = synd.reshape(Bd, *rest)

        llr0 = io_dict['llr0']
        llr0_rest = llr0.shape[1:]
        io_dict['llr0'] = llr0.unsqueeze(1).expand(B, d, *llr0_rest).reshape(Bd, *llr0_rest)

        for key in ('llr', 'converge', 'iter', 'e_v'):
            if key in io_dict and io_dict[key].shape[0] == B and io_dict[key].ndim >= 2:
                v = io_dict[key]
                io_dict[key] = v.reshape(Bd, *v.shape[2:])

        io_dict = self.decoder(io_dict)

        for key in ('e_v', 'synd', 'llr'):
            if key in io_dict and io_dict[key].shape[0] == Bd:
                v = io_dict[key]
                io_dict[key] = v.reshape(B, d, *v.shape[1:])
        for key in ('converge', 'iter'):
            if key in io_dict and io_dict[key].shape[0] == Bd:
                io_dict[key] = io_dict[key].reshape(B, d)
        io_dict['llr0'] = llr0

        return io_dict


def create_decoder(yaml_path: str=None, cfg: dict=None, **kwargs):
    """
    Create decoder(s) from a '.decoder.yaml' file or a config dict.

        create_decoder(yaml_path='bp_hx.decoder.yaml')
        create_decoder(cfg={'algorithm': 'bp_norm_min_sum', ...})
    """
    header = 'decoder'
    func_name = 'algorithm'

    if cfg is not None:
        dec_cfg = cfg
        logger.info(f'Creating decoder class from config dict.')
    else:
        logger.info(f'Creating decoder class from <{get_path(yaml_path)}>.')
        full_path = get_path(yaml_path)
        load_cfg = read_yaml(full_path)
        check_yaml_header(load_cfg, header, full_path)
        dec_cfg = load_cfg[header]

    # Read algorithm(s)
    algorithms = dec_cfg[func_name]
    if isinstance(algorithms, str):
        algorithms = [algorithms]  # wrap single decoder into a list

    MULTI_CHANNEL_DECODERS = {'bp4'}

    decoders = []
    for algo in algorithms:
        dec_cfg_copy = dec_cfg.copy()
        dec_cfg_copy[func_name] = algo
        decoder = call_func_from_cfg(dec_cfg_copy, header, func_name, os.path.dirname(__file__), **kwargs)
        decoder._base_synd_ndim = 3 if algo.lower() in MULTI_CHANNEL_DECODERS else 2
        decoders.append(RoundFlattenWrapper(decoder))

    return decoders
