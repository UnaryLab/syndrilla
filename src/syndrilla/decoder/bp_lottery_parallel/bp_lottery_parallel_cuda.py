import torch

from syndrilla.decoder.bp_lottery.bp_lottery_cuda import create as _LotteryCuda
from syndrilla.decoder.bp_lottery_parallel.bp_lottery_parallel import Parallel


class _HookExt:
    """The bp_kernel extension with vn_cn_update_csr wrapped for the replica
    knobs of dec: with syndrome_flip_last_n, the check signs read from
    dec._synd_live once the hook has set it; b_c2v kept as dec._c2v for the hook.
    Every other attribute is the extension's."""

    def __init__(self, ext, dec):
        self._ext, self._dec = ext, dec

    def __getattr__(self, name):
        return getattr(self._ext, name)

    def vn_cn_update_csr(self, l_in, s_neg, row_ptr, col, b_c2v, run, beta, N, **kw):
        d = self._dec
        s = d._synd_live
        if d._sf_rep is not None and s is not None:
            s_neg = torch.where(s == 0.0, torch.ones_like(s), -torch.ones_like(s))
        self._ext.vn_cn_update_csr(l_in, s_neg, row_ptr, col, b_c2v, run, beta, N, **kw)
        d._c2v = b_c2v


class create(Parallel, _LotteryCuda):
    """bp_lottery_parallel on the bp_lottery_cuda per-step loop. Accepts every
    bp_lottery_cuda key plus the bp_lottery_parallel keys (flip_start_iter
    defaults to 0 here); the K replicas of each shot decode as one batch of
    B * K rows, and the selection, seeds, replica knobs and flip knobs match
    the PyTorch module."""

    # system draws come from the replica generators, never the global RNG
    _global_last_draw = False

    def __init__(self, decoding_cfg: dict, **kwargs) -> None:
        super().__init__(decoding_cfg, **kwargs)
        self._init_parallel(decoding_cfg)
        if self._stall or self._sf_rep is not None:
            knobs = self._kernel_knobs()
            self._ext = _HookExt(self._ext, self)
            self._kernel_knobs = lambda: knobs
        self.algo = "bp_lottery_parallel"

    def forward(self, io_dict: dict) -> dict:
        return self._parallel_forward(io_dict, super().forward)
