"""OTS on the base CUDA BP loop, with sum-product check messages.

Dimitris Chytas, Michele Pacenti, Nithin Raveendran, Mark F. Flanagan and
Bane Vasic, "Enhanced Message-Passing Decoding of Degenerate Quantum Codes
Utilizing Trapping Set Dynamics," IEEE Communications Letters, 2024, Algorithm 1:
https://doi.org/10.1109/LCOMM.2024.3356312
An empty oscillator set biases only the least-reliable node.
"""

from pathlib import Path
import weakref

import torch

from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum_cuda import _dll_dirs, create as _CudaBP
from syndrilla.decoder.bp_ots.bp_ots import create as _Ots, _validate
from syndrilla.decoder.knobs import reject_unknown_keys

_EXT = None


def _load_ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        _load_ext._dll_handles = _dll_dirs()
        _EXT = load(name="bp_ots_prior_cuda_ext",
            sources=[str(Path(__file__).resolve().parents[1] / "cuda" / "bp_ots_kernel.cu")],
            extra_cuda_cflags=["-O3"], verbose=False)
    return _EXT


class _Kernels:
    """Replace only CN arithmetic; retain base VN, hard, syndrome and cap kernels."""

    def __init__(self, base, owner):
        self.base, self.owner = base, weakref.proxy(owner)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def vn_cn_update_csr(self, values, sign, row_ptr, col, incoming, active,
                         beta, n, **kwargs):
        self.owner._ots_ext.spa_cn(values, sign, row_ptr, col, incoming, active, n,
            self.owner.V_c_col.shape[1], kwargs.get('warp', True), kwargs.get('from_lv', True))

    def llr_hard_update_csr(self, prior, incoming, adjacency, *args):
        self.owner._ots_prior, self.owner._ots_edges = prior, incoming
        self.owner._ots_adjacency = adjacency
        return self.base.llr_hard_update_csr(prior, incoming, adjacency, *args)

    def llr_hard_update_atomic(self, prior, incoming, col, *args):
        self.owner._ots_prior, self.owner._ots_edges = prior, incoming
        self.owner._ots_adjacency = (self.owner.VN_eid_pad if self.owner._padded
                                     else self.owner.VN_eid)
        return self.base.llr_hard_update_atomic(prior, incoming, col, *args)


class create(_CudaBP):
    """Reuse the modular CUDA BP loop and all applicable optimization groups."""

    _config_keys = _Ots._config_keys
    _config_algorithm = "bp_ots"

    def __init__(self, cfg, **kwargs):
        reject_unknown_keys(
            cfg, self._base_config_keys.union(self._config_keys),
            cfg.get("algorithm", self._config_algorithm),
        )
        interval, strength = _validate(cfg)
        super().__init__(cfg, **kwargs)
        self.interval, self.strength = interval, strength
        self._ots_ext = _load_ext()
        self._ext = _Kernels(self._ext, self)
        self.algo = 'bp_ots'

    def _kernel_knobs(self):
        # The base treats quantizer proxies as fixed-default kernels. OTS supports
        # both layouts and the variable/update alternatives, so retain the flags.
        return (self._warp_per_check, self._fuse_vn, self._f64_int,
                self._padded, self._vn_gather)

    def forward(self, io_dict):
        base = io_dict['llr0'].to(self.device, self.dtype).contiguous()
        if base.ndim != 2 or base.shape[1] != self.H_shape[1]:
            raise ValueError('llr0 must have shape [batch, variables]')
        if io_dict['synd'].shape != (len(base), self.H_shape[0]):
            raise ValueError('synd must match the batch and check dimensions')
        self.cap_active_last = False
        self._ots_channel = base.clone()
        self._ots_counts = torch.zeros_like(base, dtype=torch.int64)
        self._ots_previous = base <= 0
        self._ots_current = base.clone()
        self._hook_rows = torch.arange(len(base), device=self.device)
        self.last_ots_nodes = None
        self._ots_prior = self._ots_edges = self._ots_incoming = None
        if not len(base):
            io_dict.update(llr=base.clone(), e_v=base.clone(),
                iter=torch.empty(0, device=self.device, dtype=torch.long),
                converge=torch.empty(0, device=self.device, dtype=torch.long),
                defer=torch.empty(0, device=self.device, dtype=torch.bool))
            return io_dict
        try:
            return super().forward(io_dict)
        finally:
            self._ots_prior = self._ots_edges = self._ots_incoming = None

    _iter_hook = _Ots._iter_hook

    def _switch_priors(self, rows, l_v, active):
        # The BP loop owns this prior and edge buffer. Keep its messages and
        # replace only selected priors, then recompute in the same VN order.
        adjacency = self._ots_adjacency.long()
        incoming = self._ots_edges[:, adjacency].permute(0, 2, 1).contiguous()
        prior = self._ots_current.contiguous()
        self.last_ots_nodes = self._ots_ext.switch_priors(l_v, prior, self._ots_channel,
            self._ots_counts, active.contiguous(), incoming, self.strength)
        self._ots_prior[:, :-1].copy_(prior)
