"""Guided decimation with the modular CUDA sum-product BP kernels.

Hanwen Yao et al., "Belief Propagation Decoding of Quantum LDPC Codes
with Guided Decimation," Algorithm 1 and Section 4.3, arXiv:2312.10950.
https://arxiv.org/abs/2312.10950
The torch implementation documents the deterministic choices the paper leaves
open. This module shares that round loop and decimation rule unchanged.
"""

import torch

from syndrilla.decoder.bp_gd.bp_gd import create as _Gd, _validate
from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum_cuda import create as _CudaBP
from syndrilla.decoder.bp_ots.bp_ots_cuda import _load_ext as _load_spa_ext
from syndrilla.decoder.knobs import knob, reject_unknown_keys


class create(_CudaBP):
    """Retain check messages across GD rounds and edit priors with torch ops."""

    _config_keys = _Gd._config_keys
    _config_algorithm = "bp_gd"

    def __init__(self, cfg, **kwargs):
        reject_unknown_keys(cfg, self._base_config_keys.union(self._config_keys),
                            cfg.get("algorithm", self._config_algorithm))
        _validate(cfg)
        super().__init__(cfg, **kwargs)
        self._spa_ext = _load_spa_ext()
        self.compact_frac = float(knob(cfg, "compact_frac", .75))
        if not 0 <= self.compact_frac <= .75:
            raise ValueError("compact_frac must be in [0, 0.75]")
        self.reuse_buffers = bool(knob(cfg, "reuse_buffers", True))
        self.compile = False
        self._init_gd(cfg)
        self.algo = "bp_gd"

    _init_gd = _Gd._init_gd
    forward = _Gd.forward
    _decimate = _Gd._decimate
    _random_values = _Gd._random_values

    def _new_messages(self, batch):
        edges = self.col_pad.numel() if self._padded else self.nnz
        return torch.zeros((batch, edges + 1), device=self.device, dtype=self.dtype)

    def _bp_step(self, messages, posterior, prior, syndrome_odd):
        batch = len(posterior)
        run = torch.full((batch,), -1, device=self.device, dtype=torch.long)
        if not self.reuse_buffers:
            messages = messages.clone()
        col = self.col_pad if self._padded else self.col
        adjacency = self.VN_eid_pad if self._padded else self.VN_eid
        sign = (1 - 2 * syndrome_odd.reshape(batch, self.H_shape[0]).to(self.dtype)).contiguous()
        values = posterior.contiguous()
        if not self._fuse_vn:
            values = torch.empty_like(messages)
            self._ext.vn_update_csr(posterior.contiguous(), messages, col, values, run, self.N)
        self._spa_ext.spa_cn(values, sign, self.row_ptr, col, messages, run,
            self.N, self.V_c_col.shape[1], self._warp_per_check, self._fuse_vn, self.saturation)
        new_posterior = torch.empty_like(posterior)
        hard = torch.empty_like(posterior, dtype=torch.uint8)
        if self._vn_gather:
            self._ext.llr_hard_update_csr(prior.contiguous(), messages, adjacency,
                                         new_posterior, hard, run)
        else:
            self._ext.llr_hard_update_atomic(prior.contiguous(), messages, col,
                torch.empty_like(posterior), new_posterior, hard, run)
        estimate = (hard[:, self.V_c_col].sum(2, dtype=torch.uint8) & 1).to(self.dtype)
        return messages, new_posterior, hard.to(self.dtype), estimate

    def _posterior_from_messages(self, prior, messages, out):
        adjacency = (self.VN_eid_pad if self._padded else self.VN_eid).long()
        out.copy_(prior)
        for slot in adjacency.unbind(1):
            out.add_(messages[:, slot])
        return out
