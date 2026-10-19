"""CUDA kernels for the torch SymBreak study rules, arXiv:2412.02885.

Check messages and split graphs are shot-local. Variable sums visit incident
edges in increasing slot order, matching deterministic CPU scatter addition.
The shared NMS kernels require one graph for the batch, so they cannot update
these per-shot graphs. Validation and configuration come from the torch class.
"""
import math
import os
from pathlib import Path

import torch

from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum_cuda import _dll_dirs
from syndrilla.decoder.knobs import knob
from syndrilla.decoder.bp_symbreak.bp_symbreak import create as TorchSymBreak

_EXT = None


def _load_ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        if not torch.cuda.is_available():
            raise RuntimeError('bp_symbreak CUDA kernels require a CUDA GPU')
        if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
            major, minor = torch.cuda.get_device_capability()
            os.environ['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}'
        _load_ext._dll_handles = _dll_dirs()
        _EXT = load(name='bp_symbreak_cuda_ext',
            sources=[str(Path(__file__).resolve().parents[1] / 'cuda/bp_symbreak_kernel.cu')],
            extra_cuda_cflags=['-O3', '--fmad=false'] +
                (['-Xcompiler', '/Zc:preprocessor'] if os.name == 'nt' else []),
            extra_cflags=['/Zc:preprocessor'] if os.name == 'nt' else [], verbose=False)
    return _EXT


class create(TorchSymBreak):
    def __init__(self, decoding_cfg, **kwargs):
        super().__init__(decoding_cfg, **kwargs)
        if self.device.type != 'cuda':
            raise RuntimeError('bp_symbreak CUDA kernels require device_type cuda')
        self._ext = _load_ext()
        self.vn_gather = bool(knob(decoding_cfg, "vn_gather", self.c2v_gather))
        n = self.H_shape[1]
        adjacency = self.vn_adj.reshape(-1, n + 1).t().contiguous()
        self.register_buffer('_incident', torch.where(adjacency < self.V_c_col.numel(),
            adjacency, -1))

    @torch.no_grad()
    def forward(self, io_dict):
        original = io_dict['synd'].to(device=self.device, dtype=self.dtype).contiguous()
        channel = io_dict['llr0'].to(device=self.device, dtype=self.dtype).contiguous()
        b, n = channel.shape
        m, degree = self.V_c_col.shape
        if original.shape != (b, m) or n != self.H_shape[1]:
            raise ValueError('synd and llr0 must match the matrix and batch shapes')
        if not torch.isfinite(channel).all():
            raise ValueError('llr0 must contain finite channel LLRs')
        self.batch_size = b
        checks = m + min(self.max_splits, m)
        col = torch.full((b, checks, degree), n, device=self.device, dtype=torch.long)
        col[:, :m] = self.V_c_col
        syndrome = torch.zeros((b, checks), device=self.device, dtype=self.dtype)
        syndrome[:, :m] = original
        prior = torch.cat((channel, channel.new_full((b, 1), float('inf'))), 1)
        posterior = prior.clone()
        incoming = torch.zeros_like(col, dtype=self.dtype)
        reset = torch.zeros_like(col, dtype=torch.bool)
        incident = self._incident.expand(b, -1, -1).clone()
        hard = torch.zeros_like(prior, dtype=torch.uint8)
        active = torch.ones(b, device=self.device, dtype=torch.bool)
        splits = torch.zeros(b, device=self.device, dtype=torch.long)
        iterations, converges, mismatch, monitor = (torch.zeros_like(splits) for _ in range(4))
        previous = torch.full_like(splits, -1)
        keep = torch.empty((b, degree), dtype=torch.bool, device=self.device)
        splittable = torch.zeros((b, checks), dtype=torch.bool, device=self.device)
        splittable[:, :m] = True
        state = [col, syndrome, prior, posterior, incoming, reset, incident, hard,
                 active, splits, iterations, converges, mismatch, monitor, previous,
                 original, keep, splittable]
        limit = math.sqrt(torch.finfo(self.dtype).max) / (col.shape[1] * degree + 1)
        cap_frac = (self.cap.frac if self.cap is not None and self.cap.done
                    and not self.cap_bypass else None)
        self.i, cap_stopped = 0, False
        rows = torch.arange(b, device=self.device)
        full_lv, full_hard = posterior, hard
        full_iter, full_conv, full_splits = iterations, converges, splits
        def save_outputs():
            for destination, index in [(full_lv, 3), (full_hard, 7), (full_iter, 10),
                                       (full_conv, 11), (full_splits, 9)]:
                if destination.data_ptr() != state[index].data_ptr():
                    destination[rows] = state[index]
        if b:
            for self.i in range(1, self.max_iter + 1):
                left = state[8].sum()
                if self.early_termination and 0 < left < self.compact_frac * len(rows):
                    save_outputs()
                    keep_rows = state[8].nonzero().flatten()
                    rows = rows[keep_rows]
                    state = [value[keep_rows].contiguous() for value in state]
                if not self.reuse_buffers:
                    state[3], state[4] = state[3].clone(), state[4].clone()
                boundary = self.i % self.bp_iters == 0 or self.i == self.max_iter
                self._ext.step(state, self.V_c_col, 1 - 2. ** (-self.i), limit,
                               self.i, self.early_termination, self.vn_gather,
                               self.cn_sign_parity, boundary)
                if not boundary:
                    continue
                if self.early_termination and cap_frac is not None:
                    if full_conv.data_ptr() != state[11].data_ptr():
                        full_conv[rows] = state[11]
                    if int(full_conv.sum()) >= cap_frac * b:
                        cap_stopped = bool(state[8].any() and self.i < self.max_iter)
                        break
                if self.i % self.bp_iters == 0 and self.i != self.max_iter:
                    self._ext.split(state, self.opposite_col, self.k_max,
                                    self.max_splits, self.assign == 'syndrome', self.early_termination)
                if self.early_termination and not state[8].any():
                    break
            save_outputs()
        posterior, hard = full_lv, full_hard
        iterations, converges, splits = full_iter, full_conv, full_splits
        self.cap_active_last = cap_stopped
        defer = (converges == 0) if cap_stopped else torch.zeros_like(active)
        if self.cap is not None and not self.cap.done and not self.cap_bypass and b:
            self.cap.observe(iterations, self.max_iter, b)
        io_dict.update(e_v=hard[:, :n].to(self.dtype), llr=posterior[:, :n],
                       iter=iterations, converge=converges, defer=defer, splits=splits)
        return io_dict
