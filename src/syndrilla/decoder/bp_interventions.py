"""Shared CUDA plumbing for three independent binary syndrome decoders."""

import math
from pathlib import Path

import torch
from torch import nn

from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum_cuda import (
    _build_csr,
    _build_vn_adj,
    _dll_dirs,
)

_EXT = None


def _load_ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        _load_ext._dll_handles = _dll_dirs()
        _EXT = load(
            name="bp_interventions_cuda_ext",
            sources=[str(Path(__file__).parent / "cuda" / "bp_interventions_kernel.cu")],
            extra_cuda_cflags=["-O3"],
            verbose=True,
        )
    return _EXT


class BinaryCudaDecoder(nn.Module):
    mode = None

    def __init__(self, decoding_cfg, **kwargs):
        super().__init__()
        if not torch.cuda.is_available() or decoding_cfg.get("device", {}).get(
            "device_type", "cuda"
        ) != "cuda":
            raise RuntimeError(f"{self.algo} requires a CUDA device")
        self.device = torch.device(
            "cuda", decoding_cfg.get("device", {}).get("device_idx", 0)
        )
        dtype = decoding_cfg.get("dtype", "float64")
        if dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        self.dtype = getattr(torch, dtype)
        self.max_iter = decoding_cfg.get("max_iter", 50)
        if type(self.max_iter) is not int or not 0 < self.max_iter < 2**31:
            raise ValueError("max_iter must be a positive int below 2^31")
        self.num_max_iter = self.max_iter
        self.check_type = decoding_cfg.get("check_type", "hx")
        if self.check_type not in {"hx", "hz"}:
            raise ValueError("check_type must be hx or hz")
        self.alpha = float(decoding_cfg.get("alpha", 1.0)) if self.mode == 0 else 1.0
        self.interval = decoding_cfg.get("interval", 10) if self.mode == 1 else 1
        self.strength = float(decoding_cfg.get("strength", 10.0)) if self.mode == 1 else 1.0
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if type(self.interval) is not int or not 0 < self.interval < 2**31:
            raise ValueError("interval must be a positive int below 2^31")
        if not math.isfinite(self.strength) or self.strength <= 0:
            raise ValueError("strength must be finite and positive")
        bundle = kwargs.get("bundle")
        if bundle is None:
            raise ValueError(f"{self.algo} requires a MatrixBundle")
        for name in ("Hx_matrix", "Hz_matrix", "lx_matrix", "lz_matrix"):
            setattr(self, name, getattr(bundle, name))
        self.H_shape, padded_row, padded_col, _ = bundle.select(self.check_type)
        self.register_buffer("V_c_row", padded_row.to(self.device))
        self.register_buffer("V_c_col", padded_col.to(self.device))
        self.N = self.H_shape[1]
        if self.N <= 0:
            raise ValueError("graph must have at least one variable")
        adj_c, adj_k, _ = _build_vn_adj(padded_col.cpu().numpy(), self.N)
        row, col, vn = _build_csr(padded_col.cpu().numpy(), self.N, adj_c, adj_k)
        for name, value in (("row_ptr", row), ("col", col), ("VN_eid", vn)):
            self.register_buffer(name, torch.from_numpy(value).to(self.device))
        self.cap = None
        self.cap_bypass = False
        self.cap_active_last = False
        self._ext = _load_ext()

    @torch.no_grad()
    def forward(self, io_dict):
        base = io_dict["llr0"].to(self.device, self.dtype).contiguous()
        syndrome = io_dict["synd"].to(self.device, torch.uint8).contiguous()
        if base.ndim != 2 or base.shape[1] != self.N or syndrome.shape != (
            base.shape[0], self.H_shape[0]
        ):
            raise ValueError("llr0 and synd must have matching batch and graph dimensions")
        llr, hard, iterations, converged = self._ext.decode(
            base, syndrome, self.row_ptr, self.col, self.VN_eid,
            self.max_iter, self.mode, self.alpha, self.interval, self.strength,
        )
        io_dict.update(llr=llr, e_v=hard.to(self.dtype), iter=iterations, converge=converged,
                       defer=torch.zeros_like(converged, dtype=torch.bool))
        return io_dict
