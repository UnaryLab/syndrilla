from pathlib import Path

import torch

from syndrilla.decoder.bp_norm_min_sum_layered.bp_norm_min_sum_layered import create as _Layered
from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum_cuda import _load_ext

_EXT = None


class create(_Layered):
    def __init__(self, decoding_cfg, **kwargs):
        super().__init__(decoding_cfg, **kwargs)
        if self.device.type != "cuda":
            raise ValueError("bp_norm_min_sum_layered_cuda requires a CUDA device")
        self._bp_ext = _load_ext()
        global _EXT
        if _EXT is None:
            from torch.utils.cpp_extension import load

            _EXT = load(name="bp_norm_min_sum_layered_cuda_ext",
                        sources=[str(Path(__file__).resolve().parents[1] / "cuda" / "bp_norm_min_sum_layered_kernel.cu")],
                        extra_cuda_cflags=["-O3", "--fmad=false"], verbose=True)
        self._ext = _EXT
        self.col_pad = self.V_c_col.detach().to(torch.int32).contiguous()
        # The reused padded syndrome kernel ignores this CSR row-pointer placeholder.
        self.row_ptr = torch.empty(self.H_shape[0] + 1, device=self.device, dtype=torch.int32)

    def _sweep(self, l_v, previous, syndrome):
        beta = float(1 - torch.pow(torch.tensor(2.0, dtype=self.dtype),
                                  torch.tensor(-self.i, dtype=self.dtype)))
        for layer in self.layers:
            self._ext.layer_update(l_v, previous, syndrome, self.V_c_col.detach(), layer, beta)

    def _record_convergence(self, e_v, syndrome, it_rows):
        mismatch = torch.zeros(len(it_rows), device=self.device, dtype=torch.int32)
        run = it_rows if self.early_termination else torch.full_like(it_rows, -1)
        self._bp_ext.syndrome_check_csr(e_v.to(torch.uint8).contiguous(), self.row_ptr,
                                       self.col_pad, syndrome.contiguous(), run, mismatch,
                                       self.H_shape[1])
        matches = mismatch == 0
        if self.early_termination:
            converges = torch.zeros_like(it_rows)
            self._bp_ext.convergence_flag_update(mismatch, it_rows, converges, self.i)
        return matches
