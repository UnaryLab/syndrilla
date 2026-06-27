"""
bp_norm_min_sum_quant_cuda.py — CUDA port of bp_norm_min_sum_quant (quantized NMS).

The quantized normalized-min-sum is the SAME recursion as bp_norm_min_sum with a
fixed-point round (``fp2fxp``) inserted at four points. ``fp2fxp`` is a pure-PyTorch
op, and quantization is idempotent on the per-step kernels' outputs (sums and minima
of fixed-point values stay fixed-point at float64), so this decoder reuses
bp_norm_min_sum_cuda's compiled per-step kernels UNCHANGED and applies the rounding
in PyTorch between kernel calls:

  * u_init quantized once at init,
  * a_v2c quantized after vn_update (i > 1),
  * beta pre-quantized and passed to cn_update; b_c2v quantized after cn_update,
  * l_v is already fixed-point (u_init_q + Σ b_c2v_q), so hard_decision matches.

At float64 this reproduces bp_norm_min_sum_quant bit-for-bit. The per-iteration
host rounding rules out the fused kernel, so the per-step path is always used.

YAML algorithm key: bp_norm_min_sum_quant_cuda
"""

import torch
from loguru import logger

from syndrilla.utils import fp2fxp
from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum_cuda import create as _BaseCuda


class create(_BaseCuda):
    """Quantized BP Normalized Min-Sum on CUDA kernels (per-step path).

    Accepts every bp_norm_min_sum_cuda key plus:
        int_width  : int (default 3)   integer bits of the fixed-point format
        frac_width : int (default 4)   fractional bits
    """

    def __init__(self, decoder_cfg: dict, **kwargs) -> None:
        super().__init__(decoder_cfg, **kwargs)
        self._use_fused = False  # per-iteration host rounding → no fused kernel
        self.intwidth = decoder_cfg.get("int_width", 3)
        self.fracwidth = decoder_cfg.get("frac_width", 4)
        self.algo = "bp_norm_min_sum_quant_cuda"
        logger.info(
            f"bp_norm_min_sum_quant_cuda ready (per-step, Q{self.intwidth}.{self.fracwidth})."
        )

    def _q(self, t):
        return fp2fxp(t, self.intwidth, self.fracwidth)

    def forward(self, io_dict: dict) -> dict:
        dev = self.device
        syndrome = io_dict["synd"].to(dtype=self.dtype, device=dev).contiguous()
        B, M = syndrome.shape
        self.batch_size = B

        # quantize the channel LLR once, append the +inf dummy column.
        llr0 = io_dict["llr0"].to(dtype=self.dtype, device=dev).contiguous()
        dummy_col = torch.full((B, 1), float("inf"), dtype=self.dtype, device=dev)
        u_init = torch.cat([self._q(llr0), dummy_col], dim=1).contiguous()

        syndrome_neg_bc = torch.where(
            syndrome == 0.0, torch.ones_like(syndrome), -torch.ones_like(syndrome)
        )

        e_out = torch.zeros(B, self.N_ext, dtype=self.dtype, device=dev)
        l_out = torch.zeros(B, self.N_ext, dtype=self.dtype, device=dev)
        num_iters = torch.full((B,), -1, dtype=torch.int64, device=dev)
        converges = torch.zeros(B, dtype=torch.int64, device=dev)

        cap = getattr(self, "cap", None)
        self.cap_active_last = bool(
            cap is not None and cap.done and not getattr(self, "cap_bypass", False)
        )
        cap_frac = cap.frac if self.cap_active_last else None

        D = int(self.V_c_col.shape[1])
        a_v2c = torch.zeros(B, M, D, dtype=self.dtype, device=dev)
        b_c2v = torch.zeros(B, M, D, dtype=self.dtype, device=dev)
        l_v = torch.zeros(B, self.N_ext, dtype=self.dtype, device=dev)
        e_v = torch.zeros(B, self.N_ext, dtype=self.dtype, device=dev)
        s_est = torch.zeros(B, M, dtype=self.dtype, device=dev)

        for i in range(1, self.max_iter + 1):
            # pre-quantized beta (matches fp2fxp(beta) in the reference cn_update)
            beta = float(self._q(torch.tensor(1.0 - 2.0 ** (-i), dtype=self.dtype)))
            if i == 1:
                # a_v2c = u_init[V_c_col] — already quantized, no extra rounding
                self._ext.init_messages(u_init, self.V_c_col, a_v2c)
            else:
                self._ext.vn_update(l_v, b_c2v, self.V_c_col, a_v2c, self.N)
                a_v2c = self._q(a_v2c).contiguous()  # fp2fxp(l_v_v2c - b_c2v)
            self._ext.cn_update(
                a_v2c, syndrome_neg_bc, self.V_c_col, b_c2v, beta, self.N
            )
            b_c2v = self._q(b_c2v).contiguous()  # fp2fxp(message)
            self._ext.llr_update(
                u_init, b_c2v, self.VN_adj_c, self.VN_adj_k, l_v, self.VD
            )
            l_v[:, -1] = float("inf")
            self._ext.hard_decision(l_v, e_v)  # l_v already fixed-point
            self._ext.syndrome_est(e_v, self.V_c_col, s_est, self.N)
            self._ext.convergence_update(
                s_est, syndrome, e_v, l_v, e_out, l_out, num_iters, converges, i
            )

            n_conv = int((num_iters != -1).sum())
            if n_conv == B:
                break
            if cap_frac is not None and n_conv >= cap_frac * B:
                break

        not_conv = num_iters == -1
        if not_conv.any().item():
            e_out[not_conv] = e_v[not_conv]
            l_out[not_conv] = l_v[not_conv]
            num_iters[not_conv] = (
                i  # actual stop iter (== max_iter unless the cap broke early)
            )

        if cap is not None and not cap.done and not getattr(self, "cap_bypass", False):
            cap.observe(num_iters, self.max_iter, B)

        io_dict.update(
            {
                "e_v": e_out[:, :-1],
                "iter": num_iters,
                "llr": l_out[:, :-1],
                "converge": converges,
            }
        )
        return io_dict
