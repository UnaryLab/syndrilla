"""
bp_lottery_quant_cuda.py — CUDA port of bp_lottery_quant (quantized NMS + sign-flip).

Combines the quantized per-step decode of bp_norm_min_sum_quant_cuda with the simple
lottery sign-flip applied every iteration (on a quantized Sobol stream). Reuses
bp_norm_min_sum_cuda's compiled kernels UNCHANGED — quantization (fp2fxp) and the
sign-flip both run in PyTorch between kernel calls — so at float64 it reproduces
bp_lottery_quant bit-for-bit.

Implementation reuses both parents via multiple inheritance:
  * bp_norm_min_sum_quant_cuda — quantized per-step kernel machinery and ``_q``.
  * bp_lottery_quant            — the simple ``sign_flip`` (depends only on
    self.H_matrix / self.r / self.i / self.batch_size).

YAML algorithm key: bp_lottery_quant_cuda
"""

import torch
from loguru import logger

from syndrilla.decoder.bp_lottery_quant.bp_lottery_quant import (
    create as _LotteryQuantPy,
)
from syndrilla.decoder.bp_norm_min_sum_quant.bp_norm_min_sum_quant_cuda import (
    create as _QuantCuda,
)


class create(_QuantCuda, _LotteryQuantPy):
    """Quantized lottery NMS on CUDA kernels (per-step path + sign-flip)."""

    def __init__(self, decoder_cfg: dict, **kwargs) -> None:
        _QuantCuda.__init__(
            self, decoder_cfg, **kwargs
        )  # quant + kernels; methods from _LotteryQuantPy

        self.random_machine = str(decoder_cfg.get("random_machine", "sobol")).lower()
        if self.random_machine not in {"sobol", "system"}:
            logger.warning(
                f"Invalid random_machine <{self.random_machine}>; defaulting to sobol."
            )
            self.random_machine = "sobol"

        bundle = kwargs.get("bundle")
        _, _, _, H_matrix = bundle.select(self.check_type)
        self.H_matrix = H_matrix.to(self.device, self.dtype)

        self.algo = "bp_lottery_quant"
        logger.info(
            f"bp_lottery_quant_cuda ready (per-step, Q{self.intwidth}.{self.fracwidth} + sign-flip)."
        )

    def forward(self, io_dict: dict) -> dict:
        dev = self.device
        syndrome = io_dict["synd"].to(dtype=self.dtype, device=dev).contiguous()
        B, M = syndrome.shape
        self.batch_size = B

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

        # quantized Sobol stream for the sign-flip (matches bp_lottery_quant)
        if self.random_machine == "sobol":
            sobol = torch.quasirandom.SobolEngine(dimension=1, scramble=False)
            draw_dtype = (
                self.dtype
                if self.dtype in {torch.float32, torch.float64}
                else torch.float32
            )
            self.r = self._q(
                sobol.draw(self.max_iter, dtype=draw_dtype).to(dev, self.dtype)
            )

        D = int(self.V_c_col.shape[1])
        a_v2c = torch.zeros(B, M, D, dtype=self.dtype, device=dev)
        b_c2v = torch.zeros(B, M, D, dtype=self.dtype, device=dev)
        l_v = torch.zeros(B, self.N_ext, dtype=self.dtype, device=dev)
        e_v = torch.zeros(B, self.N_ext, dtype=self.dtype, device=dev)
        s_est = torch.zeros(B, M, dtype=self.dtype, device=dev)

        for i in range(1, self.max_iter + 1):
            self.i = i
            beta = float(self._q(torch.tensor(1.0 - 2.0 ** (-i), dtype=self.dtype)))
            if i == 1:
                self._ext.init_messages(u_init, self.V_c_col, a_v2c)
            else:
                self._ext.vn_update(l_v, b_c2v, self.V_c_col, a_v2c, self.N)
                a_v2c = self._q(a_v2c).contiguous()
            self._ext.cn_update(
                a_v2c, syndrome_neg_bc, self.V_c_col, b_c2v, beta, self.N
            )
            b_c2v = self._q(b_c2v).contiguous()
            self._ext.llr_update(
                u_init, b_c2v, self.VN_adj_c, self.VN_adj_k, l_v, self.VD
            )
            l_v[:, -1] = float("inf")
            self._ext.hard_decision(l_v, e_v)
            self._ext.syndrome_est(e_v, self.V_c_col, s_est, self.N)
            self._ext.convergence_update(
                s_est, syndrome, e_v, l_v, e_out, l_out, num_iters, converges, i
            )

            n_conv = int((num_iters != -1).sum())
            if n_conv == B:
                break
            if cap_frac is not None and n_conv >= cap_frac * B:
                break

            l_v = self.sign_flip(syndrome, s_est, l_v)  # inherited, every iteration

        not_conv = num_iters == -1
        if not_conv.any().item():
            e_out[not_conv] = e_v[not_conv]
            l_out[not_conv] = l_v[not_conv]
            num_iters[not_conv] = (
                self.i
            )  # actual stop iter (== max_iter unless the cap broke early)

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
