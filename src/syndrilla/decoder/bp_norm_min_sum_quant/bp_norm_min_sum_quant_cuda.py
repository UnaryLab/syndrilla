import torch
from loguru import logger

from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum_cuda import (
    PERSISTENT_THREADS,
    create as _BaseCuda,
)
from syndrilla.utils import fp2fxp


class _QuantExt:
    """The bp_kernel extension with the fixed-point rounding on: every attribute
    is the extension's, and the two CSR decode calls (vn_cn_update_csr,
    bp_nms_persistent) get int_width and frac_width appended."""

    def __init__(self, ext, int_width, frac_width):
        self._ext, self._q = ext, (int_width, frac_width)

    def __getattr__(self, name):
        return getattr(self._ext, name)

    def vn_cn_update_csr(self, *args):
        return self._ext.vn_cn_update_csr(*args, *self._q)

    def bp_nms_persistent(self, *args):
        return self._ext.bp_nms_persistent(*args, *self._q)


class create(_BaseCuda):
    """Quantized BP normalized min-sum on CUDA: the bp_norm_min_sum_cuda per-step
    and persistent paths with the rounding of bp_norm_min_sum_quant. The channel
    LLR is rounded before decoding, cn_row rounds the v->c message, beta, the
    check-node minimum and the c->v message (fp2fxp: floor, saturating), and
    _exit_hook rounds the returned llr.

    Accepts every bp_norm_min_sum_cuda key plus:
        int_width  : int (default 3)   integer bits of the fixed-point format
        frac_width : int (default 4)   fractional bits
    """

    def __init__(self, decoding_cfg: dict, **kwargs) -> None:
        super().__init__(decoding_cfg, **kwargs)
        self.intwidth = decoding_cfg.get("int_width", 3)
        self.fracwidth = decoding_cfg.get("frac_width", 4)
        with torch.cuda.device(self.device):
            self._max_coresident = self._ext.persistent_max_blocks(
                torch.empty(0, dtype=self.dtype, device=self.device),
                PERSISTENT_THREADS,
                True,
            )
        self._ext = _QuantExt(self._ext, self.intwidth, self.fracwidth)
        self.algo = "bp_norm_min_sum_quant"
        logger.info(
            f"bp_norm_min_sum_quant_cuda ready (Q{self.intwidth}.{self.fracwidth})."
        )

    def _q(self, t):
        return fp2fxp(t, self.intwidth, self.fracwidth)

    def forward(self, io_dict: dict) -> dict:
        """bp_norm_min_sum_cuda decoding of the rounded channel LLR; io_dict keeps
        its own llr0."""
        llr0 = self._q(io_dict["llr0"].to(dtype=self.dtype, device=self.device))
        out = super().forward({**io_dict, "llr0": llr0})
        io_dict.update({k: out[k] for k in ("e_v", "iter", "llr", "converge")})
        return io_dict

    def _exit_hook(self, l_v, e_v, num_iters, converges) -> None:
        """Rounds the returned llr."""
        l_v.copy_(self._q(l_v))
        super()._exit_hook(l_v, e_v, num_iters, converges)

