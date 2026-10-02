import functools
import types

import torch

from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum import create as _NmsPy
from syndrilla.utils import fp2fxp


def _cn_quant(a, syndrome_odd, beta, mask_dummy, iw, fw):
    """Quantized check-node update on the [batch, n_checks, degree] v->c messages
    `a` (already rounded): magnitude fp2fxp(beta) * fp2fxp(minimum |a| over the
    other edges), sign from the edge's own sign bit, the parity of non-positive
    inputs on the check and the syndrome bit, dummy slots 0, then the signed
    message rounded with fp2fxp. `beta` is already rounded. Functional."""
    neg = a <= 0.0
    parity = (neg.sum(dim=2, keepdim=True, dtype=torch.uint8) & 1).bool()
    flip = neg ^ (parity ^ syndrome_odd)
    mag = a.abs()
    min_0, arg_0 = mag.min(dim=2, keepdim=True)
    min_1 = mag.scatter(2, arg_0, float("inf")).amin(dim=2, keepdim=True)
    m_0 = beta * fp2fxp(min_0, iw, fw)
    m_1 = beta * fp2fxp(min_1, iw, fw)
    msg = torch.where(flip, -m_0, m_0)
    msg = msg.scatter(2, arg_0, torch.where(flip.gather(2, arg_0), -m_1, m_1))
    return fp2fxp(msg.masked_fill(mask_dummy, 0.0), iw, fw)


def _step(l_v, c2v_prev, u_init, syndrome_odd, beta, col, vn_adj, mask_dummy, iw, fw):
    """bp_norm_min_sum's compiled iteration body with the quantized check-node
    update: v->c message fp2fxp(l_v - c2v), then _cn_quant. The c2v sum, LLR
    update, hard decision and syndrome are bp_norm_min_sum's. On iteration 1 the
    rounding of u_init (already rounded) changes only the dummy slots (+inf to the
    largest value), which never changes the rounded minimum."""
    B, N1 = l_v.shape
    M, D = mask_dummy.shape
    a = fp2fxp(l_v[:, col].view(B, M, D) - c2v_prev, iw, fw)
    msg = _cn_quant(a, syndrome_odd, beta, mask_dummy, iw, fw)
    flat = torch.cat([msg.view(B, -1), msg.new_zeros(B, 1)], 1)
    g = flat[:, vn_adj].view(B, -1, N1)
    s = torch.zeros_like(l_v)
    for k in range(g.shape[1]):
        s = s + g[:, k]
    l_new = s + u_init
    e = l_new <= 0.0
    s_est = e.to(torch.uint8)[:, col].view(B, M, D).sum(dim=2, dtype=torch.uint8) & 1
    return msg, l_new, e.to(l_v.dtype), s_est.to(l_v.dtype)


class create(_NmsPy):
    """
    Quantized BP normalized min-sum: bp_norm_min_sum (eager or compiled step, row
    compaction, hooks) with fixed-point rounding fp2fxp (floor, saturating) of
    Q(int_width).(frac_width) on the channel LLR, the v->c message, beta, the
    check-node minimum, the c->v message and the returned llr. The posterior LLR
    is a sum of rounded values, so its hard decision equals that of its rounded
    value.

    Accepts every bp_norm_min_sum key plus
        int_width : int (default 3)   integer bits of the fixed-point format
        frac_width: int (default 4)   fractional bits
    """

    def __init__(self, decoding_cfg, **kwargs) -> None:
        super().__init__(decoding_cfg, **kwargs)
        self.intwidth = decoding_cfg.get("int_width", 3)
        self.fracwidth = decoding_cfg.get("frac_width", 4)
        self.algo = "bp_norm_min_sum_quant"
        if self.compile:
            step = types.FunctionType(
                _step.__code__.replace(), _step.__globals__, _step.__name__
            )
            self._step = functools.partial(
                torch.compile(step), iw=self.intwidth, fw=self.fracwidth
            )
            self.betas = self._q(self.betas)

    def _q(self, t):
        return fp2fxp(t, self.intwidth, self.fracwidth)

    def forward(self, io_dict):
        """bp_norm_min_sum decoding of the rounded channel LLR; io_dict keeps its
        own llr0."""
        llr0 = self._q(io_dict["llr0"].to(self.device).to(self.dtype))
        out = super().forward({**io_dict, "llr0": llr0})
        io_dict.update({k: out[k] for k in ("e_v", "iter", "llr", "converge")})
        return io_dict

    def _exit_hook(self, l_v, e_v, num_iters, converges) -> None:
        """Rounds the returned llr."""
        l_v.copy_(self._q(l_v))
        super()._exit_hook(l_v, e_v, num_iters, converges)

    def vn_update(self, b_c2v, l_v_v2c):
        """Eager v->c message: the initial message on iteration 1, afterwards
        fp2fxp(l_v - c2v)."""
        a = super().vn_update(b_c2v, l_v_v2c)
        return a if self.i == 1 else self._q(a)

    def cn_update(self, a_v2c, syndrome_odd, out):
        """Eager quantized check-node update (_cn_quant), written into `out`."""
        beta = self._q(
            torch.tensor(1.0, dtype=self.dtype)
            - torch.pow(
                torch.tensor(2.0, dtype=self.dtype),
                torch.tensor(-self.i, dtype=self.dtype),
            )
        )
        msg = _cn_quant(
            a_v2c, syndrome_odd, beta, self.mask_dummy, self.intwidth, self.fracwidth
        )
        return out.copy_(msg)
