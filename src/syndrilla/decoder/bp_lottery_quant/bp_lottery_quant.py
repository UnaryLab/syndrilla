import torch

from syndrilla.decoder.bp_lottery.bp_lottery import (
    create as _LotteryPy,
    flip_rows,
    is_flip_iter,
    rand_rows,
    select_flip_vn,
)
from syndrilla.decoder.bp_norm_min_sum_quant.bp_norm_min_sum_quant import (
    create as _QuantPy,
)


class create(_LotteryPy, _QuantPy):
    """
    Quantized lottery BP: bp_norm_min_sum_quant (rounding, eager or compiled step,
    row compaction) with the quantized lottery sign-flip in _iter_hook during
    lottery phases. The Sobol value of iteration i is
    fp2fxp(r[i - 1]).

    Accepts every bp_norm_min_sum_quant and bp_lottery key, including
        random_machine: 'sobol' (default) | 'system'   RNG for the flip pick
    """

    _config_algorithm = "bp_lottery_quant"
    _config_keys = ("int_width", "frac_width")

    def __init__(self, decoding_cfg, **kwargs) -> None:
        super().__init__(decoding_cfg, **kwargs)
        self.algo = "bp_lottery_quant"

    def _native_iter_hook(self, i, l_v, e_v, active, syndrome) -> None:
        """Sign-flip at the end of each flip iteration i (is_flip_iter) on the
        unconverged rows; the next iteration reads the flipped l_v. e_v may be in
        the decoder dtype or uint8 (CUDA port)."""
        if not is_flip_iter(self, i):
            return
        self.i = i
        e_b = (e_v != 0).to(torch.uint8)
        self._flip_e_v = e_b
        s_est = (e_b[:, self.V_c_col].sum(dim=2, dtype=torch.uint8) & 1).to(self.dtype)
        self.sign_flip(syndrome, s_est, l_v, active)

    def _draw_r(self, n):
        if self.random_machine.lower() == "system":
            return rand_rows(self, n)
        if self.vn_pick == "random":
            self.r_v[self.i - 1] = self._q(self.r_v[self.i - 1])
        return self._q(self.r[self.i - 1]).repeat(n)

    def sign_flip(self, syndrome, s_est, l_v, active):
        """Select and flip VNs with the shared lottery rule."""
        unsat_cn_mask = ((syndrome + s_est) % 2.0).bool()
        chosen_cn_idx, selected_vn = select_flip_vn(
            self, unsat_cn_mask, l_v, getattr(self, "_flip_e_v", None)
        )
        self.last_flip_cn = chosen_cn_idx
        self.last_flip_vn = selected_vn
        self.last_flip_valid = unsat_cn_mask.any(dim=1) & active
        flip_rows(l_v, selected_vn, self.last_flip_valid)
        return l_v
