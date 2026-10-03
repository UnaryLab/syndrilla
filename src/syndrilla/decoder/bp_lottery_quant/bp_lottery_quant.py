import torch

from syndrilla.decoder.bp_lottery.bp_lottery import (
    create as _LotteryPy,
    flip_rows,
    rand_rows,
    vn_unsat_count,
)
from syndrilla.decoder.bp_norm_min_sum_quant.bp_norm_min_sum_quant import (
    create as _QuantPy,
)


class create(_LotteryPy, _QuantPy):
    """
    Quantized lottery BP: bp_norm_min_sum_quant (rounding, eager or compiled step,
    row compaction) with the quantized lottery sign-flip in _iter_hook from
    iteration flip_start_iter + 1 on. The Sobol value of iteration i is
    fp2fxp(r[i - 1]).

    Accepts every bp_norm_min_sum_quant key plus
        random_machine: 'sobol' (default) | 'system'   RNG for the flip pick
        flip_start_iter: int (default 4)                 flips start after this iteration
    """

    def __init__(self, decoding_cfg, **kwargs) -> None:
        super().__init__(decoding_cfg, **kwargs)
        self.algo = "bp_lottery_quant"

    def _iter_hook(self, i, l_v, e_v, active, syndrome) -> None:
        """Sign-flip at the end of iteration i > flip_start_iter on the
        unconverged rows; the next iteration reads the flipped l_v. e_v may be in
        the decoder dtype or uint8 (CUDA port)."""
        if i <= self.flip_start_iter:
            return
        self.i = i
        e_b = (e_v != 0).to(torch.uint8)
        s_est = (e_b[:, self.V_c_col].sum(dim=2, dtype=torch.uint8) & 1).to(self.dtype)
        self.sign_flip(syndrome, s_est, l_v, active)

    def sign_flip(self, syndrome, s_est, l_v, active):
        """Flip the sign of one variable's LLR per row with an unsatisfied check:
        the variable at position floor(r * (U - 1)) in the cumulative count of
        unsatisfied checks per variable, U the total count."""
        synd_diff = (syndrome + s_est) % 2.0

        temp_ls = vn_unsat_count(synd_diff.float(), self.V_c_col, self.H_shape[1])

        total_ones = temp_ls.sum(dim=1).to(self.dtype)

        valid_mask = total_ones > 0

        n = l_v.shape[0]
        if self.random_machine.lower() == "system":
            r = rand_rows(self, n)
        else:  # sobol
            r = self._q(self.r[self.i - 1]).repeat(n)

        target = torch.where(valid_mask, torch.floor(r * (total_ones - 1)), 0.0)
        target = target.unsqueeze(1)
        cumsum_x = torch.cumsum(temp_ls, dim=1)
        mask = cumsum_x >= target + 1
        selected_indices = torch.argmax(mask.float(), dim=1)

        flip_rows(l_v, selected_indices, valid_mask & active)
        return l_v
