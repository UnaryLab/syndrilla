from loguru import logger

from syndrilla.decoder.bp_lottery.bp_lottery import is_flip_iter
from syndrilla.decoder.bp_lottery.bp_lottery_cuda import create as _LotteryCuda
from syndrilla.decoder.bp_lottery_quant.bp_lottery_quant import (
    create as _LotteryQuantPy,
)
from syndrilla.decoder.bp_norm_min_sum_quant.bp_norm_min_sum_quant_cuda import (
    create as _QuantCuda,
)


class create(_LotteryCuda, _QuantCuda):
    """Quantized lottery BP on CUDA: the bp_norm_min_sum_quant_cuda per-step loop
    (rounding in cn_row and _exit_hook) with the bp_lottery_quant sign-flip in
    _iter_hook. bp_lottery_cuda supplies the knobs and the Sobol sequence.

    Accepts every bp_norm_min_sum_quant_cuda key plus
        random_machine : 'sobol' (default) | 'system'   RNG for the flip pick
        flip_start_iter: int (default 4)                 flips start after this iteration
        flip_interval  : int >= 1 (default 1)            iterations between flips
    """

    sign_flip = _LotteryQuantPy.sign_flip

    def __init__(self, decoding_cfg: dict, **kwargs) -> None:
        super().__init__(decoding_cfg, **kwargs)
        self.algo = "bp_lottery_quant"
        logger.info("bp_lottery_quant_cuda ready (per-step path + sign-flip).")

    def _iter_hook(self, i, l_v, e_v, active, syndrome) -> None:
        """The bp_lottery_quant flip on the unconverged rows at each flip
        iteration (is_flip_iter). With random_machine system, a call with no
        unconverged row returns before drawing (one host sync per call);
        forward() makes the one draw the PyTorch path makes when the last rows
        converge at a flip iteration."""
        if not is_flip_iter(self, i):
            return
        if self.random_machine == "system" and not active.any():
            return
        _LotteryQuantPy._iter_hook(self, i, l_v, e_v, active, syndrome)

    def forward(self, io_dict: dict) -> dict:
        self._B = io_dict["synd"].shape[0]
        return super().forward(io_dict)
