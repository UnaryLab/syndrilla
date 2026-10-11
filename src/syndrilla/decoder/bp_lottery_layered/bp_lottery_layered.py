import torch

from syndrilla.decoder.bp_lottery.bp_lottery import create as _Lottery
from syndrilla.decoder.bp_lottery.bp_lottery import is_flip_iter, read_study_knobs, STUDY_KNOBS
from syndrilla.decoder.bp_norm_min_sum_layered.bp_norm_min_sum_layered import create as _Layered


class create(_Lottery, _Layered):
    """Lottery flips affect the next layered sweep only.

    Undo each posterior sign flip after that sweep, before the hard decision,
    so the channel prior stays fixed. Compaction keeps pending flips in row order.
    """

    def __init__(self, decoding_cfg, **kwargs):
        knobs, _ = read_study_knobs(decoding_cfg)
        for key in ("kick", "null_mode"):
            if knobs[key] != STUDY_KNOBS[key][0]:
                raise ValueError(f"bp_lottery_layered does not support {key}={knobs[key]!r}")
        super().__init__(decoding_cfg, **kwargs)
        self.algo = "bp_lottery_layered"

    def forward(self, io_dict):
        self._pending_flip = None
        return super().forward(io_dict)

    def _iter_hook(self, i, l_v, e_v, active, syndrome):
        super()._iter_hook(i, l_v, e_v, active, syndrome)
        if is_flip_iter(self, i):
            variable = self.last_flip_vn.clone()
            delta = torch.where(self.last_flip_valid,
                                -2 * l_v.gather(1, variable.unsqueeze(1)).squeeze(1), 0)
            self._pending_flip = (self._hook_rows.clone(), variable, delta, active.clone())

    def _sweep(self, l_v, previous, syndrome):
        pending = self._pending_flip
        if pending is not None:
            rows, variable, delta, active = pending
            # Both compaction paths retain only active rows, in their original order.
            if len(l_v) != len(rows):
                rows, variable, delta = rows[active], variable[active], delta[active]
        super()._sweep(l_v, previous, syndrome)
        if pending is not None:
            l_v.scatter_add_(1, variable.unsqueeze(1), delta.unsqueeze(1))
            self._pending_flip = None
