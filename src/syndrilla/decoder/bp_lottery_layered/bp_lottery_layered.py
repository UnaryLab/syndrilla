import torch

from syndrilla.decoder.bp_lottery.bp_lottery import create as _Lottery
from syndrilla.decoder.bp_lottery.bp_lottery import is_flip_iter
from syndrilla.decoder.bp_norm_min_sum_layered.bp_norm_min_sum_layered import create as _Layered


class create(_Lottery, _Layered):
    """Lottery flips affect the next layered sweep only.

    Undo each posterior flip after that sweep, before the hard decision,
    so lottery leaves the channel prior fixed. PRE windows change their candidate prior.
    Compaction keeps pending flips in row order.
    """

    _config_algorithm = "bp_lottery_layered"

    def __init__(self, decoding_cfg, **kwargs):
        super().__init__(decoding_cfg, **kwargs)
        self.algo = "bp_lottery_layered"

    def forward(self, io_dict):
        self._pending_flip = None
        self._pending_pre = None
        try:
            return super().forward(io_dict)
        finally:
            self._pre_layered_llr = None

    def _native_iter_hook(self, i, l_v, e_v, active, syndrome):
        super()._native_iter_hook(i, l_v, e_v, active, syndrome)
        if is_flip_iter(self, i):
            variable = self.last_flip_vn.clone()
            value = l_v.gather(1, variable.unsqueeze(1)).squeeze(1)
            delta = torch.where(self.last_flip_valid,
                                -2 * value, 0)
            self._pending_flip = (self._hook_rows.clone(), variable, delta, active.clone())

    def _sweep(self, l_v, previous, syndrome):
        if self._serial_state is not None:
            self._pre_layered_llr = l_v
        pending = self._pending_flip
        if pending is not None:
            rows, variable, delta, active = pending
            # Both compaction paths retain only active rows, in their original order.
            if len(l_v) != len(rows):
                rows, variable, delta = rows[active], variable[active], delta[active]
        pre = self._pending_pre
        deferred = None
        if pre is not None:
            pre_rows, pre_nodes, pre_delta, pre_mask, pre_active = pre
            if len(l_v) != len(pre_rows):
                pre_rows, pre_nodes, pre_delta, pre_mask = (value[pre_active] for value in
                    (pre_rows, pre_nodes, pre_delta, pre_mask))
        if self._serial_state is not None:
            state = self._serial_state
            rows = self._serial_sync(len(l_v))
            prior = state["channel"][rows].clone()
            local = torch.arange(len(rows), device=self.device)
            old = state["previous"][rows]
            held = old >= 0
            prior[local[held], old[held]] = -prior[local[held], old[held]].abs()
            before = prior.clone()
            self._serial_prior(prior, self.i)
            change = prior - before
            if pre is not None:
                deferred = change.gather(1, pre_nodes[:, None]) * pre_mask[:, None]
                change.scatter_add_(1, pre_nodes[:, None], -deferred)
            l_v[:, :-1] += change
        super()._sweep(l_v, previous, syndrome)
        if pending is not None:
            l_v.scatter_add_(1, variable.unsqueeze(1), delta.unsqueeze(1))
            self._pending_flip = None
        if pre is not None:
            # Remove the one-step posterior injection, then retain its prior change.
            l_v.scatter_add_(1, pre_nodes[:, None], pre_delta[:, None] + deferred)
            self._pending_pre = None

    def _install_pre_pin(self, l_v, e_v, local, shots, nodes):
        old = l_v[local, nodes].clone()
        super()._install_pre_pin(l_v, e_v, local, shots, nodes)
        selected = torch.zeros(len(l_v), device=self.device, dtype=torch.long)
        delta = l_v.new_zeros(len(l_v))
        mask = torch.zeros(len(l_v), dtype=torch.bool, device=self.device)
        mask[local] = True
        selected[local] = nodes
        delta[local] = old - l_v[local, nodes]
        state = self._serial_state
        self._pending_pre = (self._hook_rows.clone(), selected, delta, mask,
                             state["active"][self._serial_sync(len(l_v))].clone())

    def _record_convergence(self, e_v, syndrome, it_rows):
        if self._serial_state is not None:
            # The layered loop builds its hard vector directly after each sweep.
            self._serial_after_update(self._pre_layered_llr, e_v, self.i)
        return super()._record_convergence(e_v, syndrome, it_rows)
