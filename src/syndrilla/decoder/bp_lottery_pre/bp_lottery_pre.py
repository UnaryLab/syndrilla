import torch

from syndrilla.decoder.bp_lottery.bp_lottery import create as _BpLottery


class create(_BpLottery):
    """Lottery main pass with optional flat-configured parallel PRE retries."""

    _supports_pre_parallel = True
    _config_algorithm = "bp_lottery_pre"
    _config_keys = ("pre_parallel_max_lambda", "pre_parallel_max_iter", "pre_parallel_start_iter")
    _stop_siblings = True
    _trial_state = None

    def __init__(self, decoding_cfg, **kwargs):
        super().__init__(decoding_cfg, **kwargs)
        self._init_pre(decoding_cfg)

    def _init_pre(self, cfg):
        keys = self._config_keys
        self.parallel_enabled = any(key in cfg for key in keys)
        self.max_lambda = cfg.get(keys[0], 32)
        self.pre_it = cfg.get(keys[1], self.max_iter)
        ends = [start + count * self.pre_max_iter
                for start, (_, window, count) in self._pre_windows().items() if window == 0]
        inferred = min(self.max_iter, max(ends)) if ends else self.max_iter
        self.parallel_start = cfg.get(keys[2], inferred)
        for key, value in zip(keys, (self.max_lambda, self.pre_it, self.parallel_start)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be an int >= 1")
        if self.parallel_start > self.max_iter:
            raise ValueError("pre_parallel_start_iter must not exceed max_iter")
        if keys[2] in cfg and ends and self.parallel_start < max(ends):
            raise ValueError("pre_parallel_start_iter must follow the PRE phase budget end")
        if self.parallel_enabled:
            if not self._pre_enabled:
                self._init_pre_rank()
            self.cap = None
        self.algo = "bp_lottery_pre"

    def _lottery_forward(self, io_dict):
        return super().forward(io_dict)

    def _bp_core(self, syndrome, llr0):
        owner = getattr(self, "_trial_owner", None)
        stop = self._stop_siblings and owner is not None and self.early_termination
        if not stop:
            out = self._lottery_forward({"synd": syndrome, "llr0": llr0})
            return out["e_v"], out["llr"], out["converge"], out["iter"], None
        size = len(owner)
        if stop:
            self._trial_state = dict(owner=owner, syndrome=syndrome,
                rows=torch.arange(size, device=self.device), iteration=0,
                winner=torch.full((int(owner.max()) + 1,), size, device=self.device, dtype=torch.long))
        saved_compact = getattr(self, "compact_frac", None)
        saved_step = getattr(self, "_step", None)
        try:
            if saved_compact is not None:
                self.compact_frac = 1.0
            if getattr(self, "compile", False):
                def step(*args):
                    message, values, hard, estimate = saved_step(*args)
                    return message, values, hard, self._stop_estimate(estimate)
                self._step = step
            out = self._lottery_forward({"synd": syndrome, "llr0": llr0})
            if stop:
                self.last_pre_trials = dict(owner=owner.clone(), iterations=out["iter"].clone(),
                    stopped=self._stopped_siblings(), converged=out["converge"].clone())
        finally:
            self._trial_state = None
            if saved_compact is not None:
                self.compact_frac = saved_compact
            if saved_step is not None:
                self._step = saved_step
        return out["e_v"], out["llr"], out["converge"], out["iter"], None

    def _record_matches(self, matches, rows):
        state = self._trial_state
        owner, winners = state["owner"], state["winner"]
        eligible = matches & (winners[owner[rows]] == len(owner))
        winners.scatter_reduce_(0, owner[rows], torch.where(eligible, rows, len(owner)),
                                reduce="amin", include_self=True)
        return winners[owner[rows]] < len(owner)

    def _stopped_siblings(self):
        state = self._trial_state
        owner, winner = state["owner"], state["winner"]
        index = torch.arange(len(owner), device=self.device)
        return (winner[owner] < len(owner)) & (winner[owner] != index)

    def _stop_estimate(self, estimate):
        state = self._trial_state
        if state is None or state["iteration"] == self.i:
            return estimate
        rows = state["rows"]
        if len(rows) != len(estimate):
            rows = rows[state["active"]]
            state["rows"] = rows
        target = state["syndrome"][rows]
        finished = self._record_matches((estimate == target).all(1), rows)
        state["active"], state["iteration"] = ~finished, self.i
        # The base loop retires matching rows. Only the recorded winner is a real convergence.
        return torch.where(finished[:, None], target, estimate)

    def syndrome_estimation(self, e_v):
        return self._stop_estimate(super().syndrome_estimation(e_v))

    def _exit_hook(self, l_v, e_v, num_iters, converges):
        if self._trial_state is not None:
            converges[self._stopped_siblings()] = 0

    def forward(self, io_dict):
        self.last_pre_trials = None
        self.last_pre_serial = None
        syndrome = io_dict["synd"].to(device=self.device, dtype=self.dtype)
        llr0 = io_dict["llr0"].to(device=self.device, dtype=self.dtype)
        main_budget = self.max_iter
        try:
            if self.parallel_enabled:
                self.max_iter = self.parallel_start
            self._lottery_forward(io_dict)
        finally:
            self.max_iter = main_budget
        io_dict["iter_serial"], io_dict["iter_parallel"] = self._pre_postprocess(
            syndrome, llr0, io_dict["e_v"], io_dict["llr"], io_dict["converge"], io_dict["iter"])
        return io_dict

    def _pre_postprocess(self, syndrome, llr0, e_out, l_out, converges, num_iters):
        """Serial work uses the lowest lambda observed at the sibling stop.

        A slower lower lambda is not observed. This count is an upper bound
        on the true serial model; parallel work is main plus the winner's count.
        """
        serial, parallel = num_iters.clone(), num_iters.clone()
        rows = (~converges.bool()).nonzero().flatten()
        if not self.parallel_enabled or rows.numel() == 0:
            return serial, parallel
        candidates, valid, residual = self._rank_pre(syndrome[rows], e_out[rows])
        M, N = self.H_shape
        U, C = candidates.shape
        shifts = syndrome.new_zeros((U, C, M + 1))
        shifts.scatter_(2, self.V_v_row[candidates], 1)
        shifts = shifts[:, :, :M]
        fixed = valid & (shifts == residual[:, None]).all(2)
        has_fix = fixed.any(1)
        fixed_rows = rows[has_fix]
        first_fix = fixed.long().argmax(1)
        fixed_nodes = candidates[has_fix, first_fix[has_fix]]
        e_out[fixed_rows, fixed_nodes] = 1 - e_out[fixed_rows, fixed_nodes]
        l_out[fixed_rows, fixed_nodes] = -l_out[fixed_rows, fixed_nodes]
        # Negating zero keeps hard bit 1; a new hard bit 0 needs a positive value.
        zeros = (l_out[fixed_rows, fixed_nodes] == 0) & (e_out[fixed_rows, fixed_nodes] == 0)
        l_out[fixed_rows[zeros], fixed_nodes[zeros]] = torch.finfo(l_out.dtype).tiny
        converges[fixed_rows] = 1

        trial_valid = valid & ~has_fix[:, None]
        owner, slot = trial_valid.nonzero(as_tuple=True)
        if owner.numel() == 0:
            return serial, parallel
        trial_synd = (syndrome[rows[owner]] + shifts[owner, slot]).remainder(2)
        trial_llr = llr0[rows[owner]]
        old_max = self.max_iter
        old_phases, old_bounds = self.phase_knobs, self.flip_phase_iters
        old_betas = getattr(self, "betas", None)
        try:
            self.max_iter = self.pre_it
            # Sub-lotteries keep the lottery pick policy, with no serial PRE phase.
            self.phase_knobs = [{**phase, "flip_mode": ("lottery" if phase["flip_mode"] == "pre" else phase["flip_mode"])}
                                for phase in self.phase_knobs]
            self._trial_owner = owner
            if old_betas is not None and old_betas.numel() < self.pre_it:
                i = torch.arange(1, self.pre_it + 1, dtype=self.dtype)
                self.betas = (1 - torch.pow(torch.tensor(2., dtype=self.dtype), -i)).to(self.device)
            e_r, l_r, conv_r, iter_r, _ = self._bp_core(trial_synd, trial_llr)
        finally:
            self._trial_owner = None
            self.max_iter = old_max
            self.phase_knobs, self.flip_phase_iters = old_phases, old_bounds
            if old_betas is not None:
                self.betas = old_betas

        trial_index = torch.full((U, C), -1, device=self.device, dtype=torch.long)
        trial_index[owner, slot] = torch.arange(owner.numel(), device=self.device)
        success = torch.zeros((U, C), device=self.device, dtype=torch.bool)
        success[owner, slot] = conv_r.bool()
        times = torch.full((U, C), self.pre_it + 1, device=self.device, dtype=torch.long)
        times[owner, slot] = iter_r.long()
        best = times.masked_fill(~success, self.pre_it + 1).argmin(1)
        first = success.long().argmax(1)
        has = success.any(1)
        pending = trial_valid.any(1)
        serial[rows[pending]] += self.pre_it * valid[pending].sum(1)
        parallel[rows[pending]] += self.pre_it
        winners = rows[has]
        winner_index = trial_index[has, best[has]]
        chosen = e_r[winner_index].clone()
        chosen_nodes = candidates[has, best[has]]
        chosen[torch.arange(chosen.shape[0], device=self.device), chosen_nodes] = (
            1 - chosen[torch.arange(chosen.shape[0], device=self.device), chosen_nodes])
        e_out[winners], l_out[winners], converges[winners] = chosen, l_r[winner_index], 1
        l_out[winners, chosen_nodes] = -l_out[winners, chosen_nodes]
        zeros = (l_out[winners, chosen_nodes] == 0) & (e_out[winners, chosen_nodes] == 0)
        l_out[winners[zeros], chosen_nodes[zeros]] = torch.finfo(l_out.dtype).tiny
        serial[winners] = num_iters[winners] + self.pre_it * first[has] + times[has, first[has]]
        parallel[winners] = num_iters[winners] + times[has, best[has]]
        return serial, parallel
