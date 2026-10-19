import torch

from syndrilla.decoder.bp_sf.bp_sf import create as _BpSf


class PrePhase:
    """Residual PRE ranking and serial prior flips, with score/index ties."""

    _serial_state = None

    def _init_pre_phase(self, cfg):
        self.pre_max_lambda = cfg.get("pre_max_lambda", 16)
        self.pre_max_iter = cfg.get("pre_max_iter", 5)
        has_pre = any(p["flip_mode"] == "pre" for p in self.phase_knobs)
        for key in ("pre_max_lambda", "pre_max_iter"):
            if key in cfg and not has_pre:
                raise ValueError(f"{key} requires a pre flip_mode phase")
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be an int >= 1")
        self._serial_state = None
        self.serial = None
        self._pre_enabled = bool(self._pre_windows())
        if self._pre_enabled:
            self._init_pre_rank()
            self.cap = None

    def _init_pre_rank(self):
        _BpSf._init_sf(self, {})
        self.pre_degree = (self.V_v_row < self.H_shape[0]).sum(1)

    def _rank_pre(self, syndrome, estimate, phase=None):
        from syndrilla.decoder.bp_lottery.bp_lottery import vn_unsat_count
        padded = torch.cat((estimate, estimate.new_zeros((len(estimate), 1))), 1)
        residual = (syndrome + padded[:, self.V_c_col].sum(2)).remainder(2)
        counts = vn_unsat_count(residual, self.V_c_col, self.H_shape[1])
        scores = self.pre_degree - 2 * counts.long()
        allowed = ~estimate.bool()
        scores = scores.masked_fill(~allowed, torch.iinfo(scores.dtype).max)
        count = self.max_lambda if phase is None else phase["max_lambda"]
        candidates = torch.argsort(scores, dim=1, stable=True)[:, :count]
        return candidates, allowed.gather(1, candidates), residual

    def _pre_windows(self):
        windows = {}
        starts = [1] + self.flip_phase_iters
        stops = self.flip_phase_iters + [self.max_iter]
        for phase, (knobs, start, stop) in enumerate(zip(self.phase_knobs, starts, stops)):
            if knobs["flip_mode"] != "pre":
                continue
            stop = min(stop, self.max_iter)
            if stop < 1 or (phase > 0 and stop <= start):
                continue
            start = max(1, start)
            count = min(self.pre_max_lambda, max(0, stop - start) // self.pre_max_iter)
            # The first check is at the phase boundary, before its first update.
            for window in range(max(1, count)):
                windows[start + window * self.pre_max_iter] = (phase, window, count)
        return windows

    def _serial_window(self, i):
        info = self._pre_window_map.get(i)
        if info is None:
            return None
        phase, window, count = info
        if phase != self._pre_phase:
            self._pre_phase = phase
            self.serial_windows = count
            self.serial = dict(max_lambda=self.pre_max_lambda, max_iter=self.pre_max_iter,
                               start_iter=i, rerank=True, single_bit_fix=True)
            state = self._serial_state
            state["tried"].fill_(-1)
            state["valid"].zero_()
        return window

    def _pre_main(self, io_dict, decode):
        if not self._pre_enabled or getattr(self, "_trial_owner", None) is not None:
            return decode(io_dict)
        channel = io_dict["llr0"].to(self.device, self.dtype)
        syndrome = io_dict["synd"].to(self.device, self.dtype)
        size, nodes = channel.shape
        width = min(self.pre_max_lambda, nodes)
        self._pre_window_map = self._pre_windows()
        self._pre_phase = -1
        first = min(self._pre_window_map, default=1)
        self.serial_windows = self._pre_window_map[first][2] if self._pre_window_map else 0
        self.serial = dict(max_lambda=self.pre_max_lambda, max_iter=self.pre_max_iter,
                           start_iter=first, rerank=True, single_bit_fix=True)
        self._serial_state = dict(channel=channel, syndrome=syndrome,
            rows=torch.arange(size, device=self.device),
            active=torch.ones(size, dtype=torch.bool, device=self.device),
            last_active=torch.ones(size, dtype=torch.bool, device=self.device),
            pin=torch.full((size,), -1, dtype=torch.long, device=self.device),
            previous=torch.full((size,), -1, dtype=torch.long, device=self.device),
            until=torch.zeros(size, dtype=torch.long, device=self.device),
            queue=torch.zeros((size, width), dtype=torch.long, device=self.device),
            tried=torch.full((size, width), -1, dtype=torch.long, device=self.device),
            valid=torch.zeros((size, width), dtype=torch.bool, device=self.device),
            single_bit_fix=torch.zeros(size, dtype=torch.bool, device=self.device),
            single_bit_fix_window=torch.full((size,), -1, dtype=torch.int16, device=self.device))
        step = getattr(self, "_step", None)
        try:
            if self.compile and step is not None:
                def serial_step(l_v, messages, prior, *args):
                    self._serial_prior(prior, self.i)
                    messages, l_v, e_v, estimate = step(l_v, messages, prior, *args)
                    self._serial_after_update(l_v, e_v, self.i)
                    if self._serial_window(self.i) is not None:
                        estimate = self.syndrome_estimation(e_v)
                    return messages, l_v, e_v, estimate
                self._step = serial_step
            extension = getattr(self, "_ext", None)
            if extension is not None:
                self._ext = SerialExtension(self, extension)
            return decode(io_dict)
        finally:
            if step is not None:
                self._step = step
            if extension is not None:
                self._ext = extension
            self.last_pre_serial = self._serial_state
            self._serial_state = None

    def _serial_sync(self, size):
        state = self._serial_state
        if len(state["rows"]) != size:
            state["rows"] = state["rows"][state["last_active"]]
            assert len(state["rows"]) == size
        return state["rows"]

    def _serial_prior(self, prior, i):
        state = self._serial_state
        if state is None:
            return
        rows = self._serial_sync(len(prior))
        local = torch.arange(len(rows), device=self.device)
        old = state["previous"][rows]
        release = old >= 0
        prior[local[release], old[release]] = state["channel"][rows[release], old[release]]
        nodes = state["pin"][rows]
        held = (nodes >= 0) & (i <= state["until"][rows]) & state["active"][rows]
        prior[local[held], nodes[held]] = -state["channel"][rows[held], nodes[held]].abs()
        state["previous"][rows] = torch.where(held, nodes, -1)

    def _serial_after_update(self, l_v, e_v, i):
        state = self._serial_state
        if state is None:
            return
        rows = self._serial_sync(len(l_v))
        window = self._serial_window(i)
        if window is None:
            return
        candidates, valid, residual = self._rank_pre(state["syndrome"][rows], e_v[:, :-1], self.serial)
        valid &= ~(candidates[:, :, None] == state["tried"][rows, None, :]).any(2)
        state["queue"][rows] = candidates
        state["valid"][rows] = valid
        needs = state["active"][rows] & residual.bool().any(1)
        shifts = residual.new_zeros((*candidates.shape, self.H_shape[0] + 1))
        shifts.scatter_(2, self.V_v_row[candidates], 1)
        hits = valid & needs[:, None] & (shifts[:, :, :-1] == residual[:, None]).all(2)
        found = hits.any(1)
        local = found.nonzero().flatten()
        nodes = candidates[found, hits.long().argmax(1)[found]]
        e_v[local, nodes] = 1 - e_v[local, nodes]
        l_v[local, nodes] = -l_v[local, nodes]
        zeros = (l_v[local, nodes] == 0) & (e_v[local, nodes] == 0)
        l_v[local[zeros], nodes[zeros]] = torch.finfo(l_v.dtype).tiny
        state["single_bit_fix"][rows[found]] = True
        state["single_bit_fix_window"][rows[found]] = window

    def _iter_hook(self, i, l_v, e_v, active, syndrome):
        state = self._serial_state
        chosen = None
        if state is not None:
            rows = self._serial_sync(len(l_v))
            state["active"][rows] = active
            state["last_active"] = active.clone()
            window = self._serial_window(i)
            if window is not None and window < min(self.serial_windows, state["queue"].shape[1]):
                candidates, valid = state["queue"][rows], state["valid"][rows]
                has = active & valid.any(1)
                nodes = candidates.gather(1, valid.long().argmax(1)[:, None])[:, 0]
                shots = rows[has]
                nodes = nodes[has]
                state["pin"][rows] = -1
                state["pin"][shots] = nodes
                state["until"][shots] = i + self.serial["max_iter"]
                state["tried"][shots, window] = nodes
                chosen = has.nonzero().flatten(), shots, nodes
        self._native_iter_hook(i, l_v, e_v, active, syndrome)
        if chosen is not None:
            local, shots, nodes = chosen
            self._install_pre_pin(l_v, e_v, local, shots, nodes)

    def _install_pre_pin(self, l_v, e_v, local, shots, nodes):
        l_v[local, nodes] = -self._serial_state["channel"][shots, nodes].abs()
        e_v[local, nodes] = 1

    def llr_update(self, prior, messages, out):
        self._serial_prior(prior, self.i)
        return super().llr_update(prior, messages, out)

    def hard_decision(self, l_v, out):
        super().hard_decision(l_v, out)
        self._serial_after_update(l_v, out, self.i)
        return out


class SerialExtension:
    """Run PRE check changes before the CUDA convergence check."""

    def __init__(self, decoder, extension):
        self.decoder, self.extension, self.iteration = decoder, extension, 0

    def __getattr__(self, name):
        return getattr(self.extension, name)

    def llr_hard_update_csr(self, prior, messages, vn_eid, l_v, e_v, run):
        self.iteration += 1
        self.decoder._serial_prior(prior, self.iteration)
        self.extension.llr_hard_update_csr(prior, messages, vn_eid, l_v, e_v, run)
        self.decoder._serial_after_update(l_v, e_v, self.iteration)

    def llr_hard_update_atomic(self, prior, messages, col, vn_sum, l_v, e_v, run):
        self.iteration += 1
        self.decoder._serial_prior(prior, self.iteration)
        self.extension.llr_hard_update_atomic(prior, messages, col, vn_sum, l_v, e_v, run)
        self.decoder._serial_after_update(l_v, e_v, self.iteration)
