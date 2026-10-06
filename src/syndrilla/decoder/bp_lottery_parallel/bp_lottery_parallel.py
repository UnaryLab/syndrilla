import math

import torch
from loguru import logger

from syndrilla.decoder.bp_lottery.bp_lottery import (
    cn_row_mask,
    flip_rows,
    is_flip_iter,
    vn_unsat_count,
)
from syndrilla.decoder.bp_lottery.bp_lottery import create as _LotteryPy
from syndrilla.decoder.decoder import SHARED_KEYS
from syndrilla.decoder.knobs import GROUP_OFF

# every key the decoder, its bp_lottery and bp_norm_min_sum bases (PyTorch and
# CUDA), an osd_0 stage after it and the loader read from the decoder config,
# with the GROUP_OFF group keys and members; others log a warning
KNOWN_KEYS = frozenset(
    # top of `decoding`, shared by every stage; resolve_configs drops `config`
    (set(SHARED_KEYS) - {"config"})
    | {
        # bp_norm_min_sum and bp_norm_min_sum_cuda
        "max_iter",
        "force_per_step",
        # bp_lottery
        "random_machine",
        "flip_start_iter",
        "flip_interval",
        # bp_lottery_parallel
        "num_parallel",
        "seeds",
        "select",
        "report_agreement",
        "agree_stop",
        "ensemble_llr",
        "flip_anneal",
        "flip_tiebreak",
        "stuck_check_weight",
        "flip_temperature",
        "flip_undo",
        "copy_on_stall",
        "syndrome_flip_last_n",
        "consensus_every",
        "consensus_llr",
    }
    | GROUP_OFF.keys()
    | {k for members in GROUP_OFF.values() for k in members}
)


def _flag(cfg, key):
    """cfg[key] as a bool, default False; a value other than a bool raises
    ValueError."""
    v = cfg.get(key, False)
    if not isinstance(v, bool):
        raise ValueError(f"{key} must be true or false, got <{v}>.")
    return v


def _count(cfg, key, hi=None, default=0):
    """cfg[key] as an int >= 0 (and <= hi), default default; another value
    (a float or a bool too) raises ValueError."""
    v = cfg.get(key, default)
    if isinstance(v, bool) or not isinstance(v, int) or v < 0 or v > (hi or v):
        raise ValueError(f"{key} must be an int in [0, {hi or 'inf'}], got <{v}>.")
    return v


def _real(v):
    """True when v is an int or a float, not a bool."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _unsat(e, syndrome, V_c_col):
    """[R, M] bool, True on each check the hard decision e [R, N+1] (bool)
    does not satisfy."""
    s_est = e[:, V_c_col].sum(dim=2, dtype=torch.uint8) & 1
    return s_est != (syndrome != 0)


def _lex_argmax(mask, *keys):
    """[R] per row, the first index among mask with the largest keys[0], then
    among those the largest keys[1], and so on."""
    for k in keys:
        k = k.to(torch.float64).masked_fill(~mask, -math.inf)
        mask = mask & (k == k.amax(dim=1, keepdim=True))
    return mask.to(torch.uint8).argmax(dim=1)


class Parallel:
    """The replica code shared by the PyTorch and the CUDA bp_lottery_parallel:
    config parsing, the per-row hook, the per-replica draws and the per-shot
    selection."""

    _repeat = 0  # flip repeat index within the current hook call (flip_anneal)

    def __init__(self, decoding_cfg, **kwargs):
        if "replica_knobs" in decoding_cfg:
            raise ValueError("replica_knobs is no longer supported.")
        if "syndrome_flip_replicas" in decoding_cfg:
            raise ValueError(
                "syndrome_flip_replicas is no longer supported; use syndrome_flip_last_n."
            )
        K = _count(decoding_cfg, "num_parallel", default=8)
        if K < 1:
            raise ValueError(f"num_parallel must be >= 1, got <{K}>.")
        starts = decoding_cfg.get("flip_start_iter", 0)
        fsi = starts if isinstance(starts, list) else [starts] * K
        if len(fsi) != K or any(
            isinstance(start, bool) or not isinstance(start, int) or start < 0 for start in fsi
        ):
            raise ValueError(f"flip_start_iter must be an int >= 0 or K such ints, got <{starts}>.")
        intervals = decoding_cfg.get("flip_interval", 1)
        flip_intervals = intervals if isinstance(intervals, list) else [intervals] * K
        if len(flip_intervals) != K or any(
            isinstance(interval, bool) or not isinstance(interval, int) or interval < 1
            for interval in flip_intervals
        ):
            raise ValueError(f"flip_interval must be an int >= 1 or K such ints, got <{intervals}>.")
        machine = decoding_cfg.get("random_machine", "sobol")
        machines = machine if isinstance(machine, list) else [machine] * K
        if len(machines) != K or any(
            not isinstance(sampler, str) or sampler.lower() not in {"sobol", "system"}
            for sampler in machines
        ):
            raise ValueError(f"random_machine must be sobol or system or K such strings, got <{machine}>.")
        super().__init__(
            {
                **decoding_cfg,
                "flip_start_iter": min(fsi),
                "flip_interval": min(flip_intervals),
                "random_machine": "system" if any(sampler.lower() == "system" for sampler in machines) else "sobol",
            },
            **kwargs,
        )
        self._fsi = torch.tensor(fsi, device=self.device) if isinstance(starts, list) else None
        self._fi = torch.tensor(flip_intervals, device=self.device) if isinstance(intervals, list) else None
        self._machines = [sampler.lower() for sampler in machines]

    def _init_parallel(self, decoding_cfg):
        """Read num_parallel, seeds, select, flip_start_iter, report_agreement,
        random_machine, flip_anneal and the flip knobs (flip_tiebreak,
        stuck_check_weight, flip_temperature, flip_undo, consensus_every,
        consensus_llr, copy_on_stall, syndrome_flip_last_n), and build the per-replica Sobol sequences
        self._rk [K, max_iter * most flips per call] and generators self._gens.
        Logs a warning naming the keys outside KNOWN_KEYS."""
        unknown = sorted(set(decoding_cfg) - KNOWN_KEYS)
        if unknown:
            logger.warning(
                f"bp_lottery_parallel ignores unknown config keys {unknown}."
            )
        K = _count(decoding_cfg, "num_parallel", default=8)
        if K < 1:
            raise ValueError(f"num_parallel must be >= 1, got <{K}>.")
        self.select = decoding_cfg.get("select", "min_iter")
        if self.select not in {"min_iter", "min_flip_prior", "min_flip_posterior"}:
            raise ValueError(f"select must be 'min_iter', 'min_flip_prior' or 'min_flip_posterior', got <{self.select}>.")
        self.report_agreement = _flag(decoding_cfg, "report_agreement")
        self.agree_stop = _count(decoding_cfg, "agree_stop", K)
        self._hook_before_stop = bool(self.agree_stop)
        if self.agree_stop:
            # ponytail: weighted 64-bit hashes have negligible collision probability;
            # compare full vectors if exact collision exclusion is required.
            generator = torch.Generator().manual_seed(0)
            self._ag_weights = torch.randint(
                0, 2**63 - 1, (self.H_shape[1],), generator=generator
            ).to(self.device)
        self.ensemble_llr = _flag(decoding_cfg, "ensemble_llr")
        seeds = decoding_cfg.get("seeds")
        if seeds is None:
            seeds = torch.randint(2**31, (K,)).tolist()
            logger.info(f"bp_lottery_parallel seeds: {seeds}")
        elif len(seeds) != K:
            raise ValueError(
                f"seeds has <{len(seeds)}> entries, num_parallel is <{K}>."
            )
        self.num_parallel = K
        self.seeds = [int(s) for s in seeds]

        anneal = decoding_cfg.get("flip_anneal")
        if anneal is not None:

            def pair(p):
                return (
                    isinstance(p, (list, tuple)) and len(p) == 2 and all(map(_real, p))
                )

            if pair(anneal):
                anneal = [anneal] * K
            if (
                not isinstance(anneal, (list, tuple))
                or len(anneal) != K
                or not all(map(pair, anneal))
            ):
                raise ValueError(
                    f"flip_anneal must be [T0, T1] or K such pairs, got <{anneal}>."
                )
            anneal = [(float(a), float(b)) for a, b in anneal]
        self._anneal = anneal

        self._tiebreak = decoding_cfg.get("flip_tiebreak", "llr")
        if self._tiebreak not in {"llr", "osc"}:
            raise ValueError(
                f"flip_tiebreak must be 'llr' or 'osc', got <{self._tiebreak}>."
            )
        self._stuck_on = _flag(decoding_cfg, "stuck_check_weight")
        self._temp = decoding_cfg.get("flip_temperature")
        if self._temp is not None:
            if not (_real(self._temp) and self._temp > 0):
                raise ValueError(
                    f"flip_temperature must be a number > 0, got <{self._temp}>."
                )
            self._temp = float(self._temp)
        self._accept = _flag(decoding_cfg, "flip_undo")
        self._stall = _count(decoding_cfg, "copy_on_stall")
        n_sf = _count(decoding_cfg, "syndrome_flip_last_n", K)
        # the last n_sf replicas flip a syndrome bit
        self._sf_rep = None
        if n_sf:
            self._sf_rep = torch.arange(K, device=self.device) >= K - n_sf
        self._cons_m = _count(decoding_cfg, "consensus_every")
        self._cons_llr = decoding_cfg.get("consensus_llr", 10.0)
        if not (_real(self._cons_llr) and self._cons_llr >= 0):
            raise ValueError(f"consensus_llr must be >= 0, got <{self._cons_llr}>.")
        if "consensus_llr" in decoding_cfg and not self._cons_m:
            raise ValueError("consensus_llr needs consensus_every > 0.")
        # the replicas that take part in consensus: all but the syndrome-flip ones
        self._cz_rep = None
        if self._cons_m:
            self._cz_rep = torch.ones(K, dtype=torch.bool, device=self.device)
            if self._sf_rep is not None:
                self._cz_rep = ~self._sf_rep
            if not self._cz_rep.any():
                self._cons_m = 0
        reps = 1
        if anneal is not None:
            reps = max(
                [1] + [self._n_flips(p, i) for p in anneal for i in (1, self.max_iter)]
            )

        self._rk = torch.stack(
            [
                torch.quasirandom.SobolEngine(1, scramble=True, seed=s)
                .draw(self.max_iter * reps, dtype=torch.float32)
                .flatten()
                for s in self.seeds
            ]
        ).to(self.device, self.dtype)
        self._gens = [
            torch.Generator(device=self.device).manual_seed(s) for s in self.seeds
        ]

    def _n_flips(self, pair, i):
        """Flips per unconverged row at iteration i: T0 + (T1 - T0) * (i - 1) /
        (max_iter - 1), rounded half up, at least 0."""
        t0, t1 = pair
        frac = (i - 1) / (self.max_iter - 1) if self.max_iter > 1 else 0.0
        return max(0, math.floor(t0 + (t1 - t0) * frac + 0.5))

    def _per_row(self, per_replica, n):
        """[n] per hook row from a [K] per-replica tensor: row k * B + b reads
        entry k; indexed by self._hook_rows when the n rows are a compacted
        subset."""
        r = per_replica.repeat_interleave(self._shots)
        return r if n == r.numel() else r[self._hook_rows]

    def _reset_state(self):
        """Per-forward hook state over the B * K rows: with flip_tiebreak osc,
        the oscillation count self._osc and the previous hard decision
        self._prev, per variable; with stuck_check_weight, self._stuck, per
        check the iterations it has been unsatisfied in a row; with
        flip_undo, self._fa_var, per row the variable flipped at the last
        flip iteration (-1 for none), and self._fa_u, the unsatisfied checks before
        that flip; with consensus_every, self._cz_val, self._cz_frozen and
        self._cz_dis, set by _consensus; with copy_on_stall, self._st_best,
        per row the fewest unsatisfied checks so far, and self._st_age, the
        flip iterations since that count last dropped; with syndrome_flip_last_n,
        self._sf_var, per row the variables whose syndrome columns are flipped.
        self._c2v, the check-to-variable message buffer, is set by the check
        update, self._synd_live, the decode loop's syndrome, by the hook, and
        self._flipped, under flip_anneal the variables flipped in the current
        hook call, by _apply_flip."""
        BK, N1 = self._shots * self.num_parallel, self.H_shape[1] + 1
        self._osc = self._prev = self._stuck = self._fa_var = self._fa_u = None
        self._flipped = None
        self._cz_val = self._cz_frozen = self._cz_dis = None
        self._c2v = self._synd_live = self._sf_var = None
        self._hook_stop = None
        if self.agree_stop:
            self._ag_iter = torch.zeros(BK, dtype=torch.long, device=self.device)
            self._ag_hash = torch.zeros_like(self._ag_iter)
            self._ag_stopped = torch.zeros(BK, dtype=torch.bool, device=self.device)
            self._ag_done = torch.zeros(self._shots, dtype=torch.long, device=self.device)
            self._ag_winner = torch.full_like(self._ag_done, -1)
            self._ag_count = torch.zeros_like(self._ag_done)
            self._synd_rows = None
        if self._sf_rep is not None:
            self._sf_var = torch.zeros(BK, N1, dtype=torch.bool, device=self.device)
        if self._stall:
            M = self.H_shape[0]
            self._st_best = torch.full((BK,), M + 1, device=self.device)
            self._st_age = torch.zeros(BK, dtype=torch.long, device=self.device)
        if self._accept:
            self._fa_var = torch.full((BK,), -1, dtype=torch.long, device=self.device)
            self._fa_u = torch.zeros(BK, dtype=torch.long, device=self.device)
        if self._stuck_on:
            M = self.H_shape[0]
            self._stuck = torch.zeros(BK, M, dtype=torch.int32, device=self.device)
        if self._tiebreak == "osc":
            self._osc = torch.zeros(BK, N1, dtype=torch.int32, device=self.device)
            self._prev = torch.zeros(BK, N1, dtype=torch.bool, device=self.device)

    def _rows(self, R):
        """Index of the R hook rows into the B * K row state: all rows, or
        self._hook_rows when the R rows are a compacted subset."""
        if R == self._shots * self.num_parallel:
            return slice(None)
        return self._hook_rows

    def _track(self, rows, active, e_v, syndrome):
        """Update the per-row state of the active rows at the end of an
        iteration: with flip_tiebreak osc, count per variable each change of
        the hard decision from the previous iteration (bp_sf's counter, from an
        all-zero prior); with stuck_check_weight, add 1 to each unsatisfied
        check's run and reset the satisfied ones to 0."""
        e, act = e_v != 0, active.unsqueeze(1)
        if self._stuck is not None:
            unsat = _unsat(e, syndrome, self.V_c_col)
            st = self._stuck[rows]
            self._stuck[rows] = torch.where(act, (st + 1) * unsat, st)
        if self._osc is not None:
            prev = self._prev[rows]
            self._osc[rows] += (e != prev) & act
            self._prev[rows] = torch.where(act, e, prev)

    def _consensus(self, i, rows, active, l_v):
        """consensus_every m: over the replicas in self._cz_rep (all but the
        syndrome_flip_last_n ones), at every iteration i divisible by m, over
        each shot whose such replicas are all unconverged, mark the variables
        whose hard decision all of them share and whose smallest |LLR| over
        them exceeds consensus_llr as frozen (self._cz_frozen, with their LLRs
        in self._cz_val), and the variables they disagree on as flip
        candidates (self._cz_dis), on their rows only. At every iteration, set
        the LLR of each frozen variable of an unconverged row back to its
        value at the last such iteration. A row with any variable in
        self._sf_var (an LLR row that copied a syndrome-flip row under
        copy_on_stall) takes no part: no vote, no freeze, no candidates."""
        K, B, N = self.num_parallel, self._shots, self.H_shape[1]
        if i % self._cons_m == 0:
            lv = l_v.new_zeros(K * B, N)
            lv[rows] = l_v[:, :N]
            act = active.new_zeros(K * B)
            act[rows] = active
            on = self._cz_rep.repeat_interleave(B) & self._no_sf(slice(None))
            on3 = on.view(K, B, 1)
            all_act = (act | ~on).view(K, B).all(dim=0) & on.view(K, B).any(dim=0)
            hard = lv.view(K, B, N) <= 0
            agree = ~((hard & on3).any(dim=0) & (~hard & on3).any(dim=0))
            amin = lv.view(K, B, N).abs().masked_fill(~on3, math.inf).amin(dim=0)
            strong = amin > self._cons_llr
            all_act, on = all_act.unsqueeze(1), on.unsqueeze(1)
            self._cz_frozen = (agree & strong & all_act).repeat(K, 1) & on
            self._cz_dis = (~agree & all_act).repeat(K, 1) & on
            self._cz_val = lv
        if self._cz_frozen is not None:
            f = self._cz_frozen[rows] & (active & self._no_sf(rows)).unsqueeze(1)
            l_v[:, :N] = torch.where(f, self._cz_val[rows], l_v[:, :N])

    def _no_sf(self, rows):
        """[R] bool per row in rows: True where self._sf_var marks no variable
        (all True without syndrome_flip_last_n)."""
        if self._sf_var is None:
            return torch.ones(
                self._shots * self.num_parallel, dtype=torch.bool, device=self.device
            )[rows]
        return ~self._sf_var[rows].any(dim=1)

    def _resample(self, rows, active, l_v, e_v, syndrome):
        """copy_on_stall n: on each unconverged row whose unsatisfied-check
        count has not dropped below its lowest for n flip iterations, copy l_v, e_v
        and the check-to-variable messages self._c2v (and with
        syndrome_flip_last_n the syndrome) from the other unconverged
        replica of the shot with the fewest unsatisfied checks among replicas
        on their own flip iteration (ties to the lowest replica index), when
        that replica has fewer than the row, with
        that replica's per-row knob state (self._sf_var, self._stuck,
        self._osc, self._prev, self._fa_var, self._fa_u), and restart the
        row's count of stalled iterations. A row that copies a replica of the
        other kind (syndrome-flip or LLR) keeps no pending flip_undo step."""
        K, B, M = self.num_parallel, self._shots, self.H_shape[0]
        u = _unsat(e_v != 0, syndrome, self.V_c_col).sum(dim=1)
        best, age = self._st_best[rows], self._st_age[rows]
        age = torch.where(active, torch.where(u < best, 0, age + 1), age)
        best = torch.where(active, torch.minimum(best, u), best)
        R = u.numel()
        ar = torch.arange(R, device=u.device)
        uf = u.new_full((K * B,), M + 1)
        uf[rows] = torch.where(active, u, M + 1)
        pos = torch.zeros_like(uf)
        pos[rows] = ar
        u3 = uf.view(K, B).expand(K, K, B).clone()
        u3[torch.arange(K), torch.arange(K)] = M + 1  # not the row itself
        src_u, src_k = u3.min(dim=1)
        src = (src_k * B + torch.arange(B, device=u.device)).flatten()[rows]
        src_u = src_u.flatten()[rows]
        stalled = active & (age >= self._stall)
        copy = stalled & (src_u < u)
        idx = torch.where(copy, pos[src], ar)
        l_v.copy_(l_v[idx])
        e_v.copy_(e_v[idx])
        self._c2v.copy_(self._c2v[idx])
        if self._sf_var is not None:
            syndrome.copy_(syndrome[idx])
        for x in (
            self._sf_var,
            self._stuck,
            self._osc,
            self._prev,
            self._fa_var,
            self._fa_u,
        ):
            if x is not None:
                x[rows] = x[rows][idx]
        if self._fa_var is not None and self._sf_rep is not None:
            # a pending undo is of the source's flip kind: drop it across kinds
            kind = self._per_row(self._sf_rep, R)
            cross = copy & (kind[idx] != kind)
            self._fa_var[rows] = torch.where(cross, -1, self._fa_var[rows])
        self._st_age[rows] = torch.where(stalled, 0, age)
        self._st_best[rows] = torch.where(copy, src_u, best)

    def _agree(self, i, rows, e_v, active):
        """Cache first convergence against the true syndrome, then stop active
        siblings when a shot has agree_stop matching converged replicas."""
        K, B, N = self.num_parallel, self._shots, self.H_shape[1]
        new = ~active & (self._ag_iter[rows] == 0) & ~self._ag_stopped[rows]
        canonical = e_v[:, :N] != 0
        if self._sf_var is not None:
            canonical = canonical ^ self._sf_var[rows][:, :N]
        self._ag_iter[rows] = torch.where(new, i, self._ag_iter[rows])
        hashed = (canonical.long() * self._ag_weights).sum(1)
        self._ag_hash[rows] = torch.where(new, hashed, self._ag_hash[rows])
        hashes = self._ag_hash.view(K, B)
        known = self._ag_iter.view(K, B) > 0
        same = (hashes[:, None] == hashes[None, :]) & known[:, None] & known[None, :]
        count, representative = same.sum(1).max(0)
        decided = (count >= self.agree_stop) & (self._ag_done == 0)
        shots = torch.arange(B, device=self.device)
        group = (hashes == hashes[representative, shots]) & known
        first = self._ag_iter.view(K, B).masked_fill(~group, self.max_iter + 1).argmin(0)
        self._ag_winner = torch.where(decided, first * B + shots, self._ag_winner)
        self._ag_done = torch.where(decided, i, self._ag_done)
        self._ag_count = torch.where(decided, count, self._ag_count)
        self._hook_stop = active & (self._ag_done.repeat(K)[rows] > 0)
        self._ag_stopped[rows] |= self._hook_stop
        return active & ~self._hook_stop

    def _iter_hook(self, i, l_v, e_v, active, syndrome) -> None:
        """bp_lottery's sign-flip hook on the rows whose replica flips at
        iteration i: on the replica's flip_interval grid from flip_start_iter
        + 1, and with at least one flip under flip_anneal. With a system replica,
        a call with no such row
        returns before drawing, as in bp_lottery_cuda, so the replica
        generators advance the same on both paths. Under flip_anneal, flip
        repeat j >= 1 runs bp_lottery's pick again on the rows with more than j
        flips, with the next draw of each replica, and with every check
        touching a variable flipped earlier in this call's flips (tracked in
        self._flipped, LLR and syndrome-flip rows alike) counted as satisfied,
        so each repeat flips a variable on a different unsatisfied check and
        no variable flips twice. Before max_iter, each iteration runs _track,
        then _consensus; on each row's flip grid: _resample, then the flip_undo
        step, which takes an undone row out of this iteration's flip.
        With agree_stop, first convergences vote before tracking or perturbing
        active rows, independently of each replica's flip grid.
        At i >= max_iter, the hook does nothing: the final LLRs and hard
        decisions have already been computed."""
        self._hook_stop = None
        if i >= self.max_iter:
            return
        R = active.numel()
        self._rows_now = self._rows(R)
        self._synd_live = syndrome
        if self.agree_stop:
            self._synd_rows = self._hook_rows
            active = self._agree(i, self._rows_now, e_v, active)
            if getattr(self, "_hook_final", False) or not active.any():
                return
        self._track(self._rows_now, active, e_v, syndrome)
        if self._cons_m:
            self._consensus(i, self._rows_now, active, l_v)
        if self._fsi is None and self._fi is None:
            if not is_flip_iter(self, i):
                return
        else:
            starts = self.flip_start_iter if self._fsi is None else self._per_row(self._fsi, R)
            intervals = self.flip_interval if self._fi is None else self._per_row(self._fi, R)
            on_grid = (i > starts) & ((i - starts - 1) % intervals == 0)
            if not on_grid.any():
                return
            active = active & on_grid
        if self._anneal is not None:
            nk = [self._n_flips(p, i) for p in self._anneal]
            n = self._per_row(torch.tensor(nk, device=active.device), R)
            active = active & (n > 0)
        if self._stall:
            self._resample(self._rows_now, active, l_v, e_v, syndrome)
        if self._fa_var is not None:
            # undo the last flip iteration's flip where the unsatisfied checks
            # grew; an undone row makes no new flip at this iteration
            rows = self._rows_now
            u = _unsat(e_v != 0, syndrome, self.V_c_col).sum(dim=1)
            v = self._fa_var[rows]
            undo = active & (v >= 0) & (u > self._fa_u[rows])
            self._apply_flip(l_v, v.clamp(min=0), undo)
            self._fa_var[rows] = torch.where(active, -1, v)
            active = active & ~undo
        if self.random_machine == "system" and not active.any():
            return
        if self._anneal is not None:
            self._flipped = torch.zeros_like(l_v, dtype=torch.bool)
        self.i = i
        s_est = ((e_v[:, self.V_c_col] != 0).sum(dim=2, dtype=torch.uint8) & 1).to(
            self.dtype
        )
        self._active = active
        self.sign_flip_cn_rand_new(syndrome, s_est, l_v)
        if self._anneal is None:
            return
        for j in range(1, max(nk)):
            touched = self._flipped[:, self.V_c_col].any(dim=2)
            self._active = active & (n > j)
            self._repeat = j
            self.sign_flip_cn_rand_new(
                torch.where(touched, s_est, syndrome), s_est, l_v
            )
        self._repeat = 0
        self._flipped = None

    def sign_flip_cn_rand_new(self, syndrome, s_est, l_v):
        """bp_lottery's flip on the rows in self._active, with one draw per row:
        draw an unsatisfied check (uniformly, or with stuck_check_weight with
        probability proportional to its run in self._stuck, uniformly on a row
        whose unsatisfied checks all have run 0), then among its
        variables (the consensus flip candidates among them, if any) take the
        one on the most unsatisfied checks, ties to the smallest |LLR|
        (flip_tiebreak llr) or to the most hard-decision changes, then the
        smallest |LLR| (osc), then the lowest index. With flip_temperature,
        draw the variable instead from those on an unsatisfied check (the
        consensus flip candidates among them, if any). Flip it with
        _apply_flip, and with flip_undo record it for the undo check."""
        unsat_cn_mask = ((syndrome + s_est) % 2.0).bool()
        R = unsat_cn_mask.shape[0]
        total_unsat = unsat_cn_mask.sum(dim=1)
        valid_mask = total_unsat > 0
        r = self._draw_r(R)
        N = self.H_shape[1]
        counts = vn_unsat_count(unsat_cn_mask.float(), self.V_c_col, N)
        llr = torch.abs(l_v[:, :-1])
        if self._temp is not None:
            selected = self._temperature_pick(r, self._restrict(counts > 0), llr)
        elif self._stuck is None:
            total_unsat_safe = total_unsat + (total_unsat == 0).float()
            unsat_cumsum = unsat_cn_mask.cumsum(dim=1)
            rand_pos = torch.floor(r * total_unsat_safe).long() + 1
            chosen_cn = (unsat_cumsum >= rand_pos.unsqueeze(1)) & unsat_cn_mask
            chosen_cn_idx = torch.argmax(chosen_cn.float(), dim=1)
        else:
            # first check whose cumulative weight passes r * total weight; a
            # row whose unsatisfied checks all weigh 0 draws them uniformly
            w = self._stuck[self._rows_now] * unsat_cn_mask
            zero = w.sum(dim=1, keepdim=True) == 0
            w = torch.where(zero, unsat_cn_mask.to(w.dtype), w)
            w_cum = w.cumsum(dim=1)
            pos = (r * w_cum[:, -1]).unsqueeze(1)
            chosen_cn_idx = ((w_cum > pos) & (w > 0)).to(torch.uint8).argmax(dim=1)
        if self._temp is None:
            cand = self._restrict(cn_row_mask(self.V_c_col, chosen_cn_idx, N))
            if self._osc is None:
                score = counts * 1e6 - llr
                selected = torch.argmax(score + (~cand).float() * -1e9, dim=1)
            else:
                osc = self._osc[self._rows_now][:, :-1]
                selected = _lex_argmax(cand, counts, osc, -llr)
        flip = valid_mask & self._active
        self._apply_flip(l_v, selected, flip)
        if self._fa_var is not None and self._repeat == 0:
            rows = self._rows_now
            self._fa_var[rows] = torch.where(flip, selected, self._fa_var[rows])
            self._fa_u[rows] = torch.where(flip, total_unsat, self._fa_u[rows])
        return l_v

    def _apply_flip(self, l_v, var, mask):
        """On each row in mask, negate the LLR of variable var, or on a
        syndrome_flip_last_n row, flip the syndrome bits of var's checks in
        self._synd_live and mark var in self._sf_var. Under flip_anneal, mark
        var in self._flipped on each row in mask."""
        if self._flipped is not None:
            v1 = var.unsqueeze(1)
            self._flipped.scatter_(
                1, v1, self._flipped.gather(1, v1) | mask.unsqueeze(1)
            )
        if self._sf_rep is not None:
            sf_row = self._per_row(self._sf_rep, mask.numel())
            sf = mask & sf_row
            mask = mask & ~sf_row
            hit = (self.V_c_col == var.view(-1, 1, 1)).any(dim=2) & sf.unsqueeze(1)
            s = self._synd_live
            s.copy_(torch.where(hit, 1 - s, s))
            rows = self._rows_now
            sv = self._sf_var[rows]
            v1 = var.unsqueeze(1)
            sv.scatter_(1, v1, sv.gather(1, v1) ^ sf.unsqueeze(1))
            self._sf_var[rows] = sv
        flip_rows(l_v, var, mask)

    def _exit_hook(self, l_v, e_v, num_iters, converges) -> None:
        """With syndrome_flip_last_n, flip back each variable in self._sf_var
        in e_v and negate its LLR in l_v, so e_v and llr answer the true
        syndrome: H e_v + synd is the same as before against the flipped
        syndrome, and converge stays valid."""
        if self._sf_var is not None:
            f = self._sf_var
            e_v.copy_(torch.where(f, 1 - e_v, e_v))
            l_v.copy_(torch.where(f, -l_v, l_v))

    def _restrict(self, mask):
        """[R, N] the candidate variables in mask that are in the consensus
        flip candidates self._cz_dis, on each row where that leaves any;
        mask on the other rows, on rows with a variable in self._sf_var, and
        without consensus_every."""
        if self._cz_dis is None:
            return mask
        rows = self._rows_now
        m = mask & self._cz_dis[rows] & self._no_sf(rows).unsqueeze(1)
        return torch.where(m.any(dim=1, keepdim=True), m, mask)

    def _temperature_pick(self, r, on, llr):
        """[R] per row, the variable drawn from those in on with probability
        proportional to exp(-|LLR| / flip_temperature), by inverse CDF at r."""
        a = llr.masked_fill(~on, math.inf)
        w = torch.exp((a.amin(dim=1, keepdim=True) - a) / self._temp)
        w_cum = w.masked_fill(~on, 0.0).cumsum(dim=1)
        hit = (w_cum > (r * w_cum[:, -1]).unsqueeze(1)) & on
        # r * total can round up to total: then take the last variable in on
        last = on.shape[1] - 1 - on.flip(1).to(torch.uint8).argmax(dim=1)
        return torch.where(hit.any(dim=1), hit.to(torch.uint8).argmax(dim=1), last)

    def _draw_r(self, n):
        """[n] uniform draws for the flip pick, one per hook row: replica k's row
        draws from seeds[k] (Sobol value (repeat * max_iter + iteration self.i)
        of its sequence, or the next value of its generator). Drawn for all
        B * K rows, then indexed by self._hook_rows when the n rows are a
        compacted subset."""
        B, j = self._shots, self._repeat * self.max_iter + self.i - 1
        r = torch.cat(
            [
                torch.rand(B, generator=g, device=self.device, dtype=self.dtype)
                if m == "system"
                else self._rk[k, j].expand(B)
                for k, (m, g) in enumerate(zip(self._machines, self._gens))
            ]
        )
        return r if n == r.numel() else r[self._hook_rows]

    def _parallel_forward(self, io_dict, base_forward):
        """Decode the K replicas of every shot in one base_forward call, then
        return the selected replica's e_v, llr, iter and converge per shot, and
        defer True only where every replica deferred. Among converged replicas
        the score is iter (min_iter), the llr0 sum over e_v (min_flip_prior), or
        the sum of each replica's own posterior LLRs over its flipped bits
        in float64 (min_flip_posterior). A
        shot with no converged replica scores each replica by its unsatisfied checks,
        the weight of (H e_v + synd) mod 2. The lowest score wins. With
        report_agreement, agreement [B] is the number of converged replicas
        whose e_v equals the returned e_v. With ensemble_llr, only llr changes
        to the log odds of the mean replica error probability. A shot decided
        by agree_stop returns the agreed decision, its earliest member's LLRs
        and the detection iteration, overriding select and ensemble_llr."""
        K = self.num_parallel
        synd = io_dict["synd"]
        B = synd.shape[0]
        self._shots = B
        self._reset_state()
        out = base_forward(
            {"synd": synd.repeat(K, 1), "llr0": io_dict["llr0"].repeat(K, 1)}
        )
        conv = out["converge"].view(K, B) == 1
        e_v = out["e_v"]
        if self.select == "min_flip_prior":
            llr0 = io_dict["llr0"].to(e_v)
            score = (e_v.view(K, B, -1) * llr0).sum(2)
        elif self.select == "min_flip_posterior":
            score = (e_v.double() * out["llr"].double()).view(K, B, -1).sum(2)
        else:
            score = out["iter"].view(K, B)
        # parity of e_v on each check over V_c_col; padded edges read the zero
        # column N
        e_b = torch.nn.functional.pad((e_v != 0).to(torch.uint8), (0, 1))
        synd_u8 = synd.to(e_v.device, torch.uint8)
        unsat = _unsat(e_b.bool(), synd_u8.repeat(K, 1), self.V_c_col).view(K, B, -1).sum(2)
        score = torch.where(
            conv.any(0),
            torch.where(conv, score.double(), torch.inf),
            unsat.double(),
        )
        # argmin returns the first minimum: the lowest replica index on a tie
        best = score.argmin(0)
        idx = best * B + torch.arange(B, device=best.device)
        if self.agree_stop:
            agreed = self._ag_done > 0
            idx = torch.where(agreed, self._ag_winner, idx)
        llr = out["llr"][idx]
        iterations = out["iter"][idx]
        if self.agree_stop:
            iterations = torch.where(agreed, self._ag_done.to(iterations), iterations)
        if self.ensemble_llr:
            p = torch.sigmoid(-out["llr"].view(K, B, -1)).mean(0)
            eps = torch.finfo(p.dtype).eps
            p = p.clamp(eps, 1.0 - eps)
            llr = torch.log(1.0 - p) - torch.log(p)
            if self.agree_stop:
                llr = torch.where(agreed.unsqueeze(1), out["llr"][idx], llr)
        io_dict.update(
            {
                "e_v": e_v[idx],
                "llr": llr,
                "iter": iterations,
                "converge": out["converge"][idx],
                "defer": out["defer"].view(K, B).all(0),
            }
        )
        if self.report_agreement:
            same = (e_v.view(K, B, -1) == e_v[idx]).all(2) & conv
            io_dict["agreement"] = same.sum(0)
            if self.agree_stop:
                io_dict["agreement"] = torch.where(agreed, self._ag_count, io_dict["agreement"])
        return io_dict


class create(Parallel, _LotteryPy):
    """
    Parallel lottery BP decoder: K replicas of bp_lottery per shot, each with its
    own seed, decoded as one batch of B * K rows (row k * B + b is replica k of
    shot b). Per shot it returns one converged replica picked by select, or,
    if none converged, the replica with the fewest unsatisfied checks; ties go
    to the lowest replica index.

    Accepts every bp_lottery key (flip_start_iter defaults to 0 here) plus
        num_parallel: int (default 8)   replicas per shot, K
        seeds: list of K ints           one seed per replica; absent, K seeds are
                                        drawn from the global torch RNG at init
        select: 'min_iter' (default)    the converged replica with the fewest
                                        iterations
              | 'min_flip_prior'             the converged replica with the smallest
                                        sum of llr0 over its flipped bits
                                        (prior negative log likelihood up to a constant)
              | 'min_flip_posterior'         the converged replica with the smallest
                                        sum of its own returned posterior LLRs
                                        over its flipped bits, in float64
        report_agreement: bool (default false)  adds agreement [B], the number
                                        of converged replicas whose e_v equals
                                        the returned e_v
        agree_stop: int in [0, K] (default 0, off)  stop a shot once this many
                                        converged replicas agree; return their
                                        decision, the earliest member's LLRs,
                                        and the detection iteration
        ensemble_llr: bool (default false)  llr from mean replica error
                                        probabilities; winner e_v is unchanged
        flip_start_iter: int >= 0 or list of K such ints (default 0)
                                        per-replica flip start
        flip_interval: int >= 1 or list of K such ints (default 1)
                                        per-replica grid from flip_start_iter + 1
        random_machine: 'sobol' | 'system' or list of K such strings
                                        per-replica RNG (default 'sobol')
        flip_anneal: [T0, T1] or K such pairs   flips per unconverged row per
                                        iteration, linear from T0 at iteration
                                        1 to T1 at max_iter, rounded half up,
                                        at least 0; absent, one flip
        flip_tiebreak: 'llr' (default) | 'osc'  among the chosen check's
                                        variables on the most unsatisfied
                                        checks, flip the one with the smallest
                                        |LLR|, or the one whose hard decision
                                        changed most often (bp_sf counter)
        stuck_check_weight: bool (default false)  draw the unsatisfied check
                                        with probability proportional to the
                                        iterations it has been unsatisfied in
                                        a row (uniformly when all weigh 0)
        flip_temperature: T > 0         draw the flipped variable among those
                                        on an unsatisfied check with
                                        probability proportional to
                                        exp(-|LLR| / T); absent, the check pick
        flip_undo: bool (default false)  undo a flip at the next flip iteration
                                        when the unsatisfied count grew
        consensus_every: m (default 0, off)  every m iterations, freeze the
                                        variables all replicas of a shot other
                                        than the syndrome_flip_last_n ones
                                        agree on with min |LLR| > consensus_llr
                                        (default 10.0, needs consensus_every)
                                        at their LLRs, and flip only variables
                                        they disagree on
        copy_on_stall: n (default 0, off)  a row whose unsatisfied count has
                                        not dropped for n flip iterations copies the
                                        state of the unconverged sibling with the
                                        fewest unsatisfied checks among replicas
                                        on their own flip iteration, with its
                                        per-row knob state, then flips
        syndrome_flip_last_n: n (default 0)  the last n replicas flip the
                                        syndrome bits of the picked variable's
                                        checks instead of its LLR sign; the
                                        variable is flipped back in e_v at the
                                        end
    With random_machine sobol, replica k uses a scrambled Sobol sequence seeded
    with seeds[k]; with system, a torch.Generator seeded with seeds[k].
    copy_on_stall or syndrome_flip_last_n runs the decode eager;
    syndrome_flip_last_n also runs without compaction.
    """

    def __init__(self, decoding_cfg, **kwargs) -> None:
        super().__init__(decoding_cfg, **kwargs)
        self._init_parallel(decoding_cfg)
        if self._stall or self._sf_rep is not None:
            self.compile = False
            self.cn_update = self._cn_update_parallel
        if self._sf_rep is not None:
            # never compacted, so row r is always replica r // B and the
            # hook's syndrome is the decode loop's
            self.compact_frac = 0.0
        self.algo = "bp_lottery_parallel"

    def _cn_update_parallel(self, a_v2c, syndrome_odd, out):
        """bp_norm_min_sum's cn_update for the flip knobs: with
        syndrome_flip_last_n, the check signs read from self._synd_live once
        the hook has set it; out kept as self._c2v for the hook."""
        s = self._synd_live
        if self._sf_rep is not None and s is not None:
            if self.agree_stop and s.shape[0] != out.shape[0]:
                s = s[torch.searchsorted(self._synd_rows, self._hook_rows)]
                self._synd_live, self._synd_rows = s, self._hook_rows
            syndrome_odd = (s != 0.0).unsqueeze(2)
        type(self).cn_update(self, a_v2c, syndrome_odd, out)
        self._c2v = out
        return out

    def forward(self, io_dict):
        return self._parallel_forward(io_dict, super().forward)
