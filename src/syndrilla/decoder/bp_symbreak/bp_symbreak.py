"""SymBreak, Yin et al., arXiv:2412.02885, Algorithm 1, Sec. IV-C and V.

Source: https://arxiv.org/abs/2412.02885. This is a study implementation.
Implementation contract, including deterministic choices not fixed by the paper:
* BP variant: use this repository's normalized min-sum and global beta schedule.
* Overlapping check: choose the lowest mean absolute posterior LLR, then row id.
  Equal qubit reliabilities use qubit id. BP odd rows use floor(size / 2).
  If too few non-overlap qubits exist, keep the smaller valid odd-overlap side.
* Each original check can split once. Both resulting checks are unsplittable,
  so at most twice the original number of checks can become active. Select
  only an unsplit original with positive even overlap. If none exists, skip
  the split without substituting another opposite check.
* Message reset: moved edges send their channel LLR on the next iteration;
  all other edge messages persist. Splits and their syndromes are shot-local.
* Syndrome-guided mode searches balanced qubit partitions in lexicographic
  qubit-id order, with floor(size / 2) on the first side for odd rows. Its
  overlap is k for odd k, or k - 1 or k + 1 for even k, out of 2k qubits.
  The first side must have only zero-syndrome adjacent checks other than the
  parent; no adjacent checks also qualifies. Assign it zero and the other
  side the parent's syndrome. The paper gives no rule when no partition
  qualifies; this implementation skips that split without another selection.
* The first BP block establishes the previous mismatch count. K is updated at
  later block boundaries. Convergence and cap checks use the original matrix
  only at block ends. The finite-budget convention also checks a final partial
  block at max_iter and performs no split after the last update. num_iters
  counts BP updates through these checks. Early termination false disables
  convergence and K stops, and returns the state at the total BP budget.
* Unary split checks have an infinite min-sum message. Bound real messages by
  sqrt(finfo.max) / (edge_slots + 1) to avoid inf - inf. Dummy messages stay zero.
  Exhausting the split budget continues BP until a termination condition.

Remaining deviations from the paper are the NMS variant and beta schedule,
the finite BP and split budgets, the optional cap and full-budget controls,
and finite-message clipping. The BB-specific split from Sec. VI is not used.

Optimization groups and explicit member overrides use the shared NMS rules.
Active-row compaction, BP buffer reuse and per-shot incident-edge gathering are
available. Check-reliability sums use slot order on both devices so reductions
do not choose different checks at a floating-point tie.
The scatter alternative preserves CPU edge order. On CUDA, torch
scatter uses atomics and can change floating-point sums. The kernel port uses
ordered scatter. The NMS check update is shared by the torch graph loop.
"""

import math
from itertools import combinations

import torch

from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum import create as Nms
from syndrilla.decoder.knobs import reject_unknown_keys


class create(Nms):
    _config_keys = ("bp_iters", "k_max", "max_splits", "assign")
    _config_algorithm = "bp_symbreak"

    def __init__(self, decoding_cfg, **kwargs):
        reject_unknown_keys(
            decoding_cfg, self._base_config_keys.union(self._config_keys),
            decoding_cfg.get("algorithm", self._config_algorithm),
        )
        for key, default, minimum in (("bp_iters", 10, 1), ("k_max", 3, 1),
                                      ("max_splits", 10, 0), ("max_iter", 50, 1)):
            value = decoding_cfg.get(key, default)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
            setattr(self, key, value)
        self.assign = decoding_cfg.get("assign", "bp")
        if self.assign not in ("bp", "syndrome"):
            raise ValueError("assign must be bp or syndrome")
        if decoding_cfg.get("dtype", "float64") not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if decoding_cfg.get("check_type", "hx") not in ("hx", "hz"):
            raise ValueError("check_type must be hx or hz")
        super().__init__(dict(decoding_cfg, compile=False), **kwargs)
        self.algo = "bp_symbreak"
        other = "hz" if self.check_type == "hx" else "hx"
        self.register_buffer("opposite_col", kwargs["bundle"].select(other)[2].to(self.device))
        # Masking is per shot because check edges can move to different rows.
        self.dummy_idx = self.dummy_idx[:0]

    def _split(self, row, col, syndrome, posterior, hard, incoming, reset, splits,
               splittable):
        """Split one check for one shot; all tensors except splits omit the batch."""
        n, m = self.H_shape[1], self.H_shape[0]
        reliability = posterior.abs()
        reliability = reliability.clone()
        reliability[n] = 0
        def check_sum(columns):
            total = reliability.new_zeros(len(columns))
            for qubits in columns.unbind(1):
                total.add_(reliability[qubits])
            return total
        other_valid = self.opposite_col != n
        other_degree = other_valid.sum(1)
        means = check_sum(self.opposite_col) / other_degree.clamp_min(1)
        means.masked_fill_(other_degree == 0, float("inf"))
        if means.numel() == 0 or not other_degree.any():
            return False
        g = self.opposite_col[means.argmin()]
        overlap = (col.unsqueeze(2) == g).any(2) & (col != n)
        count = overlap.sum(1)
        degree = (col != n).sum(1)
        eligible = splittable & (count > 0) & (count.remainder(2) == 0) & (degree > 1)
        if not eligible.any():
            return False
        means = check_sum(col) / degree.clamp_min(1)
        means.masked_fill_(~eligible, float("inf"))
        c = int(means.argmin())
        qubits = col[c]
        # Sort by qubit id first because moved rows need not retain slot order.
        def reliable_slots(mask):
            slots = mask.nonzero().flatten()
            slots = slots[torch.argsort(qubits[slots], stable=True)]
            return slots[torch.argsort(reliability[qubits[slots]], descending=True, stable=True)]
        k = int(count[c]) // 2
        if self.assign == "syndrome":
            slots = (qubits != n).nonzero().flatten()
            slots = slots[torch.argsort(qubits[slots], stable=True)]
            allowed_overlap = (k,) if k % 2 else (k - 1, k + 1)
            for candidate in combinations(slots.tolist(), int(degree[c]) // 2):
                first = slots.new_tensor(candidate)
                if int(overlap[c, first].sum()) not in allowed_overlap:
                    continue
                neighbours = (col.unsqueeze(2) == qubits[first]).any(2).any(1)
                neighbours[c] = False
                if not syndrome[neighbours].any():
                    break
            else:
                return False
            s1 = syndrome.new_zeros((), dtype=torch.uint8)
        else:
            keep_overlap = k if k % 2 else k - 1
            first = reliable_slots(overlap[c])[:keep_overlap]
            need = int(degree[c]) // 2 - len(first)
            first = torch.cat((first, reliable_slots((qubits != n) & ~overlap[c])[:max(need, 0)]))
            s1 = hard[qubits[first]].to(torch.uint8).sum().remainder(2)
        keep = torch.zeros_like(qubits, dtype=torch.bool)
        keep[first] = True
        moved = (qubits != n) & ~keep
        new = m + int(splits[row])
        col[new, moved] = qubits[moved]
        col[c, moved] = n
        syndrome[new] = syndrome[c].to(torch.uint8) ^ s1
        syndrome[c] = s1
        incoming[c, moved] = 0
        reset[new, moved] = True
        splittable[c] = False
        splittable[new] = False
        splits[row] += 1
        return True

    def _graph_incident(self, col):
        """Incident edge ids in row order; moving edges does not change VN degree."""
        b, c, d = col.shape
        n, vd = self.H_shape[1], self.vn_adj.numel() // (self.H_shape[1] + 1)
        batch, edge = (col.flatten(1) != n).nonzero(as_tuple=True)
        group = batch * (n + 1) + col.flatten(1)[batch, edge]
        order = torch.argsort(group, stable=True)
        group, edge = group[order], edge[order]
        counts = torch.bincount(group, minlength=b * (n + 1))
        slot = torch.arange(len(edge), device=col.device) - (counts.cumsum(0) - counts)[group]
        adjacent = torch.full((b * (n + 1), vd), c * d, device=col.device, dtype=torch.long)
        adjacent[group, slot] = edge
        return adjacent.view(b, n + 1, vd)

    def _graph_buffers(self, b, c, d, n, vd):
        return (torch.empty((b, c, d), device=self.device, dtype=self.dtype),
                torch.empty((b, c, d), device=self.device, dtype=self.dtype),
                torch.empty((b, c, d), device=self.device, dtype=self.dtype),
                torch.empty((b, n + 1), device=self.device, dtype=self.dtype),
                torch.zeros((b, c * d + 1), device=self.device, dtype=self.dtype),
                torch.empty((b, (n + 1) * vd), device=self.device, dtype=self.dtype))

    @torch.no_grad()
    def forward(self, io_dict):
        original = io_dict["synd"].to(device=self.device, dtype=self.dtype)
        channel = io_dict["llr0"].to(device=self.device, dtype=self.dtype)
        b, n = channel.shape
        m, degree = self.V_c_col.shape
        checks = m + min(self.max_splits, m)
        if original.shape != (b, m) or n != self.H_shape[1]:
            raise ValueError("synd and llr0 must match the matrix and batch shapes")
        if not torch.isfinite(channel).all():
            raise ValueError("llr0 must contain finite channel LLRs")
        self.batch_size = b
        col = torch.full((b, checks, degree), n, device=self.device, dtype=torch.long)
        col[:, :m] = self.V_c_col
        syndrome = torch.zeros((b, checks), device=self.device, dtype=self.dtype)
        syndrome[:, :m] = original
        prior = torch.cat((channel, channel.new_full((b, 1), float("inf"))), 1)
        posterior = prior.clone()
        incoming = torch.zeros_like(col, dtype=self.dtype)
        reset = torch.zeros_like(col, dtype=torch.bool)
        splittable = torch.zeros((b, checks), device=self.device, dtype=torch.bool)
        splittable[:, :m] = True
        active = torch.ones(b, device=self.device, dtype=torch.bool)
        splits = torch.zeros(b, device=self.device, dtype=torch.long)
        num_iters = torch.zeros_like(splits)
        converges = torch.zeros_like(splits)
        monitor = torch.zeros_like(splits)
        previous = torch.full_like(splits, -1)
        limit = math.sqrt(torch.finfo(self.dtype).max) / (col.shape[1] * degree + 1)
        self.i = 0
        cap_frac = (self.cap.frac if self.cap is not None and self.cap.done
                    and not self.cap_bypass else None)
        cap_stopped = False
        rows = torch.arange(b, device=self.device)
        vd = self.vn_adj.numel() // (n + 1)
        incident = None
        if self.c2v_gather:
            base = self.vn_adj.reshape(-1, n + 1).t()
            incident = torch.where(base < m * degree, base, checks * degree)
            incident = incident.expand(b, -1, -1).clone()
        buffers = None
        while b and self.i < self.max_iter and (active.any() or not self.early_termination):
            self.i += 1
            left = active[rows].sum()
            if self.early_termination and 0 < left < self.compact_frac * len(rows):
                rows = rows[active[rows]]
            r = len(rows)
            self.batch_size = r
            if buffers is None or not self.reuse_buffers:
                buffers = self._graph_buffers(r, checks, degree, n, vd)
            outgoing, updated, channel_edges, total, edge_flat, gathered = (value[:r] for value in buffers)
            full = r == b
            cols = col if full else col[rows]
            old = incoming if full else incoming[rows]
            values = posterior if full else posterior[rows]
            priors = prior if full else prior[rows]
            pending = reset if full else reset[rows]
            flat_col = cols.flatten(1)
            torch.gather(values, 1, flat_col, out=outgoing.view(r, -1))
            outgoing.sub_(old)
            torch.gather(priors, 1, flat_col, out=channel_edges.view(r, -1))
            torch.where(pending, channel_edges, outgoing, out=outgoing)
            reset[rows] = False
            self.cn_update(outgoing, syndrome[rows].bool().unsqueeze(2), updated)
            updated.clamp_(-limit, limit).masked_fill_(cols == n, 0)
            running = active[rows]
            old[running] = updated[running]
            if not full:
                incoming[rows] = old
            total.zero_()
            if self.c2v_gather:
                edges = incident if full else incident[rows]
                edge_flat[:, :-1].copy_(old.flatten(1))
                torch.gather(edge_flat, 1, edges.flatten(1), out=gathered)
                for slot in gathered.view(r, n + 1, vd).unbind(2):
                    total.add_(slot)
            else:
                total.scatter_add_(1, flat_col, old.flatten(1))
            values[running] = (total + priors)[running]
            if not full:
                posterior[rows] = values
            num_iters[active] = self.i
            if self.i % self.bp_iters and self.i != self.max_iter:
                continue
            hard = posterior <= 0
            estimated = hard[:, self.V_c_col].sum(2).remainder(2)
            mismatch = (estimated != original).sum(1)
            if self.early_termination:
                done = active & (mismatch == 0)
                converges[done] = 1
                active[done] = False
                if cap_frac is not None and int(converges.sum()) >= cap_frac * b:
                    cap_stopped = bool(active.any() and self.i < self.max_iter)
                    break
            if self.i == self.max_iter:
                continue
            seen = previous >= 0
            monitor = torch.where(active & seen & (mismatch >= previous), monitor + 1,
                torch.where(active & seen & (mismatch < previous), (monitor - 1).clamp_min(0), monitor))
            previous[active] = mismatch[active]
            if self.early_termination:
                active[monitor >= self.k_max] = False
            changed = False
            for row in (active & (mismatch > 0) & (splits < self.max_splits)).nonzero().flatten().tolist():
                changed |= self._split(row, col[row], syndrome[row], posterior[row], hard[row],
                                       incoming[row], reset[row], splits, splittable[row])
            if changed and self.c2v_gather:
                incident = self._graph_incident(col)
        if b:
            hard = posterior <= 0
        else:
            hard = torch.zeros_like(prior, dtype=torch.bool)
        if not self.early_termination:
            converges[:] = (hard[:, self.V_c_col].sum(2).remainder(2) == original).all(1)
        self.batch_size = b
        self.cap_active_last = cap_stopped
        defer = (converges == 0) if cap_stopped else torch.zeros_like(active)
        if self.cap is not None and not self.cap.done and not self.cap_bypass and b:
            self.cap.observe(num_iters, self.max_iter, b)
        io_dict.update(e_v=hard[:, :n].to(self.dtype), llr=posterior[:, :n],
                       iter=num_iters, converge=converges, defer=defer, splits=splits)
        return io_dict
