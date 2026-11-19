import torch

from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum import create as _Nms


def check_layers(cols, n):
    """Partition the checks into layers by first-fit coloring in check order.

    Layer rule, from D. E. Hocevar, US Patent 7,730,377 B2, "Layered decoding
    of low density parity check (LDPC) codes" (Texas Instruments, filed 2004):
    the rows of H are grouped into non-overlapping subsets, and each column
    has a maximum weight of one in each subset. A row is a check node, a
    column is a variable node, so the checks in one layer share no variable.
    Within a layer the check updates write disjoint posteriors and run in
    parallel; across layers the variables overlap, and each layer reads the
    posteriors the earlier layers wrote in the same iteration. Hocevar takes
    the block rows of a quasi-cyclic H as layers; this greedy coloring
    recovers the rule for an arbitrary H. Dummy variables (index >= n) are
    ignored.
    """
    layers, used = [], []
    for check, row in enumerate(cols.detach().cpu().tolist()):
        variables = {v for v in row if v < n}
        for checks, occupied in zip(layers, used):
            if variables.isdisjoint(occupied):  # layer rule: no variable shared within a layer
                checks.append(check)
                occupied.update(variables)
                break
        else:
            layers.append([check])
            used.append(variables)
    return [torch.tensor(layer, device=cols.device) for layer in layers]


class create(_Nms):
    """Normalized min-sum with one ordered sweep of disjoint check layers."""

    _config_keys = ()
    _config_algorithm = "bp_norm_min_sum_layered"

    def __init__(self, decoding_cfg, **kwargs):
        # Layer updates need eager, in-place posterior writes between layers.
        super().__init__({**decoding_cfg, "compile": False}, **kwargs)
        self.algo = "bp_norm_min_sum_layered"
        self.layers = check_layers(self.V_c_col, self.H_shape[1])
        self.layer_cols = [self.V_c_col[layer] for layer in self.layers]
        self.layer_real = [cols < self.H_shape[1] for cols in self.layer_cols]

    def _sweep(self, l_v, previous, syndrome):
        beta = 1 - torch.pow(
            torch.tensor(2.0, dtype=self.dtype),
            torch.tensor(-self.i, dtype=self.dtype),
        )
        for layer, cols, real in zip(self.layers, self.layer_cols, self.layer_real):
            old = previous[:, layer]
            a = l_v[:, cols] - old
            neg = a <= 0
            parity = (neg.sum(2, keepdim=True, dtype=torch.uint8) & 1).bool()
            flip = neg ^ (parity ^ (syndrome[:, layer] != 0).unsqueeze(2))
            mag = a.abs()
            minimum, arg = mag.min(2, keepdim=True)
            second = mag.scatter(2, arg, float("inf")).amin(2, keepdim=True)
            msg = torch.where(flip, -beta * minimum, beta * minimum)
            msg.scatter_(2, arg, torch.where(flip.gather(2, arg), -beta * second, beta * second))
            msg.masked_fill_(~real, 0)
            # Dummy slots share an index; real slots in a layer never do.
            l_v[:, cols[real]] += (msg - old)[:, real]
            previous[:, layer] = msg

    def _record_convergence(self, e_v, syndrome, it_rows):
        matches = torch.all(self.syndrome_estimation(e_v) == syndrome, dim=1)
        if self.early_termination:
            it_rows[(it_rows == -1) & matches] = self.i
        return matches

    def forward(self, io_dict):
        syndrome = io_dict["synd"].to(device=self.device, dtype=self.dtype).contiguous()
        B = self.batch_size = syndrome.shape[0]
        l_v = torch.cat((io_dict["llr0"].to(device=self.device, dtype=self.dtype),
                         torch.full((B, 1), float("inf"), device=self.device, dtype=self.dtype)), 1)
        previous = l_v.new_zeros(B, *self.V_c_col.shape)
        rows = torch.arange(B, device=self.device)
        it_rows = torch.full((B,), -1, device=self.device)
        num_iters = it_rows.clone()
        converges = torch.zeros_like(it_rows)
        l_out, e_out = torch.empty_like(l_v), torch.empty_like(l_v)
        work = l_v.new_empty(max(l_v.numel(), previous.numel())) if self.reuse_buffers else None
        capped = bool(self.cap is not None and self.cap.done and
                      self.cap.frac is not None and not self.cap_bypass)
        cap_stopped = False
        hooked = type(self)._iter_hook is not _Nms._iter_hook
        self._hook_rows = None
        for self.i in range(1, self.max_iter + 1):
            self._sweep(l_v, previous, syndrome)
            e_v = (l_v <= 0).to(self.dtype)
            was_active = it_rows == -1
            matches = self._record_convergence(e_v, syndrome, it_rows)
            if not self.early_termination:
                if hooked:
                    self._hook_rows = rows
                    self._iter_hook(self.i, l_v, e_v, ~matches, syndrome)
                continue
            new = was_active & (it_rows != -1)
            dst = rows[new]
            num_iters[dst] = self.i
            l_out[dst], e_out[dst] = l_v[new], e_v[new]
            converges[dst] = 1
            active = it_rows == -1
            n_left = int(active.sum())
            # Match NMS hook timing: stop on the sweep after the last new match.
            if not was_active.any():
                break
            if capped and int((num_iters != -1).sum()) >= self.cap.frac * B:
                cap_stopped = n_left > 0 and self.i < self.max_iter
                break
            if hooked:
                self._hook_rows = rows
                self._iter_hook(self.i, l_v, e_v, active, syndrome)
            if 0 < n_left < self.compact_frac * self.batch_size and self.i < self.max_iter:
                keep = active.nonzero().squeeze(1)
                rows, it_rows, syndrome = rows[keep], it_rows[keep], syndrome[keep]
                if self.reuse_buffers:
                    l_v = self._compact(l_v, keep, work)
                    previous = self._compact(previous, keep, work)
                else:
                    l_v, previous = l_v[keep], previous[keep]
                self.batch_size = n_left
        active = it_rows == -1
        dst = rows[active]
        l_out[dst], e_out[dst] = l_v[active], e_v[active]
        num_iters[dst] = self.i
        if not self.early_termination:
            converges[:] = matches
        self._exit_hook(l_out, e_out, num_iters, converges)
        self.cap_active_last = cap_stopped
        defer = converges == 0 if cap_stopped else torch.zeros(B, dtype=torch.bool, device=self.device)
        self.batch_size = B
        if self.cap is not None and not self.cap.done and not self.cap_bypass:
            self.cap.observe(num_iters, self.max_iter, B)
        io_dict.update(e_v=e_out[:, :-1], llr=l_out[:, :-1], iter=num_iters,
                       converge=converges, defer=defer)
        return io_dict
