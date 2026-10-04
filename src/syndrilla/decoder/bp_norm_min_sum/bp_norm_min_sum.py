import types

import torch
from loguru import logger

from syndrilla.utils import parse_device_dtype
from syndrilla.decoder.decoder import RebatchSpeedup
from syndrilla.decoder.knobs import knob

# compact the per-iteration state to the unconverged samples once fewer than this
# fraction of the rows decoded since the last compaction are still unconverged
COMPACT_FRAC = 0.75


def _step(l_v, c2v_prev, u_init, syndrome_odd, beta, col, vn_adj, mask_dummy):
    """One decoding iteration for the compiled path: v2c, vn_update, cn_update, c2v,
    llr_update, hard_decision and syndrome_estimation, with the same arithmetic as
    the eager methods. Functional: returns new (c2v_msg, l_v, e_v, s_est). On
    iteration 1, `l_v` is a copy of `u_init` and `c2v_prev` is zero, which gives the
    eager initial message bit for bit.
    """
    B, N1 = l_v.shape
    M, D = mask_dummy.shape
    a = l_v[:, col].view(B, M, D) - c2v_prev
    neg = a <= 0.0
    parity = (neg.sum(dim=2, keepdim=True, dtype=torch.uint8) & 1).bool()
    flip = neg ^ (parity ^ syndrome_odd)
    mag = a.abs()
    min_0, arg_0 = mag.min(dim=2, keepdim=True)
    min_1 = mag.scatter(2, arg_0, float("inf")).amin(dim=2, keepdim=True)
    m_0 = beta * min_0
    m_1 = beta * min_1
    msg = torch.where(flip, -m_0, m_0)
    msg = msg.scatter(2, arg_0, torch.where(flip.gather(2, arg_0), -m_1, m_1))
    msg = msg.masked_fill(mask_dummy, 0.0)
    flat = torch.cat([msg.view(B, -1), msg.new_zeros(B, 1)], 1)
    g = flat[:, vn_adj].view(B, -1, N1)
    s = torch.zeros_like(l_v)
    for k in range(g.shape[1]):
        s = s + g[:, k]
    l_new = s + u_init
    e = l_new <= 0.0
    s_est = e.to(torch.uint8)[:, col].view(B, M, D).sum(dim=2, dtype=torch.uint8) & 1
    return msg, l_new, e.to(l_v.dtype), s_est.to(l_v.dtype)


class create(torch.nn.Module):
    """
    This class creates a bp decoder on a single GPU

    A subclass can change the decoding through two hooks, with the names, argument
    order and contract of the hooks in bp_norm_min_sum_cuda: `_iter_hook` at the end
    of each iteration that does not break the loop, and `_exit_hook` once after the
    loop. The defaults do nothing; `_iter_hook` is called only when a subclass
    overrides it. Both run eagerly, on the eager and the compiled path alike.
    Both ports take the same arguments, so one override serves both. In a class
    that inherits from a CUDA decoder and a PyTorch decoder, the first base in the
    MRO supplies the hooks.
    """

    def _iter_hook(self, i, l_v, e_v, active, syndrome) -> None:
        """Called at the end of iteration i, after the convergence update and the
        early-exit checks (so not on the iteration that breaks the loop: all rows
        converged, or the rebatch cap reached) and before compaction.

        The tensors hold the rows still decoded, which after a compaction are fewer
        than the batch: row r is sample self._hook_rows[r] ([R] long, set before
        each call; the CUDA port does not compact, so there row r is sample r).
          l_v      [R, N+1] posterior LLR, column N the +inf dummy
          e_v      [R, N+1] hard decision, in the decoder dtype
          active   [R] bool, True where the row is still unconverged
          syndrome [R, M] syndrome in the decoder dtype

        The hook may change l_v in place on active rows only. The next iteration
        reads it, compaction keeps it, and an active row's l_v at loop end is the
        returned llr. Converged rows of l_v (already copied to the output), e_v,
        active, syndrome and self._hook_rows are read-only, and the hook cannot
        mark rows converged. The default does nothing.
        """

    def _exit_hook(self, l_v, e_v, num_iters, converges) -> None:
        """Called once after the decode loop, with num_iters final (no -1 left),
        before the dummy column is stripped. All four are indexed by sample over
        the whole batch: l_v [B, N+1] and e_v [B, N+1] in the decoder dtype,
        num_iters [B] and converges [B]. It may change them in place; they are the
        returned llr, e_v, iter and converge. The default does nothing.
        """

    def __init__(self, decoding_cfg, **kwargs) -> None:
        """
        Initialization for bp decoder
        Input:
            decoding_cfg: the information that come from config file (yaml)

        Parameters:
            max_iter: the number of maximum iteration of bp decoder
            i: the number of iterations running the decoder

            H_matrix: loaded ldpc matrix, either hx or hz, as 2d tensor

            V_c_row: the row index of all the variable nodes for each check node
            V_c_col: the column index of all the variable nodes for each check node

            degree: the maximum number of 1s in all check nodes in H_matrix

            compile: decoder config key, default True. Runs the iteration body
                through torch.compile, with its own compile cache per decoder
                instance. Set False to run eager. Applies on a CUDA device only,
                that is with force_pytorch or when the CUDA port is unavailable;
                ignored on CPU or without dynamo and Triton.

            Ablation knobs (decoder config keys; GROUP_OFF in
            syndrilla.decoder.knobs lists the group keys pruning_opt,
            fusion_opt, mapping_opt, gather_opt, memory_opt and rebatch_opt, default True,
            that set their members to the off value when False; an explicit
            member key wins):
            compact_frac: default COMPACT_FRAC (0.75), at most 0.75. Compact the
                state once fewer than this fraction of the rows are unconverged;
                0 never compacts.
            c2v_gather: default True. False sums the c->v messages with
                index_add_ (atomic on CUDA, so the order is not fixed) and runs
                eager. Ignored by a subclass that overrides c2v.
            cn_sign_parity: default True. False runs the product-of-signs plus
                topk(2) check-node update and runs eager. Ignored by a subclass
                that overrides cn_update.
            reuse_buffers: default True. False allocates the eager work buffers
                every iteration and compacts by plain indexing.
        """

        super(create, self).__init__()

        logger.info("Creating bp decoder.")

        # set up default device
        self.device, _ = parse_device_dtype(decoding_cfg)

        # set up default max_iter
        self.max_iter = decoding_cfg.get("max_iter", 50)
        if self.max_iter <= 0 or not isinstance(self.max_iter, int):
            logger.warning(
                f"Invalid input maximum iteration <{self.max_iter}>, default to <50>."
            )
            self.max_iter = 50

        # set up default dtype
        self.dtype = decoding_cfg.get("dtype", "float64")
        if self.dtype not in {"float32", "float64", "bfloat16", "float16"}:
            logger.warning(
                f"Invalid input data type <{self.dtype}>, default to <torch.float64>."
            )
            self.dtype = "float64"
        self.dtype = torch.__dict__[self.dtype]

        self.batch_size = 1

        self.check_type = decoding_cfg.get("check_type", "hx")
        if self.check_type.lower() not in {"hx", "hz"}:
            logger.warning(
                f"Invalid input check type <{self.check_type}>, default to <hx>."
            )
            self.check_type = "hx"

        bundle = kwargs.get("bundle")
        if bundle is None:
            raise ValueError(
                "bp_norm_min_sum requires a pre-loaded MatrixBundle via the `bundle` kwarg."
            )
        self.Hx_matrix = bundle.Hx_matrix
        self.Hz_matrix = bundle.Hz_matrix
        self.lx_matrix = bundle.lx_matrix
        self.lz_matrix = bundle.lz_matrix
        self.H_shape, self.V_c_row, self.V_c_col, self.H_matrix = bundle.select(
            self.check_type
        )

        self.mask_dummy = self.V_c_col == self.H_shape[1]
        # flat positions of the dummy slots in the [n_checks * degree] edge layout
        self.dummy_idx = self.mask_dummy.flatten().nonzero().squeeze(1)

        # variable -> check table for c2v: row n holds the flat edge ids c * degree + k
        # of variable n's edges in (c, k) order, padded with n_checks * degree, the id
        # of the zero column appended to the flat message. The dummy variable row is
        # all padding.
        edge_var = self.V_c_col.flatten().long()
        n_edges = edge_var.numel()
        edge_id = (edge_var < self.H_shape[1]).nonzero().squeeze(1)
        var = edge_var[edge_id]
        order = torch.argsort(var, stable=True)
        edge_id, var = edge_id[order], var[order]
        counts = torch.bincount(var, minlength=self.H_shape[1] + 1)
        slot = torch.arange(var.numel(), device=var.device)
        slot -= (torch.cumsum(counts, 0) - counts)[var]
        vn_adj = torch.full(
            [self.H_shape[1] + 1, max(int(counts.max()), 1)],
            n_edges,
            dtype=torch.long,
            device=var.device,
        )
        vn_adj[var, slot] = edge_id
        # flattened slot-major ([VD * (N+1)]) so one gather yields VD contiguous
        # [batch, N+1] slices
        self.vn_adj = vn_adj.t().flatten()

        # set iteration
        self.i = 0

        self.V_c_row = torch.nn.Parameter(
            self.V_c_row.to(self.device), requires_grad=False
        )
        self.V_c_col = torch.nn.Parameter(
            self.V_c_col.to(self.device), requires_grad=False
        )
        self.mask_dummy = self.mask_dummy.to(self.device)
        self.dummy_idx = self.dummy_idx.to(self.device)
        self.vn_adj = self.vn_adj.to(self.device)
        self.H_matrix = self.H_matrix.to(self.device)

        self.algo = "bp_norm_min_sum"
        self.num_max_iter = self.max_iter

        self.cap = RebatchSpeedup.from_cfg(decoding_cfg)
        self.cap_bypass = False  # set by main: True -> decode this batch uncapped
        self.cap_active_last = False  # set per forward: True if the cap stopped early

        # torch.compile of the iteration body, on by default, CUDA only; eager
        # fallback (one DEBUG line) on CPU or when dynamo/Triton is unavailable
        self.compact_frac = float(knob(decoding_cfg, "compact_frac", COMPACT_FRAC))
        if not 0 <= self.compact_frac <= 0.75:
            raise ValueError(
                f"compact_frac must be in [0, 0.75], got {self.compact_frac}"
            )
        self.reuse_buffers = bool(knob(decoding_cfg, "reuse_buffers", True))
        self.c2v_gather = (
            bool(knob(decoding_cfg, "c2v_gather", True))
            or type(self).c2v is not create.c2v
        )
        self.cn_sign_parity = (
            bool(knob(decoding_cfg, "cn_sign_parity", True))
            or type(self).cn_update is not create.cn_update
        )
        self.compile = (
            bool(knob(decoding_cfg, "compile", True))
            and self.c2v_gather
            and self.cn_sign_parity
        )
        if self.compile:
            try:
                from torch.utils._triton import has_triton

                if self.device.type != "cuda":
                    reason = f"device is {self.device.type}"
                elif not torch._dynamo.is_dynamo_supported():
                    reason = "torch.compile (dynamo) is not supported"
                elif not has_triton():
                    reason = "Triton is not available"
                else:
                    reason = None
            except Exception as e:
                reason = f"compile check failed: {e!r}"
            if reason is not None:
                logger.debug(f"compile is on but runs eager: {reason}.")
                self.compile = False
        if self.compile:
            # dynamo keeps its compile cache, and its recompile limit, per code
            # object; a copy of the code object gives each decoder its own cache
            step = types.FunctionType(
                _step.__code__.replace(), _step.__globals__, _step.__name__
            )
            self._step = torch.compile(step)
            i = torch.arange(1, self.max_iter + 1).to(self.dtype)
            # beta of every iteration, same formula as cn_update
            self.betas = (
                torch.tensor(1.0, dtype=self.dtype)
                - torch.pow(torch.tensor(2.0, dtype=self.dtype), -i)
            ).to(self.device)

        logger.info("Complete.")

    def forward(self, io_dict):
        """Iterative bp (normalized min sum) decoding algorithm
        Input:
            syndrome: estimated syndrome for c-th code node

        Output:
            e_v: estimated error for c-th code node at i-th iteration

        Parameters:
            llr:  Log-likelihood Ratio (LLR) for each v-th variable node (initialization)
            l_v: Log-likelihood Ratio (LLR) for v-th variable node at i-th iteration
            u_init: Log-likelihood Ratio (LLR) for v-th variable node (initialization)

            a_v2c: Message from the v-th variable node to c-th check node at i-th iteration
            b_c2v: Message from the c-th check node to v-th variable node at i-th iteration
            message: used to represent both a_v2c and b_c2v

            s_est:  estimated syndrome for c-th code node at i-th iteration

        Hooks: _iter_hook(i, l_v, e_v, active, syndrome) at the end of each
        iteration that does not break the loop, and _exit_hook(l_v, e_v, num_iters,
        converges) once after it; see their docstrings. They have the arguments of
        the bp_norm_min_sum_cuda hooks, and the sample index of each compacted row
        is self._hook_rows. A class with a CUDA and a PyTorch base takes the hooks
        of the first base in its MRO.
        """
        logger.info("Initializing bp (normailized min sum) decoding.")

        syndrome = io_dict["synd"].to(dtype=self.dtype).to(self.device)

        self.batch_size, _ = syndrome.size()
        B = self.batch_size

        # add a dummy element at the end in case the H (ldpc matrix) does not have the same number of 1s in each check node
        N_extended = self.H_shape[1] + 1
        l_v = torch.zeros(
            [self.batch_size, N_extended], dtype=self.dtype, device=self.device
        )
        e_v = torch.zeros(
            [self.batch_size, N_extended], dtype=self.dtype, device=self.device
        )
        s_est = torch.zeros(
            [self.batch_size, self.H_shape[0]], dtype=self.dtype, device=self.device
        )

        # add dummy column
        dummy_column = torch.full(
            [self.batch_size, 1], float("inf"), dtype=self.dtype, device=self.device
        )
        u_init = torch.cat(
            (io_dict["llr0"].to(self.device).to(self.dtype), dummy_column), dim=1
        )
        e_out = torch.zeros(
            [self.batch_size, N_extended], dtype=self.dtype, device=self.device
        )
        l_out = torch.zeros(
            [self.batch_size, N_extended], dtype=self.dtype, device=self.device
        )
        num_iters = torch.full([self.batch_size], -1, device=self.device)
        converges = torch.full([self.batch_size], 0, device=self.device)
        # the per-iteration state holds only the rows still decoded: row r is sample
        # rows[r], and it_rows[r] is its convergence iteration (-1 while unconverged)
        rows = torch.arange(B, device=self.device)
        it_rows = num_iters.clone()

        # syndrome bit per check (True where nonzero), flips the check message sign
        syndrome_odd = (syndrome != 0.0).unsqueeze(2)

        n_checks, degree = self.V_c_col.shape
        if self.compile:
            # iteration 1 of the compiled body: l_v a copy of u_init (not the same
            # tensor, which would add an alias guard) and a zero c->v message. The
            # batch dim is dynamic, so compaction and new batch sizes >= 2 reuse the
            # first graph.
            l_v = u_init.clone()
            c2v_msg = torch.zeros(
                [B, n_checks, degree], dtype=self.dtype, device=self.device
            )
            if B > 1:
                for t in (l_v, c2v_msg, u_init, syndrome_odd):
                    torch._dynamo.mark_dynamic(t, 0)
            col = self.V_c_col.detach().flatten()
        else:
            # set up initialization for all parameters for decoding process
            # message is a in place version of a_v2c and b_c2v
            message = u_init[:, self.V_c_col]

        if not self.compile and self.reuse_buffers:
            c2v_flat, c2v_msg, c2v_sum, work, v2c_buf, c2v_buf = self._work_buffers(l_v)

        logger.info("Complete.")

        logger.info("Starting decoding iterations.")

        # adaptive cap: once warm-up has chosen a stop fraction, break this batch as
        # soon as that fraction has converged (unless main asked for an uncapped pass).
        cap_applied = bool(
            self.cap is not None
            and self.cap.done
            and self.cap.frac is not None
            and not self.cap_bypass
        )
        cap_frac = self.cap.frac if cap_applied else None
        cap_stopped = False
        hooked = type(self)._iter_hook is not create._iter_hook
        self._hook_rows = None

        self.i = 0
        while self.i < self.max_iter:
            # variable node update update v2c
            self.i += 1

            if self.compile:
                c2v_msg, l_v, e_v, s_est = self._step(
                    l_v,
                    c2v_msg,
                    u_init,
                    syndrome_odd,
                    self.betas[self.i - 1],
                    col,
                    self.vn_adj,
                    self.mask_dummy,
                )
            else:
                if not self.reuse_buffers:
                    bufs = self._work_buffers(l_v)
                    c2v_flat, c2v_msg, c2v_sum, _, v2c_buf, c2v_buf = bufs
                # v2c: gather the per-variable LLR into the check-grouped layout
                l_v_v2c = self.v2c(l_v, v2c_buf)

                # variable node update
                message = self.vn_update(message, l_v_v2c)

                # check node update (min-sum), in the [batch, n_checks, degree] layout
                message = self.cn_update(message, syndrome_odd, c2v_msg)

                # c2v: convert the check messages back to the per-variable layout
                message_c2v = self.c2v(c2v_flat, c2v_sum, c2v_buf)

                # elementwise LLR update
                self.llr_update(u_init, message_c2v, l_v)

                # hard decision: map posterior LLRs to a binary error estimate
                self.hard_decision(l_v, e_v)

                s_est = self.syndrome_estimation(e_v)

            # different samples from the same batch may terminated at different iteration (pick the smallest one)
            indices = torch.all(s_est == syndrome, 1).nonzero()
            checker = torch.where(it_rows == -1.0)[0]
            indices = indices[torch.isin(indices, checker)]
            if indices.size()[0] > 0:
                it_rows[indices] = self.i
                dst = rows[indices]
                num_iters[dst] = self.i
                e_out[dst] = e_v[indices]
                l_out[dst] = l_v[indices]
                converges[dst] = 1

            # do the early termination if all batch satisfy the condition
            if checker.size()[0] == 0:
                break

            # adaptive cap: stop once >= cap_frac of the batch has converged; the
            # unconverged remainder (converge == 0) becomes main's deferred tail.
            if cap_frac is not None:
                n_conv = int((num_iters != -1).sum())
                if n_conv >= cap_frac * B:
                    cap_stopped = n_conv < B and self.i < self.max_iter
                    break

            if hooked:
                self._hook_rows = rows
                self._iter_hook(self.i, l_v, e_v, it_rows == -1, syndrome)

            # compaction: keep only the unconverged rows once they drop below
            # compact_frac of the current rows (never to zero rows, never after the
            # last iteration; compiled, never to one row, which would recompile).
            # Eager with reuse_buffers: kept rows move to the front of each buffer
            # through the work buffer, which is free here, and the buffers become
            # prefix views.
            n_left = checker.size()[0] - indices.size()[0]
            if (
                int(self.compile) < n_left < self.compact_frac * self.batch_size
                and self.i < self.max_iter
            ):
                keep = (it_rows == -1).nonzero().squeeze(1)
                rows, it_rows = rows[keep], it_rows[keep]
                syndrome = syndrome[keep]
                syndrome_odd = syndrome_odd[keep]
                self.batch_size = n_left
                if self.compile:
                    l_v, u_init, c2v_msg = l_v[keep], u_init[keep], c2v_msg[keep]
                    continue
                if not self.reuse_buffers:
                    l_v, u_init, e_v = l_v[keep], u_init[keep], e_v[keep]
                    message = message[keep]
                    continue
                l_v = self._compact(l_v, keep, work)
                u_init = self._compact(u_init, keep, work)
                c2v_flat = self._compact(c2v_flat, keep, work)
                e_v = e_v.view(-1)[: l_v.numel()].view(l_v.shape)
                c2v_sum = c2v_sum.view(-1)[: l_v.numel()].view(l_v.shape)
                c2v_msg = c2v_flat[:, :-1].view(self.batch_size, n_checks, degree)
                message = c2v_msg
                v2c_buf = work[: self.batch_size * n_checks * degree].view(
                    self.batch_size, n_checks, degree
                )
                c2v_buf = work[: self.batch_size * self.vn_adj.numel()].view(
                    self.batch_size, -1
                )

        checker = torch.where(it_rows == -1)[0]
        dst = rows[checker]
        e_out[dst] = e_v[checker]
        l_out[dst] = l_v[checker]
        num_iters[dst] = self.i  # actual stop iter (== max_iter unless the cap broke early)
        self._exit_hook(l_out, e_out, num_iters, converges)
        self.cap_active_last = cap_stopped
        # rows main drops and decodes again: the unconverged rows of a batch the
        # cap stopped before max_iter
        defer = (
            converges.flatten() == 0
            if cap_stopped
            else torch.zeros(B, dtype=torch.bool, device=self.device)
        )
        e_out = e_out[:, :-1]
        l_out = l_out[:, :-1]
        self.batch_size = B

        # warm-up: observe this batch's iteration distribution (decides k + the cap).
        if self.cap is not None and not self.cap.done and not self.cap_bypass:
            self.cap.observe(num_iters, self.max_iter, B)

        logger.info("Complete.")
        logger.info(f"Decoding iterations: <{(self.i)}>.")
        io_dict.update(
            {
                "e_v": e_out,
                "iter": num_iters,
                "llr": l_out,
                "converge": converges,
                "defer": defer,
            }
        )
        return io_dict

    def _work_buffers(self, l_v):
        """Eager work buffers for a batch of self.batch_size rows: the flat c->v
        messages c2v_flat plus one zero column that the c2v padding ids point
        at, its [batch, n_checks, degree] view c2v_msg, the c2v sum (like l_v),
        and one work buffer shared by the v2c output v2c_buf (overwritten in
        place by vn_update and cn_update) and the c2v gather c2v_buf, whose
        lifetimes do not overlap. Returns (c2v_flat, c2v_msg, c2v_sum, work,
        v2c_buf, c2v_buf)."""
        B = self.batch_size
        n_checks, degree = self.V_c_col.shape
        c2v_flat = torch.zeros(
            [B, n_checks * degree + 1], dtype=self.dtype, device=self.device
        )
        c2v_msg = c2v_flat[:, :-1].view(B, n_checks, degree)
        n_v2c = B * n_checks * degree
        n_gather = B * self.vn_adj.numel()
        work = torch.empty(max(n_v2c, n_gather), dtype=self.dtype, device=self.device)
        v2c_buf = work[:n_v2c].view(B, n_checks, degree)
        c2v_buf = work[:n_gather].view(B, -1)
        return c2v_flat, c2v_msg, torch.empty_like(l_v), work, v2c_buf, c2v_buf

    def _compact(self, x, keep, work):
        """Moves rows `keep` (ascending) of the contiguous [batch, ...] buffer `x` to
        its front and returns them as a prefix view of `x`'s storage. The rows are
        staged in `work`, so no buffer is allocated. The widest rows are c2v_flat's,
        n_checks * degree + 1 each; `work` holds at least B * n_checks * degree
        elements and n_left < compact_frac * B <= 0.75 * B (compact_frac is
        checked in __init__), so the kept c2v rows always fit.
        """
        n = keep.numel() * x[0].numel()
        staged = work[:n].view(keep.numel(), *x.shape[1:])
        torch.index_select(x, 0, keep, out=staged)
        return x.view(-1)[:n].view(staged.shape).copy_(staged)

    def v2c(self, l_v, out):
        """Format conversion (variable -> check layout).

        Gathers the per-variable LLR vector `l_v` ([batch, N+1]) into the
        check-node-grouped layout ([batch, n_checks, degree]) that the variable-node
        and check-node updates operate on. Each edge (c, v) picks up `l_v[:, v]`.
        Writes into and returns `out`.
        """
        col = self.V_c_col.flatten()
        flat = out.view(self.batch_size, -1)
        # index_select is the faster gather on CUDA, gather the faster one on CPU
        if l_v.is_cuda:
            torch.index_select(l_v, 1, col, out=flat)
        else:
            torch.gather(l_v, 1, col.expand(self.batch_size, -1), out=flat)
        return out

    def vn_update(self, b_c2v, l_v_v2c):
        """Variable-node update: produce the v->c messages a_v2c.

        On the first iteration there is no incoming c->v message yet, so the
        initialized message is passed through. Afterwards each v->c message is the
        current per-variable LLR (already v2c-gathered into the check layout) minus
        the incoming c->v message on that same edge (extrinsic information), written
        in place into `l_v_v2c`.
        """
        if self.i == 1:
            return b_c2v
        else:
            return torch.sub(l_v_v2c, b_c2v, out=l_v_v2c)

    def cn_update(self, a_v2c, syndrome_odd, out, beta=None):
        """Check-node update (normalized min-sum): produce the c->v messages b_c2v.

        Each message has magnitude beta * (minimum |a_v2c| over the other edges of
        its check). An input a_v2c <= 0 (including -0.0) counts as negative; the
        message is negative when the edge's own sign bit, the parity of negative
        inputs on the check and the syndrome bit XOR to 1. Dummy slots are set to 0.
        Overwrites `a_v2c` with |a_v2c|. Writes into and returns `out`, which must
        not alias `a_v2c`. With cn_sign_parity False, runs _cn_update_topk;
        beta defaults to the current iteration's normalization.
        """
        if beta is None:
            base = torch.tensor(2.0, dtype=self.dtype)
            exponent = torch.tensor(-(self.i), dtype=self.dtype)
            beta = torch.tensor(1.0, dtype=self.dtype) - torch.pow(base, exponent)
        if not self.cn_sign_parity:
            return self._cn_update_topk(a_v2c, syndrome_odd, out, beta)

        # sign: parity of the negative inputs per check, XOR the syndrome bit
        # (uint8 sum: wraparound mod 256 keeps the parity)
        neg = a_v2c <= 0.0
        parity = (neg.sum(dim=2, keepdim=True, dtype=torch.uint8) & 1).bool()
        flip = neg ^ (parity ^ syndrome_odd)

        # magnitude: the edge at the minimum gets the second minimum, all others
        # get the minimum (a tied minimum gives min_1 == min_0)
        mag = a_v2c.abs_()
        min_0, arg_0 = mag.min(dim=2, keepdim=True)
        mag.scatter_(2, arg_0, float("inf"))
        min_1 = mag.amin(dim=2, keepdim=True)
        m_0 = beta * min_0
        m_1 = beta * min_1

        # signed message: one broadcast select for the minimum, then the signed
        # second minimum scattered onto the edge at the minimum
        torch.where(flip, -m_0, m_0, out=out)
        out.scatter_(2, arg_0, torch.where(flip.gather(2, arg_0), -m_1, m_1))
        out.view(self.batch_size, -1).index_fill_(1, self.dummy_idx, 0.0)
        return out

    def _cn_update_topk(self, a_v2c, syndrome_odd, out, beta):
        """Check-node update with the same messages as cn_update for finite inputs:
        sign of each edge (0 counts as -1) times the product of the signs on its
        check times the syndrome sign; magnitude the second minimum (topk(2)) on
        the edges equal to the minimum, else the minimum, scaled by beta. Writes
        into and returns `out`."""
        sign = torch.sgn(a_v2c)
        sign = torch.where(sign == 0.0, -1.0, sign)
        sign_prod = torch.prod(sign, dim=2, keepdim=True)
        Q_sign = torch.where(syndrome_odd, -1.0, 1.0).to(self.dtype) * sign_prod

        abs_a_v2c = torch.abs(a_v2c)
        mins, _ = torch.topk(abs_a_v2c, 2, dim=2, largest=False)
        min_0 = mins[:, :, 0].unsqueeze(2)
        min_1 = mins[:, :, 1].unsqueeze(2)
        min_result = torch.where(abs_a_v2c == min_0, min_1, min_0)

        out.copy_(beta * sign * Q_sign * min_result)
        out.view(self.batch_size, -1).index_fill_(1, self.dummy_idx, 0.0)
        return out

    def c2v(self, c2v_flat, out, gathered):
        """Format conversion (check -> variable layout).

        Sums the c->v messages of each variable into the per-variable layout
        ([batch, N+1]) that the LLR update adds to the channel LLR. `c2v_flat` is
        the [batch, n_checks * degree + 1] flat message with a zero last column.
        One gather through `self.vn_adj` fills `gathered` ([batch, VD * (N+1)]);
        each column `v` then starts at +0.0 and adds its edge messages one at a
        time in (c, k) order, padding adding the zero column. The dummy variable
        (last column) sums to 0. Without c2v_gather, index_add_ sums the
        messages into `out` instead. Writes into and returns `out`.
        """
        if not self.c2v_gather:
            out.zero_()
            return out.index_add_(1, self.V_c_col.flatten(), c2v_flat[:, :-1])
        torch.gather(
            c2v_flat, 1, self.vn_adj.expand(self.batch_size, -1), out=gathered
        )
        out.zero_()
        for slot in gathered.view(self.batch_size, -1, out.shape[1]).unbind(1):
            out.add_(slot)
        return out

    def llr_update(self, u_init, b_c2v, out):
        """Elementwise LLR update: posterior LLR = sum of incoming c->v + channel LLR.

        Takes the already c2v-converted per-variable messages and adds the channel
        (initialization) LLR last. The dummy variable (last column) gets +inf, since
        its channel LLR is +inf and its message sum is 0, so it is never decoded as
        an error. Writes into and returns `out`.
        """
        return torch.add(b_c2v, u_init, out=out)

    def hard_decision(self, l_v, out):
        """Hard decision: map posterior LLRs to a binary error estimate.

        A non-positive LLR (<= 0) means the bit is more likely 1, so it is set to
        1; otherwise 0. Writes into and returns `out` in the decoder's dtype.
        """
        return torch.le(l_v, 0.0, out=out)

    def syndrome_estimation(self, e_v):
        """Syndrome of the hard decision: parity of the error bits on each check.

        The dummy variable (last column) is always 0 since its LLR is +inf. Counts in
        uint8 (wraparound mod 256 keeps the parity). Returns the syndrome in the
        decoder's dtype.
        """
        e_b = (e_v != 0.0).to(torch.uint8)
        return (e_b[:, self.V_c_col].sum(dim=2, dtype=torch.uint8) & 1).to(self.dtype)
