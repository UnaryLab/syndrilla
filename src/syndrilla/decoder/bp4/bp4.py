import types

import torch
from loguru import logger

from syndrilla.utils import parse_device_dtype
from syndrilla.decoder.decoder import RebatchSpeedup

# compact the per-iteration state to the unconverged samples once fewer than this
# fraction of the rows decoded since the last compaction are still unconverged
COMPACT_FRAC = 0.75


def _step(
    message, oldbitnode, chan, check_node, d, eps, col, vn_adj, mask_dummy, synd_bits
):
    """One bp4 iteration after the variable-node update: check-node update, c2v,
    posterior normalization, hard decision, syndrome check, then the variable-node
    update that gives the next iteration's v->c messages. Functional; used by the
    eager path and, compiled, on CUDA.

      message    [B, 2, M, D] v->c LLRs (channel 0: X checks, 1: Z checks)
      oldbitnode [B, 4, N+1]  damped posterior memory; chan [B, 4, N+1] channel prior
      check_node [B, 2, M, 1] 1 - 2 * syndrome
      col        [2 * M * D]  variable of each edge, (channel, check, slot) order
      vn_adj     [VD * (N+1)] slot-major edge ids per variable (see create.__init__)
      synd_bits  [B, 2, M]    syndrome as bool

    Returns (next message, normalized posterior, converged [B] bool,
    joint-Pauli hard decisions [B, 2, N+1] in Hx/Hz sector order).
    """
    B, C, M, D = message.shape

    # check-node update (quaternary min-sum); dummy slots (Hx padding) set to 0
    sign = torch.sgn(message)
    sign_prod = torch.prod(sign, dim=3, keepdim=True)
    mag = torch.abs(message)
    srt, _ = torch.sort(mag, dim=3)
    min_result = torch.where(mag == srt[..., :1], srt[..., 1:2], srt[..., :1])
    msg = (check_node * sign_prod * sign * min_result).masked_fill(mask_dummy, 0.0)

    # c2v: per-edge quaternary factors [B, 4 (I, X, Y, Z), 2, M, D], multiplied
    # into the damped prior of each variable in (channel, check, slot) order
    err_neg = 0.5 / (1.0 + torch.exp(-msg))
    err_pos = 0.5 / (1.0 + torch.exp(msg))
    n0, n1 = err_neg.unbind(1)
    p0, p1 = err_pos.unbind(1)
    err = torch.stack(
        [torch.stack(pair, 1) for pair in ((n0, n1), (n0, p1), (p0, p1), (p0, n1))], 1
    )
    flat = torch.cat([err.view(B, 4, -1), err.new_ones(B, 4, 1)], 2)
    g = flat[:, :, vn_adj].view(B, 4, -1, chan.shape[2])
    bitnode = torch.pow(chan, 1.0 - d) * torch.pow(oldbitnode, d)
    for k in range(g.shape[2]):
        bitnode = bitnode * g[:, :, k]
    normalized = bitnode / bitnode.sum(dim=1, keepdim=True).clamp_min(eps)

    # hard decision and syndrome: X checks see the Z bits and Z checks the X bits
    qubits = torch.argmax(bitnode, dim=1)
    x_bits = ((qubits == 1) | (qubits == 2)).to(torch.uint8)
    z_bits = ((qubits == 2) | (qubits == 3)).to(torch.uint8)
    col3 = col.view(C, M, D)
    x_checks = z_bits[:, col3[0]].sum(dim=2, dtype=torch.uint8) & 1
    z_checks = x_bits[:, col3[1]].sum(dim=2, dtype=torch.uint8) & 1
    converged = (x_checks.bool() == synd_bits[:, 0]).all(1) & (
        z_checks.bool() == synd_bits[:, 1]
    ).all(1)

    # variable-node update: divide each edge's own factor out of the posterior and
    # map back to X / Z LLRs
    gathered = bitnode[:, :, col].view(B, 4, C, M, D) / err.clamp_min(eps)
    num0 = gathered[:, 0, 0] + gathered[:, 1, 0] + eps
    den0 = gathered[:, 2, 0] + gathered[:, 3, 0] + eps
    num1 = gathered[:, 0, 1] + gathered[:, 3, 1] + eps
    den1 = gathered[:, 1, 1] + gathered[:, 2, 1] + eps
    message = torch.stack([torch.log(num0 / den0), torch.log(num1 / den1)], 1)
    return message, normalized, converged, torch.stack((z_bits, x_bits), 1)


class create(torch.nn.Module):
    """
    This class creates a quaternary bp (bp4) decoder.

    Each iteration runs `_step` (eager, or through torch.compile on CUDA). The
    per-variable product of the check factors is a gather through a fixed
    variable-to-edge table, multiplied in (channel, check, slot) order, so the
    result does not depend on the device's scatter order. Rows are compacted to the
    unconverged samples as in bp_norm_min_sum. The returned e_v [B, 2, N] is the
    joint-Pauli hard decision projected onto Hx/Hz sectors; llr holds the sector
    marginal log odds from the same stop iteration. Their signs need not agree
    with e_v because marginal MAP and projected joint MAP can differ.
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

            compile: decoder config key, default True. Runs `_step` through
                torch.compile on a CUDA device when dynamo and Triton are
                available; eager otherwise.
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

        self.d = decoding_cfg.get("damping_factor", 0.1)
        if self.d <= 0 or self.d > 1:
            logger.warning(
                f"Invalid input damping factor <{self.d}>, default to <0.1>."
            )
            self.max_iter = 50

        bundle = kwargs.get("bundle")
        if bundle is None:
            raise ValueError(
                "bp4 requires a pre-loaded MatrixBundle via the `bundle` kwarg."
            )
        self.Hx_matrix = bundle.Hx_matrix
        self.Hz_matrix = bundle.Hz_matrix
        self.lx_matrix = bundle.lx_matrix
        self.lz_matrix = bundle.lz_matrix

        # bp4 needs indices from both Hx and Hz (no check_type selection)
        self.H_shape, self.Hx_V_c_row, self.Hx_V_c_col, _ = self.Hx_matrix.get_index()
        _, self.Hz_V_c_row, self.Hz_V_c_col, _ = self.Hz_matrix.get_index()

        self.mask_dummy = self.Hx_V_c_col == self.H_shape[1]

        # set iteration
        self.i = 0

        self.H_matrix = torch.stack((self.Hx_V_c_row, self.Hz_V_c_row))

        # convert to as the parameters in a model
        self.V_c_row = torch.nn.Parameter(
            torch.stack((self.Hx_V_c_row, self.Hz_V_c_row)), requires_grad=False
        )
        self.V_c_col = torch.nn.Parameter(
            torch.stack((self.Hx_V_c_col, self.Hz_V_c_col)), requires_grad=False
        )

        # variable -> edge table for c2v: row n holds the flat edge ids
        # (channel * M + check) * D + slot of variable n's edges in that order,
        # padded with 2 * M * D, the id of the ones column appended to the flat
        # factors. The dummy variable N has no edges: its prior is +inf, then NaN
        # after normalization, and a positive finite factor leaves either as is.
        edge_var = self.V_c_col.detach().flatten().long()
        edge_id = (edge_var < self.H_shape[1]).nonzero().squeeze(1)
        var = edge_var[edge_id]
        order = torch.argsort(var, stable=True)
        edge_id, var = edge_id[order], var[order]
        counts = torch.bincount(var, minlength=self.H_shape[1] + 1)
        slot = torch.arange(var.numel(), device=var.device)
        slot -= (torch.cumsum(counts, 0) - counts)[var]
        vn_adj = torch.full(
            [self.H_shape[1] + 1, int(counts.max())],
            edge_var.numel(),
            dtype=torch.long,
            device=var.device,
        )
        vn_adj[var, slot] = edge_id
        # slot-major, so one gather yields VD contiguous [batch, 4, N+1] slices
        self.vn_adj = vn_adj.t().flatten()

        self.algo = "bp4"
        self.num_max_iter = self.max_iter

        self.cap = RebatchSpeedup.from_cfg(decoding_cfg)
        self.cap_bypass = False  # set by main: True -> decode this batch uncapped
        self.cap_active_last = False  # set per forward: True if the cap was applied

        # torch.compile of _step, used when forward runs on CUDA
        self._compiled = None
        if decoding_cfg.get("compile", True):
            try:
                from torch.utils._triton import has_triton

                if torch._dynamo.is_dynamo_supported() and has_triton():
                    # a copy of the code object gives each decoder its own compile cache
                    self._compiled = torch.compile(
                        types.FunctionType(
                            _step.__code__.replace(), _step.__globals__, _step.__name__
                        )
                    )
            except Exception as e:
                logger.debug(f"compile is on but runs eager: {e!r}.")

        logger.info("Complete.")

    def forward(self, io_dict):
        """Iterative bp4 (Quaternary BP) decoding algorithm
        Input:
            synd: [B, 2, M] syndrome of the X and Z checks
            llr0: [B, 4, N] channel probabilities of I, X, Y, Z

        Output:
            e_v: [B, 2, N] joint-Pauli decisions (Z|Y for Hx, X|Y for Hz)
            llr: [B, 2, N] sector marginal log odds; llr <= 0 can differ from e_v
            iter: iteration at which the sample converged (the stop iteration if not)
            converge: 1 where the hard decision matches both syndromes
        """
        logger.info("Initializing bp4 (Quaternary BP) decoding.")

        syndrome = io_dict["synd"].to(dtype=self.dtype).to(self.device)

        self.batch_size, self.number_channel, _ = syndrome.size()
        B = self.batch_size
        dev = syndrome.device

        # add a dummy element at the end in case the H (ldpc matrix) does not have the same number of 1s in each check node
        self.N_extended = self.H_shape[1] + 1

        dummy_column = torch.full([B, 4, 1], float("inf"), dtype=self.dtype, device=dev)
        chan = torch.cat((io_dict["llr0"].to(dev).to(self.dtype), dummy_column), dim=2)
        oldbitnode = chan
        num_iters = torch.full([B], -1, device=dev)
        converges = torch.full([B], 0, device=dev)
        e_out = torch.zeros([B, 2, self.H_shape[1]], dtype=self.dtype, device=dev)
        posterior_out = torch.zeros([B, 4, self.H_shape[1]], dtype=self.dtype, device=dev)

        col = self.V_c_col.detach().to(dev)
        vn_adj = self.vn_adj.to(dev)
        mask_dummy = self.mask_dummy.to(dev)
        self.eps = 1e-40

        # initialize messages
        pI, pX, pY, pZ = chan[:, 0], chan[:, 1], chan[:, 2], chan[:, 3]
        x_msg = torch.log((pI + pX + self.eps) / (pY + pZ + self.eps))
        z_msg = torch.log((pI + pZ + self.eps) / (pX + pY + self.eps))
        message = torch.stack((x_msg[:, col[0]], z_msg[:, col[1]]), 1)

        check_node = (1.0 - 2.0 * syndrome).unsqueeze(3)
        synd_bits = syndrome != 0.0
        col = col.flatten()

        step = (
            self._compiled
            if self._compiled is not None and dev.type == "cuda"
            else _step
        )
        if step is not _step and B > 1:
            for t in (message, oldbitnode, chan, check_node, synd_bits):
                torch._dynamo.mark_dynamic(t, 0)

        # the per-iteration state holds only the rows still decoded: row r is sample
        # rows[r], and it_rows[r] is its convergence iteration (-1 while unconverged)
        rows = torch.arange(B, device=dev)
        it_rows = num_iters.clone()

        logger.info("Complete.")

        logger.info("Starting decoding iterations.")

        # adaptive cap: once warm-up has chosen a stop fraction, break this batch as
        # soon as that fraction has converged (unless main asked for an uncapped pass).
        self.cap_active_last = bool(
            self.cap is not None and self.cap.done and self.cap.frac is not None
            and not self.cap_bypass
        )
        cap_frac = self.cap.frac if self.cap_active_last else None

        self.i = 0
        while self.i < self.max_iter:
            self.i += 1

            message, oldbitnode, conv, hard = step(
                message,
                oldbitnode,
                chan,
                check_node,
                self.d,
                self.eps,
                col,
                vn_adj,
                mask_dummy,
                synd_bits,
            )

            active = (it_rows == -1)[:, None, None]
            e_out[rows] = torch.where(active, hard[:, :, :-1], e_out[rows])
            posterior_out[rows] = torch.where(
                active, oldbitnode[:, :, :-1], posterior_out[rows]
            )

            # record the rows that converge in this iteration, without a host sync
            new = conv & (it_rows == -1)
            it_rows = torch.where(new, self.i, it_rows)
            num_iters.index_copy_(0, rows, torch.where(new, self.i, num_iters[rows]))
            converges.index_copy_(0, rows, torch.where(new, 1, converges[rows]))

            n_left = int((it_rows == -1).sum())
            if n_left == 0:
                break

            # adaptive cap: stop once >= cap_frac of the batch has converged; the
            # unconverged remainder (converge == 0) becomes main's deferred tail.
            if cap_frac is not None and B - n_left >= cap_frac * B:
                break

            # compaction: keep only the unconverged rows once they drop below
            # COMPACT_FRAC of the current rows (never after the last iteration;
            # compiled, never to one row, which would recompile)
            if (
                int(step is not _step) < n_left < COMPACT_FRAC * self.batch_size
                and self.i < self.max_iter
            ):
                keep = (it_rows == -1).nonzero().squeeze(1)
                rows, it_rows = rows[keep], it_rows[keep]
                message, oldbitnode, chan = message[keep], oldbitnode[keep], chan[keep]
                check_node, synd_bits = check_node[keep], synd_bits[keep]
                self.batch_size = n_left

        # actual stop iter (== max_iter unless the cap broke early)
        num_iters[num_iters == -1] = self.i
        self.batch_size = B
        pI, pX, pY, pZ = posterior_out.unbind(1)
        eps = max(self.eps, torch.finfo(self.dtype).tiny)
        llr = torch.stack((pI + pX, pI + pZ), 1).clamp_min(eps).log()
        llr -= torch.stack((pZ + pY, pX + pY), 1).clamp_min(eps).log()

        # warm-up: observe this batch's iteration distribution (decides k + the cap).
        if self.cap is not None and not self.cap.done and not self.cap_bypass:
            self.cap.observe(num_iters, self.max_iter, B)

        logger.info("Complete.")
        logger.info(f"Decoding iterations: <{(self.i)}>.")
        io_dict.update(
            {"e_v": e_out, "iter": num_iters, "llr": llr, "converge": converges}
        )
        return io_dict
