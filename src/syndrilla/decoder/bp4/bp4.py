import torch
from loguru import logger

from syndrilla.decoder.decoder import RebatchSpeedup


class create(torch.nn.Module):
    """
    This class creates a bp decoder on a single GPU
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
        """

        super(create, self).__init__()

        logger.info("Creating bp decoder.")

        # set up default device
        device_cfg = decoding_cfg.get("device", {})
        self.device = device_cfg.get(
            "device_type", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device not in {
            "cuda",
            "cpu",
            torch.device("cuda"),
            torch.device("cpu"),
        }:
            logger.warning(
                f"Invalid input device <{self.device}>, default to avaliable device in your machine."
            )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device == "cuda":
            device_idx = device_cfg.get("device_idx", 0)
            if device_idx >= torch.cuda.device_count():
                logger.warning(
                    f"Invalid input device index <{device_idx}>, default to avaliable device in your machine."
                )
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device(f"cuda:{device_idx}")

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

        self.algo = "bp4"
        self.num_max_iter = self.max_iter

        self.cap = RebatchSpeedup.from_cfg(decoding_cfg.get("rebatch_speedup"))
        self.cap_bypass = False  # set by main: True -> decode this batch uncapped
        self.cap_active_last = False  # set per forward: True if the cap was applied

        logger.info("Complete.")

    def forward(self, io_dict):
        """Iterative bp4 (Quaternary BP) decoding algorithm
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
        """
        logger.info("Initializing bp4 (Quaternary BP) decoding.")

        syndrome = io_dict["synd"].to(dtype=self.dtype).to(self.device)

        self.batch_size, self.number_channel, _ = syndrome.size()

        torch.set_default_dtype(self.dtype)

        # add a dummy element at the end in case the H (ldpc matrix) does not have the same number of 1s in each check node
        self.N_extended = self.H_shape[1] + 1
        l_v = torch.zeros(
            [self.batch_size, self.number_channel, self.N_extended],
            dtype=self.dtype,
            device=self.device,
        )
        e_v = torch.zeros(
            [self.batch_size, self.number_channel, self.N_extended],
            dtype=self.dtype,
            device=self.device,
        )

        # add dummy column
        dummy_column = torch.full(
            [self.batch_size, 4, 1], float("inf"), dtype=self.dtype, device=self.device
        )

        u_init = torch.cat(
            (io_dict["llr0"].to(self.device).to(self.dtype), dummy_column), dim=2
        )
        e_out = torch.zeros(
            [self.batch_size, self.number_channel, self.N_extended],
            dtype=self.dtype,
            device=self.device,
        )
        l_out = torch.zeros(
            [self.batch_size, self.number_channel, self.N_extended],
            dtype=self.dtype,
            device=self.device,
        )
        num_iters = torch.full([self.batch_size], -1, device=self.device)
        converges = torch.full([self.batch_size], 0, device=self.device)

        # set up initialization for all parameters for decoding process
        # message is a in place version of a_v2c and b_c2v
        message = torch.zeros_like(
            self.V_c_row.unsqueeze(0), dtype=self.dtype, device=self.device
        ).repeat(self.batch_size, 1, 1, 1)

        chan = torch.cat(
            (io_dict["llr0"].to(self.device).to(self.dtype), dummy_column), dim=2
        )
        bitnode = torch.cat(
            (io_dict["llr0"].to(self.device).to(self.dtype), dummy_column), dim=2
        )
        oldbitnode = torch.cat(
            (io_dict["llr0"].to(self.device).to(self.dtype), dummy_column), dim=2
        )

        # initialize messages
        self.eps = 1e-40
        pI, pX, pY, pZ = u_init[:, 0], u_init[:, 1], u_init[:, 2], u_init[:, 3]
        x_msg = torch.log((pI + pX + self.eps) / (pY + pZ + self.eps))
        z_msg = torch.log((pI + pZ + self.eps) / (pX + pY + self.eps))
        message[:, 0] = x_msg[:, self.V_c_col[0]]  # channel 0: X
        message[:, 1] = z_msg[:, self.V_c_col[1]]

        logger.info("Complete.")

        logger.info("Starting decoding iterations.")

        # adaptive cap: once warm-up has chosen a stop fraction, break this batch as
        # soon as that fraction has converged (unless main asked for an uncapped pass).
        self.cap_active_last = bool(
            self.cap is not None and self.cap.done and not self.cap_bypass
        )
        cap_frac = self.cap.frac if self.cap_active_last else None

        # per-edge extrinsic factors produced by c2v; consumed by the next vn_update
        new_err = None
        self.i = 0
        while self.i < self.max_iter:
            self.i += 1

            # v2c + variable-node update: recover the v->c messages from the previous
            # posterior (pass-through on the first iteration, before any posterior exists)
            message = self.vn_update(message, bitnode, new_err)

            # check node update (min-sum), still in the [batch, 2, n_checks, degree] layout
            message = self.cn_update(message, syndrome)

            # c2v: scatter the check messages into the per-variable posterior (prob domain)
            bitnode, new_err = self.c2v(message, oldbitnode, chan)

            # damped posterior memory carried into the next iteration
            oldbitnode = self.normalize_posterior(bitnode)

            # hard decision: map the posterior probabilities to a binary error estimate
            x_bits, z_bits = self.hard_decision(bitnode)

            convergent_mask = self.syndrome_estimation(x_bits, z_bits, syndrome)

            # different samples from the same batch may terminated at different iteration (pick the smallest one)
            indices = torch.nonzero(convergent_mask == 1)
            checker = torch.where(num_iters == -1.0)[0]

            indices = indices[torch.isin(indices, checker)]
            if indices.size()[0] > 0:
                num_iters[indices] = self.i
                e_out[indices] = e_v[indices]
                l_out[indices] = l_v[indices]
                converges[indices] = 1
            # do the early termination if all batch satisfy the condition
            if checker.size()[0] == 0:
                break

            # adaptive cap: stop once >= cap_frac of the batch has converged; the
            # unconverged remainder (converge == 0) becomes main's deferred tail.
            if (
                cap_frac is not None
                and int((num_iters != -1).sum()) >= cap_frac * self.batch_size
            ):
                break

        checker = torch.where(num_iters == -1)[0]
        e_out[checker] = e_v[checker]
        l_out[checker] = l_v[checker]
        num_iters[checker] = (
            self.i
        )  # actual stop iter (== max_iter unless the cap broke early)
        e_out = e_out[:, :, :-1]
        l_out = l_out[:, :, :-1]

        # warm-up: observe this batch's iteration distribution (decides k + the cap).
        if self.cap is not None and not self.cap.done and not self.cap_bypass:
            self.cap.observe(num_iters, self.max_iter, self.batch_size)

        logger.info("Complete.")
        logger.info(f"Decoding iterations: <{(self.i)}>.")
        io_dict.update(
            {"e_v": e_out, "iter": num_iters, "llr": l_out, "converge": converges}
        )
        return io_dict

    def cn_update(self, a_v2c, syndrome):
        """Check-node update (quaternary min-sum) in the [batch, 2, n_checks, degree]
        layout, then zero the dummy edges padded onto irregular checks."""
        # checks
        check_node = 1.0 - 2.0 * syndrome.to(self.dtype)
        channel_idx = torch.arange(self.number_channel, device=check_node.device)
        # compute sgn
        sign = torch.sgn(a_v2c)

        check_node = check_node[:, channel_idx[:, None, None], self.V_c_row]

        sign_prod = torch.prod(sign, dim=3, keepdim=True)

        # compute min
        abs_a_v2c = torch.abs(a_v2c)
        sorted, _ = torch.sort(abs_a_v2c, dim=3)
        min_0 = sorted[:, :, :, 0].unsqueeze(3)
        min_1 = sorted[:, :, :, 1].unsqueeze(3)
        min_result = torch.where(abs_a_v2c == min_0, min_1, min_0)
        message = check_node * sign_prod * sign * min_result
        message[:, :, self.mask_dummy] = float(0.0)
        return message

    def c2v(self, a_v2c, oldbitnode, chan):
        """Check-to-variable update: turn the check messages into per-edge quaternary
        error probabilities and scatter-multiply them into the per-variable posterior
        (probability domain), seeded by the damped channel/memory prior. Returns the
        posterior `bitnode` and the per-edge factors `new_err` (consumed by vn_update).
        """
        bitnode = torch.pow(chan, 1.0 - self.d) * torch.pow(oldbitnode, self.d)
        err_neg = 0.5 / (1.0 + torch.exp(-a_v2c))
        err_pos = 0.5 / (1.0 + torch.exp(a_v2c))
        new_err = torch.zeros(
            (self.batch_size, 2, 4, self.H_shape[0], 4),
            dtype=a_v2c.dtype,
            device=a_v2c.device,
        )
        new_err[:, 0, 0] = err_neg[:, 0]
        new_err[:, 0, 1] = err_neg[:, 0]
        new_err[:, 0, 2] = err_pos[:, 0]
        new_err[:, 0, 3] = err_pos[:, 0]

        new_err[:, 1, 0] = err_neg[:, 1]
        new_err[:, 1, 1] = err_pos[:, 1]
        new_err[:, 1, 2] = err_pos[:, 1]
        new_err[:, 1, 3] = err_neg[:, 1]

        data_flat = (
            new_err.flatten(start_dim=3)
            .permute(0, 2, 1, 3)
            .reshape(self.batch_size, 4, 8 * self.H_shape[0])
        )

        partitions_flat = self.V_c_col.flatten(start_dim=1).unsqueeze(1).repeat(1, 4, 1)
        partitions_flat = partitions_flat.unsqueeze(0).repeat(self.batch_size, 1, 1, 1)
        partitions_flat = partitions_flat.permute(0, 2, 1, 3).reshape(
            self.batch_size, 4, 8 * self.H_shape[0]
        )
        partitions_flat_expanded = partitions_flat.expand(self.batch_size, -1, -1)

        sum_b_c2v = torch.zeros(
            [self.batch_size, 4, self.H_shape[1] + 1],
            dtype=self.dtype,
            device=self.device,
        )

        sum_b_c2v = bitnode + sum_b_c2v
        sum_b_c2v.scatter_reduce_(2, partitions_flat_expanded, data_flat, reduce="prod")
        return sum_b_c2v, new_err

    def normalize_posterior(self, bitnode):
        """Normalize the quaternary posterior into the damped memory carried to the next
        iteration: divide each variable's [I, X, Z, Y] probabilities by their sum (clamped
        to `eps` to avoid divide-by-zero) so they form a per-variable distribution."""
        return bitnode / bitnode.sum(dim=1, keepdim=True).clamp_min(self.eps)

    def vn_update(self, message, bitnode, new_err):
        """Variable-node update: produce the v->c messages from the current posterior.

        On the first iteration there is no posterior yet, so the initialized message is
        passed through. Afterwards each per-edge v->c LLR is recovered from the posterior
        `bitnode` by dividing out that edge's own contribution (`new_err`, the quaternary
        extrinsic information) and mapping back to X/Z log-likelihood ratios."""
        if self.i == 1:
            return message
        idx = self.V_c_col.unsqueeze(0).unsqueeze(2)
        idx = idx.expand(self.batch_size, 2, 4, self.H_shape[0], 4)

        bitnode_expanded = (
            bitnode.unsqueeze(1).unsqueeze(3).expand(-1, 2, -1, self.H_shape[0], -1)
        )
        bitnode_gathered = torch.gather(bitnode_expanded, dim=-1, index=idx)
        bitnode_gathered = bitnode_gathered / new_err.clamp_min(self.eps)

        num0 = (
            bitnode_gathered[:, 0, 0, :, :] + bitnode_gathered[:, 0, 1, :, :] + self.eps
        )
        den0 = (
            bitnode_gathered[:, 0, 2, :, :] + bitnode_gathered[:, 0, 3, :, :] + self.eps
        )

        num1 = (
            bitnode_gathered[:, 1, 0, :, :] + bitnode_gathered[:, 1, 3, :, :] + self.eps
        )
        den1 = (
            bitnode_gathered[:, 1, 1, :, :] + bitnode_gathered[:, 1, 2, :, :] + self.eps
        )

        # message: [batch, 2, 9, 4]
        message = torch.empty(
            (
                bitnode_gathered.size(0),
                2,
                bitnode_gathered.size(3),
                bitnode_gathered.size(4),
            ),
            device=bitnode_gathered.device,
            dtype=bitnode_gathered.dtype,
        )

        message[:, 0] = torch.log(num0 / den0)
        message[:, 1] = torch.log(num1 / den1)
        return message

    def hard_decision(self, bitnode):
        qubits = torch.argmax(bitnode, dim=1)
        x_bits = torch.zeros(
            (self.batch_size, self.N_extended),
            dtype=bitnode.dtype,
            device=bitnode.device,
        )
        z_bits = torch.zeros(
            (self.batch_size, self.N_extended),
            dtype=bitnode.dtype,
            device=bitnode.device,
        )
        x_bits[(qubits == 1) | (qubits == 2)] = 1
        z_bits[(qubits == 2) | (qubits == 3)] = 1
        return x_bits, z_bits

    def syndrome_estimation(self, x_bits, z_bits, syndrome):
        # Output check tensors
        x_checks = torch.zeros(
            (self.batch_size, self.H_shape[0]), dtype=x_bits.dtype, device=x_bits.device
        )
        z_checks = torch.zeros(
            (self.batch_size, self.H_shape[0]), dtype=x_bits.dtype, device=x_bits.device
        )

        # Expand row indices per batch
        idx0 = self.V_c_row[0].flatten().unsqueeze(0).expand(self.batch_size, -1)
        idx1 = self.V_c_row[1].flatten().unsqueeze(0).expand(self.batch_size, -1)

        # Gather source bits per batch
        src0 = z_bits.gather(
            1, self.V_c_col[0].flatten().unsqueeze(0).expand(self.batch_size, -1)
        )
        src1 = x_bits.gather(
            1, self.V_c_col[1].flatten().unsqueeze(0).expand(self.batch_size, -1)
        )

        # XOR accumulation (sum then mod 2)
        x_checks.scatter_add_(1, idx0, src0)
        z_checks.scatter_add_(1, idx1, src1)

        x_checks %= 2
        z_checks %= 2

        x_match = (x_checks == syndrome[:, 0, :]).all(dim=1)
        z_match = (z_checks == syndrome[:, 1, :]).all(dim=1)

        # A batch is convergent if both match
        convergent_mask = x_match & z_match

        return convergent_mask.int()
