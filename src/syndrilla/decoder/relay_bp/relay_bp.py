import types

import torch
from loguru import logger

from syndrilla.utils import parse_device_dtype
from syndrilla.decoder.decoder import RebatchSpeedup
from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum import create as _NmsPy
from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum_cuda import _build_vn_adj


def _step(l_v, c2v_prev, bias, syndrome_odd, alpha, col, vn_adj, mask_dummy):
    """One relay iteration for the compiled path, with the arithmetic of the eager
    methods: v2c message l_v[col] - c2v_prev, check-node update with alpha,
    marginal bias + c2v (bias first, then the edges in (c, k) order), hard decision
    and syndrome. Functional: returns new (c2v_msg, l_v, e_v, s_est). At a leg
    start `l_v` is u_init and `c2v_prev` is zero.
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
    m_0 = alpha * min_0
    m_1 = alpha * min_1
    msg = torch.where(flip, -m_0, m_0)
    msg = msg.scatter(2, arg_0, torch.where(flip.gather(2, arg_0), -m_1, m_1))
    msg = msg.masked_fill(mask_dummy, 0.0)
    flat = torch.cat([msg.view(B, -1), msg.new_zeros(B, 1)], 1)
    g = flat[:, vn_adj].view(B, -1, N1)
    l_new = bias
    for k in range(g.shape[1]):
        l_new = l_new + g[:, k]
    e = l_new <= 0.0
    s_est = e.to(torch.uint8)[:, col].view(B, M, D).sum(dim=2, dtype=torch.uint8) & 1
    return msg, l_new, e.to(l_v.dtype), s_est.to(l_v.dtype)


class create(torch.nn.Module):
    """
    This class creates a bp decoder on a single GPU
    """
    def __init__(self,
                 decoding_cfg,
                 **kwargs) -> None:
        """
        Initialization for bp decoder
        Input:
            decoding_cfg: the information that come from config file (yaml)

        Parameters:
            max_iter: the number of maximum iteration of bp decoder
            i: the number of iterations running the decoder
            center: the center of the memory strength distribution
            width: the width of the memory strength distribution
            solution: the number of solutions to find
            legs: the number of relay legs
            iteration_initial: the number of initial iterations
            iteration_count: the number of iterations for each relay leg after the initial one

            H_matrix: loaded ldpc matrix, either hx or hz, as 2d tenso

            V_c_row: the row index of all the variable nodes for each check node
            V_c_col: the column index of all the variable nodes for each check node

            degree: the maximum number of 1s in all check nodes in H_matrix

            compile: decoder config key, default True. Runs the iteration body
                through torch.compile, as bp_norm_min_sum does, on a CUDA device
                only; ignored on CPU or without dynamo and Triton.
        """

        super(create, self).__init__()

        logger.info('Creating bp decoder.')

        # Defaults match the relay-bp crate's gamma_dist_interval = (-0.24, 0.66),
        # expressed here as center = (low + high) / 2 and width = high - low.
        self.center = decoding_cfg.get('center', 0.21)
        if not isinstance(self.center, float):
            logger.warning(f'Invalid input center <{self.center}>, default to <0.21>.')
            self.center = 0.21

        self.width = decoding_cfg.get('width', 0.9)
        if not isinstance(self.width, float):
            logger.warning(f'Invalid input width <{self.width}>, default to <0.9>.')
            self.width = 0.9

        self.init_mem_strength = decoding_cfg.get('init_mem_strength', 0.35)

        self.alpha_const = decoding_cfg.get('alpha', 0.0)
        if not isinstance(self.alpha_const, (int, float)) or isinstance(self.alpha_const, bool):
            logger.warning(f'Invalid input alpha <{self.alpha_const}>, default to <0.0>.')
            self.alpha_const = 0.0
        self.alpha_const = float(self.alpha_const)

        self.alpha_scaling = decoding_cfg.get('alpha_scaling', 1.0)
        if not isinstance(self.alpha_scaling, (int, float)) or isinstance(self.alpha_scaling, bool) or self.alpha_scaling <= 0:
            logger.warning(f'Invalid input alpha_scaling <{self.alpha_scaling}>, default to <1.0>.')
            self.alpha_scaling = 1.0
        self.alpha_scaling = float(self.alpha_scaling)

        # set up default device. Accept either a `device:` block {device_type, device_idx}
        # (as main passes via the decoding yaml) or a plain string / torch.device (direct use).
        device_cfg = decoding_cfg.get('device') or {}
        if not isinstance(device_cfg, dict):
            try:
                device = torch.device(device_cfg)
                device_cfg = {'device_type': device.type, 'device_idx': device.index or 0}
            except (RuntimeError, TypeError):
                logger.warning(f'Invalid input device <{device_cfg}>, default to avaliable device in your machine.')
                device_cfg = {}
        self.device, _ = parse_device_dtype({**decoding_cfg, 'device': device_cfg})

        #set up default solution number
        self.solution = decoding_cfg.get('solution', 5)
        if self.solution <= 0 or not isinstance(self.solution, int):
            logger.warning(f'Invalid input solution number <{self.solution}>, default to <5>.')
            self.solution = 5

        #initial iteration count
        self.iteration_initial = decoding_cfg.get('iteration_initial', 80)
        if self.iteration_initial < 0 or not isinstance(self.iteration_initial, int):
            logger.warning(f'Invalid input iteration initial <{self.iteration_initial}>, default to <0>.')
            self.iteration_initial = 80

        #set up default iteration count
        self.iteration_count = decoding_cfg.get('iteration_count', 60)
        if self.iteration_count <= 0 or not isinstance(self.iteration_count, int):
            logger.warning(f'Invalid input iteration count <{self.iteration_count}>, default to <60>.')
            self.iteration_count = 60

        # set up default R
        self.legs = decoding_cfg.get('legs', 20)
        if self.legs <= 0 or not isinstance(self.legs, int):
            logger.warning(f'Invalid input R <{self.legs}>, default to <1>.')
            self.legs = 20

        # set up default dtype
        self.dtype = decoding_cfg.get('dtype', 'float64')
        if self.dtype not in {'float32', 'float64', 'bfloat16', 'float16'}:
            logger.warning(f'Invalid input type <{self.dtype}>, default to <torch.float64>.')
            self.dtype = 'float64'
        self.dtype = torch.__dict__[self.dtype]

        self.batch_size = 1

        self.type = decoding_cfg.get('type', 'hx')
        if self.type.lower() not in {'hx', 'hz'}:
            logger.warning(f'Invalid input type <{self.type}>, default to <hx>.')
            self.type = 'hx'

        self.check_type = decoding_cfg.get('check_type', 'hx')
        if self.check_type.lower() not in {'hx', 'hz'}:
            logger.warning(f'Invalid input check type <{self.check_type}>, default to <hx>.')
            self.check_type = 'hx'

        bundle = kwargs.get('bundle')
        if bundle is None:
            raise ValueError('bp_norm_min_sum requires a pre-loaded MatrixBundle via the `bundle` kwarg.')
        self.Hx_matrix = bundle.Hx_matrix
        self.Hz_matrix = bundle.Hz_matrix
        self.lx_matrix = bundle.lx_matrix
        self.lz_matrix = bundle.lz_matrix
        self.H_shape, self.V_c_row, self.V_c_col, self.H_matrix = bundle.select(self.check_type)

        self.mask_dummy = (self.V_c_col == self.H_shape[1]).to(self.device)

        # variable -> check table for the c2v sum, slot-major ([VD * (N+1)]): slot vd
        # of variable n holds the flat edge id c * degree + k of its vd-th edge in
        # (c, k) order, padded with n_checks * degree, the id of the zero column
        # appended to the flat message
        n_checks, degree = self.V_c_col.shape
        adj_c, adj_k, _ = _build_vn_adj(self.V_c_col.cpu().numpy(), self.H_shape[1])
        adj_c, adj_k = torch.from_numpy(adj_c), torch.from_numpy(adj_k)
        vn_adj = torch.where(adj_c >= 0, adj_c * degree + adj_k, n_checks * degree)
        self.vn_adj = vn_adj.t().flatten().to(self.device)

        # convert to as the parameters in a model
        self.V_c_row = torch.nn.Parameter(self.V_c_row, requires_grad=False)
        self.V_c_col = torch.nn.Parameter(self.V_c_col.to(self.device), requires_grad=False)

        self.algo = 'bp_relay'
        self.num_max_iter = self.iteration_initial + (self.legs - 1) * self.iteration_count

        self.cap = RebatchSpeedup.from_cfg(decoding_cfg)
        self.cap_bypass = False       # set by main: True -> decode this batch uncapped
        self.cap_active_last = False  # set per forward: True if the cap was applied

        # torch.compile of the iteration body, on by default, CUDA only; eager
        # fallback (one DEBUG line) on CPU or when dynamo/Triton is unavailable
        self.compile = bool(decoding_cfg.get('compile', True))
        if self.compile:
            try:
                from torch.utils._triton import has_triton

                if self.device.type != 'cuda':
                    reason = f'device is {self.device.type}'
                elif not torch._dynamo.is_dynamo_supported():
                    reason = 'torch.compile (dynamo) is not supported'
                elif not has_triton():
                    reason = 'Triton is not available'
                else:
                    reason = None
            except Exception as e:
                reason = f'compile check failed: {e!r}'
            if reason is not None:
                logger.debug(f'compile is on but runs eager: {reason}.')
                self.compile = False
        if self.compile:
            # a copy of the code object gives each decoder its own dynamo cache
            step = types.FunctionType(_step.__code__.replace(), _step.__globals__, _step.__name__)
            self._step = torch.compile(step)

        logger.info('Complete.')


    def forward(self, io_dict):

        logger.info('Initializing bp-relay decoding.')

        syndrome = io_dict['synd'].to(dtype=self.dtype).to(self.device)

        self.batch_size, _ = syndrome.size()

        # add a dummy element at the end in case the H (ldpc matrix) does not have the same number of 1s in each check node
        N_extended = self.H_shape[1] + 1
        solutions = torch.zeros([self.batch_size], dtype=self.dtype, device=self.device)
        e_solutions = torch.full([self.batch_size], float('inf'), dtype=self.dtype, device=self.device)
        e_best = torch.zeros([self.batch_size, self.H_shape[1]], dtype=self.dtype, device=self.device)
        solution_sum = 0

        # add dummy column
        dummy_column = torch.full([self.batch_size,1], float('inf'), dtype=self.dtype, device=self.device)
        u_init = torch.cat((io_dict['llr0'].to(self.device).to(self.dtype), dummy_column), dim=1)
        logger.info(str(u_init.size()))
        e_out = torch.zeros([self.batch_size, N_extended], dtype=self.dtype, device=self.device)
        l_out = torch.zeros([self.batch_size, N_extended], dtype=self.dtype, device=self.device)
        num_iters = torch.full([self.batch_size], 1, device=self.device)
        converges = torch.full([self.batch_size], 0, device=self.device)

        # syndrome bit per check (True where nonzero), flips the check message sign
        syndrome_odd = (syndrome != 0.0).unsqueeze(2)

        # buffers reused by every iteration (eager): the flat c->v messages plus one
        # zero column that the c2v padding ids point at, the posterior LLR, the hard
        # decision, and one work buffer shared by the v2c message and the c2v gather,
        # whose lifetimes do not overlap
        n_checks, degree = self.V_c_col.shape
        l_v = torch.empty_like(u_init)
        if self.compile:
            col = self.V_c_col.detach().flatten()
            if self.batch_size > 1:
                for t in (u_init, syndrome_odd):
                    torch._dynamo.mark_dynamic(t, 0)
        else:
            c2v_flat = torch.zeros([self.batch_size, n_checks * degree + 1], dtype=self.dtype, device=self.device)
            c2v_msg = c2v_flat[:, :-1].view(self.batch_size, n_checks, degree)
            e_v = torch.empty_like(u_init)
            n_v2c = self.batch_size * n_checks * degree
            n_gather = self.batch_size * self.vn_adj.numel()
            work = torch.empty(max(n_v2c, n_gather), dtype=self.dtype, device=self.device)
            v2c_buf = work[:n_v2c].view(self.batch_size, n_checks, degree)
            c2v_buf = work[:n_gather].view(self.batch_size, -1)

        logger.info('Complete.')

        logger.info('Starting decoding iterations.')

        # adaptive cap: once warm-up has chosen a stop fraction, end the leg ensemble as soon
        # as that fraction has converged (unless main asked for an uncapped pass).
        self.cap_active_last = bool(self.cap is not None and self.cap.done and self.cap.frac is not None and not self.cap_bypass)
        cap_frac = self.cap.frac if self.cap_active_last else None

        self.r = 0
        self.s = 0
        while self.r < self.legs:
            self.r += 1
            self.i = 0
            num_iters_local = torch.full([self.batch_size], -1, device=self.device)
            logger.info(f'Starting leg <{self.r}> decoding.')

            if self.r == 1:
                self.max_iter = self.iteration_initial
                memory_strengths = torch.full([self.batch_size, N_extended], self.init_mem_strength, dtype=self.dtype, device=self.device)
                # full prior LLR as its bias, matching the crate's set_posterior_ratios_to_priors.
                l_v.copy_(u_init)
            else:
                self.max_iter = self.iteration_count
                memory_strengths = self.create_memory_strengths(self.batch_size, N_extended, self.center, self.width)
            # leg start: zero c->v messages, so the first v->c message is u_init - 0 = u_init
            if self.compile:
                c2v_msg = torch.zeros([self.batch_size, n_checks, degree], dtype=self.dtype, device=self.device)
                if self.batch_size > 1:
                    torch._dynamo.mark_dynamic(c2v_msg, 0)
            else:
                c2v_flat.zero_()

            while self.i < self.max_iter:
                # variable node update update v2c
                self.i += 1

                # memory bias (variable prior with disordered memory), from the
                # posterior of the previous iteration
                bias = self.bias_update(memory_strengths, l_v, u_init)
                alpha = self.compute_alpha()
                l_src = u_init if self.i == 1 else l_v

                if self.compile:
                    if self.batch_size > 1:
                        torch._dynamo.mark_dynamic(bias, 0)
                    c2v_msg, l_v, e_v, s_est = self._step(
                        l_src, c2v_msg, bias, syndrome_odd, alpha.to(self.device),
                        col, self.vn_adj, self.mask_dummy)
                else:
                    # variable node update: v->c message l_v[col] - c2v (extrinsic)
                    message = torch.sub(self.v2c(l_src, v2c_buf), c2v_msg, out=v2c_buf)

                    # check node update c2v (dummy edges zeroed inside cn_update)
                    self.cn_update(message, syndrome_odd, alpha, c2v_msg)

                    # marginal / posterior LLR update (dummy variable +inf from bias)
                    self.marginal_update(bias, c2v_flat, l_v, c2v_buf)

                    # hard decision: map posterior LLRs to a binary error estimate
                    self.hard_decision(l_v, e_v)

                    s_est = self.syndrome_estimation(e_v)

                # different samples from the same batch may terminated at different iteration (pick the smallest one)
                indices = torch.all(s_est == syndrome, 1).nonzero()
                checker = torch.where(num_iters_local == -1.0)[0]
                indices = indices[torch.isin(indices, checker)]
                if indices.size()[0] > 0:
                    num_iters[indices] += self.i
                    num_iters_local[indices] = self.i
                    e_out[indices] = e_v[indices]
                    l_out[indices] = l_v[indices]
                    converges[indices] = 1

                # do the early termination if all batch satisfy the condition
                if checker.size()[0] == indices.size()[0]:
                    break

            valid_mask = (converges == 1) & (solutions < self.solution)

            # decoding quality = sum of the prior LLRs over the decoded support
            new_e_weight_all = (e_out[:, :-1] * u_init[:, :-1].abs()).sum(dim=1)
            solutions = solutions + valid_mask.to(solutions.dtype)

            # keep the lowest-quality (best) converged solution seen so far
            improve_mask = valid_mask & (new_e_weight_all < e_solutions)
            e_solutions = torch.where(improve_mask, new_e_weight_all, e_solutions)
            e_best[improve_mask, :] = e_out[improve_mask, :-1]

            # stop once every sample has found `solution` converged solutions
            solution_sum = solutions.sum()
            if solution_sum >= self.batch_size * self.solution:
                break

            # adaptive cap: stop the leg ensemble once >= cap_frac of the batch has converged;
            # the unconverged remainder (converge == 0) becomes main's deferred tail.
            if cap_frac is not None and int((converges == 1).sum()) >= cap_frac * self.batch_size:
                break

        # warm-up: observe this batch's iters-to-converge distribution (decides k + the cap).
        # Non-converged samples ran the full leg budget; clamp so every histogram shares a length.
        if self.cap is not None and not self.cap.done and not self.cap_bypass:
            obs = num_iters.clone().clamp(max=self.num_max_iter)
            obs[converges == 0] = self.num_max_iter
            self.cap.observe(obs, self.num_max_iter, self.batch_size)

        logger.info('Complete.')
        logger.info(f'Decoding iterations: <{(self.i)}>.')
        io_dict.update({
            'e_v': e_best,
            'iter': num_iters,
            'llr': l_out,
            'converge': converges
        })
        return io_dict


    v2c = _NmsPy.v2c
    hard_decision = _NmsPy.hard_decision
    syndrome_estimation = _NmsPy.syndrome_estimation


    def compute_alpha(self):
        if self.alpha_const == 0.0:
            alpha = 1.0 - 2.0 ** (-(self.i / self.alpha_scaling))
        else:
            alpha = self.alpha_const
        if alpha < 0.0:
            alpha = 1.0
        return torch.tensor(alpha, dtype=self.dtype)


    def cn_update(self, a_v2c, syndrome_odd, alpha, out):
        """Check-node update (normalized min-sum with factor alpha), the
        bp_norm_min_sum cn_update: magnitude alpha * (minimum |a_v2c| over the other
        edges of the check); an input <= 0 counts as negative, and the message is
        negative when the edge's own sign bit, the parity of negative inputs on the
        check and the syndrome bit XOR to 1. Dummy slots are set to 0. Overwrites
        `a_v2c` with |a_v2c|. Writes into and returns `out`.
        """
        neg = a_v2c <= 0.0
        parity = (neg.sum(dim=2, keepdim=True, dtype=torch.uint8) & 1).bool()
        flip = neg ^ (parity ^ syndrome_odd)

        mag = a_v2c.abs_()
        min_0, arg_0 = mag.min(dim=2, keepdim=True)
        mag.scatter_(2, arg_0, float('inf'))
        min_1 = mag.amin(dim=2, keepdim=True)
        m_0 = alpha * min_0
        m_1 = alpha * min_1

        torch.where(flip, -m_0, m_0, out=out)
        out.scatter_(2, arg_0, torch.where(flip.gather(2, arg_0), -m_1, m_1))
        out.masked_fill_(self.mask_dummy, 0.0)
        return out


    def marginal_update(self, bias, c2v_flat, out, gathered):
        """Posterior LLR: bias + sum of the c->v messages of each variable, adding
        the bias first and then the edges in (c, k) order through one gather of
        `c2v_flat` ([batch, n_checks * degree + 1], zero last column) into
        `gathered`. The dummy variable gets the +inf of its bias. Writes into and
        returns `out`.
        """
        torch.gather(c2v_flat, 1, self.vn_adj.expand(self.batch_size, -1), out=gathered)
        out.copy_(bias)
        for slot in gathered.view(self.batch_size, -1, out.shape[1]).unbind(1):
            out.add_(slot)
        return out


    def bias_update(self, memory_strengths, marginals, u_init):
        marginal_strength = memory_strengths * marginals
        sub = 1 - memory_strengths
        bias = (sub * u_init) + marginal_strength
        bias[:, -1] = u_init[:, -1]
        return bias


    def create_memory_strengths(self, rows, cols, center, width):
        memory_strengths = ((center + width/2) - (center-width/2)) * torch.rand([rows, cols], dtype=self.dtype, device=self.device) + (center - width/2)
        return memory_strengths
