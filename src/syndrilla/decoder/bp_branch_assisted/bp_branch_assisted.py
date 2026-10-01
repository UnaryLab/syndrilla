import types

import torch
from loguru import logger

from syndrilla.utils import parse_device_dtype
from syndrilla.decoder.bp_lottery.bp_lottery import vn_unsat_count
from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum_cuda import _build_vn_adj
from syndrilla.decoder.relay_bp.relay_bp import _step, create as _RelayPy


class create(torch.nn.Module):
    """
    This class creates a bsfbp decoder on a single GPU
    based on this paper: Branch-Assisted Sign-Flipping Belief Propagation Decoding for Topological Quantum Codes Based on Hypergraph Product Structure
    """
    def __init__(self,
                 decoding_cfg,
                 **kwargs) -> None:
        """
        Initialization for bsfbp decoder
        Input:
            decoding_cfg: the information that come from config file (yaml)

        Parameters:
            max_iter: the number of maximum iteration of bsfbp decoder
            i: the number of iterations running the decoder

            H_matrix: loaded ldpc matrix, either hx or hz, as 2d tensor

            V_c_row: the row index of all the variable nodes for each check node
            V_c_col: the column index of all the variable nodes for each check node

            degree: the maximum number of 1s in all check nodes in H_matrix

            compile: decoder config key, default True. Runs the iteration body
                through torch.compile, as bp_norm_min_sum does, on a CUDA device
                only; ignored on CPU or without dynamo and Triton.
        """

        super(create, self).__init__()

        logger.info('Creating bsfbp decoder.')

        # set up default device
        self.device, _ = parse_device_dtype(decoding_cfg)

        # set up default max_iter
        self.max_iter = decoding_cfg.get('max_iter', 50)
        if self.max_iter <= 0 or not isinstance(self.max_iter, int):
            logger.warning(f'Invalid input maximum iteration <{self.max_iter}>, default to <50>.')
            self.max_iter = 50

        # set up default max_iter
        self.max_b_iter = decoding_cfg.get('max_b_iter', 12)
        if self.max_b_iter <= 0 or not isinstance(self.max_b_iter, int):
            logger.warning(f'Invalid input maximum iteration <{self.max_b_iter}>, default to <12>.')
            self.max_b_iter = 50

        # set up default dtype
        self.dtype = decoding_cfg.get('dtype', 'float64')
        if self.dtype not in {'float32', 'float64', 'bfloat16', 'float16'}:
            logger.warning(f'Invalid input data type <{self.dtype}>, default to <torch.float64>.')
            self.dtype = 'float64'
        self.dtype = torch.__dict__[self.dtype]

        self.batch_size = 1

        self.check_type = decoding_cfg.get('check_type', 'hx')
        if self.check_type.lower() not in {'hx', 'hz'}:
            logger.warning(f'Invalid input check type <{self.check_type}>, default to <hx>.')
            self.check_type = 'hx'

        self.random_machine = decoding_cfg.get('random_machine', 'sobol')
        if self.random_machine.lower() not in {'sobol', 'system'}:
            logger.warning(f'Invalid input machine type <{self.random_machine}>, default to <sobol>.')
            self.random_machine = 'sobol'

        bundle = kwargs.get('bundle')
        if bundle is None:
            raise ValueError('bp_branch_assisted requires a pre-loaded MatrixBundle via the `bundle` kwarg.')
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

        # set iteration may not needed
        self.i = 0

        # convert to as the parameters in a model
        self.V_c_row = torch.nn.Parameter(self.V_c_row, requires_grad=False)
        self.V_c_col = torch.nn.Parameter(self.V_c_col.to(self.device), requires_grad=False)

        self.algo = 'bp_branch_assisted'
        self.num_max_iter = self.max_iter * self.max_b_iter

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
        """Iterative bsfbp (normalized min sum) decoding algorithm
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
        logger.info('Initializing bsfbp (normailized min sum) decoding.')

        syndrome = io_dict['synd'].to(self.device).to(self.dtype)
        self.batch_size, _ = syndrome.size()

        # add a dummy element at the end in case the H (ldpc matrix) does not have the same number of 1s in each check node
        N_extended = self.H_shape[1] + 1
        e_v_saver = torch.zeros([self.batch_size, N_extended], dtype=self.dtype, device=self.device)

        # tensor needed for BSFBP
        l_saver = torch.zeros([self.batch_size, N_extended], dtype=self.dtype, device=self.device)
        syndrome_saver = torch.zeros([self.batch_size, self.H_shape[0]], dtype = self.dtype, device=self.device)
        s_est_saver = torch.zeros([self.batch_size, self.H_shape[0]], dtype = self.dtype, device=self.device)
        iteration_saver = torch.full([self.batch_size], 0, device=self.device)
        index_saver = torch.tensor([], dtype=int, device=self.device)
        curr_iters = torch.full([self.batch_size], 0, dtype=int, device=self.device)
        s0_est_comp = torch.full([self.batch_size], 0, device=self.device)

        # add dummy column
        dummy_column = torch.full([self.batch_size,1], float('inf'), dtype=self.dtype, device=self.device)
        u_init = torch.cat((io_dict['llr0'].to(self.device).to(self.dtype), dummy_column), dim=1)
        e_out = torch.zeros([self.batch_size, N_extended], dtype=self.dtype, device=self.device)
        l_out = torch.zeros([self.batch_size, N_extended], dtype=self.dtype, device=self.device)
        num_iters = torch.full([self.batch_size], -1, device=self.device)
        converges = torch.full([self.batch_size], 0, device=self.device)

        # A sample at its first iteration (start or branch) has l_v = u_init and a
        # zero c->v message, so its first v->c message is u_init. Buffers reused by
        # every iteration (eager): the flat c->v messages plus one zero column that
        # the c2v padding ids point at, the posterior LLR, the hard decision, and one
        # work buffer shared by the v2c message and the c2v gather, whose lifetimes
        # do not overlap
        n_checks, degree = self.V_c_col.shape
        message_saved = torch.zeros([self.batch_size, n_checks, degree], dtype=self.dtype, device=self.device)
        l_v = u_init.clone()
        if self.compile:
            col = self.V_c_col.detach().flatten()
            c2v_msg = torch.zeros_like(message_saved)
            if self.batch_size > 1:
                for t in (l_v, c2v_msg, u_init):
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

        if self.random_machine.lower() == 'sobol':
            sobol = torch.quasirandom.SobolEngine(dimension=1, scramble=False)
            self.r = sobol.draw(self.max_iter*self.max_iter).to(self.device).to(self.dtype)

        logger.info('Complete.')

        logger.info('Starting decoding iterations.')

        self.i = 0
        checker = torch.where(num_iters == -1)[0]
        while (checker.size()[0] != 0):
            self.i += 1
            curr_iters += 1

            # per-sample normalization factor from the sample's own iteration count
            beta = (1.0 - torch.pow(2.0, -curr_iters.to(self.dtype))).view(-1, 1, 1)
            syndrome_odd = (syndrome != 0.0).unsqueeze(2)

            if self.compile:
                if self.batch_size > 1:
                    for t in (beta, syndrome_odd):
                        torch._dynamo.mark_dynamic(t, 0)
                c2v_msg, l_v, e_v, s_est = self._step(
                    l_v, c2v_msg, u_init, syndrome_odd, beta,
                    col, self.vn_adj, self.mask_dummy)
            else:
                # variable node update: v->c message l_v[col] - c2v (extrinsic)
                message = torch.sub(self.v2c(l_v, v2c_buf), c2v_msg, out=v2c_buf)

                # check node update c2v (dummy edges zeroed inside cn_update)
                self.cn_update(message, syndrome_odd, beta, c2v_msg)

                # LLR update u_init + c2v, u_init first (dummy variable +inf)
                self.marginal_update(u_init, c2v_flat, l_v, c2v_buf)

                # hard decision
                self.hard_decision(l_v, e_v)

                s_est = self.syndrome_estimation(e_v)

            if (curr_iters == 1).all():
                s0_est_comp = torch.sum((s_est + syndrome) % 2, 1)

            # check success cases
            mask = torch.ones(curr_iters.size(0), dtype=torch.bool, device=curr_iters.device)
            mask[index_saver] = False

            condition = (curr_iters == self.max_iter) | torch.all(s_est == syndrome, dim=1)

            # Combine the mask and the condition, then get the indices where both are True.
            indices = torch.where(mask & condition)[0]

            converges_index = torch.where(torch.all(s_est == syndrome, 1))[0]

            checker = torch.where(num_iters == -1)[0]
            indices = indices[torch.isin(indices, checker)]

            if indices.size()[0] > 0:
                num_iters[indices] = self.i
                e_out[indices] = e_v[indices]
                l_out[indices] = l_v[indices]

                converges[converges_index] = 1

                keep_mask = ~torch.isin(index_saver, indices)
                remove_mask = index_saver[torch.isin(index_saver, indices)]

                if remove_mask.size()[0] > 0:
                    num_iters[remove_mask] = self.i
                    temp_l_v = l_saver[remove_mask]
                    l_out[remove_mask] = l_v[remove_mask] + temp_l_v
                    e_out[remove_mask] = (e_out[remove_mask] + e_v_saver[remove_mask])%2

                index_saver = index_saver[keep_mask]

            checker = torch.where(num_iters == -1)[0]

            if checker.size()[0] == 0:
                e_out = e_out[:, :-1]
                l_out = l_out[:, :-1]
                logger.info('Complete.')
                logger.info(f'Decoding iterations: <{(curr_iters)}>.')
                io_dict.update({
                    'e_v': e_out,
                    'iter': num_iters,
                    'llr': l_out,
                    'converge': converges
                })
                return io_dict


            # else satisfy C1 or c2
            # C.1 w(ˆsk ⊕ s) ≤ w(b ⊕ s)
            # C.2 Iˆsk
            sk_est_comp = (s_est + syndrome) % 2
            c1_results = (torch.sum(sk_est_comp, 1) <= s0_est_comp).int()
            c2_results = torch.sum(((s_est == 1) & (syndrome == 0)).int(),1)

            branch_index = torch.where((c1_results == 1) & (c2_results == 0) &(num_iters == -1))[0]

            mask = ~torch.isin(branch_index, index_saver)

            not_in_index_saver = branch_index[mask]

            branched = not_in_index_saver.numel() != 0 and self.i != 0
            if branched:
                syndrome_saver[not_in_index_saver] = syndrome[not_in_index_saver]
                syndrome[not_in_index_saver] = sk_est_comp[not_in_index_saver]

                e_v_saver[not_in_index_saver] = e_v[not_in_index_saver]

                s_est_saver[not_in_index_saver] = s_est[not_in_index_saver]

                iteration_saver[not_in_index_saver] = curr_iters[not_in_index_saver]
                curr_iters[not_in_index_saver] = 0

                l_saver[not_in_index_saver] = l_v[not_in_index_saver]

                message_saved[not_in_index_saver] = c2v_msg[not_in_index_saver]
                c2v_msg[not_in_index_saver] = 0.0
                index_saver = torch.cat([not_in_index_saver, index_saver], dim=0)

            b_finish = (curr_iters[index_saver] >= self.max_b_iter)

            if torch.any(b_finish):
                last_b_iter_ind = index_saver[torch.where(b_finish)[0].int()]

                c2v_msg[last_b_iter_ind] = message_saved[last_b_iter_ind]
                syndrome[last_b_iter_ind] = syndrome_saver[last_b_iter_ind]
                s0_est_comp[last_b_iter_ind] = torch.sum((s_est_saver[last_b_iter_ind] + syndrome[last_b_iter_ind]) % 2, 1)

                l_v[last_b_iter_ind] = l_saver[last_b_iter_ind]

                # remove index of branch
                curr_iters[last_b_iter_ind] = iteration_saver[last_b_iter_ind]
                mark = ~torch.isin(index_saver, last_b_iter_ind)
                index_saver = index_saver[mark]

            # sign flip
            l_v = self.sign_flip(syndrome, s_est, l_v)
            if branched:
                # new branch: l_v = u_init, set after the sign flip so the flip skips it
                l_v[not_in_index_saver] = u_init[not_in_index_saver]
            checker = torch.where(num_iters == -1)[0]

        e_out[checker] = e_v[checker]
        l_out[checker] = l_v[checker]
        num_iters[checker] = self.i
        e_out = e_out[:, :-1]
        l_out = l_out[:, :-1]
        logger.info('Complete.')
        logger.info(f'Decoding iterations: <{(self.i)}>.')
        io_dict.update({
            'e_v': e_out,
            'iter': num_iters,
            'llr': l_out,
            'converge': converges
        })
        return io_dict


    v2c = _RelayPy.v2c
    hard_decision = _RelayPy.hard_decision
    syndrome_estimation = _RelayPy.syndrome_estimation
    cn_update = _RelayPy.cn_update
    marginal_update = _RelayPy.marginal_update


    def sign_flip(self, syndrome, s_est, l_v):
        synd_diff = (syndrome + s_est)%2.0

        temp_ls = vn_unsat_count(synd_diff.float(), self.V_c_col, self.H_shape[1])

        total_ones = temp_ls.sum(dim=1).to(self.dtype)

        valid_mask = total_ones > 0

        if self.random_machine.lower() == 'system':
            r = torch.rand(self.batch_size, device=self.device, dtype=self.dtype)
        elif self.random_machine.lower() == 'sobol':
            r = self.r[self.i-1].repeat(self.batch_size)

        target = torch.zeros_like(total_ones).to(self.dtype)
        target[valid_mask] = torch.floor(r[valid_mask] * (total_ones[valid_mask] - 1))
        target = target.unsqueeze(1)
        cumsum_x = torch.cumsum(temp_ls, dim=1)
        mask = cumsum_x >= target + 1
        selected_indices = torch.argmax(mask.float(), dim=1)

        l_v[valid_mask, selected_indices[valid_mask]] *= -1.0
        return l_v

