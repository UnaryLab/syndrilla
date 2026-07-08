import torch

from loguru import logger

from syndrilla.decoder.decoder import RebatchSpeedup


class create(torch.nn.Module):
    """
    This class creates a lotterybp decoder on a single GPU
    """
    def __init__(self,
                 decoder_cfg,
                 **kwargs) -> None:
        """
        Initialization for lotterybp decoder
        Input:
            decoder_cfg: the information that come from config file (yaml)

        Parameters:
            max_iter: the number of maximum iteration of lotterybp decoder
            i: the number of iterations running the decoder

            H_matrix: loaded ldpc matrix, either hx or hz, as 2d tensor

            V_c_row: the row index of all the variable nodes for each check node
            V_c_col: the column index of all the variable nodes for each check node

            degree: the maximum number of 1s in all check nodes in H_matrix
        """

        super(create, self).__init__()

        logger.info(f'Creating lotterybp decoder.')

        # set up default device
        device_cfg = decoder_cfg.get('device', {})
        self.device = device_cfg.get('device_type', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        if self.device not in {'cuda', 'cpu', torch.device('cuda'), torch.device('cpu')}:
            logger.warning(f'Invalid input device <{self.device}>, default to avaliable device in your machine.')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.device == 'cuda':
            device_idx = device_cfg.get('device_idx', 0)
            if device_idx >= torch.cuda.device_count():
                logger.warning(f'Invalid input device index <{device_idx}>, default to avaliable device in your machine.')
                self.device = torch.device(f'cuda:0')
            else:
                self.device = torch.device(f'cuda:{device_idx}')

        # set up default max_iter
        self.max_iter = decoder_cfg.get('max_iter', 50)
        if self.max_iter <= 0 or not isinstance(self.max_iter, int):
            logger.warning(f'Invalid input maximum iteration <{self.max_iter}>, default to <50>.')
            self.max_iter = 50
        
        # set up default dtype
        self.dtype = decoder_cfg.get('dtype', 'float64')
        if self.dtype not in {'float32', 'float64', 'bfloat16', 'float16'}: 
            logger.warning(f'Invalid input data type <{self.dtype}>, default to <torch.float64>.')
            self.dtype = 'float64'
        self.dtype = torch.__dict__[self.dtype]

        self.batch_size = 1

        self.check_type = decoder_cfg.get('check_type', 'hx')
        if self.check_type.lower() not in {'hx', 'hz'}: 
            logger.warning(f'Invalid input check type <{self.check_type}>, default to <hx>.')
            self.check_type = 'hx'

        self.random_machine = decoder_cfg.get('random_machine', 'sobol')
        if self.random_machine.lower() not in {'sobol', 'system'}:
            logger.warning(f'Invalid input machine type <{self.random_machine}>, default to <sobol>.')
            self.random_machine = 'sobol'

        # Sign-flip policy:
        self._sign_flip_policies = {
            'Proposed',
            'global_optimal',
            'global_connectivity',
            'global_weighted_random',
            'local_random',
            'local_reliable',
            'local_connectivity',
        }
        self.sign_flip_policy = decoder_cfg.get('sign_flip_policy', 'Proposed')
        if self.sign_flip_policy not in self._sign_flip_policies: 
            logger.warning(f'Invalid sign_flip_policy <{self.sign_flip_policy}>, defaulting to <Proposed>. Allowed: {sorted(self._sign_flip_policies)}.')
            self.sign_flip_policy = 'Proposed'

        bundle = kwargs.get('bundle')
        if bundle is None:
            raise ValueError('bp_lottery_policy requires a pre-loaded MatrixBundle via the `bundle` kwarg.')
        self.Hx_matrix = bundle.Hx_matrix
        self.Hz_matrix = bundle.Hz_matrix
        self.lx_matrix = bundle.lx_matrix
        self.lz_matrix = bundle.lz_matrix
        self.H_shape, self.V_c_row, self.V_c_col, self.H_matrix = bundle.select(self.check_type)

        self.mask_dummy = (self.V_c_col == self.H_shape[1])

        # set iteration
        self.i = 0

        # convert to as the parameters in a model
        self.V_c_row = torch.nn.Parameter(self.V_c_row, requires_grad=False)
        self.V_c_col = torch.nn.Parameter(self.V_c_col, requires_grad=False)

        self.algo = 'bp_lottery_policy'
        self.num_max_iter = self.max_iter

        # opt-in adaptive iteration speedup (no-op unless an `rebatch_speedup` block is
        # in the config). When active it stops a batch once a learned % has converged
        # and leaves the rest unconverged for main's extra queue. See decoder.py.
        self.cap = RebatchSpeedup.from_cfg(decoder_cfg.get('rebatch_speedup'))
        self.cap_bypass = False       # set by main: True -> decode this batch uncapped
        self.cap_active_last = False  # set per forward: True if the cap was applied

        logger.info(f'Complete.')


    def forward(self, io_dict):
        """Iterative lotterybp (normalized min sum) decoding algorithm
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

        logger.info(f'Initializing lotterybp (normailized min sum) decoding.')
        syndrome = io_dict['synd'].to(dtype=self.dtype).to(self.device)
        
        self.batch_size, _ = syndrome.size()
        
        torch.set_default_dtype(self.dtype)

        # add a dummy element at the end in case the H (ldpc matrix) does not have the same number of 1s in each check node
        N_extended = self.H_shape[1] + 1 
        l_v = torch.zeros([self.batch_size, N_extended], dtype=self.dtype, device=self.device)
        e_v = torch.zeros([self.batch_size, N_extended], dtype=self.dtype, device=self.device)
        s_est = torch.zeros([self.batch_size, self.H_shape[0]], dtype=self.dtype, device=self.device)
        
        # add dummy column
        dummy_column = torch.full([self.batch_size,1], float('inf'), dtype=self.dtype, device=self.device)
        u_init = torch.cat((io_dict['llr0'].to(self.device).to(self.dtype), dummy_column), dim=1)
        e_out = torch.zeros([self.batch_size, N_extended], dtype=self.dtype, device=self.device)
        l_out = torch.zeros([self.batch_size, N_extended], dtype=self.dtype, device=self.device)
        num_iters = torch.full([self.batch_size], -1, device=self.device)
        converges = torch.full([self.batch_size], 0, device=self.device)

        # set up initialization for all parameters for decoding process 
        # message is a in place version of a_v2c and b_c2v
        message = torch.zeros_like(self.V_c_row.unsqueeze(0), dtype=self.dtype, device=self.device).repeat(self.batch_size, 1, 1)
        message = u_init[:, self.V_c_col]

        # compute syndrome for multiplication
        self.syndrome_neg = torch.where(syndrome == 0.0, 1.0, -1.0).to(self.dtype)
        self.syndrome_neg = self.syndrome_neg[:, self.V_c_row]

        if self.random_machine.lower() == 'sobol':
            if self.dtype in {'float32', 'float64'}:
                sobol = torch.quasirandom.SobolEngine(dimension=1, scramble=False)
                self.r = sobol.draw(self.max_iter).to(self.device).to(self.dtype)
            else: 
                sobol = torch.quasirandom.SobolEngine(dimension=1, scramble=False)
                self.r = sobol.draw(self.max_iter, dtype=torch.float32).to(self.device).to(self.dtype)
        
        logger.info(f'Complete.')

        logger.info(f'Starting decoding iterations.')

        # adaptive cap: once warm-up has chosen a stop fraction, break this batch as
        # soon as that fraction has converged (unless main asked for an uncapped pass).
        self.cap_active_last = bool(self.cap is not None and self.cap.done and not self.cap_bypass)
        cap_frac = self.cap.frac if self.cap_active_last else None

        self.i = 0
        while self.i < self.max_iter:
            self.i += 1

            # v2c: gather the per-variable LLR into the check-grouped layout
            l_v_v2c = self.v2c(l_v)

            # variable node update
            message = self.vn_update(message, l_v_v2c)

            # check node update (min-sum), still in the [batch, n_checks, degree] layout
            message = self.cn_update(message)

            # c2v: convert the check messages back to the per-variable layout
            message_c2v = self.c2v(message)

            # elementwise LLR update
            l_v = self.llr_update(u_init, message_c2v)

            # hard decision: map posterior LLRs to a binary error estimate
            e_v = self.hard_decision(l_v)

            s_est = self.syndrome_estimation(e_v)

            # different samples from the same batch may terminated at different iteration (pick the smallest one)
            indices = torch.all(s_est == syndrome, 1).nonzero()
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
            if cap_frac is not None and int((num_iters != -1).sum()) >= cap_frac * self.batch_size:
                break

            if self.sign_flip_policy == 'Proposed':
                l_v = self.sign_flip_lottery(syndrome, s_est, l_v)
            elif self.sign_flip_policy == 'global_optimal':
                l_v = self.sign_flip_global_optimal(syndrome, s_est, l_v)
            elif self.sign_flip_policy == 'global_connectivity':
                l_v = self.sign_flip_global_connectivity(syndrome, s_est, l_v)
            elif self.sign_flip_policy == 'global_weighted_random':
                l_v = self.sign_flip_global_weighted_random(syndrome, s_est, l_v)
            elif self.sign_flip_policy == 'local_random':
                l_v = self.sign_flip_local_random(syndrome, s_est, l_v)
            elif self.sign_flip_policy == 'local_reliable':
                l_v = self.sign_flip_local_reliable(syndrome, s_est, l_v)
            elif self.sign_flip_policy == 'local_connectivity':
                l_v = self.sign_flip_local_connectivity(syndrome, s_est, l_v)

        checker = torch.where(num_iters == -1)[0]
        e_out[checker] = e_v[checker]
        l_out[checker] = l_v[checker]
        num_iters[checker] = self.i
        e_out = e_out[:, :-1]
        l_out = l_out[:, :-1]

        # warm-up: observe this batch's iteration distribution (decides k + the cap).
        if self.cap is not None and not self.cap.done and not self.cap_bypass:
            self.cap.observe(num_iters, self.max_iter, self.batch_size)

        logger.info(f'Complete.')
        logger.info(f'Decoding iterations: <{(self.i)}>.')
        io_dict.update({
            'e_v': e_out,
            'iter': num_iters,
            'llr': l_out,
            'converge': converges
        })
        return io_dict


    def v2c(self, l_v):
        """Format conversion (variable -> check layout).

        Gathers the per-variable LLR vector `l_v` ([batch, N+1]) into the
        check-node-grouped layout ([batch, n_checks, degree]) that the variable-node
        and check-node updates operate on. Each edge (c, v) picks up `l_v[:, v]`.
        """
        return l_v[:, self.V_c_col]


    def vn_update(self, b_c2v, l_v_v2c):
        """Variable-node update: produce the v->c messages a_v2c.

        On the first iteration there is no incoming c->v message yet, so the
        initialized message is passed through. Afterwards each v->c message is the
        current per-variable LLR (already v2c-gathered into the check layout) minus
        the incoming c->v message on that same edge (extrinsic information).
        """
        if self.i == 1:
            return b_c2v
        else:
            return l_v_v2c - b_c2v


    def cn_update(self, a_v2c):
        base = torch.tensor(2.0, dtype=self.dtype)
        exponent = torch.tensor(-(self.i), dtype=self.dtype) 

        # Compute the power in PyTorch:
        beta = torch.tensor(1.0, dtype=self.dtype) - torch.pow(base, exponent)
        # beta = 0.75
        # compute sgn
        sign = torch.sgn(a_v2c)
        sign = torch.where(sign == 0.0, -1.0, sign)
        sign_prod = torch.prod(sign, dim=2, keepdim=True)
        Q_sign = self.syndrome_neg * sign_prod
        
        # compute min
        abs_a_v2c = torch.abs(a_v2c)
        sorted, _ = torch.sort(abs_a_v2c, dim=2)
        min_0 = sorted[:, :, 0].unsqueeze(2)
        min_1 = sorted[:, :, 1].unsqueeze(2)
        min_result = torch.where(abs_a_v2c == min_0, min_1, min_0)

        message = beta * sign * Q_sign * min_result
        message[:, self.mask_dummy] = 0.0
        return message


    def c2v(self, b_c2v):
        """Format conversion (check -> variable layout).

        Scatters the check-node-grouped c->v messages ([batch, n_checks, degree])
        back into the per-variable layout ([batch, N+1]) that the LLR update sums
        over: every edge (c, v) accumulates its message into column `v`.
        """
        # set up the format for both data and partition so they can matching each other
        data_flat = b_c2v.flatten(start_dim=1)
        partitions_flat = self.V_c_col.flatten().repeat(self.batch_size, 1)
        sum_b_c2v = torch.zeros([self.batch_size, self.H_shape[1] + 1], dtype=self.dtype, device=self.device)

        sum_b_c2v.scatter_add_(1, partitions_flat, data_flat)

        return sum_b_c2v


    def llr_update(self, u_init, b_c2v):
        """Elementwise LLR update: posterior LLR = channel LLR + sum of incoming c->v.

        Takes the already c2v-converted per-variable messages and adds the channel
        (initialization) LLR. The dummy variable (last column) is pinned to +inf so
        it is never decoded as an error.
        """
        l_v = u_init + b_c2v
        l_v[:, -1] = float('inf')
        return l_v


    def hard_decision(self, l_v):
        """Hard decision: map posterior LLRs to a binary error estimate.

        A non-positive LLR (<= 0) means the bit is more likely 1, so it is set to
        1; otherwise 0. Returns the estimate in the decoder's dtype.
        """
        return torch.where(l_v <= 0.0, 1.0, 0.0).to(self.dtype)

    
    def syndrome_estimation(self, e_v):
        # calculate the syndrome by summing the number of 1s in each column in e
        temp_e = e_v
        temp_e[:, -1] = 0.0
        estimated_syndrome = temp_e[:, self.V_c_col].sum(dim = 2).to(dtype = self.dtype)
        
        return torch.where((estimated_syndrome%2) > 0.0, 1.0, 0.0)
    

    def sign_flip_global_weighted_random(self, syndrome, s_est, l_v):
        # Outside the paper's five-policy. Globally random VN
        # selection weighted by per-VN unsatisfied-CN count: no CN is picked
        # first, and VNs touching more unsatisfied CNs are more likely to be
        # chosen. No |LLR| consideration.
        synd_diff = (syndrome + s_est)%2.0

        temp_ls = torch.matmul(synd_diff.float(), self.H_matrix.unsqueeze(0).float())
        temp_ls = temp_ls.squeeze(0)

        total_ones = temp_ls.sum(dim=1).to(self.dtype)

        valid_mask = total_ones > 0

        if self.random_machine.lower() == 'system':
            r = torch.rand(self.batch_size, device=self.device, dtype=self.dtype)
        elif self.random_machine.lower() == 'sobol':
            r = self.r[(self.i-1)].repeat(self.batch_size)

        target = torch.zeros_like(total_ones).to(self.dtype)

        target[valid_mask] = torch.floor(r[valid_mask] * (total_ones[valid_mask] - 1))
        target = target.unsqueeze(1)
        cumsum_x = torch.cumsum(temp_ls, dim=1)
        mask = cumsum_x >= target + 1
        selected_indices = torch.argmax(mask.float(), dim=1)

        l_v[valid_mask, selected_indices[valid_mask]] *= -1.0
        return l_v
    

    def sign_flip_global_optimal(self, syndrome, s_est, l_v):
        # Paper policy (1) "Global optimal": among all VNs with the most
        # unsatisfied CNs, flip the sign of the one with the minimum
        # absolute LLR. Builds an upper bound but is impractical for
        # hardware due to the global search.
        synd_diff = (syndrome + s_est)%2.0
        
        temp_ls = torch.matmul(synd_diff.float(), self.H_matrix.unsqueeze(0).float()) 
        temp_ls = temp_ls.squeeze(0)

        total_ones = temp_ls.sum(dim=1).to(self.dtype) 

        valid_mask = total_ones > 0

        max_vals, _ = temp_ls.max(dim=1, keepdim=True)

        mask_max = (temp_ls == max_vals)

        abs_llr = torch.abs(l_v[:, :-1])
        mask_max = (temp_ls == temp_ls.max(dim=1, keepdim=True).values)

        masked_abs_llr = abs_llr + (~mask_max) * 1e9

        selected_indices = torch.argmin(masked_abs_llr, dim=1)

        l_v[valid_mask, selected_indices[valid_mask]] *= -1.0
        return l_v


    def sign_flip_global_connectivity(self, syndrome, s_est, l_v):
        # Paper policy (2) "Global connectivity only": among all VNs with the
        # most unsatisfied CNs, flip the sign of a random one. Ignores |LLR|;
        # demonstrates the importance of reliability guidance vs. global_optimal.
        synd_diff = (syndrome + s_est) % 2.0
        
        temp_ls = torch.matmul(synd_diff.float(), self.H_matrix.unsqueeze(0).float()) 
        temp_ls = temp_ls.squeeze(0)

        total_ones = temp_ls.sum(dim=1).to(self.dtype) 
        valid_mask = total_ones > 0

        max_vals, _ = temp_ls.max(dim=1, keepdim=True)

        candidates_mask = (temp_ls == max_vals) & (temp_ls > 0)

        if self.random_machine.lower() == 'system':
            r = torch.rand(self.batch_size, device=self.device, dtype=self.dtype)
        elif self.random_machine.lower() == 'sobol': 
            r = self.r[(self.i-1)].repeat(self.batch_size)

        num_candidates = candidates_mask.sum(dim=1)
        
        safe_num_candidates = num_candidates.clone()
        safe_num_candidates[safe_num_candidates == 0] = 1

        target_k = torch.floor(r * safe_num_candidates).long()

        candidates_cumsum = candidates_mask.cumsum(dim=1)
        
        selected_mask = (candidates_cumsum == (target_k.unsqueeze(1) + 1)) & candidates_mask
        
        selected_indices = torch.argmax(selected_mask.float(), dim=1)

        l_v[valid_mask, selected_indices[valid_mask]] *= -1.0
        
        return l_v
    

    def sign_flip_local_random(self, syndrome, s_est, l_v):
        # Paper policy (3) "Local random": for a randomly selected
        # unsatisfied CN, flip the sign of a random neighboring VN. Lowest
        # hardware complexity but low accuracy due to no reliability guidance.
        synd_diff = (syndrome + s_est) % 2.0
        unsat_cn_mask = synd_diff.bool()  # [B, M]

        batch_size, M = unsat_cn_mask.shape

        if self.random_machine.lower() == 'system':
            r1 = torch.rand(batch_size, device=self.device)
        elif self.random_machine.lower() == 'sobol':
            r1 = self.r[(self.i-1)].repeat(batch_size)

        unsat_cumsum = unsat_cn_mask.cumsum(dim=1)  # [B, M]
        total_unsat = unsat_cn_mask.sum(dim=1)      # [B]
        
        valid_mask = total_unsat > 0

        rand_pos_cn = torch.floor(r1 * total_unsat).long() + 1
        rand_pos_cn = torch.clamp(rand_pos_cn, max=total_unsat)

        chosen_cn_onehot = ((unsat_cumsum >= rand_pos_cn.unsqueeze(1)) & unsat_cn_mask).float()
        chosen_cn_idx = torch.argmax(chosen_cn_onehot, dim=1)  # [B]

        H_expanded = self.H_matrix.bool().unsqueeze(0).expand(batch_size, -1, -1) # [B, M, N]
        connected_vn_mask = H_expanded[torch.arange(batch_size), chosen_cn_idx, :] 

        num_connected_vns = connected_vn_mask.sum(dim=1) # [B]

        if self.random_machine.lower() == 'system':
            r2 = torch.rand(batch_size, device=self.device)
        elif self.random_machine.lower() == 'sobol':
            r2 = torch.rand(batch_size, device=self.device) 


        rand_pos_vn = torch.floor(r2 * num_connected_vns).long() + 1
        
        vn_cumsum = connected_vn_mask.cumsum(dim=1) # [B, N]
        
        chosen_vn_onehot = (vn_cumsum == rand_pos_vn.unsqueeze(1)) & connected_vn_mask
        selected_vn_idx = torch.argmax(chosen_vn_onehot.float(), dim=1) # [B]


        l_v[valid_mask, selected_vn_idx[valid_mask]] *= -1.0

        return l_v
    


        
    def sign_flip_local_reliable(self, syndrome, s_est, l_v):
        # Paper policy (4) "Local reliable": for a randomly selected
        # unsatisfied CN, flip the sign of its neighboring VN with the
        # minimum absolute LLR. No connectivity-based prioritization.
        # synd_diff: [B, M]
        synd_diff = (syndrome + s_est) % 2.0
        unsat_cn_mask = synd_diff.bool()  # [B, M]

        batch_size, M = unsat_cn_mask.shape

        if self.random_machine.lower() == 'system':
            r = torch.rand(batch_size, device=self.device)
        elif self.random_machine.lower() == 'sobol':
            r = self.r[(self.i-1)].repeat(batch_size)

        unsat_cumsum = unsat_cn_mask.cumsum(dim=1)  # [B, M]
        total_unsat = unsat_cn_mask.sum(dim=1)      # [B]

        rand_pos = torch.floor(r * total_unsat).long() + 1
        rand_pos = torch.clamp(rand_pos, max=total_unsat)

        chosen_cn = ((unsat_cumsum >= rand_pos.unsqueeze(1)) & unsat_cn_mask).float()
        chosen_cn_idx = torch.argmax(chosen_cn, dim=1)  # [B]

        H = self.H_matrix.bool().unsqueeze(0).expand(batch_size, -1, -1)  # [B, M, N]
        cn_mask = H[torch.arange(batch_size), chosen_cn_idx, :]             # [B, N]

        llr = l_v[:, :-1]  # [B, N]
        masked_llr = llr + (~cn_mask).float() * 1e9

        selected_vn = torch.argmin(masked_llr, dim=1)

        l_v[torch.arange(batch_size), selected_vn] *= -1.0

        return l_v
    

    def sign_flip_lottery(self, syndrome, s_est, l_v):
        # Paper policy (5) "Proposed lottery policy": two-tier local policy
        # combining connectivity-based prioritization with reliability
        # guidance inside a local neighborhood.
        #   - randomly pick an unsatisfied CN c*
        #   - among c*'s neighboring VNs, take the subset with the most
        #     unsatisfied connecting CNs
        #   - within that subset, flip the sign of the VN with the minimum
        #     absolute LLR
        # Achieves accuracy comparable to global_optimal at low hardware cost.
        # Syndrome Residual
        synd_diff = (syndrome + s_est) % 2.0  # [B, M]
        unsat_cn_mask = synd_diff.bool()
        
        batch_size, M = unsat_cn_mask.shape
        
        # random
        if self.random_machine.lower() == 'system':
            r = torch.rand(batch_size, device=self.device)
        else: # sobol
            r = self.r[(self.i-1)].repeat(batch_size)
            
        total_unsat = unsat_cn_mask.sum(dim=1)
        total_unsat = total_unsat + (total_unsat == 0).float() 
        
        unsat_cumsum = unsat_cn_mask.cumsum(dim=1)
        rand_pos = torch.floor(r * total_unsat).long() + 1
        chosen_cn = ((unsat_cumsum >= rand_pos.unsqueeze(1)) & unsat_cn_mask).float() # [B, M]
        chosen_cn_idx = torch.argmax(chosen_cn, dim=1) # [B]
        
        H_expanded = self.H_matrix.unsqueeze(0).expand(batch_size, -1, -1) # [B, M, N]
        candidate_vn_mask = H_expanded[torch.arange(batch_size), chosen_cn_idx, :].bool()
        
        vn_unsat_counts = torch.matmul(unsat_cn_mask.float(), self.H_matrix.float())
        abs_llr = torch.abs(l_v[:, :-1]) # [B, N]
        score = vn_unsat_counts * 1e6 - abs_llr
        masked_score = score + (~candidate_vn_mask).float() * -1e9
        
        selected_vn = torch.argmax(masked_score, dim=1)
        
        l_v[torch.arange(batch_size), selected_vn] *= -1.0
        
        return l_v
    

    def sign_flip_local_connectivity(self, syndrome, s_est, l_v):
        # Outside the paper's five-policy taxonomy. Local analogue of
        # global_connectivity: random unsatisfied CN, then max unsat-CN VN
        # among that CN's neighbors, with no |LLR| tiebreak (argmax breaks
        # ties by lowest index).
        synd_diff = (syndrome + s_est) % 2.0  # [B, M]
        unsat_cn_mask = synd_diff.bool()
        
        batch_size, M = unsat_cn_mask.shape
        
        if self.random_machine.lower() == 'system':
            r = torch.rand(batch_size, device=self.device)
        else: # sobol
            r = self.r[(self.i-1)].repeat(batch_size)
            
        total_unsat = unsat_cn_mask.sum(dim=1)
        total_unsat = total_unsat + (total_unsat == 0).float() 
        
        unsat_cumsum = unsat_cn_mask.cumsum(dim=1)
        rand_pos = torch.floor(r * total_unsat).long() + 1
        
        chosen_cn = ((unsat_cumsum >= rand_pos.unsqueeze(1)) & unsat_cn_mask).float() # [B, M]
        chosen_cn_idx = torch.argmax(chosen_cn, dim=1) # [B]
        
        H_expanded = self.H_matrix.unsqueeze(0).expand(batch_size, -1, -1) # [B, M, N]
        candidate_vn_mask = H_expanded[torch.arange(batch_size), chosen_cn_idx, :].bool()
        
        vn_unsat_counts = torch.matmul(unsat_cn_mask.float(), self.H_matrix.float())
        
        score = vn_unsat_counts

        masked_score = score + (~candidate_vn_mask).float() * -1e9
        
        selected_vn = torch.argmax(masked_score, dim=1)
        
        l_v[torch.arange(batch_size), selected_vn] *= -1.0
        
        return l_v