import torch

from loguru import logger


class create(torch.nn.Module):
    """
    This class creates a bsfbp decoder on a single GPU
    based on this paper: Branch-Assisted Sign-Flipping Belief Propagation Decoding for Topological Quantum Codes Based on Hypergraph Product Structure
    """
    def __init__(self,
                 decoder_cfg,
                 **kwargs) -> None:
        """
        Initialization for bsfbp decoder
        Input:
            decoder_cfg: the information that come from config file (yaml)

        Parameters:
            max_iter: the number of maximum iteration of bsfbp decoder
            i: the number of iterations running the decoder

            H_matrix: loaded ldpc matrix, either hx or hz, as 2d tensor

            V_c_row: the row index of all the variable nodes for each check node
            V_c_col: the column index of all the variable nodes for each check node

            degree: the maximum number of 1s in all check nodes in H_matrix
        """

        super(create, self).__init__()

        logger.info(f'Creating bsfbp decoder.')
        
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

        # set up default max_iter
        self.max_b_iter = decoder_cfg.get('max_b_iter', 12)
        if self.max_b_iter <= 0 or not isinstance(self.max_b_iter, int):
            logger.warning(f'Invalid input maximum iteration <{self.max_b_iter}>, default to <12>.')
            self.max_b_iter = 50
        
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

        bundle = kwargs.get('bundle')
        if bundle is None:
            raise ValueError('bp_branch_assisted requires a pre-loaded MatrixBundle via the `bundle` kwarg.')
        self.Hx_matrix = bundle.Hx_matrix
        self.Hz_matrix = bundle.Hz_matrix
        self.lx_matrix = bundle.lx_matrix
        self.lz_matrix = bundle.lz_matrix
        self.H_shape, self.V_c_row, self.V_c_col, self.H_matrix = bundle.select(self.check_type)

        self.mask_dummy = (self.V_c_col == self.H_shape[1])

        # set iteration may not needed
        self.i = 0
        
        # convert to as the parameters in a model
        self.V_c_row = torch.nn.Parameter(self.V_c_row, requires_grad=False)
        self.V_c_col = torch.nn.Parameter(self.V_c_col, requires_grad=False)

        self.algo = 'bp_branch_assisted'
        self.num_max_iter = self.max_iter * self.max_b_iter
        
        logger.info(f'Complete.')


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
        logger.info(f'Initializing bsfbp (normailized min sum) decoding.')

        syndrome = io_dict['synd'].to(self.device).to(self.dtype)
        self.batch_size, _ = syndrome.size()
        
        # add a dummy element at the end in case the H (ldpc matrix) does not have the same number of 1s in each check node
        N_extended = self.H_shape[1] + 1 
        l_v = torch.zeros([self.batch_size, N_extended], dtype=self.dtype, device=self.device)
        e_v = torch.zeros([self.batch_size, N_extended], dtype=self.dtype, device=self.device)
        e_v_saver = torch.zeros([self.batch_size, N_extended], dtype=self.dtype, device=self.device)
        s_est = torch.zeros([self.batch_size, self.H_shape[0]], dtype=self.dtype, device=self.device)
        
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
        u_init = torch.cat((io_dict['llr0'].to(self.device), dummy_column), dim=1)
        e_out = torch.zeros([self.batch_size, N_extended], dtype=self.dtype, device=self.device)
        l_out = torch.zeros([self.batch_size, N_extended], dtype=self.dtype, device=self.device)
        num_iters = torch.full([self.batch_size], -1, device=self.device)
        converges = torch.full([self.batch_size], 0, device=self.device)
        
        # set up initialization for all parameters for decoding process 
        # message is a in place version of a_v2c and b_c2v
        message = torch.zeros_like(self.V_c_col.unsqueeze(0), dtype=self.dtype, device=self.device).repeat(self.batch_size, 1, 1)
        message_saved = torch.zeros_like(self.V_c_col.unsqueeze(0), dtype=self.dtype, device=self.device).repeat(self.batch_size, 1, 1)
        message = u_init[:, self.V_c_col]

        if self.random_machine.lower() == 'sobol':
            sobol = torch.quasirandom.SobolEngine(dimension=1, scramble=False)
            self.r = sobol.draw(self.max_iter*self.max_iter).to(self.device).to(self.dtype)

        logger.info(f'Complete.')

        logger.info(f'Starting decoding iterations.')
        
        self.i = 0
        checker = torch.where(num_iters == -1)[0]
        while (checker.size()[0] != 0):
            self.i += 1
            curr_iters += 1
            
            # variable node update update v2c
            message = self.vn_update(message, l_v, curr_iters)

            # check node update c2v
            message = self.cn_update(message, syndrome, curr_iters)
            message[:, self.mask_dummy] = float(0.0)

            # elementwise LLR update
            l_v = self.llr_update(u_init, message)
            
            # hard decision
            e_v = torch.where(l_v <= 0, 1, 0).to(self.dtype)

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
                logger.info(f'Complete.')
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
            # s0_est_comp = torch.sum((zeroth_saver + syndrome) % 2, 1)
            sk_est_comp = (s_est + syndrome) % 2
            c1_results = (torch.sum(sk_est_comp, 1) <= s0_est_comp).int()
            c2_results = torch.sum(((s_est == 1) & (syndrome == 0)).int(),1)

            branch_index = torch.where((c1_results == 1) & (c2_results == 0) &(num_iters == -1))[0]
        
            mask = ~torch.isin(branch_index, index_saver)
            
            not_in_index_saver = branch_index[mask]
            
            if not_in_index_saver.numel() != 0 and self.i != 0:
                syndrome_saver[not_in_index_saver] = syndrome[not_in_index_saver]
                syndrome[not_in_index_saver] = sk_est_comp[not_in_index_saver]

                e_v_saver[not_in_index_saver] = e_v[not_in_index_saver]
                
                s_est_saver[not_in_index_saver] = s_est[not_in_index_saver]

                iteration_saver[not_in_index_saver] = curr_iters[not_in_index_saver]
                curr_iters[not_in_index_saver] = 0

                l_saver[not_in_index_saver] = l_v[not_in_index_saver]
                l_v[not_in_index_saver] = u_init[not_in_index_saver]
                
                message_saved[not_in_index_saver] = message[not_in_index_saver]
                message[not_in_index_saver] = u_init[not_in_index_saver][:, self.V_c_col]
                index_saver = torch.cat([not_in_index_saver, index_saver], dim=0)
            
            b_finish = (curr_iters[index_saver] >= self.max_b_iter)
            
            if torch.any(b_finish):
                last_b_iter_ind = index_saver[torch.where(b_finish)[0].int()]
                
                message[last_b_iter_ind] = message_saved[last_b_iter_ind]
                syndrome[last_b_iter_ind] = syndrome_saver[last_b_iter_ind]
                s0_est_comp[last_b_iter_ind] = torch.sum((s_est_saver[last_b_iter_ind] + syndrome[last_b_iter_ind]) % 2, 1)
                
                l_v[last_b_iter_ind] = l_saver[last_b_iter_ind] 
                
                # remove index of branch
                curr_iters[last_b_iter_ind] = iteration_saver[last_b_iter_ind]
                mark = ~torch.isin(index_saver, last_b_iter_ind)
                index_saver = index_saver[mark]

            # sign flip
            l_v = self.sign_flip(syndrome, s_est, l_v)
            checker = torch.where(num_iters == -1)[0]
                   
        e_out[checker] = e_v[checker]
        l_out[checker] = l_v[checker]
        num_iters[checker] = self.i
        e_out = e_out[:, :-1]
        l_out = l_out[:, :-1]
        logger.info(f'Complete.')
        logger.info(f'Decoding iterations: <{(self.i)}>.')
        io_dict.update({
            'e_v': e_out,
            'iter': num_iters,
            'llr': l_out,
            'converge': converges
        })
        return io_dict


    def vn_update(self, b_c2v, l_v, curr_iters):
        result = torch.empty_like(b_c2v)
        mask = (curr_iters == 1)

        result[mask] = b_c2v[mask]
        result[~mask] = l_v[~mask][:, self.V_c_col] - b_c2v[~mask]
        return result
        

    def cn_update(self, a_v2c, syndrome, curr_iters):
        # compute syndrome for multiplication
        syndrome_neg = torch.where(syndrome != 0, -1.0, 1.0)
        syndrome_neg = syndrome_neg[:, self.V_c_row]

        beta = 1.0 - torch.pow(2.0, -curr_iters).to(self.dtype)
        beta = beta.unsqueeze(1).unsqueeze(2)
    
        # compute sgn
        sign = torch.sgn(a_v2c)
        sign = torch.where(sign == 0.0, -1.0, sign)
        sign_prod = torch.prod(sign, dim=2, keepdim=True)
        Q_sign = syndrome_neg * sign_prod
        
        # compute min
        abs_a_v2c = torch.abs(a_v2c)
        sorted, _ = torch.sort(abs_a_v2c, dim=2)
        min_0 = sorted[:, :, 0].unsqueeze(2)
        min_1 = sorted[:, :, 1].unsqueeze(2)
        min_result = torch.where(abs_a_v2c == min_0, min_1, min_0)

        return beta * sign * Q_sign  * min_result


    def llr_update(self, u_init, b_c2v):
        # set up the format for both data and partition so they can matching each other
        data_flat = b_c2v.flatten(start_dim=1)
        partitions_flat = self.V_c_col.flatten().repeat(self.batch_size, 1)

        sum_b_c2v = torch.zeros([self.batch_size, self.H_shape[1] + 1], dtype=self.dtype, device=self.device)

        # Use index_add to accumulate sums in the result tensor
        sum_b_c2v = u_init + sum_b_c2v
        sum_b_c2v.scatter_add_(1, partitions_flat, data_flat)

        return sum_b_c2v
    

    def syndrome_estimation(self, e_v):
        # calculate the syndrome by summing the number of 1s in each column in e
        temp_e = e_v
        temp_e[:, -1] = 0.0
        estimated_syndrome = temp_e[:, self.V_c_col].sum(dim = 2).to(dtype = self.dtype)
        
        return torch.where((estimated_syndrome%2) > 0.0, 1.0, 0.0).to(dtype = self.dtype)
    
    
    def sign_flip(self, syndrome, s_est, l_v):
        synd_diff = (syndrome + s_est)%2.0
        
        temp_ls = torch.matmul(synd_diff.float(), self.H_matrix.unsqueeze(0).float()) 
        temp_ls = temp_ls.squeeze(0)

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
    
