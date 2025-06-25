import torch

from loguru import logger

import numpy as np

from src.matrix import create_parity_matrix
from src.utils import compute_lz


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
        self.device = decoder_cfg.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        if self.device not in {'cuda', 'cpu', torch.device('cuda'), torch.device('cpu')}:
            logger.warning(f'Invalid input device <{self.device}>, default to avaliable device in your machine.')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # set up default max_iter
        self.max_iter = decoder_cfg.get('max_iter', 50)
        if self.max_iter <= 0 or not isinstance(self.max_iter, int):
            logger.warning(f'Invalid input maximum iteration <{self.max_iter}>, default to <50>.')
            self.max_iter = 50
        
        # set up default dtype
        self.dtype = decoder_cfg.get('dtype', 'float64')
        if self.dtype not in {'float32', 'float64', 'bfloat16', 'float16'}: 
            logger.warning(f'Invalid input type <{self.dtype}>, default to <torch.float64>.')
            self.dtype = 'float64'
        self.dtype = torch.__dict__[self.dtype]

        self.batch_size = 1

        self.type = decoder_cfg.get('type', 'hx')
        if self.type.lower() not in {'hx', 'hz'}: 
            logger.warning(f'Invalid input type <{self.type}>, default to <hx>.')
            self.dtype = 'hx'

        self.random_machine = decoder_cfg.get('random_machine', 'sobol')
        if self.random_machine.lower() not in {'sobol', 'system'}: 
            logger.warning(f'Invalid input type <{self.random_machine}>, default to <sobol>.')
            self.random_machine = 'sobol'

        # get the column and row index for all 1s in parity check matrix
        logger.info(f'Creating hx parity check matrix.')
        self.Hx_matrix = create_parity_matrix(yaml_path=decoder_cfg['parity_matrix_hx'], device=self.device, dtype=self.dtype)

        logger.info(f'Creating hz parity check matrix.')
        self.Hz_matrix = create_parity_matrix(yaml_path=decoder_cfg['parity_matrix_hz'], device=self.device, dtype=self.dtype)

        if self.type.lower() == 'hx':
            self.H_shape, self.V_c_row, self.V_c_col, self.H_matrix = self.Hx_matrix.get_index()
        else: 
            self.H_shape, self.V_c_row, self.V_c_col, self.H_matrix = self.Hz_matrix.get_index()

        # compute lx, switch hx and hz position can compute lz
        # currently, lx and lz are following bposd, https://github.com/quantumgizmos/bp_osd
        logger.info(f'Creating lx and lz parity check matrix.')
        logical_check_matrix =  decoder_cfg.get('logical_check_matrix', False)
        if logical_check_matrix:
            self.lx_matrix = create_parity_matrix(yaml_path=decoder_cfg['logical_check_lx'], device=self.device, dtype=self.dtype).get_dense()
            self.lz_matrix = create_parity_matrix(yaml_path=decoder_cfg['logical_check_lz'], device=self.device, dtype=self.dtype).get_dense()
        else:
            self.lx_matrix = compute_lz(self.Hz_matrix.get_dense(), self.Hx_matrix.get_dense())
            self.lz_matrix = compute_lz(self.Hx_matrix.get_dense(), self.Hz_matrix.get_dense())

        self.mask_dummy = (self.V_c_col == self.H_shape[1])
        
        # set iteration
        self.i = 0
        
        # convert to as the parameters in a model
        self.V_c_row = torch.nn.Parameter(self.V_c_row, requires_grad=False)
        self.V_c_col = torch.nn.Parameter(self.V_c_col, requires_grad=False)

        self.algo = 'bp_lottery'
        
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

        self.i = 0
        while self.i < self.max_iter:
            self.i += 1

            # variable node update update v2c
            message = self.vn_update(message, l_v)

            # check node update c2v
            message = self.cn_update(message)
            message[:, self.mask_dummy] = float(0.0)

            # elementwise LLR update
            l_v = self.llr_update(u_init, message)
            l_v[:, -1] = float('inf')

            e_v = torch.where(l_v <= 0.0, 1.0, 0.0).to(self.dtype)
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


    def vn_update(self, b_c2v, l_v):
        # updating the a_v2c by b_c2v
        if self.i == 1:
            return b_c2v
        else:
            return l_v[:, self.V_c_col] - b_c2v
        

    def cn_update(self, a_v2c):
        base = torch.tensor(2.0, dtype=self.dtype)
        exponent = torch.tensor(-(self.i), dtype=self.dtype) 

        # Compute the power in PyTorch:
        beta = torch.tensor(1.0, dtype=self.dtype) - torch.pow(base, exponent)

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
        
        return torch.where((estimated_syndrome%2) > 0.0, 1.0, 0.0)
    

    def sign_flip(self, syndrome, s_est, l_v):
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
    
