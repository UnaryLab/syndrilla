import torch

from loguru import logger

from syndrilla.utils import parse_device_dtype

import numpy as np



class create(torch.nn.Module):
    """
    This class creates a probability-domain sum-product (SPA) bp decoder on a single GPU.

    This is the deterministic reference oracle for the stochastic-computing decoder
    `bp_sum_prod_sc`: it computes the exact probability-domain message-passing equations
    that the stochastic XOR/equality logic gates approximate. The only difference from
    `bp_norm_min_sum` is the check-node update, which uses the syndrome-folded
    probability-domain XOR rule (sum-product / tanh rule) instead of normalized min-sum.

    References:
    R. G. Gallager, Low-Density Parity-Check Codes, MIT Press, 1963. doi:10.7551/mitpress/4347.001.0001
    F. R. Kschischang, B. J. Frey, H.-A. Loeliger, "Factor graphs and the sum-product algorithm," IEEE Trans. Inf. Theory, vol. 47, no. 2, pp. 498-519, 2001. doi:10.1109/18.910572
    D. Poulin, Y. Chung, "On the iterative decoding of sparse quantum codes," Quantum Inf. Comput., vol. 8, no. 10, pp. 987-1000, 2008. arXiv:0801.1241
    """
    def __init__(self,
                 decoder_cfg,
                 **kwargs) -> None:
        """
        Initialization for the probability-domain SPA bp decoder.
        Input:
            decoder_cfg: the information that come from config file (yaml)

        Parameters:
            max_iter: the number of maximum iteration of bp decoder
            i: the number of iterations running the decoder

            H_matrix: loaded ldpc matrix, either hx or hz, as 2d tensor

            V_c_row: the row index of all the variable nodes for each check node
            V_c_col: the column index of all the variable nodes for each check node

            degree: the maximum number of 1s in all check nodes in H_matrix
        """

        super(create, self).__init__()

        logger.info(f'Creating bp_sum_prod (probability-domain sum-product) decoder.')

        # set up default device
        self.device, _ = parse_device_dtype(decoder_cfg)

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

        bundle = kwargs.get('bundle')
        if bundle is None:
            raise ValueError('bp_sum_prod requires a pre-loaded MatrixBundle via the `bundle` kwarg.')
        self.Hx_matrix = bundle.Hx_matrix
        self.Hz_matrix = bundle.Hz_matrix
        self.lx_matrix = bundle.lx_matrix
        self.lz_matrix = bundle.lz_matrix
        self.H_shape, self.V_c_row, self.V_c_col, self.H_matrix = bundle.select(self.check_type)

        self.mask_dummy = (self.V_c_col == self.H_shape[1])

        # small constant to keep probabilities away from 0/1 (avoids inf in the prob<->llr maps)
        self.eps = 1e-12

        # set iteration
        self.i = 0

        # convert to as the parameters in a model
        self.V_c_row = torch.nn.Parameter(self.V_c_row, requires_grad=False)
        self.V_c_col = torch.nn.Parameter(self.V_c_col, requires_grad=False)

        self.algo = 'bp_sum_prod'
        self.num_max_iter = self.max_iter

        logger.info(f'Complete.')


    def forward(self, io_dict):
        """Iterative probability-domain sum-product (SPA) decoding algorithm.
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
        logger.info(f'Initializing bp_sum_prod (probability-domain sum-product) decoding.')


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

        # compute syndrome sign (1 - 2 s_j) on each edge; folds the QEC syndrome into the XOR check
        self.syndrome_neg = torch.where(syndrome == 0.0, 1.0, -1.0).to(self.dtype)
        self.syndrome_neg = self.syndrome_neg[:, self.V_c_row]

        logger.info(f'Complete.')

        logger.info(f'Starting decoding iterations.')

        self.i = 0
        while self.i < self.max_iter:
            # variable node update update v2c
            self.i += 1

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

        checker = torch.where(num_iters == -1)[0]
        e_out[checker] = e_v[checker]
        l_out[checker] = l_v[checker]
        num_iters[checker] = self.max_iter
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
        # updating the a_v2c by b_c2v (extrinsic LLR), identical to normalized min-sum
        if self.i == 1:
            return b_c2v
        else:
            return l_v[:, self.V_c_col] - b_c2v


    def cn_update(self, a_v2c):
        """Probability-domain sum-product (XOR) check-node update with syndrome folding.

        For check j and incident variable i:
            Q_{j->i} = (1 - (1 - 2 s_j) * prod_{k!=i} (1 - 2 p_{k->j})) / 2
        where p = P(bit = 1) = sigmoid(-LLR). This is the exact tanh rule the stochastic
        XOR gate approximates; min-sum replaces the product by its dominant term.
        The result is returned as a c2v LLR so the rest of the pipeline matches min-sum.
        """
        # incoming v2c messages are LLRs; map to P(bit = 1) and to t = 1 - 2p = tanh(LLR/2)
        p = torch.sigmoid(-a_v2c).clamp(self.eps, 1.0 - self.eps)
        t = 1.0 - 2.0 * p

        # leave-one-out product of t over the incident variables (dim=2), robust to zeros
        t_excl = self._loo_prod(t)

        # fold the syndrome sign and map back to a probability, then to an LLR
        Q = ((1.0 - self.syndrome_neg * t_excl) / 2.0).clamp(self.eps, 1.0 - self.eps)
        return torch.log((1.0 - Q) / Q)


    def _loo_prod(self, t):
        # leave-one-out product along dim=2 via exclusive prefix * suffix products
        ones = torch.ones_like(t[:, :, :1])
        prefix = torch.cumprod(t, dim=2)
        prefix_excl = torch.cat([ones, prefix[:, :, :-1]], dim=2)
        suffix = torch.flip(torch.cumprod(torch.flip(t, dims=[2]), dim=2), dims=[2])
        suffix_excl = torch.cat([suffix[:, :, 1:], ones], dim=2)
        return prefix_excl * suffix_excl


    def llr_update(self, u_init, b_c2v):
        # set up the format for both data and partition so they can matching each other
        data_flat = b_c2v.flatten(start_dim=1)
        partitions_flat = self.V_c_col.flatten().repeat(self.batch_size, 1)
        sum_b_c2v = torch.zeros([self.batch_size, self.H_shape[1] + 1], dtype=self.dtype, device=self.device)

        sum_b_c2v = u_init + sum_b_c2v
        sum_b_c2v.scatter_add_(1, partitions_flat, data_flat)

        return sum_b_c2v


    def syndrome_estimation(self, e_v):
        # calculate the syndrome by summing the number of 1s in each column in e
        temp_e = e_v
        temp_e[:, -1] = 0.0
        estimated_syndrome = temp_e[:, self.V_c_col].sum(dim = 2).to(dtype = self.dtype)

        return torch.where((estimated_syndrome%2) > 0.0, 1.0, 0.0)
