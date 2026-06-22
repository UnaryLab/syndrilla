import torch
import random

from loguru import logger

import numpy as np

from syndrilla.decoder.decoder import IterSpeedup


class create(torch.nn.Module):
    """
    This class creates a bp decoder on a single GPU
    """
    def __init__(self,
                 decoder_cfg,
                 **kwargs) -> None:
        """
        Initialization for bp decoder
        Input:
            decoder_cfg: the information that come from config file (yaml)

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
        """

        super(create, self).__init__()

        logger.info(f'Creating bp decoder.')

        # Defaults match the relay-bp crate's gamma_dist_interval = (-0.24, 0.66),
        # expressed here as center = (low + high) / 2 and width = high - low.
        self.center = decoder_cfg.get('center', 0.21)
        if not isinstance(self.center, float):
            logger.warning(f'Invalid input center <{self.center}>, default to <0.21>.')
            self.center = 0.21

        self.width = decoder_cfg.get('width', 0.9)
        if not isinstance(self.width, float):
            logger.warning(f'Invalid input width <{self.width}>, default to <0.9>.')
            self.width = 0.9

        self.init_mem_strength = decoder_cfg.get('init_mem_strength', 0.35)

        # Min-sum normalization factor (alpha). Matches the relay-bp crate semantics:
        #   alpha == 0.0  -> adaptive schedule 1 - 2^(-i / alpha_scaling) (crate's Some(0.))
        #   alpha < 0.0   -> falls back to 1.0 (crate behavior)
        #   otherwise     -> constant alpha (e.g. 1.0 for plain min-sum, the crate default)
        # Default 0.0 preserves this decoder's original adaptive normalized min-sum.
        self.alpha_const = decoder_cfg.get('alpha', 0.0)
        if not isinstance(self.alpha_const, (int, float)) or isinstance(self.alpha_const, bool):
            logger.warning(f'Invalid input alpha <{self.alpha_const}>, default to <0.0>.')
            self.alpha_const = 0.0
        self.alpha_const = float(self.alpha_const)

        self.alpha_scaling = decoder_cfg.get('alpha_scaling', 1.0)
        if not isinstance(self.alpha_scaling, (int, float)) or isinstance(self.alpha_scaling, bool) or self.alpha_scaling <= 0:
            logger.warning(f'Invalid input alpha_scaling <{self.alpha_scaling}>, default to <1.0>.')
            self.alpha_scaling = 1.0
        self.alpha_scaling = float(self.alpha_scaling)

        # set up default device. Accept either a `device:` block {device_type, device_idx}
        # (as main passes via the decoder yaml) or a plain string / torch.device (direct use).
        device_cfg = decoder_cfg.get('device', {})
        if isinstance(device_cfg, dict):
            self.device = device_cfg.get('device_type', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            device_idx = device_cfg.get('device_idx', 0)
        else:
            self.device, device_idx = device_cfg, 0
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

        #set up default solution number
        self.solution = decoder_cfg.get('solution', 5)
        if self.solution <= 0 or not isinstance(self.solution, int):
            logger.warning(f'Invalid input solution number <{self.solution}>, default to <5>.')
            self.solution = 5
        
        #initial iteration count
        self.iteration_initial = decoder_cfg.get('iteration_initial', 80)
        if self.iteration_initial < 0 or not isinstance(self.iteration_initial, int):
            logger.warning(f'Invalid input iteration initial <{self.iteration_initial}>, default to <0>.')
            self.iteration_initial = 80

        #set up default iteration count
        self.iteration_count = decoder_cfg.get('iteration_count', 60)
        if self.iteration_count <= 0 or not isinstance(self.iteration_count, int):
            logger.warning(f'Invalid input iteration count <{self.iteration_count}>, default to <60>.')
            self.iteration_count = 60

        # set up default R
        self.legs = decoder_cfg.get('legs', 20)
        if self.legs <= 0 or not isinstance(self.legs, int):
            logger.warning(f'Invalid input R <{self.legs}>, default to <1>.')
            self.legs = 20
        
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
            self.type = 'hx'
        
        self.check_type = decoder_cfg.get('check_type', 'hx')
        if self.check_type.lower() not in {'hx', 'hz'}: 
            logger.warning(f'Invalid input check type <{self.check_type}>, default to <hx>.')
            self.check_type = 'hx'

        # Matrices: use a pre-loaded bundle when one is provided (this is how main builds every
        # decoder -- create_decoder(cfg=..., bundle=bundle)), like the bp_norm_min_sum family.
        # Otherwise load them from the yaml paths in the config (direct instantiation in tests).
        bundle = kwargs.get('bundle')
        if bundle is not None:
            self.Hx_matrix = bundle.Hx_matrix
            self.Hz_matrix = bundle.Hz_matrix
            self.lx_matrix = bundle.lx_matrix
            self.lz_matrix = bundle.lz_matrix
            self.H_shape, self.V_c_row, self.V_c_col, self.H_matrix = bundle.select(self.check_type)
        else:
            # get the column and row index for all 1s in parity check matrix
            logger.info(f'Creating hx parity check matrix.')
            self.Hx_matrix = create_parity_matrix(yaml_path=decoder_cfg['parity_matrix_hx'], device=self.device, dtype=self.dtype)

            logger.info(f'Creating hz parity check matrix.')
            self.Hz_matrix = create_parity_matrix(yaml_path=decoder_cfg['parity_matrix_hz'], device=self.device, dtype=self.dtype)

            if self.check_type.lower() == 'hx':
                self.H_shape, self.V_c_row, self.V_c_col, self.H_matrix = self.Hx_matrix.get_index()
            else:
                self.H_shape, self.V_c_row, self.V_c_col, self.H_matrix = self.Hz_matrix.get_index()

            # compute lx, switch hx and hz position can compute lz
            # currently, lx and lz are following bposd, https://github.com/quantumgizmos/bp_osd
            logger.info(f'Creating lx and lz parity check matrix.')
            logical_check_matrix = decoder_cfg.get('logical_check_matrix', False)
            if logical_check_matrix:
                self.lx_matrix = create_parity_matrix(yaml_path=decoder_cfg['logical_check_lx'], device=self.device, dtype=self.dtype).get_dense()
                self.lz_matrix = create_parity_matrix(yaml_path=decoder_cfg['logical_check_lz'], device=self.device, dtype=self.dtype).get_dense()
            else:
                self.lx_matrix = compute_lz(self.Hz_matrix.get_dense(), self.Hx_matrix.get_dense())
                self.lz_matrix = compute_lz(self.Hx_matrix.get_dense(), self.Hz_matrix.get_dense())

        self.mask_dummy = (self.V_c_col == self.H_shape[1])

        # convert to as the parameters in a model
        self.V_c_row = torch.nn.Parameter(self.V_c_row, requires_grad=False)
        self.V_c_col = torch.nn.Parameter(self.V_c_col, requires_grad=False)

        self.algo = 'bp_relay'
        self.num_max_iter = self.iteration_initial + (self.legs - 1) * self.iteration_count

        # opt-in adaptive iteration speedup (no-op unless an `iter_speedup` block is in the
        # config). When active it stops the leg ensemble once a learned % of the batch has
        # converged and leaves the rest unconverged for main's deferred extra queue. Mirrors
        # bp_norm_min_sum; here the cap is at the LEG granularity (each leg is a full BP run).
        self.cap = IterSpeedup.from_cfg(decoder_cfg.get('iter_speedup'))
        self.cap_bypass = False       # set by main: True -> decode this batch uncapped
        self.cap_active_last = False  # set per forward: True if the cap was applied

        logger.info(f'Complete.')


    def forward(self, io_dict):
        
        logger.info(f'Initializing bp-relay decoding.')

        syndrome = io_dict['synd'].to(dtype=self.dtype).to(self.device)
    
        self.batch_size, _ = syndrome.size()
    
        torch.set_default_dtype(self.dtype)

        # add a dummy element at the end in case the H (ldpc matrix) does not have the same number of 1s in each check node
        N_extended = self.H_shape[1] + 1 
        solutions = torch.zeros([self.batch_size], dtype=self.dtype, device=self.device)
        e_solutions = torch.full([self.batch_size], float('inf'), dtype=self.dtype, device=self.device)
        e_best = torch.zeros([self.batch_size, self.H_shape[1]], dtype=self.dtype, device=self.device)
        l_v = torch.zeros([self.batch_size, N_extended], dtype=self.dtype, device=self.device)
        e_v = torch.zeros([self.batch_size, N_extended], dtype=self.dtype, device=self.device)
        s_est = torch.zeros([self.batch_size, self.H_shape[0]], dtype=self.dtype, device=self.device)
        solution_sum = 0
        
        # add dummy column
        dummy_column = torch.full([self.batch_size,1], float('inf'), dtype=self.dtype, device=self.device)
        u_init = torch.cat((io_dict['llr0'].to(self.device).to(self.dtype), dummy_column), dim=1)
        logger.info(str(u_init.size()))
        e_out = torch.zeros([self.batch_size, N_extended], dtype=self.dtype, device=self.device)
        l_out = torch.zeros([self.batch_size, N_extended], dtype=self.dtype, device=self.device)
        num_iters = torch.full([self.batch_size], 1, device=self.device)
        converges = torch.full([self.batch_size], 0, device=self.device)

        # set up initialization for all parameters for decoding process 
        # message is a in place version of a_v2c and b_c2v
        message = torch.zeros_like(self.V_c_row.unsqueeze(0), dtype=self.dtype, device=self.device).repeat(self.batch_size, 1, 1)
        message = u_init[:, self.V_c_col]

        # compute syndrome for multiplication
        self.syndrome_neg = torch.where(syndrome == 0.0, 1.0, -1.0).to(self.dtype)
        self.syndrome_neg = self.syndrome_neg[:, self.V_c_row]

        logger.info(f'Complete.')

        logger.info(f'Starting decoding iterations.')

        # adaptive cap: once warm-up has chosen a stop fraction, end the leg ensemble as soon
        # as that fraction has converged (unless main asked for an uncapped pass).
        self.cap_active_last = bool(self.cap is not None and self.cap.done and not self.cap_bypass)
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
                l_v = u_init.clone()
            else:
                self.max_iter = self.iteration_count
                memory_strengths = self.create_memory_strengths(self.batch_size, N_extended, self.center, self.width)
                message = u_init[:, self.V_c_col]
            
            
            while self.i < self.max_iter:
                # variable node update update v2c
                self.i += 1

                # memory bias (variable prior with disordered memory), from the
                # posterior of the previous iteration
                bias = self.bias_update(memory_strengths, l_v, u_init)

                # check node update c2v: consumes the variable->check messages carried
                # from the previous iteration (dummy edges zeroed inside cn_update).
                c2v = self.cn_update(message)

                # marginal / posterior LLR update (dummy variable pinned to +inf inside)
                l_v = self.marginal_update(c2v, bias)

                # hard decision: map posterior LLRs to a binary error estimate
                e_v = self.hard_decision(l_v)

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

                # variable node update: produce the variable->check messages for the
                # next iteration (extrinsic: bias + sum of OTHER incoming c2v messages)
                message = self.vn_update(c2v, bias)
            
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

        logger.info(f'Complete.')
        logger.info(f'Decoding iterations: <{(self.i)}>.')
        io_dict.update({
            'e_v': e_best,
            'iter': num_iters,
            'llr': l_out,
            'converge': converges
        })
        return io_dict
     
        
    def sum_func(self, bias, b_c2v):
        data_flat = b_c2v.flatten(start_dim=1)
        partitions_flat = self.V_c_col.flatten().repeat(self.batch_size, 1)
        sum_b_c2v = torch.zeros([self.batch_size, self.H_shape[1] + 1], dtype=self.dtype, device=self.device)
        sum_b_c2v = bias + sum_b_c2v
        sum_b_c2v.scatter_add_(1, partitions_flat, data_flat)
        return sum_b_c2v


    def vn_update(self, b_c2v, bias):
        # extrinsic variable->check message: the variable prior (with memory) plus the
        # sum of all OTHER incoming check->variable messages on each edge. Called once
        # per iteration to build the messages consumed by the next check-node update.
        sum_b_c2v = self.sum_func(bias, b_c2v)
        return sum_b_c2v[:, self.V_c_col] - b_c2v


    def compute_alpha(self):
        if self.alpha_const == 0.0:
            alpha = 1.0 - 2.0 ** (-(self.i / self.alpha_scaling))
        else:
            alpha = self.alpha_const
        if alpha < 0.0:
            alpha = 1.0
        return torch.tensor(alpha, dtype=self.dtype)


    def cn_update(self, a_v2c):
        # normalization factor (alpha); for alpha == 0 this is the adaptive 1 - 2^(-i) schedule
        beta = self.compute_alpha()

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
        message[:, self.mask_dummy] = 0.0          # non-existent (padded) edges carry no message
        return message


    def marginal_update(self, b_c2v, bias):
        sum_b_c2v = self.sum_func(bias, b_c2v)
        sum_b_c2v[:, -1] = float('inf')            # dummy variable never decoded as an error
        return sum_b_c2v


    def hard_decision(self, l_v):
        """Hard decision: a non-positive posterior LLR (<= 0) decodes to 1, else 0."""
        return torch.where(l_v <= 0.0, 1.0, 0.0).to(self.dtype)


    def syndrome_estimation(self, e_v):
        # calculate the syndrome by summing the number of 1s in each column in e
        temp_e = e_v
        temp_e[:, -1] = 0.0
        estimated_syndrome = temp_e[:, self.V_c_col].sum(dim = 2).to(dtype = self.dtype)
        
        return torch.where((estimated_syndrome%2) > 0.0, 1.0, 0.0)
    

    def bias_update(self, memory_strengths, marginals, u_init):
        marginal_strength = memory_strengths * marginals
        sub = 1 - memory_strengths
        bias = (sub * u_init) + marginal_strength
        bias[:, -1] = u_init[:, -1]
        return bias
    

    def create_memory_strengths(self, rows, cols, center, width):
        memory_strengths = ((center + width/2) - (center-width/2)) * torch.rand([rows, cols], dtype=self.dtype, device=self.device) + (center - width/2)
        return memory_strengths
