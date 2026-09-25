import math
import torch

from loguru import logger

from syndrilla.utils import parse_device_dtype

import numpy as np



class create(torch.nn.Module):
    """
    This class creates a bit-stream stochastic-computing (SC) bp decoder on a single GPU.

    Messages are carried as Bernoulli bit streams: one bit per decoding cycle (DC) on each
    edge. Check nodes are XOR gates (syndrome-folded for QEC); variable nodes are
    equality/AND logic whose hold state is resolved by a per-VN up/down saturating counter
    that tracks the regenerated bit probability and breaks the latching/correlation problem.
    The counter is LUT-initialized from the channel prior and its midpoint gives the hard
    decision. See `bp_sum_prod` for the deterministic probability-domain equations this
    stochastic logic approximates, and examples/alist/bp_sum_prod_sc_hx.decoding.yaml for a config.

    Modeling choices (core decoder, v1): a single up/down counter per variable node serves
    as both the conservative-bit source on holds and the posterior/hard-decision tracker;
    `max_iter` is the decoding-cycle budget.
    """
    def __init__(self,
                 decoder_cfg,
                 **kwargs) -> None:
        """
        Initialization for the bit-stream SC bp decoder.
        Input:
            decoder_cfg: the information that come from config file (yaml)

        Parameters:
            max_iter: the number of maximum decoding cycles (DCs) of the SC decoder
            i: the number of DCs running the decoder

            H_matrix: loaded ldpc matrix, either hx or hz, as 2d tensor

            V_c_row: the row index of all the variable nodes for each check node
            V_c_col: the column index of all the variable nodes for each check node

            counter_width: bit width of the per-VN up/down saturating counter (default 8)
            random_machine: Bernoulli bit source, 'system' or 'sobol' (uniform < p), or a
                hardware TRNG model 'smtj', 'ro', 'latch', or 'memristor' (see `_bernoulli`)
            smtj_rho: 'smtj' lag-1 correlation exp(-T/tau_c) (default 0.5)
            ro_q: 'ro' per-cycle phase-jitter variance Q (default 0.012)
            ro_nu: 'ro' per-cycle mean phase advance nu (default 0.0)
            ro_bits: 'ro' comparator width k (default 8)
            latch_offset_sigma: 'latch' std of the static per-site offset d (default 0.1)
            memristor_tau_sigma: 'memristor' log-std of the static per-site tau (default 0.0)
            sobol_dim: 'sobol' Sobol dimension used, 1-indexed, 1 to 21201 (default 1, see `_bernoulli`)
        """

        super(create, self).__init__()

        logger.info(f'Creating bp_sum_prod_sc (bit-stream stochastic computing) decoder.')

        # set up default device
        self.device, _ = parse_device_dtype(decoder_cfg)

        # numeric config value: bool, non-numeric, or out-of-range input warns and uses the default
        def cfg_num(key, default, valid, kind=(int, float)):
            x = decoder_cfg.get(key, default)
            if isinstance(x, bool) or not isinstance(x, kind) or not valid(x):
                logger.warning(f'Invalid input {key} <{x}>, default to <{default}>.')
                x = default
            return x

        # set up default max_iter (number of decoding cycles)
        self.max_iter = cfg_num('max_iter', 50, lambda x: x > 0, int)

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

        # per-VN up/down saturating counter width (replaces the 64-bit edge memory)
        self.counter_width = cfg_num('counter_width', 8, lambda x: x > 0, int)
        self.max_count = float(2 ** self.counter_width - 1)

        self.random_machine = str(decoder_cfg.get('random_machine', 'system'))
        if self.random_machine.lower() not in {'sobol', 'system', 'smtj', 'ro', 'latch', 'memristor'}:
            logger.warning(f'Invalid input machine type <{self.random_machine}>, default to <system>.')
            self.random_machine = 'system'

        # hardware TRNG model parameters
        self.smtj_rho = cfg_num('smtj_rho', 0.5, lambda x: 0.0 <= x < 1.0)
        self.ro_q = cfg_num('ro_q', 0.012, lambda x: x >= 0.0)
        self.ro_nu = cfg_num('ro_nu', 0.0, lambda x: True)
        self.ro_bits = cfg_num('ro_bits', 8, lambda x: 0 < x <= 24, int)
        self.latch_offset_sigma = cfg_num('latch_offset_sigma', 0.1, lambda x: x >= 0.0)
        self.memristor_tau_sigma = cfg_num('memristor_tau_sigma', 0.0, lambda x: x >= 0.0)
        self.sobol_dim = cfg_num('sobol_dim', 1, lambda x: 1 <= x <= torch.quasirandom.SobolEngine.MAXDIM, int)
        self._rng_state = {}
        if self.random_machine.lower() == 'sobol':
            # one unscrambled Sobol value per decoding cycle, 2**ceil(log2(max_iter)) values; a plain
            # float64 attribute (not a buffer) so a module-level .to(dtype) never casts it
            L = 1 << (self.max_iter - 1).bit_length()
            seq = torch.quasirandom.SobolEngine(dimension=self.sobol_dim, scramble=False).draw(L, dtype=torch.float64)
            self.sobol_seq = seq[:, self.sobol_dim - 1].to(self.device)

        bundle = kwargs.get('bundle')
        if bundle is None:
            raise ValueError('bp_sum_prod_sc requires a pre-loaded MatrixBundle via the `bundle` kwarg.')
        self.Hx_matrix = bundle.Hx_matrix
        self.Hz_matrix = bundle.Hz_matrix
        self.lx_matrix = bundle.lx_matrix
        self.lz_matrix = bundle.lz_matrix
        self.H_shape, self.V_c_row, self.V_c_col, self.H_matrix = bundle.select(self.check_type)

        self.mask_dummy = (self.V_c_col == self.H_shape[1])

        # static per-site hardware variation, drawn once per decoder and shared across the batch:
        # channel site [1, N+1], edge site [1, M, degree]
        site_shapes = {'channel': [1, self.H_shape[1] + 1], 'edge': [1, self.H_shape[0], self.V_c_col.shape[1]]}
        self._static = {}
        if self.random_machine.lower() == 'latch':
            self._static = {k: torch.randn(v, dtype=torch.float32, device=self.device) * self.latch_offset_sigma for k, v in site_shapes.items()}
        elif self.random_machine.lower() == 'memristor':
            # tau_nom / tau_site = exp(-sigma g)
            self._static = {k: torch.exp(-self.memristor_tau_sigma * torch.randn(v, dtype=torch.float32, device=self.device)) for k, v in site_shapes.items()}

        self.eps = 1e-12

        # per-VN degree (number of incident checks); dummy variable gets degree 0
        vn_degree = self.H_matrix.to(self.dtype).sum(dim=0)
        vn_degree = torch.cat([vn_degree, torch.zeros(1, dtype=self.dtype, device=vn_degree.device)])

        # set iteration
        self.i = 0

        # convert to as the parameters in a model
        self.V_c_row = torch.nn.Parameter(self.V_c_row, requires_grad=False)
        self.V_c_col = torch.nn.Parameter(self.V_c_col, requires_grad=False)
        self.vn_degree = torch.nn.Parameter(vn_degree.to(self.device), requires_grad=False)

        self.algo = 'bp_sum_prod_sc'
        self.num_max_iter = self.max_iter

        logger.info(f'Complete.')


    def _bernoulli(self, p, site):
        """Draw one bit per element of the target probability p (bool tensor of p.shape).

        site ('channel' or 'edge') keys the per-stream state, so channel and edge streams never
        share state. State lives on self.device in float32 (probability math is done in float32
        for every dtype). Dynamic state (smtj x, ro phases) is reset at the start of each
        forward(); static per-site variation (latch d, memristor tau) is drawn once per decoder.

        system: bit = 1[u < p], u uniform [0, 1) from torch.rand, compared with p in float32.

        sobol: all streams share one uniform per cycle, an extreme case of the RNG sharing in SC
            decoder hardware (Wu et al., TCAS-II 2016, Fig. 3, 64 LFSRs for 2048 variable nodes):
            at decoding cycle i all streams of both sites share u = sobol_seq[i - 1] and
            bit = 1[u < p] in float64. sobol_seq is dimension `sobol_dim` (1-indexed) of an
            unscrambled, unseeded Sobol sequence, as in napl (github.com/UnaryLab/napl, commit
            9a4a2ef, src/napl/module/encoder.py, the Sobol draw in gen_num_seq, lines 71-72), so
            every decode replays it. Its first value is 0.0, so at
            cycle 1 every p > 0 gives bit 1.

        smtj: superparamagnetic-MTJ p-bit as a sampled two-state Markov chain. With state x
            (x ~ Bernoulli(p) on first use), x' = 1[u < p + (x - p) * rho], rho = exp(-T/tau_c)
            (`smtj_rho`). Stationary mean p and lag correlation rho: the exp(-(r01 + r10) T) form of
            Vodenicarevic et al. 2017 (Phys. Rev. Applied 8, 054045, Sec. III) and the two-state
            model of Daniels et al. 2020 (Phys. Rev. Applied 13, 034016); tunable p-bit definition
            of Camsari et al. 2017 (Phys. Rev. X 7, 031014, Eq. 1). p changes every cycle; the
            model assumes the device follows p within one cycle.

        ro: ring-oscillator jitter TRNG with a k-bit comparator (k = `ro_bits`). Each of k phases
            follows the Wiener-process model phi <- phi + nu + sqrt(Q) z, z ~ N(0, 1), with fair
            bit 1[frac(phi) >= 1/2] (Baudet et al. 2011, J. Cryptology 24:398-425, Eq. 1,
            Q = sigma^2 dt (`ro_q`, default 0.012 as measured on Stratix II, Table 1),
            nu = mu dt (`ro_nu`)). The k fair bits form R in [0, 2^k) and bit = 1[R < floor(p 2^k)]
            (weighted-binary SNG, p = B/2^k, of Luo et al. 2025, Supercond. Sci. Technol.,
            "True stochastic number generator using AQFP logic").

        latch: metastable latch / gray-zone comparator with a static per-site offset d:
            bit = 1[z < Phi^-1(p) + d], z ~ N(0, 1), so P(1) = Phi(Phi^-1(p) + d). d ~ N(0, sigma)
            (`latch_offset_sigma`) is drawn once per decoder per hardware site and shared across the batch.
            Gray-zone model of Ben Romdhane 2014 PhD thesis (Telecom ParisTech), Eq. 3.15,
            p_Q = 1/2 [1 - erf((dt - Tsetup0) / (sigma sqrt 2))], with die-to-die spread
            46.77-55.72% ones at the best setting (Table 4.7); same erf-shaped gray zone as the
            Josephson comparator TRNG of Sugiura et al. 2011 (IEEE TASC 21(3), Fig. 2).

        memristor: pulse-programmed Poisson switching, P(switch in pulse t) = 1 - exp(-t/tau)
            (Gaba et al. 2013, Nanoscale 5, 5872, Eq. 2). The target p sets the pulse via
            t/tau_nom = -ln(1 - p); each hardware site has a static tau_site = tau_nom exp(sigma g),
            g ~ N(0, 1) drawn once per decoder and shared across the batch, so
            bit = 1[u < 1 - (1 - p)^(tau_nom/tau_site)]. The paper gives no device spread, so
            sigma (`memristor_tau_sigma`) has no published value; sigma = 0 equals 'system'.
            p = 0 and p = 1 give exactly all-0 and all-1 bits. The modeled non-volatile
            Ag/a-Si device of Gaba et al. 2013 is reset to OFF after every trial (Fig. 2c/2e,
            Methods). The volatile diffusive-memristor TRNG of Jiang et al. 2017 (Nat. Commun. 8,
            882) needs no reset but reports 6 kb/s (Methods: 300 us pulses at 1 kHz, 6 low-order
            bits) and endurance about 10^7 cycles with stuck-ON failure (Discussion).
        """
        machine = self.random_machine.lower()
        if machine == 'system':
            return torch.rand(p.shape, dtype=torch.float32, device=self.device) < p.to(self.device, torch.float32)
        if machine == 'sobol':
            if self.i < 1:
                raise RuntimeError('_bernoulli with sobol was called outside a decoding cycle: self.i must be the 1-indexed cycle')
            return self.sobol_seq[self.i - 1] < p.to(self.device, torch.float64)

        pf = p.to(self.device, torch.float32)
        state = self._rng_state.get(site)

        if machine == 'smtj':
            u = torch.rand(pf.shape, device=self.device)
            thr = pf if state is None else pf + (state - pf) * self.smtj_rho
            bit = u < thr
            self._rng_state[site] = bit.float()
            return bit

        if machine == 'ro':
            if state is None:
                state = torch.rand([self.ro_bits, *pf.shape], device=self.device)
            state = torch.remainder(state + self.ro_nu + math.sqrt(self.ro_q) * torch.randn_like(state), 1.0)
            self._rng_state[site] = state
            weights = 2.0 ** torch.arange(self.ro_bits, device=self.device, dtype=torch.float32)
            R = ((state >= 0.5).float() * weights.view(-1, *[1] * pf.dim())).sum(dim=0)
            return R < torch.floor(pf * 2 ** self.ro_bits)

        if machine == 'memristor':
            return torch.rand(pf.shape, device=self.device) < 1.0 - torch.pow(1.0 - pf, self._static[site])

        # latch
        return torch.randn(pf.shape, device=self.device) < torch.special.ndtri(pf) + self._static[site]


    def forward(self, io_dict):
        """Bit-stream stochastic-computing decoding over decoding cycles (DCs).
        Input:
            syndrome: target syndrome for each check node

        Output:
            e_v: estimated error for each variable node

        Parameters:
            p_init: channel prior probability P(error bit = 1) per variable node
            cnt: per-VN up/down saturating counter (LUT-initialized from p_init)
            v2c_bit / c2v_bit: one Bernoulli message bit per edge per DC
            s_est: estimated syndrome for the current hard decision
        """
        logger.info(f'Initializing bp_sum_prod_sc (bit-stream stochastic computing) decoding.')

        syndrome = io_dict['synd'].to(dtype=self.dtype).to(self.device)

        self.batch_size, _ = syndrome.size()

        torch.set_default_dtype(self.dtype)

        self._rng_state = {}

        N_extended = self.H_shape[1] + 1
        degree = self.V_c_col.shape[1]

        # channel prior P(error = 1) = sigmoid(-LLR); dummy variable has prior 0 (never an error)
        p_init = torch.sigmoid(-io_dict['llr0'].to(self.device).to(self.dtype)).clamp(self.eps, 1.0 - self.eps)
        dummy_column = torch.zeros([self.batch_size, 1], dtype=self.dtype, device=self.device)
        p_init = torch.cat((p_init, dummy_column), dim=1)

        # LUT initialization of the per-VN counter
        cnt = torch.round(p_init * self.max_count)

        # syndrome bit on each edge (folded into the XOR check node)
        s_edge = syndrome[:, self.V_c_row]

        # constant per-edge "number of other inputs" for the leave-one-out equality
        vn_deg_edge = self.vn_degree[self.V_c_col]
        deg_full = self.vn_degree + 1.0

        # message bits, hard-decision / output buffers
        c2v_bit = torch.zeros([self.batch_size, self.H_shape[0], degree], dtype=self.dtype, device=self.device)
        e_v = torch.zeros([self.batch_size, N_extended], dtype=self.dtype, device=self.device)
        e_out = torch.zeros([self.batch_size, N_extended], dtype=self.dtype, device=self.device)
        l_out = torch.zeros([self.batch_size, N_extended], dtype=self.dtype, device=self.device)
        num_iters = torch.full([self.batch_size], -1, device=self.device)
        converges = torch.full([self.batch_size], 0, device=self.device)

        ones_e = torch.ones([self.batch_size, self.H_shape[0], degree], dtype=self.dtype, device=self.device)
        zeros_e = torch.zeros_like(ones_e)

        logger.info(f'Complete.')

        logger.info(f'Starting decoding cycles.')

        self.i = 0
        while self.i < self.max_iter:
            self.i += 1

            # channel bit stream for this DC (dummy variable prior is 0 -> always bit 0)
            b_ch = self._bernoulli(p_init, 'channel').to(self.dtype)

            if self.i == 1:
                # no check messages yet: variable node just forwards the channel bit
                v2c_bit = b_ch[:, self.V_c_col]
                v2c_bit[:, self.mask_dummy] = 0.0
            else:
                # n1 = channel bit + sum of incident check-to-variable bits, per variable node
                n1 = b_ch.clone()
                data_flat = c2v_bit.flatten(start_dim=1)
                partitions_flat = self.V_c_col.flatten().repeat(self.batch_size, 1)
                n1.scatter_add_(1, partitions_flat, data_flat)

                # variable-node equality (leave-one-out over the OTHER inputs of each edge)
                n1_edge = n1[:, self.V_c_col]
                others_ones = n1_edge - c2v_bit
                all_others_one = (others_ones == vn_deg_edge)
                all_others_zero = (others_ones == 0.0)
                cnt_edge = cnt[:, self.V_c_col]
                conservative = self._bernoulli(cnt_edge / self.max_count, 'edge').to(self.dtype)
                v2c_bit = torch.where(all_others_one, ones_e, torch.where(all_others_zero, zeros_e, conservative))
                v2c_bit[:, self.mask_dummy] = 0.0

                # update the per-VN counter from the full-input equality (once per DC)
                all_ones_vn = (n1 == deg_full)
                all_zeros_vn = (n1 == 0.0)
                push = torch.where(all_ones_vn, torch.ones_like(cnt),
                                   torch.where(all_zeros_vn, -torch.ones_like(cnt), torch.zeros_like(cnt)))
                cnt = (cnt + push).clamp(0.0, self.max_count)
                cnt[:, -1] = 0.0

            # check-node XOR with syndrome folding: c2v = (XOR of other v2c bits) XOR s_j
            c2v_bit = self.cn_update(v2c_bit, s_edge)

            # hard decision from the counter midpoint, expressed as an LLR for the contract
            p_post = (cnt / self.max_count).clamp(self.eps, 1.0 - self.eps)
            l_v = torch.log((1.0 - p_post) / p_post)
            l_v[:, -1] = float('inf')
            e_v = torch.where(l_v <= 0.0, 1.0, 0.0).to(self.dtype)

            s_est = self.syndrome_estimation(e_v)

            # latch the first DC at which a sample's hard decision satisfies the syndrome
            indices = torch.all(s_est == syndrome, 1).nonzero()
            checker = torch.where(num_iters == -1.0)[0]
            indices = indices[torch.isin(indices, checker)]
            if indices.size()[0] > 0:
                num_iters[indices] = self.i
                e_out[indices] = e_v[indices]
                l_out[indices] = l_v[indices]
                converges[indices] = 1

            if checker.size()[0] == 0:
                e_out = e_out[:, :-1]
                l_out = l_out[:, :-1]
                logger.info(f'Complete.')
                logger.info(f'Decoding cycles: <{(self.i)}>.')
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
        logger.info(f'Decoding cycles: <{(self.i)}>.')
        io_dict.update({
            'e_v': e_out,
            'iter': num_iters,
            'llr': l_out,
            'converge': converges
        })
        return io_dict


    def cn_update(self, v2c_bit, s_edge):
        # XOR gate: bit to variable i = (parity of the other incident v2c bits) XOR s_j
        tot_par = v2c_bit.sum(dim=2, keepdim=True)
        others_par = tot_par - v2c_bit
        c2v_bit = torch.remainder(others_par + s_edge, 2.0)
        c2v_bit[:, self.mask_dummy] = 0.0
        return c2v_bit


    def syndrome_estimation(self, e_v):
        # calculate the syndrome by summing the number of 1s in each column in e
        temp_e = e_v
        temp_e[:, -1] = 0.0
        estimated_syndrome = temp_e[:, self.V_c_col].sum(dim = 2).to(dtype = self.dtype)

        return torch.where((estimated_syndrome%2) > 0.0, 1.0, 0.0)
