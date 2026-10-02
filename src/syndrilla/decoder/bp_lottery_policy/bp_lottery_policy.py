import torch
from loguru import logger

from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum import create as _NmsPy
from syndrilla.decoder.bp_lottery.bp_lottery import (
    cn_row_mask,
    flip_rows,
    rand_rows,
    vn_unsat_count,
)


class create(_NmsPy):
    """
    Lottery BP decoder with a selectable sign-flip policy: bp_norm_min_sum (eager
    or compiled step, row compaction) with the policy's sign-flip in _iter_hook
    every iteration.

    Accepts every bp_norm_min_sum key plus
        random_machine  : 'sobol' (default) | 'system'
        sign_flip_policy: one of _sign_flip_policies (default Proposed)
    """

    _sign_flip_policies = {
        'Proposed',
        'global_optimal',
        'global_connectivity',
        'global_weighted_random',
        'local_random',
        'local_reliable',
        'local_connectivity',
    }

    def __init__(self,
                 decoding_cfg,
                 **kwargs) -> None:
        super().__init__(decoding_cfg, **kwargs)

        self.random_machine = decoding_cfg.get('random_machine', 'sobol')
        if self.random_machine.lower() not in {'sobol', 'system'}:
            logger.warning(f'Invalid input machine type <{self.random_machine}>, default to <sobol>.')
            self.random_machine = 'sobol'

        self.sign_flip_policy = decoding_cfg.get('sign_flip_policy', 'Proposed')
        if self.sign_flip_policy not in self._sign_flip_policies:
            logger.warning(f'Invalid sign_flip_policy <{self.sign_flip_policy}>, defaulting to <Proposed>. Allowed: {sorted(self._sign_flip_policies)}.')
            self.sign_flip_policy = 'Proposed'

        self.algo = 'bp_lottery_policy'


    def forward(self, io_dict):
        """bp_norm_min_sum decoding with the policy sign-flip in _iter_hook."""
        self._prepare(io_dict)
        return super().forward(io_dict)


    def _prepare(self, io_dict):
        """Per-forward state of the flip: the batch size and the Sobol sequence
        (one value per iteration)."""
        self._B = io_dict['synd'].shape[0]
        if self.random_machine.lower() == 'sobol':
            sobol = torch.quasirandom.SobolEngine(dimension=1, scramble=False)
            self.r = sobol.draw(self.max_iter, dtype=torch.float32).to(self.device).to(self.dtype)


    def _iter_hook(self, i, l_v, e_v, active, syndrome) -> None:
        """Policy sign-flip at the end of iteration i on the unconverged rows; the
        next iteration reads the flipped l_v."""
        self.i = i
        self._active = active
        self._apply_policy(syndrome, self.syndrome_estimation(e_v), l_v)


    def _apply_policy(self, syndrome, s_est, l_v):
        p = self.sign_flip_policy
        if p == 'Proposed':
            return self.sign_flip_lottery(syndrome, s_est, l_v)
        if p == 'global_optimal':
            return self.sign_flip_global_optimal(syndrome, s_est, l_v)
        if p == 'global_connectivity':
            return self.sign_flip_global_connectivity(syndrome, s_est, l_v)
        if p == 'global_weighted_random':
            return self.sign_flip_global_weighted_random(syndrome, s_est, l_v)
        if p == 'local_random':
            return self.sign_flip_local_random(syndrome, s_est, l_v)
        if p == 'local_reliable':
            return self.sign_flip_local_reliable(syndrome, s_est, l_v)
        if p == 'local_connectivity':
            return self.sign_flip_local_connectivity(syndrome, s_est, l_v)
        return l_v


    def sign_flip_global_weighted_random(self, syndrome, s_est, l_v):
        # Outside the paper's five-policy. Globally random VN
        # selection weighted by per-VN unsatisfied-CN count: no CN is picked
        # first, and VNs touching more unsatisfied CNs are more likely to be
        # chosen. No |LLR| consideration.
        synd_diff = (syndrome + s_est)%2.0

        temp_ls = vn_unsat_count(synd_diff.float(), self.V_c_col, self.H_shape[1])

        total_ones = temp_ls.sum(dim=1).to(self.dtype)

        valid_mask = total_ones > 0

        if self.random_machine.lower() == 'system':
            r = rand_rows(self, self.batch_size)
        elif self.random_machine.lower() == 'sobol':
            r = self.r[(self.i-1)].repeat(self.batch_size)

        target = torch.zeros_like(total_ones).to(self.dtype)

        target[valid_mask] = torch.floor(r[valid_mask] * (total_ones[valid_mask] - 1))
        target = target.unsqueeze(1)
        cumsum_x = torch.cumsum(temp_ls, dim=1)
        mask = cumsum_x >= target + 1
        selected_indices = torch.argmax(mask.float(), dim=1)

        flip_rows(l_v, selected_indices, valid_mask & self._active)
        return l_v


    def sign_flip_global_optimal(self, syndrome, s_est, l_v):
        # Paper policy (1) "Global optimal": among all VNs with the most
        # unsatisfied CNs, flip the sign of the one with the minimum
        # absolute LLR. Builds an upper bound but is impractical for
        # hardware due to the global search.
        synd_diff = (syndrome + s_est)%2.0

        temp_ls = vn_unsat_count(synd_diff.float(), self.V_c_col, self.H_shape[1])

        total_ones = temp_ls.sum(dim=1).to(self.dtype)

        valid_mask = total_ones > 0

        max_vals, _ = temp_ls.max(dim=1, keepdim=True)

        mask_max = (temp_ls == max_vals)

        abs_llr = torch.abs(l_v[:, :-1])
        mask_max = (temp_ls == temp_ls.max(dim=1, keepdim=True).values)

        masked_abs_llr = abs_llr + (~mask_max).to(self.dtype) * 1e9

        selected_indices = torch.argmin(masked_abs_llr, dim=1)

        flip_rows(l_v, selected_indices, valid_mask & self._active)
        return l_v


    def sign_flip_global_connectivity(self, syndrome, s_est, l_v):
        # Paper policy (2) "Global connectivity only": among all VNs with the
        # most unsatisfied CNs, flip the sign of a random one. Ignores |LLR|;
        # demonstrates the importance of reliability guidance vs. global_optimal.
        synd_diff = (syndrome + s_est) % 2.0

        temp_ls = vn_unsat_count(synd_diff.float(), self.V_c_col, self.H_shape[1])

        total_ones = temp_ls.sum(dim=1).to(self.dtype)
        valid_mask = total_ones > 0

        max_vals, _ = temp_ls.max(dim=1, keepdim=True)

        candidates_mask = (temp_ls == max_vals) & (temp_ls > 0)

        if self.random_machine.lower() == 'system':
            r = rand_rows(self, self.batch_size)
        elif self.random_machine.lower() == 'sobol':
            r = self.r[(self.i-1)].repeat(self.batch_size)

        num_candidates = candidates_mask.sum(dim=1)

        safe_num_candidates = num_candidates.clone()
        safe_num_candidates[safe_num_candidates == 0] = 1

        target_k = torch.floor(r * safe_num_candidates).long()

        candidates_cumsum = candidates_mask.cumsum(dim=1)

        selected_mask = (candidates_cumsum == (target_k.unsqueeze(1) + 1)) & candidates_mask

        selected_indices = torch.argmax(selected_mask.float(), dim=1)

        flip_rows(l_v, selected_indices, valid_mask & self._active)

        return l_v


    def sign_flip_local_random(self, syndrome, s_est, l_v):
        # Paper policy (3) "Local random": for a randomly selected
        # unsatisfied CN, flip the sign of a random neighboring VN. Lowest
        # hardware complexity but low accuracy due to no reliability guidance.
        synd_diff = (syndrome + s_est) % 2.0
        unsat_cn_mask = synd_diff.bool()  # [B, M]

        batch_size, M = unsat_cn_mask.shape

        if self.random_machine.lower() == 'system':
            r1 = rand_rows(self, batch_size)
        elif self.random_machine.lower() == 'sobol':
            r1 = self.r[(self.i-1)].repeat(batch_size)

        unsat_cumsum = unsat_cn_mask.cumsum(dim=1)  # [B, M]
        total_unsat = unsat_cn_mask.sum(dim=1)      # [B]

        valid_mask = total_unsat > 0

        rand_pos_cn = torch.floor(r1 * total_unsat).long() + 1
        rand_pos_cn = torch.clamp(rand_pos_cn, max=total_unsat)

        chosen_cn_onehot = ((unsat_cumsum >= rand_pos_cn.unsqueeze(1)) & unsat_cn_mask).float()
        chosen_cn_idx = torch.argmax(chosen_cn_onehot, dim=1)  # [B]

        connected_vn_mask = cn_row_mask(self.V_c_col, chosen_cn_idx, self.H_shape[1])  # [B, N]

        num_connected_vns = connected_vn_mask.sum(dim=1) # [B]

        if self.random_machine.lower() == 'system':
            r2 = rand_rows(self, batch_size)
        elif self.random_machine.lower() == 'sobol':
            r2 = rand_rows(self, batch_size)


        rand_pos_vn = torch.floor(r2 * num_connected_vns).long() + 1

        vn_cumsum = connected_vn_mask.cumsum(dim=1) # [B, N]

        chosen_vn_onehot = (vn_cumsum == rand_pos_vn.unsqueeze(1)) & connected_vn_mask
        selected_vn_idx = torch.argmax(chosen_vn_onehot.float(), dim=1) # [B]


        flip_rows(l_v, selected_vn_idx, valid_mask & self._active)

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
            r = rand_rows(self, batch_size)
        elif self.random_machine.lower() == 'sobol':
            r = self.r[(self.i-1)].repeat(batch_size)

        unsat_cumsum = unsat_cn_mask.cumsum(dim=1)  # [B, M]
        total_unsat = unsat_cn_mask.sum(dim=1)      # [B]

        rand_pos = torch.floor(r * total_unsat).long() + 1
        rand_pos = torch.clamp(rand_pos, max=total_unsat)

        chosen_cn = ((unsat_cumsum >= rand_pos.unsqueeze(1)) & unsat_cn_mask).float()
        chosen_cn_idx = torch.argmax(chosen_cn, dim=1)  # [B]

        cn_mask = cn_row_mask(self.V_c_col, chosen_cn_idx, self.H_shape[1])  # [B, N]

        llr = l_v[:, :-1]  # [B, N]
        masked_llr = llr + (~cn_mask).float() * 1e9

        selected_vn = torch.argmin(masked_llr, dim=1)

        flip_rows(l_v, selected_vn, self._active)

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
            r = rand_rows(self, batch_size)
        else: # sobol
            r = self.r[(self.i-1)].repeat(batch_size)

        total_unsat = unsat_cn_mask.sum(dim=1)
        total_unsat = total_unsat + (total_unsat == 0).float()

        unsat_cumsum = unsat_cn_mask.cumsum(dim=1)
        rand_pos = torch.floor(r * total_unsat).long() + 1
        chosen_cn = ((unsat_cumsum >= rand_pos.unsqueeze(1)) & unsat_cn_mask).float() # [B, M]
        chosen_cn_idx = torch.argmax(chosen_cn, dim=1) # [B]

        N = self.H_shape[1]
        candidate_vn_mask = cn_row_mask(self.V_c_col, chosen_cn_idx, N) # [B, N]

        vn_unsat_counts = vn_unsat_count(unsat_cn_mask.float(), self.V_c_col, N)
        abs_llr = torch.abs(l_v[:, :-1]) # [B, N]
        score = vn_unsat_counts * 1e6 - abs_llr
        masked_score = score + (~candidate_vn_mask).float() * -1e9

        selected_vn = torch.argmax(masked_score, dim=1)

        flip_rows(l_v, selected_vn, self._active)

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
            r = rand_rows(self, batch_size)
        else: # sobol
            r = self.r[(self.i-1)].repeat(batch_size)

        total_unsat = unsat_cn_mask.sum(dim=1)
        total_unsat = total_unsat + (total_unsat == 0).float()

        unsat_cumsum = unsat_cn_mask.cumsum(dim=1)
        rand_pos = torch.floor(r * total_unsat).long() + 1

        chosen_cn = ((unsat_cumsum >= rand_pos.unsqueeze(1)) & unsat_cn_mask).float() # [B, M]
        chosen_cn_idx = torch.argmax(chosen_cn, dim=1) # [B]

        N = self.H_shape[1]
        candidate_vn_mask = cn_row_mask(self.V_c_col, chosen_cn_idx, N) # [B, N]

        vn_unsat_counts = vn_unsat_count(unsat_cn_mask.float(), self.V_c_col, N)

        score = vn_unsat_counts

        masked_score = score + (~candidate_vn_mask).float() * -1e9

        selected_vn = torch.argmax(masked_score, dim=1)

        flip_rows(l_v, selected_vn, self._active)

        return l_v
