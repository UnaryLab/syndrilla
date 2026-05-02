import torch
from loguru import logger

from syndrilla.vote import create_vote


class create():

    def __init__(self, check_cfg, **kwargs) -> None:
        self.voter = create_vote(cfg={'method': 'majority_vote'})

    def check(self, e_v_total, observable_flips, l_matrix, converge=None):
        """
        Args:
            e_v_total:        [B, num_errors]        decoded error-mechanism vector
            observable_flips: [B, num_obs] or [B, d, num_obs]  true observable outcomes
            l_matrix:         [num_obs, num_errors]   observable matrix (L)
            converge:         [B] or None             convergence flags

        Returns:
            logical_check:    [B]  1 = logical error, 0 = success
        """
        logger.info(f'Measuring stim logical check rate.')

        device = e_v_total.device
        e_v = e_v_total.to(device).to(torch.float32)
        L = torch.tensor(l_matrix, device=device, dtype=torch.float32)

        predicted_obs = (e_v @ L.T) % 2

        rounds = observable_flips.shape[1] if observable_flips.ndim > 2 else 1

        if rounds > 1:
            rounds_dim = 1
            round_checks = []
            for r in range(rounds):
                obs_r = observable_flips.select(rounds_dim, r).to(device).to(torch.float32)
                round_checks.append(self._check_single(predicted_obs, obs_r, converge))
            logical_check = self.voter.apply(
                torch.stack(round_checks, dim=1), number_channel=1, rounds=rounds,
                vote_stage='check', current_stage='check'
            )
        else:
            obs_flips = observable_flips.to(device).to(torch.float32)
            logical_check = self._check_single(predicted_obs, obs_flips, converge)

        logger.info(f'Stim logical check rate measurement complete.')
        return logical_check

    def _check_single(self, predicted_obs, obs_flips, converge):
        logical_check = torch.where(
            torch.any(predicted_obs != obs_flips, dim=1), 1, 0
        )
        if converge is not None:
            unconverged = torch.where(converge == 0)[0]
            logical_check[unconverged] = 1
        return logical_check
