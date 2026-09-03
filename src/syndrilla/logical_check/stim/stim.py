import torch
from loguru import logger


class create():

    def __init__(self, check_cfg, **kwargs) -> None:
        pass

    def check(self, e_v_total, observable_flips, l_matrix, converge=None):
        """
        Args:
            e_v_total:        [B, num_errors]        decoded error-mechanism vector
            observable_flips: [B, num_obs] or [B, d, num_obs]  true observable outcomes
            l_matrix:         [num_obs, num_errors]   observable matrix (L)
            converge:         [B] or None             convergence flags

        Returns:
            logical_check:    [B]  1 = logical error, 0 = success

        A multi-round `observable_flips` is judged against its **final** round, which is
        the outcome stim's own decoders are compared against. Earlier rounds are the
        state on the way there, not competing verdicts to be reconciled.
        """
        logger.info('Measuring stim logical check rate.')

        device = e_v_total.device
        e_v = e_v_total.to(device).to(torch.float32)
        L = torch.tensor(l_matrix, device=device, dtype=torch.float32)

        predicted_obs = (e_v @ L.T) % 2

        if observable_flips.ndim > 2:
            observable_flips = observable_flips[:, -1]
        obs_flips = observable_flips.to(device).to(torch.float32)
        logical_check = self._check_single(predicted_obs, obs_flips, converge)

        logger.info('Stim logical check rate measurement complete.')
        return logical_check

    def _check_single(self, predicted_obs, obs_flips, converge):
        logical_check = torch.where(
            torch.any(predicted_obs != obs_flips, dim=1), 1, 0
        )
        if converge is not None:
            unconverged = torch.where(converge == 0)[0]
            logical_check[unconverged] = 1
        return logical_check
