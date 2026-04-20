import torch
from loguru import logger


class create():

    def __init__(self, vote_cfg, **kwargs) -> None:
        self._best_round_idx = None

    def apply(self, tensor, number_channel=1, d_rounds=1, vote_stage='syndrome', current_stage='syndrome'):
        """
        Apply majority vote to collapse the rounds dimension (always dim=1).

        Returns:
            tensor with the rounds dimension collapsed, or unchanged
        """
        if d_rounds <= 1 or vote_stage != current_stage:
            return tensor

        rounds_dim = 1
        voted = (tensor.sum(dim=rounds_dim) > tensor.size(rounds_dim) / 2).to(tensor.dtype)

        if tensor.ndim >= 3:
            flat = tensor.flatten(start_dim=2)
            voted_flat = voted.flatten(start_dim=1)
            hamming = (flat != voted_flat.unsqueeze(1)).sum(dim=-1)
            self._best_round_idx = hamming.argmin(dim=1)
        else:
            self._best_round_idx = None

        return voted

    def select_round(self, tensor, d_rounds=1, vote_stage='syndrome', current_stage='syndrome'):
        """
        Select the round that best matches the voted result.

        Use this for metadata (llr, converge, iter) after apply() has been
        called on the primary tensor (e_v).

        Args:
            tensor: [B, d, ...] tensor to select from

        Returns:
            [B, ...] tensor from the best-matching round per sample
        """
        if d_rounds <= 1 or vote_stage != current_stage:
            return tensor
        if self._best_round_idx is None:
            return tensor[:, 0]

        idx = self._best_round_idx
        B = tensor.shape[0]
        if tensor.ndim == 2:
            return tensor[torch.arange(B, device=tensor.device), idx]
        else:
            idx_shape = [B] + [1] * (tensor.ndim - 2)
            expand_shape = [B] + list(tensor.shape[2:])
            return tensor.gather(1, idx.view(*idx_shape).expand(*expand_shape).unsqueeze(1)).squeeze(1)
