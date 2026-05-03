import torch
from loguru import logger


class create():

    def __init__(self, vote_cfg, **kwargs) -> None:
        self._best_round_idx = None
        self.last_match_counts = None
        self.last_sample_count = 0
        self.last_rounds = 0
        self.last_voted_stage = None

    def apply(self, tensor, number_channel=1, rounds=1, vote_stage='syndrome', current_stage='syndrome'):
        """
        Apply majority vote to collapse the rounds dimension (always dim=1).

        Returns:
            tensor with the rounds dimension collapsed, or unchanged
        """
        if rounds <= 1 or vote_stage != current_stage:
            self.last_match_counts = None
            self.last_sample_count = 0
            self.last_rounds = 0
            self.last_voted_stage = None
            return tensor

        rounds_dim = 1
        d = tensor.size(rounds_dim)
        voted = (tensor.sum(dim=rounds_dim) > d / 2).to(tensor.dtype)

        # Cumulative-prefix convergence: for each k in 1..d, compare the majority
        # vote over rounds 1..k against the full vote over rounds 1..d.
        # thresholds[k-1] = k/2 — strict majority, matching the full-vote rule.
        thresholds = torch.arange(1, d + 1, device=tensor.device, dtype=torch.float32) / 2.0

        if tensor.ndim >= 3:
            flat = tensor.flatten(start_dim=2)                   # [B, d, F]
            voted_flat = voted.flatten(start_dim=1).unsqueeze(1) # [B, 1, F]
            cumsum = torch.cumsum(flat.float(), dim=1)           # [B, d, F]
            partial_votes = (cumsum > thresholds.view(1, d, 1)).to(tensor.dtype)  # [B, d, F]
            match_per_k = (partial_votes == voted_flat).all(dim=-1)               # [B, d]

            eq = (flat == voted_flat)
            hamming = (~eq).sum(dim=-1)
            self._best_round_idx = hamming.argmin(dim=1)
        else:
            cumsum = torch.cumsum(tensor.float(), dim=1)               # [B, d]
            partial_votes = (cumsum > thresholds.view(1, d)).to(tensor.dtype)
            match_per_k = (partial_votes == voted.unsqueeze(1))         # [B, d]
            self._best_round_idx = None

        self.last_match_counts = match_per_k.sum(dim=0).int().tolist()
        self.last_sample_count = int(tensor.size(0))
        self.last_rounds = int(d)
        self.last_voted_stage = current_stage

        return voted

    def select_round(self, tensor, rounds=1, vote_stage='syndrome', current_stage='syndrome'):
        """
        Select the round that best matches the voted result.

        Use this for metadata (llr, converge, iter) after apply() has been
        called on the primary tensor (e_v).

        Args:
            tensor: [B, d, ...] tensor to select from

        Returns:
            [B, ...] tensor from the best-matching round per sample
        """
        if rounds <= 1 or vote_stage != current_stage:
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
