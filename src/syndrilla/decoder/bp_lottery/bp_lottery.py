import torch
from loguru import logger

from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum import create as _NmsPy


def vn_unsat_count(cn_val, V_c_col, N):
    """cn_val [B, M] @ H as [B, N]: per variable, the sum of cn_val over its checks.

    Scatters each edge's check value onto its variable over V_c_col; padded edges
    land on the dummy column N, which is dropped. Output dtype follows cn_val.
    """
    B, D = cn_val.shape[0], V_c_col.shape[1]
    src = cn_val.unsqueeze(2).expand(-1, -1, D).reshape(B, -1)
    return cn_val.new_zeros(B, N + 1).index_add_(1, V_c_col.reshape(-1), src)[:, :N]


def cn_row_mask(V_c_col, cn_idx, N):
    """Bool [B, N] mask of the variables on check cn_idx[b], i.e. H[cn_idx].bool()."""
    mask = torch.zeros(cn_idx.shape[0], N + 1, dtype=torch.bool, device=V_c_col.device)
    return mask.scatter_(1, V_c_col[cn_idx], True)[:, :N]


def flip_rows(l_v, idx, mask):
    """Negate l_v[b, idx[b]] in place for every row b where mask[b] is True,
    without a host sync."""
    sel = idx.unsqueeze(1)
    v = l_v.gather(1, sel)
    l_v.scatter_(1, sel, torch.where(mask.unsqueeze(1), -v, v))


def rand_rows(dec, n):
    """n uniform draws in dec.dtype from the global RNG, one per hook row: drawn
    for the whole batch of dec._B samples, then indexed by dec._hook_rows when
    the n rows are a compacted subset, so a sample's draw does not depend on
    which samples are still decoded."""
    r = torch.rand(dec._B, device=dec.device, dtype=dec.dtype)
    return r if n == dec._B else r[dec._hook_rows]


class create(_NmsPy):
    """
    Lottery BP decoder: bp_norm_min_sum (eager or compiled step, row compaction)
    with the lottery sign-flip in _iter_hook from iteration flip_start_iter + 1 on.

    Accepts every bp_norm_min_sum key plus
        random_machine: 'sobol' (default) | 'system'   RNG for the flip pick
        flip_start_iter: int (default 4)                 flips start after this iteration
    """

    def __init__(self, decoding_cfg, **kwargs) -> None:
        super().__init__(decoding_cfg, **kwargs)

        self.random_machine = decoding_cfg.get("random_machine", "sobol")
        if self.random_machine.lower() not in {"sobol", "system"}:
            logger.warning(
                f"Invalid input machine type <{self.random_machine}>, default to <sobol>."
            )
            self.random_machine = "sobol"
        self.flip_start_iter = int(decoding_cfg.get("flip_start_iter", 4))

        self.algo = "bp_lottery"

    def forward(self, io_dict):
        """bp_norm_min_sum decoding with the lottery sign-flip in _iter_hook.
        Builds the Sobol sequence (one value per iteration) first."""
        self._B = io_dict["synd"].shape[0]
        if self.random_machine.lower() == "sobol":
            sobol = torch.quasirandom.SobolEngine(dimension=1, scramble=False)
            self.r = (
                sobol.draw(self.max_iter, dtype=torch.float32)
                .to(self.device)
                .to(self.dtype)
            )
        return super().forward(io_dict)

    def _iter_hook(self, i, l_v, e_v, active, syndrome) -> None:
        """Lottery sign-flip at the end of iteration i > flip_start_iter on the
        unconverged rows; the next iteration reads the flipped l_v."""
        if i <= self.flip_start_iter:
            return
        self._active = active
        self.sign_flip_cn_rand_new(syndrome, self.syndrome_estimation(e_v), l_v)

    def _draw_r(self, n):
        """[n] uniform draws for the flip pick, one per hook row."""
        if self.random_machine.lower() == "system":
            return rand_rows(self, n)
        return self.r[(self.i - 1)].repeat(n)  # sobol

    def sign_flip_cn_rand_new(self, syndrome, s_est, l_v):
        # Syndrome Residual
        synd_diff = (syndrome + s_est) % 2.0  # [B, M]
        unsat_cn_mask = synd_diff.bool()

        batch_size, M = unsat_cn_mask.shape

        total_unsat = unsat_cn_mask.sum(dim=1)
        valid_mask = total_unsat > 0

        # random selection setup
        r = self._draw_r(batch_size)

        total_unsat_safe = total_unsat + (total_unsat == 0).float()

        unsat_cumsum = unsat_cn_mask.cumsum(dim=1)

        # CN Selection: Random
        rand_pos = torch.floor(r * total_unsat_safe).long() + 1
        chosen_cn = (
            (unsat_cumsum >= rand_pos.unsqueeze(1)) & unsat_cn_mask
        ).float()  # [B, M]
        chosen_cn_idx = torch.argmax(chosen_cn, dim=1)  # [B]

        # VN Selection: 1. Unsat CN count priority ->  2. Min LLR priority
        N = self.H_shape[1]
        candidate_vn_mask = cn_row_mask(self.V_c_col, chosen_cn_idx, N)  # [B, N]

        vn_unsat_counts = vn_unsat_count(unsat_cn_mask.float(), self.V_c_col, N)

        llr = torch.abs(l_v[:, :-1])  # [B, N]
        score = vn_unsat_counts * 1e6 - llr

        masked_score = score + (~candidate_vn_mask).float() * -1e9
        selected_vn = torch.argmax(masked_score, dim=1)

        flip_rows(l_v, selected_vn, valid_mask & self._active)

        return l_v
