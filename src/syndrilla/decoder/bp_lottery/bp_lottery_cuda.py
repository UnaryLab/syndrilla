import torch
from loguru import logger

from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum_cuda import create as _BaseCuda
from syndrilla.decoder.bp_lottery.bp_lottery import (
    cn_row_mask,
    flip_rows,
    is_flip_iter,
    read_flip_interval,
    vn_unsat_count,
)


class create(_BaseCuda):
    """BP Normalized Min-Sum lottery decoder on CUDA kernels: the
    bp_norm_min_sum_cuda per-step loop with the sign-flip in _iter_hook.

    Accepts every bp_norm_min_sum_cuda key plus the lottery knobs:
        random_machine : 'sobol' (default) | 'system'   RNG for the flip pick
        flip_start_iter: int (default 4)                 flips start after this iteration
        flip_interval  : int >= 1 (default 1)            iterations between flips
    """

    # forward() makes the PyTorch path's last system draw from the global RNG
    _global_last_draw = True

    def __init__(self, decoding_cfg: dict, **kwargs) -> None:
        super().__init__(decoding_cfg, **kwargs)

        # lottery knobs
        self.random_machine = str(decoding_cfg.get("random_machine", "sobol")).lower()
        if self.random_machine not in {"sobol", "system"}:
            logger.warning(
                f"Invalid random_machine <{self.random_machine}>; defaulting to sobol."
            )
            self.random_machine = "sobol"
        self.flip_start_iter = int(decoding_cfg.get("flip_start_iter", 4))
        self.flip_interval = read_flip_interval(decoding_cfg)

        self.algo = "bp_lottery"
        logger.info("bp_lottery_cuda decoder ready (per-step path + sign-flip).")

    def _iter_hook(self, i, l_v, e_v, active, syndrome) -> None:
        """Lottery sign-flip at the end of each flip iteration i (is_flip_iter),
        on the rows still unconverged; the next iteration's check update reads
        the flipped l_v. With random_machine system, a call with no unconverged
        row returns before drawing (one host sync per call); forward() makes the
        one draw the PyTorch path makes when the last rows converge at a flip
        iteration."""
        if not is_flip_iter(self, i):
            return
        if self.random_machine == "system" and not active.any():
            return
        self.i = i
        s_est = (e_v[:, self.V_c_col].sum(dim=2, dtype=torch.uint8) & 1).to(self.dtype)
        self._active = active
        self.sign_flip_cn_rand_new(syndrome, s_est, l_v)

    def forward(self, io_dict: dict) -> dict:
        """bp_norm_min_sum_cuda per-step decode with the lottery sign-flip in
        _iter_hook. Builds the Sobol sequence (one value per iteration) first."""
        if self.random_machine == "sobol":
            sobol = torch.quasirandom.SobolEngine(dimension=1, scramble=False)
            draw_dtype = (
                self.dtype
                if self.dtype in {torch.float32, torch.float64}
                else torch.float32
            )
            self.r = sobol.draw(self.max_iter, dtype=draw_dtype).to(
                self.device, self.dtype
            )
        # read before decoding: cap.observe() may choose the cap during this batch
        capped = bool(
            self.cap is not None
            and self.cap.done
            and self.cap.frac is not None
            and not self.cap_bypass
        )
        out = super().forward(io_dict)
        # system: the PyTorch path also draws B values at the iteration where the
        # last rows converge (its loop breaks one iteration later) when that is a
        # flip iteration, unless the cap stopped the loop; draw them here so both
        # paths leave the global RNG equal.
        if (
            self.random_machine == "system"
            and self._global_last_draw
            and not capped
            and out["converge"].numel() > 0
            and bool(out["converge"].all())
            and is_flip_iter(self, int(out["iter"].max()))
        ):
            torch.rand(out["converge"].numel(), device=self.device, dtype=self.dtype)
        return out

    def _draw_r(self, n):
        """[n] uniform draws for the flip pick, one per row."""
        if self.random_machine == "system":
            return torch.rand(n, device=self.device, dtype=self.dtype)
        return self.r[(self.i - 1)].repeat(n)  # sobol

    def sign_flip_cn_rand_new(self, syndrome, s_est, l_v):
        """Flip one variable node's LLR sign per stuck sample.

        Pick a random unsatisfied check, then among its variable nodes pick the one
        with the highest unsatisfied-check connectivity (ties broken by smallest
        |LLR|), and flip its posterior LLR sign. Only rows in self._active flip.
        """
        synd_diff = (syndrome + s_est) % 2.0  # [B, M]
        unsat_cn_mask = synd_diff.bool()
        batch_size, M = unsat_cn_mask.shape

        total_unsat = unsat_cn_mask.sum(dim=1)
        valid_mask = total_unsat > 0

        r = self._draw_r(batch_size)

        total_unsat_safe = total_unsat + (total_unsat == 0).float()
        unsat_cumsum = unsat_cn_mask.cumsum(dim=1)

        # check selection: random among unsatisfied
        rand_pos = torch.floor(r * total_unsat_safe).long() + 1
        chosen_cn = ((unsat_cumsum >= rand_pos.unsqueeze(1)) & unsat_cn_mask).float()
        chosen_cn_idx = torch.argmax(chosen_cn, dim=1)  # [B]

        # variable selection: 1) max unsatisfied-CN connectivity 2) min |LLR|
        candidate_vn_mask = cn_row_mask(self.V_c_col, chosen_cn_idx, self.N)
        vn_unsat_counts = vn_unsat_count(
            unsat_cn_mask.to(self.dtype), self.V_c_col, self.N
        )

        llr = torch.abs(l_v[:, :-1])  # [B, N]
        score = vn_unsat_counts * 1e6 - llr
        masked_score = score + (~candidate_vn_mask).float() * -1e9
        selected_vn = torch.argmax(masked_score, dim=1)

        flip_rows(l_v, selected_vn, valid_mask & self._active)
        return l_v
