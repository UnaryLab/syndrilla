import torch

from syndrilla.decoder.bp_lottery.pre import PrePhase
from loguru import logger

from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum_cuda import create as _BaseCuda
from syndrilla.decoder.bp_lottery.bp_lottery import (
    LOTTERY_CONFIG_KEYS,
    flip_with_mode,
    is_flip_iter,
    read_phase_knobs,
    read_vn_pick_tie_break,
    reset_phase_state,
    set_phase,
)


class create(PrePhase, _BaseCuda):
    """BP Normalized Min-Sum lottery decoder on CUDA kernels: the
    bp_norm_min_sum_cuda per-step loop with the sign-flip in _iter_hook.

    Accepts every bp_norm_min_sum_cuda key plus the lottery knobs:
        random_machine : 'sobol' (default) | 'system'   RNG for the flip pick
    """

    # forward() makes the PyTorch path's last system draw from the global RNG
    _global_last_draw = True
    _config_algorithm = "bp_lottery"
    _config_keys = ()
    _base_config_keys = LOTTERY_CONFIG_KEYS

    def __init__(self, decoding_cfg: dict, **kwargs) -> None:
        boundaries, phases = read_phase_knobs(
            {"algorithm": self._config_algorithm, **decoding_cfg}, self._config_keys)
        self.vn_pick_tie_break = read_vn_pick_tie_break(decoding_cfg)
        super().__init__(decoding_cfg, **kwargs)
        self.flip_phase_iters, self.phase_knobs, self._phase_index = boundaries, phases, -1
        set_phase(self, 1)
        self.mask_dummy = self.V_c_col == self.N

        # lottery knobs
        self.random_machine = str(decoding_cfg.get("random_machine", "sobol")).lower()
        if self.random_machine not in {"sobol", "system"}:
            logger.warning(
                f"Invalid random_machine <{self.random_machine}>; defaulting to sobol."
            )
            self.random_machine = "sobol"

        self._init_pre_phase(decoding_cfg)
        self.algo = "bp_lottery"
        logger.info("bp_lottery_cuda decoder ready (per-step path + sign-flip).")

    def _native_iter_hook(self, i, l_v, e_v, active, syndrome) -> None:
        """Lottery sign-flip at the end of each flip iteration i (is_flip_iter),
        on the rows still unconverged; the next iteration's check update reads
        the flipped l_v. With early termination and random_machine system,
        a call with no unconverged
        row returns before drawing (one host sync per call); forward() makes the
        one draw the PyTorch path makes when the last rows converge at a flip
        iteration."""
        self.i = i
        if not is_flip_iter(self, i):
            return
        if self.early_termination and self.random_machine == "system" and not active.any():
            return
        self.i = i
        s_est = (e_v[:, self.V_c_col].sum(dim=2, dtype=torch.uint8) & 1).to(self.dtype)
        self._active = active
        self._flip_e_v = e_v
        self.sign_flip_cn_rand_new(syndrome, s_est, l_v)

    def forward(self, io_dict: dict) -> dict:
        """bp_norm_min_sum_cuda per-step decode with the lottery sign-flip in
        _iter_hook. Builds the Sobol sequence (one value per iteration) first."""
        self._B = io_dict["synd"].shape[0]
        reset_phase_state(self)
        if self.random_machine == "sobol":
            sobol = torch.quasirandom.SobolEngine(dimension=1, scramble=False)
            self.r = sobol.draw(self.max_iter, dtype=torch.float32).to(
                self.device, self.dtype
            )
            if any(p["vn_pick"] == "random" for p in self.phase_knobs):
                neighbors = torch.quasirandom.SobolEngine(dimension=2, scramble=False)
                self.r_v = neighbors.draw(self.max_iter, dtype=torch.float32)[:, 1:].to(
                    self.device, self.dtype
                )
        # read before decoding: cap.observe() may choose the cap during this batch
        capped = bool(
            self.cap is not None
            and self.cap.done
            and self.cap.frac is not None
            and not self.cap_bypass
        )
        out = self._pre_main(io_dict, super().forward)
        # system: the PyTorch path also draws B values at the iteration where the
        # last rows converge (its loop breaks one iteration later) when that is a
        # flip iteration, unless the cap stopped the loop; draw them here so both
        # paths leave the global RNG equal.
        if (
            self.random_machine == "system"
            and self.early_termination
            and self._global_last_draw
            and not capped
            and out["converge"].numel() > 0
            and bool(out["converge"].all())
            and is_flip_iter(self, int(out["iter"].max()))
        ):
            torch.rand(out["converge"].numel(), device=self.device, dtype=self.dtype)
            if self.vn_pick == "random":
                torch.rand(out["converge"].numel(), device=self.device, dtype=self.dtype)
        return out

    def _draw_r(self, n):
        """[n] uniform draws for the flip pick, one per row."""
        if self.random_machine == "system":
            return torch.rand(n, device=self.device, dtype=self.dtype)
        return self.r[(self.i - 1)].repeat(n)  # sobol

    def sign_flip_cn_rand_new(self, syndrome, s_est, l_v):
        """Select and flip VN posteriors on rows in self._active."""
        return flip_with_mode(self, syndrome, s_est, l_v)
