from loguru import logger

from syndrilla.decoder.bp_lottery_policy.bp_lottery_policy import create as _PolicyPy
from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum_cuda import create as _NmsCuda


class create(_NmsCuda, _PolicyPy):
    """bp_lottery_policy on CUDA kernels: the bp_norm_min_sum_cuda per-step loop
    with the policy sign-flip in _iter_hook.

    Accepts every bp_norm_min_sum_cuda key plus:
        random_machine  : 'sobol' (default) | 'system'
        sign_flip_policy: one of bp_lottery_policy's seven policies (default Proposed)
    """

    def __init__(self, decoding_cfg: dict, **kwargs) -> None:
        # Only the CUDA parent's __init__ runs (kernels, adjacency, V_c_col on
        # device, dtype, N_ext, …). The PyTorch parent contributes methods only.
        _NmsCuda.__init__(self, decoding_cfg, **kwargs)

        self.random_machine = str(decoding_cfg.get("random_machine", "sobol")).lower()
        if self.random_machine not in {"sobol", "system"}:
            logger.warning(
                f"Invalid random_machine <{self.random_machine}>; defaulting to sobol."
            )
            self.random_machine = "sobol"

        self.sign_flip_policy = decoding_cfg.get("sign_flip_policy", "Proposed")
        if self.sign_flip_policy not in self._sign_flip_policies:
            logger.warning(
                f"Invalid sign_flip_policy <{self.sign_flip_policy}>; defaulting to "
                f"<Proposed>. Allowed: {sorted(self._sign_flip_policies)}."
            )
            self.sign_flip_policy = "Proposed"

        self.algo = "bp_lottery_policy"
        logger.info(
            f"bp_lottery_policy_cuda ready (per-step path, policy={self.sign_flip_policy})."
        )

    def _iter_hook(self, i, l_v, e_v, active, syndrome) -> None:
        """The bp_lottery_policy flip on the unconverged rows. When the policy
        draws from the global RNG (random_machine system, or local_random), a
        call with no unconverged row returns before drawing (one host sync per
        call), so the RNG advances as if the loop had stopped when the last row
        converged."""
        draws = (
            self.random_machine == "system" or self.sign_flip_policy == "local_random"
        )
        if draws and not active.any():
            return
        _PolicyPy._iter_hook(self, i, l_v, e_v, active, syndrome)

    def forward(self, io_dict: dict) -> dict:
        """bp_norm_min_sum_cuda per-step decode with the policy sign-flip in
        _iter_hook, after the flip's per-forward state (_PolicyPy._prepare)."""
        self._prepare(io_dict)
        return _NmsCuda.forward(self, io_dict)
