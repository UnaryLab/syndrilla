"""
bp_lottery_policy_cuda.py — CUDA port of bp_lottery_policy.

Same per-iteration BP math as bp_norm_min_sum, plus the configurable lottery
sign-flip *policy* (one of seven strategies) applied every iteration. Like
bp_lottery_cuda it reuses bp_norm_min_sum_cuda's compiled per-step kernels for the
BP steps and runs the sign-flip in PyTorch between iterations (a per-iteration host
op the fused kernel cannot express).

Implementation reuses BOTH parents directly via multiple inheritance:
  * bp_norm_min_sum_cuda   — device/dtype/kernels/adjacency setup + the per-step
    kernel entry points (self._ext.*).
  * bp_lottery_policy       — the seven sign_flip_* methods and policy validation,
    which depend only on self.H_matrix / self.r / self.i / self.batch_size and so
    work unchanged once H_matrix is staged on-device here.

YAML algorithm key: bp_lottery_policy_cuda
"""

import torch
from loguru import logger

from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum_cuda import create as _NmsCuda
from syndrilla.decoder.bp_lottery_policy.bp_lottery_policy import create as _PolicyPy


class create(_NmsCuda, _PolicyPy):
    """bp_lottery_policy on CUDA kernels (per-step path).

    Accepts every bp_norm_min_sum_cuda key plus:
        random_machine  : 'sobol' (default) | 'system'
        sign_flip_policy: one of bp_lottery_policy's seven policies (default Proposed)
    """

    def __init__(self, decoder_cfg: dict, **kwargs) -> None:
        # Only the CUDA parent's __init__ runs (kernels, adjacency, V_c_col on
        # device, dtype, N_ext, …). The PyTorch parent contributes methods only.
        _NmsCuda.__init__(self, decoder_cfg, **kwargs)

        # sign-flip is a per-iteration host op → no fused kernel.
        self._use_fused = False

        self.random_machine = str(decoder_cfg.get("random_machine", "sobol")).lower()
        if self.random_machine not in {"sobol", "system"}:
            logger.warning(
                f"Invalid random_machine <{self.random_machine}>; defaulting to sobol."
            )
            self.random_machine = "sobol"

        self.sign_flip_policy = decoder_cfg.get("sign_flip_policy", "Proposed")
        if self.sign_flip_policy not in self._sign_flip_policies:
            logger.warning(
                f"Invalid sign_flip_policy <{self.sign_flip_policy}>; defaulting to "
                f"<Proposed>. Allowed: {sorted(self._sign_flip_policies)}."
            )
            self.sign_flip_policy = "Proposed"

        # Dense [M, N] parity matrix on-device — the inherited sign_flip_* methods
        # index self.H_matrix (bp_norm_min_sum_cuda kept only V_c_col).
        bundle = kwargs.get("bundle")
        _, _, _, H_matrix = bundle.select(self.check_type)
        self.H_matrix = H_matrix.to(self.device, self.dtype)

        self.algo = "bp_lottery_policy"
        logger.info(
            f"bp_lottery_policy_cuda ready (per-step path, policy={self.sign_flip_policy})."
        )

    # _PolicyPy defines these only as an instance attr (in its __init__, which we
    # skip), so declare the valid set here for validation. The seven sign_flip_*
    # methods themselves are inherited from _PolicyPy.
    _sign_flip_policies = {
        "Proposed",
        "global_optimal",
        "global_connectivity",
        "global_weighted_random",
        "local_random",
        "local_reliable",
        "local_connectivity",
    }

    def _apply_policy(self, syndrome, s_est, l_v):
        p = self.sign_flip_policy
        if p == "Proposed":
            return self.sign_flip_lottery(syndrome, s_est, l_v)
        if p == "global_optimal":
            return self.sign_flip_global_optimal(syndrome, s_est, l_v)
        if p == "global_connectivity":
            return self.sign_flip_global_connectivity(syndrome, s_est, l_v)
        if p == "global_weighted_random":
            return self.sign_flip_global_weighted_random(syndrome, s_est, l_v)
        if p == "local_random":
            return self.sign_flip_local_random(syndrome, s_est, l_v)
        if p == "local_reliable":
            return self.sign_flip_local_reliable(syndrome, s_est, l_v)
        if p == "local_connectivity":
            return self.sign_flip_local_connectivity(syndrome, s_est, l_v)
        return l_v

    def forward(self, io_dict: dict) -> dict:
        """Per-step CUDA BP decode with the policy sign-flip applied each iteration."""
        dev = self.device
        syndrome = io_dict["synd"].to(dtype=self.dtype, device=dev).contiguous()
        B, M = syndrome.shape
        self.batch_size = B

        llr0 = io_dict["llr0"].to(dtype=self.dtype, device=dev).contiguous()
        dummy_col = torch.full((B, 1), float("inf"), dtype=self.dtype, device=dev)
        u_init = torch.cat([llr0, dummy_col], dim=1)

        syndrome_neg_bc = torch.where(
            syndrome == 0.0, torch.ones_like(syndrome), -torch.ones_like(syndrome)
        )

        e_out = torch.zeros(B, self.N_ext, dtype=self.dtype, device=dev)
        l_out = torch.zeros(B, self.N_ext, dtype=self.dtype, device=dev)
        num_iters = torch.full((B,), -1, dtype=torch.int64, device=dev)
        converges = torch.zeros(B, dtype=torch.int64, device=dev)

        cap = getattr(self, "cap", None)
        self.cap_active_last = bool(
            cap is not None and cap.done and not getattr(self, "cap_bypass", False)
        )
        cap_frac = cap.frac if self.cap_active_last else None

        if self.random_machine == "sobol":
            sobol = torch.quasirandom.SobolEngine(dimension=1, scramble=False)
            draw_dtype = (
                self.dtype
                if self.dtype in {torch.float32, torch.float64}
                else torch.float32
            )
            self.r = sobol.draw(self.max_iter, dtype=draw_dtype).to(dev, self.dtype)

        D = int(self.V_c_col.shape[1])
        a_v2c = torch.zeros(B, M, D, dtype=self.dtype, device=dev)
        b_c2v = torch.zeros(B, M, D, dtype=self.dtype, device=dev)
        l_v = torch.zeros(B, self.N_ext, dtype=self.dtype, device=dev)
        e_v = torch.zeros(B, self.N_ext, dtype=self.dtype, device=dev)
        s_est = torch.zeros(B, M, dtype=self.dtype, device=dev)

        for i in range(1, self.max_iter + 1):
            self.i = i
            beta = 1.0 - 2.0 ** (-i)
            if i == 1:
                self._ext.init_messages(u_init, self.V_c_col, a_v2c)
            else:
                self._ext.vn_update(l_v, b_c2v, self.V_c_col, a_v2c, self.N)
            self._ext.cn_update(
                a_v2c, syndrome_neg_bc, self.V_c_col, b_c2v, beta, self.N
            )
            self._ext.llr_update(
                u_init, b_c2v, self.VN_adj_c, self.VN_adj_k, l_v, self.VD
            )
            l_v[:, -1] = float("inf")
            self._ext.hard_decision(l_v, e_v)
            self._ext.syndrome_est(e_v, self.V_c_col, s_est, self.N)
            self._ext.convergence_update(
                s_est, syndrome, e_v, l_v, e_out, l_out, num_iters, converges, i
            )

            n_conv = int((num_iters != -1).sum())
            if n_conv == B:
                break
            if cap_frac is not None and n_conv >= cap_frac * B:
                break

            # policy sign-flip every iteration (matches bp_lottery_policy)
            l_v = self._apply_policy(syndrome, s_est, l_v)

        not_conv = num_iters == -1
        if not_conv.any().item():
            e_out[not_conv] = e_v[not_conv]
            l_out[not_conv] = l_v[not_conv]
            num_iters[not_conv] = (
                self.i
            )  # actual stop iter (== max_iter unless the cap broke early)

        if cap is not None and not cap.done and not getattr(self, "cap_bypass", False):
            cap.observe(num_iters, self.max_iter, B)

        io_dict.update(
            {
                "e_v": e_out[:, :-1],
                "iter": num_iters,
                "llr": l_out[:, :-1],
                "converge": converges,
            }
        )
        return io_dict
