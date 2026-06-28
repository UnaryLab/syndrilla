"""
bp_lottery_cuda.py — BP Normalized Min-Sum "lottery" decoder backed by CUDA kernels.

CUDA port of bp_lottery (pure PyTorch). The per-iteration BP math is identical to
bp_norm_min_sum, so this subclasses bp_norm_min_sum_cuda and reuses its compiled
modular kernels (init_messages / vn_update / cn_update / llr_update /
hard_decision / syndrome_est / convergence_update) unchanged. The only addition is
the lottery "sign-flip" heuristic: after `flip_start_iter`, each iteration picks
one variable node attached to a random unsatisfied check and flips its posterior
LLR sign, nudging stuck samples toward a valid syndrome.

That sign-flip is a per-iteration HOST operation that mutates l_v between BP steps,
so — unlike bp_norm_min_sum_cuda — this decoder cannot use the single-launch fused
kernel (which has no host break point mid-loop). It always runs the PER-STEP
kernels, doing the sign-flip in PyTorch between iterations. The BP kernels still
replace the eager-PyTorch ops, and at float64 the result matches bp_lottery
bit-for-bit (the sign-flip math is the same PyTorch, fed bit-identical l_v/s_est).

YAML algorithm key: bp_lottery_cuda
"""

import torch
from loguru import logger

from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum_cuda import create as _BaseCuda


class create(_BaseCuda):
    """BP Normalized Min-Sum lottery decoder on CUDA kernels (per-step path).

    Accepts every bp_norm_min_sum_cuda key plus the lottery knobs:
        random_machine : 'sobol' (default) | 'system'   RNG for the flip pick
        flip_start_iter: int (default 4)                 first iteration that flips
    """

    def __init__(self, decoder_cfg: dict, **kwargs) -> None:
        super().__init__(decoder_cfg, **kwargs)

        # The sign-flip is a host op between BP steps; the fused kernel cannot
        # express it, so always take the per-step kernel path.
        self._use_fused = False

        # lottery knobs
        self.random_machine = str(decoder_cfg.get("random_machine", "sobol")).lower()
        if self.random_machine not in {"sobol", "system"}:
            logger.warning(
                f"Invalid random_machine <{self.random_machine}>; defaulting to sobol."
            )
            self.random_machine = "sobol"
        self.flip_start_iter = int(decoder_cfg.get("flip_start_iter", 4))

        # Dense [M, N] parity-check matrix on-device for the sign-flip scoring
        # (bp_norm_min_sum_cuda keeps only V_c_col; the sign-flip needs the matrix).
        bundle = kwargs.get("bundle")
        _, _, _, H_matrix = bundle.select(self.check_type)
        self.H_dense = H_matrix.to(self.device, self.dtype)

        self.algo = "bp_lottery"
        logger.info("bp_lottery_cuda decoder ready (per-step path + sign-flip).")

    # ── Forward pass ──────────────────────────────────────────────────────────
    def forward(self, io_dict: dict) -> dict:
        """Per-step CUDA BP decode with the lottery sign-flip between iterations.

        Mirrors bp_norm_min_sum_cuda's per-step path, inserting
        ``sign_flip_cn_rand_new`` after ``flip_start_iter`` so the modified l_v
        feeds the next iteration's vn_update — exactly as bp_lottery does.
        """
        dev = self.device
        syndrome = io_dict["synd"].to(dtype=self.dtype, device=dev).contiguous()
        B, M = syndrome.shape
        self.batch_size = B

        llr0 = io_dict["llr0"].to(dtype=self.dtype, device=dev).contiguous()
        dummy_col = torch.full((B, 1), float("inf"), dtype=self.dtype, device=dev)
        u_init = torch.cat([llr0, dummy_col], dim=1)  # [B, N_ext]

        syndrome_neg_bc = torch.where(
            syndrome == 0.0, torch.ones_like(syndrome), -torch.ones_like(syndrome)
        )

        e_out = torch.zeros(B, self.N_ext, dtype=self.dtype, device=dev)
        l_out = torch.zeros(B, self.N_ext, dtype=self.dtype, device=dev)
        num_iters = torch.full((B,), -1, dtype=torch.int64, device=dev)
        converges = torch.zeros(B, dtype=torch.int64, device=dev)

        # adaptive cap (built by bp_norm_min_sum_cuda.__init__ from the rebatch_speedup
        # block; None when that block is absent → uncapped)
        cap = getattr(self, "cap", None)
        self.cap_active_last = bool(
            cap is not None and cap.done and not getattr(self, "cap_bypass", False)
        )
        cap_frac = cap.frac if self.cap_active_last else None

        # Sobol draw for the flip pick — one quasi-random value per iteration.
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
            # adaptive cap: stop once the learned fraction has converged; the
            # unconverged tail (converge == 0) is deferred to main's extra queue.
            if cap_frac is not None and n_conv >= cap_frac * B:
                break

            # lottery sign-flip: nudge stuck (unconverged) samples. Converged
            # samples have no unsatisfied checks, so they are left untouched.
            if i > self.flip_start_iter:
                l_v = self.sign_flip_cn_rand_new(syndrome, s_est, l_v)

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

    # ── lottery sign-flip (verbatim math from bp_lottery, on H_dense) ──────────
    def sign_flip_cn_rand_new(self, syndrome, s_est, l_v):
        """Flip one variable node's LLR sign per stuck sample.

        Pick a random unsatisfied check, then among its variable nodes pick the one
        with the highest unsatisfied-check connectivity (ties broken by smallest
        |LLR|), and flip its posterior LLR sign. Identical to bp_lottery.
        """
        synd_diff = (syndrome + s_est) % 2.0  # [B, M]
        unsat_cn_mask = synd_diff.bool()
        batch_size, M = unsat_cn_mask.shape

        total_unsat = unsat_cn_mask.sum(dim=1)
        valid_mask = total_unsat > 0

        if self.random_machine == "system":
            r = torch.rand(batch_size, device=self.device)
        else:  # sobol
            r = self.r[(self.i - 1)].repeat(batch_size)

        total_unsat_safe = total_unsat + (total_unsat == 0).float()
        unsat_cumsum = unsat_cn_mask.cumsum(dim=1)

        # check selection: random among unsatisfied
        rand_pos = torch.floor(r * total_unsat_safe).long() + 1
        chosen_cn = ((unsat_cumsum >= rand_pos.unsqueeze(1)) & unsat_cn_mask).float()
        chosen_cn_idx = torch.argmax(chosen_cn, dim=1)  # [B]

        # variable selection: 1) max unsatisfied-CN connectivity 2) min |LLR|
        H_expanded = self.H_dense.unsqueeze(0).expand(batch_size, -1, -1)
        candidate_vn_mask = H_expanded[
            torch.arange(batch_size), chosen_cn_idx, :
        ].bool()
        vn_unsat_counts = torch.matmul(unsat_cn_mask.to(self.dtype), self.H_dense)

        llr = torch.abs(l_v[:, :-1])  # [B, N]
        score = vn_unsat_counts * 1e6 - llr
        masked_score = score + (~candidate_vn_mask).float() * -1e9
        selected_vn = torch.argmax(masked_score, dim=1)

        l_v[valid_mask, selected_vn[valid_mask]] *= -1.0
        return l_v
