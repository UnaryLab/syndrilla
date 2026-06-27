"""
bp_branch_assisted_cuda.py — CUDA port of bp_branch_assisted (BSFBP).

Branch-assisted sign-flipping BP runs normalized min-sum with a per-sample branching
search: when a sample meets a local condition it branches (snapshots state, decodes
the syndrome residual for up to max_b_iter steps, then restores). The inner BP math
is the SAME min-sum as bp_norm_min_sum, so this reuses bp_norm_min_sum_cuda's compiled
per-step kernels UNCHANGED. The branch-specific pieces stay in PyTorch:

  * per-sample beta: call cn_update with beta=1.0, then scale b_c2v by
    (1 - 2^(-curr_iters)) per sample (curr_iters is a PER-SAMPLE counter reset on
    each branch),
  * per-sample "iteration 1" passthrough: override a_v2c for samples at curr_iters==1
    with the carried message,
  * dynamic syndrome: recompute the per-check ±1 signs each iteration from the
    (branch-mutated) syndrome,
  * all branch save/restore bookkeeping and the sign-flip.

At float64 this reproduces bp_branch_assisted bit-for-bit. The pure-PyTorch
bp_branch_assisted has no rebatch_speedup cap; this CUDA port adds one — supply an
rebatch_speedup block under decoder: to activate it, omit it to run uncapped.

YAML algorithm key: bp_branch_assisted_cuda
"""

import torch
import torch.nn as nn
from loguru import logger

from syndrilla.decoder.decoder import RebatchSpeedup
from syndrilla.decoder.bp_branch_assisted.bp_branch_assisted import create as _BranchPy
from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum_cuda import (
    _load_ext,
    _build_vn_adj,
)


class create(_BranchPy):
    """bp_branch_assisted on CUDA kernels (per-step path). Accepts every
    bp_branch_assisted key, plus the optional ``rebatch_speedup`` block to enable the
    adaptive cap (omit it to run uncapped).
    """

    def __init__(self, decoder_cfg: dict, **kwargs) -> None:
        super().__init__(
            decoder_cfg, **kwargs
        )  # branch params + helpers (sign_flip, etc.)

        if not torch.cuda.is_available():
            raise RuntimeError("bp_branch_assisted_cuda requires a CUDA GPU.")

        self._ext = _load_ext()
        self.N = self.H_shape[1]
        self.N_ext = self.N + 1
        self.V_c_col = nn.Parameter(
            self.V_c_col.detach().to(self.device).long(), requires_grad=False
        )
        adj_c, adj_k, self.VD = _build_vn_adj(self.V_c_col.cpu().numpy(), self.N)
        self.VN_adj_c = nn.Parameter(
            torch.from_numpy(adj_c).to(self.device), requires_grad=False
        )
        self.VN_adj_k = nn.Parameter(
            torch.from_numpy(adj_k).to(self.device), requires_grad=False
        )
        # dense [M, N] matrix for the inherited sign_flip
        self.H_matrix = self.H_matrix.to(self.device, self.dtype)

        # pure-PyTorch branch builds no cap; enable the adaptive cap here when an
        # rebatch_speedup block is present (the forward below already honors it).
        self.cap = RebatchSpeedup.from_cfg(decoder_cfg.get("rebatch_speedup"))
        self.cap_bypass = False
        self.cap_active_last = False
        if self.cap is None:
            logger.info(
                "bp_branch_assisted_cuda: no rebatch_speedup block — running uncapped."
            )

        self.algo = "bp_branch_assisted_cuda"
        logger.info(
            "bp_branch_assisted_cuda decoder ready (per-step kernels + branching)."
        )

    def forward(self, io_dict: dict) -> dict:
        dev = self.device
        dt = self.dtype
        syndrome = io_dict["synd"].to(dtype=dt, device=dev).contiguous()
        B, M = syndrome.shape
        self.batch_size = B
        N_ext = self.N_ext

        l_v = torch.zeros(B, N_ext, dtype=dt, device=dev)
        e_v = torch.zeros(B, N_ext, dtype=dt, device=dev)
        e_v_saver = torch.zeros(B, N_ext, dtype=dt, device=dev)
        s_est = torch.zeros(B, M, dtype=dt, device=dev)
        l_saver = torch.zeros(B, N_ext, dtype=dt, device=dev)
        syndrome_saver = torch.zeros(B, M, dtype=dt, device=dev)
        s_est_saver = torch.zeros(B, M, dtype=dt, device=dev)
        iteration_saver = torch.full((B,), 0, device=dev)
        index_saver = torch.tensor([], dtype=torch.long, device=dev)
        curr_iters = torch.full((B,), 0, dtype=torch.long, device=dev)
        s0_est_comp = torch.full((B,), 0, device=dev)

        dummy_col = torch.full((B, 1), float("inf"), dtype=dt, device=dev)
        u_init = torch.cat([io_dict["llr0"].to(dev, dt), dummy_col], dim=1).contiguous()
        e_out = torch.zeros(B, N_ext, dtype=dt, device=dev)
        l_out = torch.zeros(B, N_ext, dtype=dt, device=dev)
        num_iters = torch.full((B,), -1, device=dev)
        converges = torch.zeros(B, dtype=torch.int64, device=dev)

        D = int(self.V_c_col.shape[1])
        message = u_init[:, self.V_c_col].contiguous()  # carried b_c2v (init)
        message_saved = torch.zeros(B, M, D, dtype=dt, device=dev)
        a_v2c = torch.zeros(B, M, D, dtype=dt, device=dev)
        b_c2v = torch.zeros(B, M, D, dtype=dt, device=dev)

        sobol = torch.quasirandom.SobolEngine(dimension=1, scramble=False)
        self.r = sobol.draw(self.max_iter * self.max_iter).to(dev, dt)

        cap = getattr(self, "cap", None)
        self.cap_active_last = bool(
            cap is not None and cap.done and not getattr(self, "cap_bypass", False)
        )
        cap_frac = cap.frac if self.cap_active_last else None

        self.i = 0
        checker = torch.where(num_iters == -1)[0]
        while checker.size()[0] != 0:
            self.i += 1
            curr_iters = curr_iters + 1

            # ── vn_update with per-sample iteration-1 passthrough ───────────────
            self._ext.vn_update(l_v, message, self.V_c_col, a_v2c, self.N)
            curr1 = curr_iters == 1
            if curr1.any():
                a_v2c[curr1] = message[curr1]

            # ── cn_update (beta=1) then per-sample beta scale + dynamic syndrome ─
            syndrome_neg_bc = torch.where(
                syndrome == 0.0, torch.ones_like(syndrome), -torch.ones_like(syndrome)
            )
            self._ext.cn_update(
                a_v2c, syndrome_neg_bc, self.V_c_col, b_c2v, 1.0, self.N
            )
            beta_b = (1.0 - torch.pow(2.0, -curr_iters.to(dt))).view(B, 1, 1)
            b_c2v = b_c2v * beta_b
            message = b_c2v  # carry for next iteration

            # ── marginal LLR + hard decision + syndrome ─────────────────────────
            self._ext.llr_update(
                u_init, b_c2v, self.VN_adj_c, self.VN_adj_k, l_v, self.VD
            )
            l_v[:, -1] = float("inf")
            e_v = torch.where(l_v <= 0, 1.0, 0.0).to(dt)
            self._ext.syndrome_est(e_v, self.V_c_col, s_est, self.N)

            if (curr_iters == 1).all():
                s0_est_comp = torch.sum((s_est + syndrome) % 2, 1)

            mask = torch.ones(B, dtype=torch.bool, device=dev)
            mask[index_saver] = False
            condition = (curr_iters == self.max_iter) | torch.all(
                s_est == syndrome, dim=1
            )
            indices = torch.where(mask & condition)[0]
            converges_index = torch.where(torch.all(s_est == syndrome, 1))[0]
            checker = torch.where(num_iters == -1)[0]
            indices = indices[torch.isin(indices, checker)]

            if indices.size()[0] > 0:
                num_iters[indices] = self.i
                e_out[indices] = e_v[indices]
                l_out[indices] = l_v[indices]
                converges[converges_index] = 1

                keep_mask = ~torch.isin(index_saver, indices)
                remove_mask = index_saver[torch.isin(index_saver, indices)]
                if remove_mask.size()[0] > 0:
                    num_iters[remove_mask] = self.i
                    temp_l_v = l_saver[remove_mask]
                    l_out[remove_mask] = l_v[remove_mask] + temp_l_v
                    e_out[remove_mask] = (
                        e_out[remove_mask] + e_v_saver[remove_mask]
                    ) % 2
                index_saver = index_saver[keep_mask]

            checker = torch.where(num_iters == -1)[0]
            if checker.size()[0] == 0:
                break
            if cap_frac is not None and int((num_iters != -1).sum()) >= cap_frac * B:
                break

            # ── branch decision (C1/C2) and save/restore ────────────────────────
            sk_est_comp = (s_est + syndrome) % 2
            c1_results = (torch.sum(sk_est_comp, 1) <= s0_est_comp).int()
            c2_results = torch.sum(((s_est == 1) & (syndrome == 0)).int(), 1)
            branch_index = torch.where(
                (c1_results == 1) & (c2_results == 0) & (num_iters == -1)
            )[0]
            not_in = branch_index[~torch.isin(branch_index, index_saver)]

            if not_in.numel() != 0 and self.i != 0:
                syndrome_saver[not_in] = syndrome[not_in]
                syndrome[not_in] = sk_est_comp[not_in]
                e_v_saver[not_in] = e_v[not_in]
                s_est_saver[not_in] = s_est[not_in]
                iteration_saver[not_in] = curr_iters[not_in]
                curr_iters[not_in] = 0
                l_saver[not_in] = l_v[not_in]
                l_v[not_in] = u_init[not_in]
                message_saved[not_in] = message[not_in]
                message[not_in] = u_init[not_in][:, self.V_c_col]
                index_saver = torch.cat([not_in, index_saver], dim=0)

            b_finish = curr_iters[index_saver] >= self.max_b_iter
            if torch.any(b_finish):
                last = index_saver[torch.where(b_finish)[0]]
                message[last] = message_saved[last]
                syndrome[last] = syndrome_saver[last]
                s0_est_comp[last] = torch.sum(
                    (s_est_saver[last] + syndrome[last]) % 2, 1
                )
                l_v[last] = l_saver[last]
                curr_iters[last] = iteration_saver[last]
                index_saver = index_saver[~torch.isin(index_saver, last)]

            l_v = self.sign_flip(syndrome, s_est, l_v)  # inherited
            checker = torch.where(num_iters == -1)[0]

        checker = torch.where(num_iters == -1)[0]
        e_out[checker] = e_v[checker]
        l_out[checker] = l_v[checker]
        num_iters[checker] = self.i

        if cap is not None and not cap.done and not getattr(self, "cap_bypass", False):
            cap.observe(num_iters, self.num_max_iter, B)

        io_dict.update(
            {
                "e_v": e_out[:, :-1],
                "iter": num_iters,
                "llr": l_out[:, :-1],
                "converge": converges,
            }
        )
        return io_dict
