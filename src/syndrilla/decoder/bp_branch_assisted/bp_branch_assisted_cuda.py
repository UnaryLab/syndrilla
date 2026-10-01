import torch
import torch.nn as nn
from loguru import logger

from syndrilla.decoder.bp_branch_assisted.bp_branch_assisted import create as _BranchPy
from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum_cuda import (
    _build_csr,
    _build_vn_adj,
    _load_ext,
)
from syndrilla.decoder.decoder import RebatchSpeedup


class create(_BranchPy):
    """bp_branch_assisted on the bp_norm_min_sum_cuda per-step CSR kernels, with
    the per-sample beta applied in torch. Accepts every bp_branch_assisted key, plus
    ``rebatch_opt`` (default true; false runs uncapped) and the optional
    ``rebatch_opt_params`` block that tunes the adaptive cap.
    """

    def __init__(self, decoding_cfg: dict, **kwargs) -> None:
        super().__init__(
            decoding_cfg, **kwargs
        )  # branch params + helpers (sign_flip, etc.)

        if not torch.cuda.is_available():
            raise RuntimeError("bp_branch_assisted_cuda requires a CUDA GPU.")

        self._ext = _load_ext()
        self.N = self.H_shape[1]
        self.N_ext = self.N + 1
        self.V_c_col = nn.Parameter(
            self.V_c_col.detach().to(self.device).long(), requires_grad=False
        )
        V_c_col_np = self.V_c_col.cpu().numpy()
        adj_c, adj_k, _ = _build_vn_adj(V_c_col_np, self.N)
        row_ptr, col, vn_eid = _build_csr(V_c_col_np, self.N, adj_c, adj_k)
        self.nnz = len(col)
        self.row_ptr = nn.Parameter(
            torch.from_numpy(row_ptr).to(self.device), requires_grad=False
        )
        self.col = nn.Parameter(
            torch.from_numpy(col).to(self.device), requires_grad=False
        )
        self.VN_eid = nn.Parameter(
            torch.from_numpy(vn_eid).to(self.device), requires_grad=False
        )

        # pure-PyTorch branch builds no cap; build the adaptive cap here unless
        # rebatch_opt is false (the forward below already honors it).
        self.cap = RebatchSpeedup.from_cfg(decoding_cfg)
        self.cap_bypass = False
        self.cap_active_last = False
        if self.cap is None:
            logger.info(
                "bp_branch_assisted_cuda: rebatch_opt false, running uncapped."
            )

        self.algo = "bp_branch_assisted"
        logger.info(
            "bp_branch_assisted_cuda decoder ready (per-step CSR kernels + branching)."
        )

    def forward(self, io_dict: dict) -> dict:
        dev = self.device
        dt = self.dtype
        syndrome = io_dict["synd"].to(dtype=dt, device=dev).contiguous()
        B, M = syndrome.shape
        self.batch_size = B
        N_ext = self.N_ext

        e_v_saver = torch.zeros(B, N_ext, dtype=torch.uint8, device=dev)
        l_saver = torch.zeros(B, N_ext, dtype=dt, device=dev)
        syndrome_saver = torch.zeros(B, M, dtype=dt, device=dev)
        s_est_saver = torch.zeros(B, M, dtype=dt, device=dev)
        iteration_saver = torch.full((B,), 0, device=dev)
        index_saver = torch.tensor([], dtype=torch.long, device=dev)
        curr_iters = torch.full((B,), 0, dtype=torch.long, device=dev)
        s0_est_comp = torch.full((B,), 0, device=dev)

        dummy_col = torch.full((B, 1), float("inf"), dtype=dt, device=dev)
        u_init = torch.cat([io_dict["llr0"].to(dev, dt), dummy_col], dim=1).contiguous()
        e_out = torch.zeros(B, N_ext, dtype=torch.uint8, device=dev)
        l_out = torch.zeros(B, N_ext, dtype=dt, device=dev)
        num_iters = torch.full((B,), -1, device=dev)
        converges = torch.zeros(B, dtype=torch.int64, device=dev)

        # CSR edge buffer [B, nnz + 1] of check->variable messages (last slot always
        # zero), posterior LLR and uint8 hard decision. A sample at its first
        # iteration (start or branch) has l_v = u_init and b_c2v = 0, so its first
        # variable->check message is u_init. Every sample runs every iteration, so
        # the kernels get an all -1 skip flag.
        b_c2v = torch.zeros(B, self.nnz + 1, dtype=dt, device=dev)
        message_saved = torch.zeros_like(b_c2v)
        l_v = u_init.clone()
        e_v = torch.empty(B, N_ext, dtype=torch.uint8, device=dev)
        run_all = torch.full((B,), -1, dtype=torch.int64, device=dev)

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

            # variable->check message l_v[col] - b_c2v and check-node update with
            # beta 1 in place on b_c2v, then the per-sample beta scale
            syndrome_neg_bc = torch.where(
                syndrome == 0.0, torch.ones_like(syndrome), -torch.ones_like(syndrome)
            )
            self._ext.vn_cn_update_csr(
                l_v, syndrome_neg_bc, self.row_ptr, self.col, b_c2v, run_all, 1.0, self.N
            )
            beta_b = (1.0 - torch.pow(2.0, -curr_iters.to(dt))).view(B, 1)
            b_c2v.mul_(beta_b)

            # l_v = sum of b_c2v + u_init (the dummy column gets +inf), hard decision
            self._ext.llr_hard_update_csr(u_init, b_c2v, self.VN_eid, l_v, e_v, run_all)
            s_est = (e_v[:, self.V_c_col].sum(2, dtype=torch.uint8) & 1).to(dt)

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

            sk_est_comp = (s_est + syndrome) % 2
            c1_results = (torch.sum(sk_est_comp, 1) <= s0_est_comp).int()
            c2_results = torch.sum(((s_est == 1) & (syndrome == 0)).int(), 1)
            branch_index = torch.where(
                (c1_results == 1) & (c2_results == 0) & (num_iters == -1)
            )[0]
            not_in = branch_index[~torch.isin(branch_index, index_saver)]

            branched = not_in.numel() != 0 and self.i != 0
            if branched:
                syndrome_saver[not_in] = syndrome[not_in]
                syndrome[not_in] = sk_est_comp[not_in]
                e_v_saver[not_in] = e_v[not_in]
                s_est_saver[not_in] = s_est[not_in]
                iteration_saver[not_in] = curr_iters[not_in]
                curr_iters[not_in] = 0
                l_saver[not_in] = l_v[not_in]
                message_saved[not_in] = b_c2v[not_in]
                b_c2v[not_in] = 0.0
                index_saver = torch.cat([not_in, index_saver], dim=0)

            b_finish = curr_iters[index_saver] >= self.max_b_iter
            if torch.any(b_finish):
                last = index_saver[torch.where(b_finish)[0]]
                b_c2v[last] = message_saved[last]
                syndrome[last] = syndrome_saver[last]
                s0_est_comp[last] = torch.sum(
                    (s_est_saver[last] + syndrome[last]) % 2, 1
                )
                l_v[last] = l_saver[last]
                curr_iters[last] = iteration_saver[last]
                index_saver = index_saver[~torch.isin(index_saver, last)]

            l_v = self.sign_flip(syndrome, s_est, l_v)  # inherited
            if branched:
                # new branch: l_v = u_init, set after the sign flip so the flip skips it
                l_v[not_in] = u_init[not_in]
            checker = torch.where(num_iters == -1)[0]

        checker = torch.where(num_iters == -1)[0]
        e_out[checker] = e_v[checker]
        l_out[checker] = l_v[checker]
        num_iters[checker] = self.i

        if cap is not None and not cap.done and not getattr(self, "cap_bypass", False):
            cap.observe(num_iters, self.num_max_iter, B)

        io_dict.update(
            {
                "e_v": e_out[:, :-1].to(dt),
                "iter": num_iters,
                "llr": l_out[:, :-1],
                "converge": converges,
            }
        )
        return io_dict
