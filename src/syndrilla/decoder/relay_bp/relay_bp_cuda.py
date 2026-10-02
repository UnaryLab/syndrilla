import torch
import torch.nn as nn
from loguru import logger

from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum_cuda import (
    _build_csr,
    _build_vn_adj,
    _load_ext,
)
from syndrilla.decoder.relay_bp.relay_bp import create as _RelayPy


class create(_RelayPy):
    """relay_bp on the bp_norm_min_sum_cuda per-step CSR kernels, with alpha as
    their beta and the memory bias in place of the channel LLR. Accepts every
    relay_bp key."""

    def __init__(self, decoding_cfg: dict, **kwargs) -> None:
        super().__init__(decoding_cfg, **kwargs)  # all relay params + helper methods

        if not torch.cuda.is_available():
            raise RuntimeError("relay_bp_cuda requires a CUDA GPU.")

        self._ext = _load_ext()
        self.N = self.H_shape[1]
        self.N_ext = self.N + 1
        # V_c_col [M, D] int64 on-device (relay kept it as a Parameter)
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

        self.algo = "relay_bp"
        logger.info("relay_bp_cuda decoder ready (per-step CSR kernels + relay legs).")

    def forward(self, io_dict: dict) -> dict:
        dev = self.device
        dt = self.dtype
        syndrome = io_dict["synd"].to(dtype=dt, device=dev).contiguous()
        B, M = syndrome.shape
        self.batch_size = B
        N_ext = self.N_ext

        solutions = torch.zeros(B, dtype=dt, device=dev)
        e_solutions = torch.full((B,), float("inf"), dtype=dt, device=dev)
        e_best = torch.zeros(B, self.N, dtype=dt, device=dev)

        dummy_col = torch.full((B, 1), float("inf"), dtype=dt, device=dev)
        u_init = torch.cat([io_dict["llr0"].to(dev, dt), dummy_col], dim=1).contiguous()
        e_out = torch.zeros(B, N_ext, dtype=dt, device=dev)
        l_out = torch.zeros(B, N_ext, dtype=dt, device=dev)
        num_iters = torch.full((B,), 1, device=dev)
        converges = torch.zeros(B, dtype=torch.int64, device=dev)

        # per-check ±1 syndrome signs for the check-node update
        syndrome_neg_bc = torch.where(
            syndrome == 0.0, torch.ones_like(syndrome), -torch.ones_like(syndrome)
        )

        cap = getattr(self, "cap", None)
        self.cap_active_last = bool(
            cap is not None and cap.done and cap.frac is not None
            and not getattr(self, "cap_bypass", False)
        )
        cap_frac = cap.frac if self.cap_active_last else None

        # CSR edge buffer [B, nnz + 1] of check->variable messages (last slot always
        # zero), posterior LLR and uint8 hard decision. Every sample runs every
        # iteration of a leg, also after it converged in that leg, so the kernels
        # get an all -1 skip flag; leg_iters records the convergence iteration.
        b_c2v = torch.zeros(B, self.nnz + 1, dtype=dt, device=dev)
        l_v = torch.empty(B, N_ext, dtype=dt, device=dev)
        e_v = torch.empty(B, N_ext, dtype=torch.uint8, device=dev)
        run_all = torch.full((B,), -1, dtype=torch.int64, device=dev)
        mismatch = torch.zeros(B, dtype=torch.int32, device=dev)

        self.r = 0
        while self.r < self.legs:
            self.r += 1
            self.i = 0
            leg_iters = torch.full((B,), -1, dtype=torch.int64, device=dev)
            mismatch.zero_()

            if self.r == 1:
                max_iter = self.iteration_initial
                memory_strengths = torch.full(
                    (B, N_ext), self.init_mem_strength, dtype=dt, device=dev
                )
                l_v.copy_(u_init)
            else:
                max_iter = self.iteration_count
                memory_strengths = self.create_memory_strengths(
                    B, N_ext, self.center, self.width
                )
            # leg start: zero messages, so the first variable->check message is
            # u_init - 0 = u_init
            b_c2v.zero_()

            while self.i < max_iter:
                self.i += 1

                # memory bias (variable prior with disordered memory)
                bias = self.bias_update(memory_strengths, l_v, u_init).contiguous()
                alpha = float(self.compute_alpha())

                # variable->check message l_v[col] - b_c2v and check-node update,
                # in place on b_c2v
                self._ext.vn_cn_update_csr(
                    u_init if self.i == 1 else l_v,
                    syndrome_neg_bc,
                    self.row_ptr,
                    self.col,
                    b_c2v,
                    run_all,
                    alpha,
                    self.N,
                )
                # marginal: l_v = sum of b_c2v + bias (the dummy column gets +inf
                # from bias), and the hard decision
                self._ext.llr_hard_update_csr(
                    bias, b_c2v, self.VN_eid, l_v, e_v, run_all
                )
                self._ext.syndrome_check_csr(
                    e_v, self.row_ptr, self.col, syndrome, run_all, mismatch, self.N
                )
                self._ext.convergence_flag_update(
                    mismatch, leg_iters, converges, self.i
                )
                new = (leg_iters == self.i).unsqueeze(1)
                torch.where(new, e_v, e_out, out=e_out)
                torch.where(new, l_v, l_out, out=l_out)

                if not (leg_iters == -1).any():
                    break

            num_iters += leg_iters.clamp(min=0)
            valid_mask = (converges == 1) & (solutions < self.solution)
            new_e_weight_all = (e_out[:, :-1] * u_init[:, :-1].abs()).sum(dim=1)
            solutions = solutions + valid_mask.to(solutions.dtype)
            improve_mask = valid_mask & (new_e_weight_all < e_solutions)
            e_solutions = torch.where(improve_mask, new_e_weight_all, e_solutions)
            e_best[improve_mask, :] = e_out[improve_mask, :-1]

            if solutions.sum() >= B * self.solution:
                break
            if cap_frac is not None and int((converges == 1).sum()) >= cap_frac * B:
                break

        if cap is not None and not cap.done and not getattr(self, "cap_bypass", False):
            obs = num_iters.clone().clamp(max=self.num_max_iter)
            obs[converges == 0] = self.num_max_iter
            cap.observe(obs, self.num_max_iter, B)

        io_dict.update(
            {"e_v": e_best, "iter": num_iters, "llr": l_out, "converge": converges}
        )
        return io_dict
