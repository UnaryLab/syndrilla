import torch
import torch.nn as nn
from loguru import logger

from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum_cuda import (
    _build_vn_adj,
    _load_ext,
)
from syndrilla.decoder.relay_bp.relay_bp import create as _RelayPy


class create(_RelayPy):
    """relay_bp on CUDA kernels (per-step path). Accepts every relay_bp key."""

    def __init__(self, decoder_cfg: dict, **kwargs) -> None:
        super().__init__(decoder_cfg, **kwargs)  # all relay params + helper methods

        if not torch.cuda.is_available():
            raise RuntimeError("relay_bp_cuda requires a CUDA GPU.")

        self._ext = _load_ext()
        self.N = self.H_shape[1]
        self.N_ext = self.N + 1
        # V_c_col [M, D] int64 on-device (relay kept it as a Parameter)
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

        self.algo = "relay_bp"
        logger.info("relay_bp_cuda decoder ready (per-step kernels + relay legs).")

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
        l_v = torch.zeros(B, N_ext, dtype=dt, device=dev)
        e_v = torch.zeros(B, N_ext, dtype=dt, device=dev)
        s_est = torch.zeros(B, M, dtype=dt, device=dev)

        dummy_col = torch.full((B, 1), float("inf"), dtype=dt, device=dev)
        u_init = torch.cat([io_dict["llr0"].to(dev, dt), dummy_col], dim=1).contiguous()
        e_out = torch.zeros(B, N_ext, dtype=dt, device=dev)
        l_out = torch.zeros(B, N_ext, dtype=dt, device=dev)
        num_iters = torch.full((B,), 1, device=dev)
        converges = torch.zeros(B, dtype=torch.int64, device=dev)

        # per-check ±1 syndrome signs for the cn_update kernel
        syndrome_neg_bc = torch.where(
            syndrome == 0.0, torch.ones_like(syndrome), -torch.ones_like(syndrome)
        )
        # carried variable->check message (a_v2c), init from the channel prior
        message = u_init[:, self.V_c_col].contiguous()

        cap = getattr(self, "cap", None)
        self.cap_active_last = bool(
            cap is not None and cap.done and not getattr(self, "cap_bypass", False)
        )
        cap_frac = cap.frac if self.cap_active_last else None

        D = int(self.V_c_col.shape[1])
        c2v = torch.zeros(B, M, D, dtype=dt, device=dev)

        self.r = 0
        while self.r < self.legs:
            self.r += 1
            self.i = 0
            num_iters_local = torch.full((B,), -1, device=dev)

            if self.r == 1:
                max_iter = self.iteration_initial
                memory_strengths = torch.full(
                    (B, N_ext), self.init_mem_strength, dtype=dt, device=dev
                )
                l_v = u_init.clone()
            else:
                max_iter = self.iteration_count
                memory_strengths = self.create_memory_strengths(
                    B, N_ext, self.center, self.width
                )
                message = u_init[:, self.V_c_col].contiguous()

            while self.i < max_iter:
                self.i += 1

                # memory bias (variable prior with disordered memory)
                bias = self.bias_update(memory_strengths, l_v, u_init).contiguous()
                alpha = float(self.compute_alpha())

                # check-node update on the carried message → c2v
                self._ext.cn_update(
                    message, syndrome_neg_bc, self.V_c_col, c2v, alpha, self.N
                )
                # marginal: l_v = bias + Σ c2v   (bias plays the role of u_init)
                self._ext.llr_update(
                    bias, c2v, self.VN_adj_c, self.VN_adj_k, l_v, self.VD
                )
                l_v[:, -1] = float("inf")

                self._ext.hard_decision(l_v, e_v)
                self._ext.syndrome_est(e_v, self.V_c_col, s_est, self.N)

                indices = torch.all(s_est == syndrome, 1).nonzero()
                checker = torch.where(num_iters_local == -1.0)[0]
                indices = indices[torch.isin(indices, checker)]
                if indices.size()[0] > 0:
                    num_iters[indices] += self.i
                    num_iters_local[indices] = self.i
                    e_out[indices] = e_v[indices]
                    l_out[indices] = l_v[indices]
                    converges[indices] = 1

                if checker.size()[0] == indices.size()[0]:
                    break

                # next leg-message: l_v[V_c_col] - c2v
                self._ext.vn_update(l_v, c2v, self.V_c_col, message, self.N)

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
