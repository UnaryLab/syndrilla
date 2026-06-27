"""
bp_sf_cuda.py — CUDA port of bp_sf (normalized min-sum BP + syndrome flipping).

BP-SF's belief-propagation core is the SAME normalized min-sum as bp_norm_min_sum,
so this reuses bp_norm_min_sum_cuda's compiled kernels UNCHANGED — no new .cu code.
Only the BP loop (`_bp_core`) is moved onto the GPU kernels; the syndrome-flipping
post-processing (`_sf_postprocess`) is already batched PyTorch and runs on-device as
inherited, so the whole decoder follows the convention used by the other *_cuda ports
(bp_branch_assisted_cuda, bp_lottery_cuda, …): subclass the pure-PyTorch decoder and
swap the message passing for kernels.

Two BP execution paths, picked per call:

  • PER-STEP (main pass, `track_osc=True`): the seven modular kernels
    (init_messages, vn_update, cn_update, llr_update, hard_decision, syndrome_est,
    convergence_update) driven by a host loop. Oscillation tracking — the per-bit
    hard-decision flip count BP-SF needs to rank flip candidates — is a cheap
    elementwise compare in PyTorch between kernel steps, so it stays on the GPU
    without touching the .cu code. This path is mandatory for the main pass because
    the fused kernel does not emit oscillation counts.

  • FUSED (SF retries, `track_osc=False`): one kernel launch (bp_nms_fused) decodes
    a whole retry batch in shared memory with per-sample early termination. The SF
    retries re-run BP many times and never need oscillation counts, so they take the
    fast single-launch path whenever the working set fits shared memory. Disable with
    `force_per_step: true` (also forces the modular kernels for retries).

At float64 the BP core is numerically identical to bp_norm_min_sum_cuda's (which is
validated bit-for-bit against the PyTorch reference), so bp_sf_cuda reproduces bp_sf's
per-sample decoding outcomes. The SF logic itself is inherited verbatim.

BP-SF uses no rebatch_speedup cap (the SF retry replaces the deferred-tail mechanism), so
— unlike the other *_cuda ports — this decoder builds no adaptive cap.

YAML algorithm key: bp_sf  (with device.device_type: cuda → this file is dispatched)
"""

import math

import torch
import torch.nn as nn
from loguru import logger

from syndrilla.decoder.bp_sf.bp_sf import create as _BpSfPy
from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum_cuda import (
    _load_ext,
    _build_vn_adj,
)


class create(_BpSfPy):
    """bp_sf on CUDA kernels. Accepts every bp_sf key (check_type, max_iter, dtype,
    the ``sf:`` block) plus the optional ``force_per_step: true`` debug knob that
    routes the SF retries through the modular kernels instead of the fused one."""

    def __init__(self, decoder_cfg: dict, **kwargs) -> None:
        # Build all SF params, H_dense, and the message-passing helpers. The base
        # also resolves self.device to cuda:idx (device_type=cuda in the YAML).
        super().__init__(decoder_cfg, **kwargs)

        if not torch.cuda.is_available():
            raise RuntimeError(
                "bp_sf_cuda requires a CUDA-capable GPU. Use bp_sf on a cpu device."
            )

        self._ext = _load_ext()

        # ── kernel scaffolding (mirrors bp_norm_min_sum_cuda.__init__) ──────────
        self.N = self.H_shape[1]
        self.N_ext = self.N + 1
        # V_c_col as int64 on device — every kernel indexes it as the check→var map.
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

        # ── fused-kernel block size + availability (used by the SF retries) ─────
        M_val, D = self.V_c_col.shape
        min_threads = M_val * D
        self._block_size = min(max(32 * math.ceil(min_threads / 32), 64), 512)
        smem_needed = self._ext.fused_smem_bytes(
            M_val, D, self.N_ext, self.VD, self.dtype.itemsize
        )
        smem_limit = self._ext.fused_smem_limit()
        self._use_fused = smem_needed <= smem_limit and not decoder_cfg.get(
            "force_per_step", False
        )
        if not self._use_fused:
            logger.info(
                f"bp_sf_cuda: SF retries use the per-step path "
                f"(fused needs {smem_needed} B shared memory, limit {smem_limit} B)."
            )

        self.algo = "bp_sf"
        logger.info(
            "bp_sf_cuda decoder ready (per-step BP + oscillation tracking; "
            f"fused SF retries={'on' if self._use_fused else 'off'})."
        )

    # ── BP core (overrides the PyTorch loop; forward + _sf_postprocess inherited) ──
    def _bp_core(self, syndrome, llr0, track_osc=False):
        """Run normalized min-sum BP on a batch of syndromes via CUDA kernels.

        Returns ``(e_out, l_out, converges, num_iters, osc)`` over the real N bits,
        matching the PyTorch bp_sf._bp_core contract. The main pass (track_osc=True)
        must use the per-step path so it can accumulate per-bit oscillation counts;
        the SF retries (track_osc=False) take the fused single-launch path when it is
        available (much faster across the many retry batches)."""
        if track_osc or not self._use_fused:
            return self._bp_core_perstep(syndrome, llr0, track_osc)
        return self._bp_core_fused(syndrome, llr0)

    def _bp_core_perstep(self, syndrome, llr0, track_osc):
        """Host loop over the modular kernels, tracking oscillation in PyTorch.

        The message-passing math is identical to bp_norm_min_sum_cuda's per-step path
        (so bit-for-bit equal to bp_sf's PyTorch loop at float64). Oscillation tracking
        replicates the reference: per bit, count how often the hard decision differs
        from the previous iteration, starting from an all-zero prior."""
        dev, dt = self.device, self.dtype
        N, N_ext, VD = self.N, self.N_ext, self.VD
        syndrome = syndrome.to(dev, dt).contiguous()
        B, M = syndrome.shape
        D = int(self.V_c_col.shape[1])

        dummy_col = torch.full((B, 1), float("inf"), dtype=dt, device=dev)
        u_init = torch.cat([llr0.to(dev, dt), dummy_col], dim=1).contiguous()
        syndrome_neg_bc = torch.where(
            syndrome == 0.0, torch.ones_like(syndrome), -torch.ones_like(syndrome)
        )

        a_v2c = torch.zeros(B, M, D, dtype=dt, device=dev)
        b_c2v = torch.zeros(B, M, D, dtype=dt, device=dev)
        l_v = torch.zeros(B, N_ext, dtype=dt, device=dev)
        e_v = torch.zeros(B, N_ext, dtype=dt, device=dev)
        s_est = torch.zeros(B, M, dtype=dt, device=dev)
        e_out = torch.zeros(B, N_ext, dtype=dt, device=dev)
        l_out = torch.zeros(B, N_ext, dtype=dt, device=dev)
        num_iters = torch.full((B,), -1, device=dev)
        converges = torch.zeros(B, dtype=torch.int64, device=dev)

        if track_osc:
            prev_hard = torch.zeros(B, N_ext, dtype=dt, device=dev)
            osc = torch.zeros(B, N_ext, dtype=torch.long, device=dev)
        else:
            osc = None

        i = 0
        for i in range(1, self.max_iter + 1):
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

            if track_osc:
                # e_v is written in place by the kernel, so prev_hard must be a copy.
                osc += (e_v != prev_hard).to(torch.long)
                prev_hard = e_v.clone()

            self._ext.convergence_update(
                s_est, syndrome, e_v, l_v, e_out, l_out, num_iters, converges, i
            )
            if not (num_iters == -1).any().item():
                break

        # Unconverged tail: snapshot final state, stamp the iteration reached.
        not_conv = num_iters == -1
        if not_conv.any().item():
            e_out[not_conv] = e_v[not_conv]
            l_out[not_conv] = l_v[not_conv]
            num_iters[not_conv] = i

        osc_out = osc[:, :N] if track_osc else None
        return e_out[:, :N], l_out[:, :N], converges, num_iters, osc_out

    def _bp_core_fused(self, syndrome, llr0):
        """Single fused kernel launch — used for the SF retries (no osc needed).

        Uncapped (conv_count=None): the kernel skips every rebatch_speedup cap branch,
        so it runs each sample to convergence or max_iter exactly like the reference."""
        dev, dt = self.device, self.dtype
        syndrome = syndrome.to(dev, dt).contiguous()
        B, M = syndrome.shape

        dummy_col = torch.full((B, 1), float("inf"), dtype=dt, device=dev)
        u_init = torch.cat([llr0.to(dev, dt), dummy_col], dim=1).contiguous()
        syndrome_neg_bc = torch.where(
            syndrome == 0.0, torch.ones_like(syndrome), -torch.ones_like(syndrome)
        )

        e_out = torch.zeros(B, self.N_ext, dtype=dt, device=dev)
        l_out = torch.zeros(B, self.N_ext, dtype=dt, device=dev)
        num_iters = torch.zeros(B, dtype=torch.int64, device=dev)
        converges = torch.zeros(B, dtype=torch.int64, device=dev)

        self._ext.bp_nms_fused(
            u_init,
            syndrome_neg_bc,
            self.V_c_col,
            self.VN_adj_c,
            self.VN_adj_k,
            e_out,
            l_out,
            num_iters,
            converges,
            self.N,
            self.VD,
            self.max_iter,
            self._block_size,
            None,  # conv_count=None ⇒ uncapped
            0,
        )
        return e_out[:, : self.N], l_out[:, : self.N], converges, num_iters, None
