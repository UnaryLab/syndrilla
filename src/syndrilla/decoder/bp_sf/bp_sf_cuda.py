import torch
from loguru import logger

from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum_cuda import (
    PERSISTENT_THREADS,
    create as _BaseCuda,
)
from syndrilla.decoder.bp_sf.bp_sf import create as _BpSfPy


class create(_BaseCuda):
    """bp_sf on CUDA kernels. Accepts every bp_sf key (check_type, max_iter, dtype,
    the ``sf:`` block). The main pass runs the bp_norm_min_sum_cuda per-step loop
    and counts oscillations in _iter_hook; the SF retries run on bp_nms_persistent,
    or on the per-step loop with force_per_step or when no persistent block fits.
    bp_sf never applies the rebatch cap."""

    _init_sf = _BpSfPy._init_sf
    _sf_postprocess = _BpSfPy._sf_postprocess

    def __init__(self, decoding_cfg: dict, **kwargs) -> None:
        super().__init__(decoding_cfg, **kwargs)
        self._init_sf(decoding_cfg)

        self.cap = None

        self.algo = "bp_sf"

    def _iter_hook(self, i, l_v, e_v, active, syndrome) -> None:
        """Per bit, count how often the hard decision differs from the previous
        iteration, starting from an all-zero prior. Retries count nothing."""
        if getattr(self, "_osc", None) is None:
            return
        self._osc += e_v != self._prev_hard
        self._prev_hard.copy_(e_v)

    def forward(self, io_dict: dict) -> dict:
        """Normalized min-sum BP with oscillation counts, then SF post-processing
        on the samples BP left unconverged."""
        syndrome = io_dict["synd"].to(dtype=self.dtype, device=self.device)
        llr0 = io_dict["llr0"].to(dtype=self.dtype, device=self.device)
        B = syndrome.shape[0]
        self._osc = torch.zeros(B, self.N_ext, dtype=torch.long, device=self.device)
        self._prev_hard = torch.zeros(
            B, self.N_ext, dtype=torch.uint8, device=self.device
        )
        super().forward(io_dict)
        osc = self._osc[:, : self.N]
        del self._osc, self._prev_hard

        if self.w_max > 0:
            self._sf_postprocess(
                syndrome,
                llr0,
                io_dict["e_v"],
                io_dict["llr"],
                io_dict["converge"],
                io_dict["iter"],
                osc,
            )
        return io_dict

    def _bp_core(self, syndrome, llr0, track_osc=False):
        """SF retry decode in one uncapped bp_nms_persistent launch, or on the
        per-step loop with force_per_step or when no persistent block fits. Returns
        ``(e_out, l_out, converges, num_iters, None)`` over the real N bits."""
        if self._force_per_step or self._max_coresident == 0:
            out = _BaseCuda.forward(self, {"synd": syndrome, "llr0": llr0})
            return out["e_v"], out["llr"], out["converge"], out["iter"], None
        dev, dt = self.device, self.dtype
        syndrome = syndrome.to(dev, dt).contiguous()
        B = syndrome.shape[0]

        dummy_col = torch.full((B, 1), float("inf"), dtype=dt, device=dev)
        u_init = torch.cat([llr0.to(dev, dt), dummy_col], dim=1).contiguous()
        syndrome_neg_bc = torch.where(
            syndrome == 0.0, torch.ones_like(syndrome), -torch.ones_like(syndrome)
        )

        b_c2v = torch.zeros(B, self.nnz + 1, dtype=dt, device=dev)
        l_v = torch.empty(B, self.N_ext, dtype=dt, device=dev)
        e_v = torch.empty(B, self.N_ext, dtype=torch.uint8, device=dev)
        num_iters = torch.zeros(B, dtype=torch.int64, device=dev)
        converges = torch.zeros(B, dtype=torch.int64, device=dev)

        self._ext.bp_nms_persistent(
            u_init,
            syndrome_neg_bc,
            syndrome,
            self.row_ptr,
            self.col,
            self.VN_eid,
            b_c2v,
            l_v,
            e_v,
            num_iters,
            converges,
            self.N,
            self.max_iter,
            PERSISTENT_THREADS,
            None,
            0,
        )
        return e_v[:, : self.N].to(dt), l_v[:, : self.N], converges, num_iters, None
