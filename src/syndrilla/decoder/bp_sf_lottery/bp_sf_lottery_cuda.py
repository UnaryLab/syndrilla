import torch
from loguru import logger

from syndrilla.decoder.bp_lottery.bp_lottery_cuda import create as _BpLotteryCuda
from syndrilla.decoder.bp_sf.bp_sf import create as _BpSf


class create(_BpLotteryCuda):
    """CUDA lottery BP with syndrome-flipping retries on the per-step loop."""

    _init_sf = _BpSf._init_sf
    _sf_postprocess = _BpSf._sf_postprocess
    _config_algorithm = "bp_sf_lottery"
    _config_keys = ("sf", "topk", "w_min", "w_max", "n_sample")

    def __init__(self, decoding_cfg, **kwargs) -> None:
        super().__init__(decoding_cfg, **kwargs)
        self._init_sf(decoding_cfg)
        self.cap = None
        self._osc = None
        self.algo = "bp_sf_lottery"
        logger.info("bp_sf_lottery_cuda decoder ready.")

    def _iter_hook(self, i, l_v, e_v, active, syndrome) -> None:
        if self._osc is not None:
            self._osc += e_v != self._prev_hard
            self._prev_hard.copy_(e_v)
        super()._iter_hook(i, l_v, e_v, active, syndrome)

    def forward(self, io_dict: dict) -> dict:
        syndrome = io_dict["synd"].to(dtype=self.dtype, device=self.device)
        llr0 = io_dict["llr0"].to(dtype=self.dtype, device=self.device)
        B = syndrome.shape[0]
        self._osc = torch.zeros(B, self.N_ext, dtype=torch.long, device=self.device)
        self._prev_hard = torch.zeros(
            B, self.N_ext, dtype=torch.uint8, device=self.device
        )
        super().forward(io_dict)
        osc = self._osc[:, : self.N]
        self._osc = self._prev_hard = None

        io_dict["iter_serial"], io_dict["iter_parallel"] = self._sf_postprocess(
            syndrome,
            llr0,
            io_dict["e_v"],
            io_dict["llr"],
            io_dict["converge"],
            io_dict["iter"],
            osc,
        )
        return io_dict

    def _bp_core(self, syndrome, llr0):
        out = super().forward({"synd": syndrome, "llr0": llr0})
        return out["e_v"], out["llr"], out["converge"], out["iter"], None
