import torch
from loguru import logger

from syndrilla.decoder.bp_lottery.bp_lottery import create as _BpLottery
from syndrilla.decoder.bp_sf.bp_sf import create as _BpSf


class create(_BpLottery):
    """Lottery BP with syndrome-flipping retries that also use lottery BP."""

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
        logger.info("bp_sf_lottery decoder ready.")

    def _iter_hook(self, i, l_v, e_v, active, syndrome) -> None:
        if self._osc is not None:
            rows = self._hook_rows
            self._osc[rows] += e_v != self._prev_hard[rows]
            self._prev_hard[rows] = e_v
        super()._iter_hook(i, l_v, e_v, active, syndrome)

    def forward(self, io_dict):
        syndrome = io_dict["synd"].to(dtype=self.dtype, device=self.device)
        llr0 = io_dict["llr0"].to(dtype=self.dtype, device=self.device)
        B = syndrome.shape[0]
        N_extended = self.H_shape[1] + 1
        self._osc = torch.zeros([B, N_extended], dtype=torch.long, device=self.device)
        self._prev_hard = torch.zeros([B, N_extended], dtype=self.dtype, device=self.device)
        super().forward(io_dict)
        osc = self._osc[:, :-1]
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
