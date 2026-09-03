import torch
from loguru import logger

from syndrilla.decoder.bp4.bp4 import create as _BP4Py


class create(_BP4Py):
    """Quaternary BP on CUDA. PyTorch-on-CUDA port (custom kernel pending)."""

    def __init__(self, decoding_cfg: dict, **kwargs) -> None:
        super().__init__(decoding_cfg, **kwargs)
        if not torch.cuda.is_available():
            raise RuntimeError("bp4_cuda requires a CUDA GPU.")
        if str(self.device) == "cpu":
            logger.warning("bp4_cuda configured on cpu; forcing cuda:0.")
            self.device = torch.device("cuda:0")
        self.algo = "bp4"
        logger.info(
            "bp4_cuda ready (PyTorch-on-CUDA; custom quaternary kernel still pending)."
        )
