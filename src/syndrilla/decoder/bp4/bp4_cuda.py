"""
bp4_cuda.py — CUDA decoder for bp4 (quaternary BP).

IMPORTANT — current status: this is a PyTorch-on-CUDA port, NOT a custom-kernel port.
Unlike the other *_cuda decoders (bp_lottery, bp_norm_min_sum_quant, relay_bp, …) which
reuse bp_norm_min_sum_cuda's fused/per-step kernels, bp4's recursion has no reuse path:
it is probability-domain quaternary BP (sigmoid err factors, scatter_reduce(prod) into a
4-class posterior, normalize_posterior, argmax hard-decision, 3D [B,2,M] syndrome) and
its cn_update uses torch.sgn (sign(0)=0), which differs from the binary log-domain
min-sum the existing kernels implement. A dedicated fused quaternary CUDA kernel is
required and must be developed/compiled/bit-tested on a GPU.

Until that kernel exists, this decoder runs bp4's existing quaternary BP as PyTorch
tensor ops on the CUDA device (scatter_reduce / sigmoid / argmax are already GPU ops),
with the rebatch_speedup cap active. It is therefore bit-identical to bp4 and provides the
cap on GPU, but NOT a custom-kernel speedup. The custom kernel is tracked as the
remaining work (see tests/test_bp4_cuda.py and the bp4_cuda GPU goal).

YAML algorithm key: bp4_cuda
"""

import torch
from loguru import logger

from syndrilla.decoder.bp4.bp4 import create as _BP4Py


class create(_BP4Py):
    """Quaternary BP on CUDA. PyTorch-on-CUDA port (custom kernel pending)."""

    def __init__(self, decoder_cfg: dict, **kwargs) -> None:
        super().__init__(decoder_cfg, **kwargs)
        if not torch.cuda.is_available():
            raise RuntimeError("bp4_cuda requires a CUDA GPU.")
        if str(self.device) == "cpu":
            logger.warning("bp4_cuda configured on cpu; forcing cuda:0.")
            self.device = torch.device("cuda:0")
        self.algo = "bp4"
        logger.info(
            "bp4_cuda ready (PyTorch-on-CUDA; custom quaternary kernel still pending)."
        )
