"""
Stim-based error model (setup only).

Reads a stim circuit and provides LLR priors from its detector error model.
Does NOT sample — sampling is handled by the stim syndrome measurer
(``syndrome/stim``).  ``inject_error`` returns dummy zero errors and the DEM
prior LLR so the decoder has physically-meaningful channel information.

YAML config::

    error:
      model: stim_circuit
      circuit: <inline stim circuit string>
      number_channel: 1
      device:
        device_type: cpu
"""

import math

import torch
from loguru import logger

from syndrilla.utils import dataset
from syndrilla.interface.stim.stim import get_stim_circuit


class create():

    def __init__(self, error_model_cfg, **kwargs) -> None:
        circuit_str = error_model_cfg.get('circuit', None)
        circuit = get_stim_circuit(circuit_str=circuit_str)

        # device
        device_cfg = error_model_cfg.get('device', {})
        self.device = device_cfg.get(
            'device_type',
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        )
        if self.device not in {'cuda', 'cpu', torch.device('cuda'), torch.device('cpu')}:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device == 'cuda':
            device_idx = device_cfg.get('device_idx', 0)
            if device_idx >= torch.cuda.device_count():
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device(f'cuda:{device_idx}')

        self.number_channel = error_model_cfg.get('number_channel', 1)

        # extract DEM priors
        self.dem = circuit.detector_error_model(decompose_errors=False)
        self.num_errors = sum(1 for inst in self.dem.flattened() if inst.type == 'error')
        self._prior_llr = self._build_prior_llr()

        self.dtype = torch.float64
        self.rate = float(error_model_cfg.get('rate', self._estimate_avg_rate()))


    # ------------------------------------------------------------------
    # LLR from DEM
    # ------------------------------------------------------------------
    def _estimate_avg_rate(self) -> float:
        total_p, n = 0.0, 0
        for inst in self.dem.flattened():
            if inst.type == 'error':
                total_p += inst.args_copy()[0]
                n += 1
        return total_p / max(n, 1)

    def _build_prior_llr(self) -> torch.Tensor:
        """Per-error-mechanism LLR: log((1-p)/p) for each column of H."""
        priors = []
        for inst in self.dem.flattened():
            if inst.type != 'error':
                continue
            p = inst.args_copy()[0]
            p = min(max(p, 1e-12), 1 - 1e-12)
            priors.append(math.log((1 - p) / p))
        return torch.tensor(priors, dtype=torch.float64)


    # ------------------------------------------------------------------
    # error model interface
    # ------------------------------------------------------------------
    def inject_error(self, codeword, batch_size: int = 0):
        """
        Return dummy zero errors and DEM-derived LLR priors.

        The actual error sampling is done by the stim syndrome measurer
        (``syndrome/stim``). This method exists so main.py's pipeline gets
        the LLR it needs without any code changes.
        """
        codeword = codeword.to(self.device)
        if batch_size == 0:
            batch_size = codeword.size(0)
        self.dtype = codeword.dtype

        # dummy error — zeros of length num_errors (the DEM column count)
        error = torch.zeros([batch_size, self.num_errors], dtype=self.dtype, device=self.device)

        llr = self._prior_llr.to(self.device).to(self.dtype)
        llr = llr.unsqueeze(0).expand(batch_size, -1).contiguous()

        ds = dataset(error, llr, torch.arange(batch_size))
        dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
        return error, dataloader

    def get_llr(self, error):
        llr = self._prior_llr.to(self.device).to(self.dtype)
        return llr.unsqueeze(0).expand(error.shape[0], -1).contiguous()
