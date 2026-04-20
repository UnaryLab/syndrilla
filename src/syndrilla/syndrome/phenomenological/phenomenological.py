"""
Phenomenological syndrome measurer.

Computes the true syndrome (H*e), then replicates it ``d_rounds`` times
with independent random bit flips at rate ``measurement_error_rate``.

YAML config::

    syndrome:
      measure: phenomenological
      d_rounds: 3
      measurement_error_rate: 0.01
"""

import torch
from loguru import logger


class create():

    def __init__(self, syndrome_cfg, **kwargs) -> None:
        self.d_rounds = int(syndrome_cfg.get('d_rounds', 1))
        self.measurement_error_rate = float(syndrome_cfg.get('measurement_error_rate', 0.0))
        self.observable_flips = None
        self.syndrome_actual = None

        logger.info(
            f'Phenomenological syndrome measurer ready: '
            f'{self.d_rounds} round(s), '
            f'measurement_error_rate={self.measurement_error_rate}.'
        )

    def measure_syndrome(self, error, decoder):
        logger.info(f'Measuring syndrome.')

        if error.ndim == 2:
            dummy_column = torch.zeros([error.shape[0], 1], dtype=error.dtype, device=error.device)
            error_ext = torch.cat((error, dummy_column), dim=1)

            v_c_col = decoder.V_c_col.to(error.device)
            syndrome = error_ext[:, v_c_col].sum(dim=2)
            syndrome = torch.where((syndrome % 2) > 0, 1, 0)
        else:
            dummy_column = torch.zeros([error.shape[0], error.shape[1], 1], dtype=error.dtype, device=error.device)
            error_ext = torch.cat((error, dummy_column), dim=2)

            v_c_col = decoder.V_c_col.to(error.device)
            v_c_col_expanded = v_c_col.unsqueeze(0).expand(error.size(0), -1, -1, -1)
            error_expanded = error_ext.unsqueeze(2).expand(-1, -1, v_c_col.size(1), -1)
            syndrome = torch.gather(error_expanded, dim=3, index=v_c_col_expanded).sum(dim=3)
            syndrome = torch.where((syndrome % 2) > 0, 1, 0)

        self.syndrome_actual = syndrome
        logger.info(f'Syndrome measurement complete.')

        d = self.d_rounds
        if d <= 1:
            return self._apply_noise(syndrome)

        B = syndrome.shape[0]
        repeated = syndrome.unsqueeze(1).expand(*([B, d] + list(syndrome.shape[1:]))).clone()
        noisy = self._apply_noise(repeated)

        logger.info(f'Phenomenological measurement complete: {d} rounds, output shape {list(noisy.shape)}.')
        return noisy

    def _apply_noise(self, syndrome):
        if self.measurement_error_rate <= 0:
            return syndrome
        flip_mask = torch.bernoulli(torch.full(syndrome.shape, self.measurement_error_rate,dtype=torch.float32, device=syndrome.device)).to(syndrome.dtype)
        return (syndrome + flip_mask) % 2
