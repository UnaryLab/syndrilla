"""
Phenomenological syndrome measurer.

When the error has a rounds dimension [B, rounds, N] (e.g. from BSC with
rounds > 1), computes per-round syndrome directly and applies measurement
noise — no duplication needed.

When the error is 2-D [B, N] (a single round), computes one syndrome and
applies measurement noise.

YAML config::

    syndrome:
      measure: phenomenological
      measurement_error_rate: 0.01
"""

import torch
from loguru import logger


class create():

    def __init__(self, syndrome_cfg, **kwargs) -> None:
        self.rounds = int(syndrome_cfg.get('rounds', 1))
        self.measurement_error_rate = float(syndrome_cfg.get('measurement_error_rate', 0.0))
        self.observable_flips = None
        self.syndrome_actual = None

        logger.info(
            f'Phenomenological syndrome measurer ready: '
            f'{self.rounds} round(s), '
            f'measurement_error_rate={self.measurement_error_rate}.'
        )

    def measure_syndrome(self, error, decoder):
        logger.info(f'Measuring syndrome.')

        if error.ndim == 2:
            # [B, N] — single round. (rounds > 1 always arrives as a 3-D
            # [B, rounds, N] error from the BSC model, handled by the branch
            # below, so no per-round replication is needed here.)
            dummy_column = torch.zeros([error.shape[0], 1], dtype=error.dtype, device=error.device)
            error_ext = torch.cat((error, dummy_column), dim=1)

            v_c_col = decoder.V_c_col.to(error.device)
            syndrome = error_ext[:, v_c_col].sum(dim=2)
            syndrome = torch.where((syndrome % 2) > 0, 1, 0)

            self.syndrome_actual = syndrome
            logger.info(f'Syndrome measurement complete.')
            return self._apply_noise(syndrome)
        else:
            # [B, rounds, N] — error already carries per-round data
            dummy_column = torch.zeros([error.shape[0], error.shape[1], 1], dtype=error.dtype, device=error.device)
            error_ext = torch.cat((error, dummy_column), dim=2)

            v_c_col = decoder.V_c_col.to(error.device)
            syndrome = error_ext[:, :, v_c_col].sum(dim=3)
            syndrome = torch.where((syndrome % 2) > 0, 1, 0)

            self.syndrome_actual = syndrome
            logger.info(f'Syndrome measurement complete.')

            noisy = self._apply_noise(syndrome)
            logger.info(f'Phenomenological measurement complete: {self.rounds} rounds, output shape {list(noisy.shape)}.')
            return noisy

    def _apply_noise(self, syndrome):
        if self.measurement_error_rate <= 0:
            return syndrome
        flip_mask = torch.bernoulli(torch.full(syndrome.shape, self.measurement_error_rate,dtype=torch.float32, device=syndrome.device)).to(syndrome.dtype)
        return (syndrome + flip_mask) % 2

    def adjust_llr0(self, llr0):
        """Fold the measurement-error rate q into the per-data-qubit channel prior.

        A noisy syndrome bit looks like an apparent extra data flip, so the
        effective data-error probability is p_eff = p + q - 2*p*q.  The incoming
        prior encodes p (llr0 = log((1-p)/p)  =>  p = sigmoid(-llr0)); we inflate
        it to p_eff and rebuild the prior.  Elementwise, so it works for any llr0
        shape, and a zero rate leaves llr0 untouched."""
        q = self.measurement_error_rate
        if q <= 0:
            return llr0
        p = torch.sigmoid(-llr0)
        p_eff = p + q - 2 * p * q
        return torch.log((1 - p_eff) / p_eff)
