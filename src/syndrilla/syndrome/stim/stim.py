import torch
from loguru import logger

from syndrilla.interface.stim.stim import get_stim_circuit


class create():

    def __init__(self, syndrome_cfg, **kwargs) -> None:
        circuit_str = syndrome_cfg.get('circuit', None)
        self.circuit = get_stim_circuit(circuit_str=circuit_str)
        self.path = '<inline>'

        self.sampler = self.circuit.compile_detector_sampler()
        self.num_detectors = self.circuit.num_detectors
        self.num_observables = self.circuit.num_observables

        self.d_rounds = int(syndrome_cfg.get('d_rounds', 1))
        self.number_channel = int(syndrome_cfg.get('number_channel', 1))

        self.observable_flips = None
        self.syndrome_actual = None

        logger.info(
            f'Stim syndrome measurer ready: '
            f'{self.num_detectors} detectors, {self.num_observables} observables, '
            f'{self.d_rounds} round(s).'
        )

    def measure_syndrome(self, error, decoder):
        """
        Sample syndromes from the stim circuit.

        Returns:
            d_rounds == 1: syndrome [B, num_detectors]
            d_rounds >  1: syndrome [B, d_rounds, num_detectors]

        Side-effect:
            ``self.observable_flips`` is set with matching shape:
            d_rounds == 1: [B, num_observables]
            d_rounds >  1: [B, d_rounds, num_observables]
        """
        batch_size = error.shape[0]
        device = error.device
        d = self.d_rounds

        total_shots = batch_size * d
        logger.info(f'Sampling {total_shots} shots ({batch_size} x {d} rounds) from stim circuit.')

        det_np, obs_np = self.sampler.sample(
            shots=total_shots, separate_observables=True,
        )

        det = torch.from_numpy(det_np.astype('int64')).to(device)
        obs = torch.from_numpy(obs_np.astype('uint8')).to(device)

        if d > 1:
            syndrome = det.reshape(batch_size, d, self.num_detectors)
            self.observable_flips = obs.reshape(batch_size, d, self.num_observables)
        else:
            syndrome = det
            self.observable_flips = obs

        self.syndrome_actual = syndrome

        logger.info(f'Stim syndrome measurement complete.')
        return syndrome
