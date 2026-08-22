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

        # qec_rounds: number of QEC rounds baked into the stim circuit. The
        # circuit's detectors already cover all rounds, so the syndrome output
        # has no separate rounds dimension. Exposed under a different name so
        # main.py's getattr(syndrome_generator, 'rounds', 1) returns 1 and
        # nothing downstream looks for a rounds axis that isn't there.
        self.qec_rounds = int(syndrome_cfg.get('rounds', 1))
        self.number_channel = int(syndrome_cfg.get('number_channel', 1))

        self.observable_flips = None
        self.syndrome_actual = None

        logger.info(
            f'Stim syndrome measurer ready: '
            f'{self.num_detectors} detectors (across {self.qec_rounds} QEC round(s)), '
            f'{self.num_observables} observables.'
        )

    def measure_syndrome(self, error, decoder):
        """
        Sample syndromes from the stim circuit.

        The stim detector vector already encodes every QEC round of the
        circuit, so the output is always 2-D:

            syndrome:        [B, num_detectors]
            observable_flips:[B, num_observables]
        """
        batch_size = error.shape[0]
        device = error.device

        logger.info(f'Sampling {batch_size} shots from stim circuit.')

        det_np, obs_np = self.sampler.sample(
            shots=batch_size, separate_observables=True,
        )

        syndrome = torch.from_numpy(det_np.astype('int64')).to(device)
        self.observable_flips = torch.from_numpy(obs_np.astype('uint8')).to(device)
        self.syndrome_actual = syndrome

        logger.info(f'Stim syndrome measurement complete.')
        return syndrome
