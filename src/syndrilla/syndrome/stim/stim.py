import torch
from loguru import logger

from syndrilla.interface.stim.stim import get_stim_circuit


class create:
    """Detectors and observable flips of an error the stim error model sampled.

    A detector error model *defines* a detector as the parity of the mechanisms that
    flip it, so the syndrome of a mechanism vector `e` is exactly ``H @ e`` over GF(2),
    and its observable flips are ``L @ e``. Reading both off the error the error model
    drew, rather than taking an independent draw from the circuit's own sampler, is what
    ties the three together: the syndrome the decoder sees, the observables it is scored
    against, and the ground truth a training loss supervises with all describe one shot.

    It is also what makes `measure_syndrome`'s `error` argument mean something. A decode
    run's deferred-sample queue re-measures errors it stored earlier; against a fresh
    circuit sample those would come back with an unrelated syndrome.
    """

    def __init__(self, syndrome_cfg, **kwargs) -> None:
        # imported here rather than at module scope: matrix/stim imports the stim
        # interface, which imports this package, and the cycle only closes at import time
        from syndrilla.matrix.stim.stim import _build_dem_matrices

        circuit_str = syndrome_cfg.get("circuit", None)
        self.circuit = get_stim_circuit(circuit_str=circuit_str)
        self.path = "<inline>"

        H, obs_mat, _ = _build_dem_matrices(self.circuit)
        self._H = torch.from_numpy(H)
        self._L = torch.from_numpy(obs_mat)

        self.num_detectors = self.circuit.num_detectors
        self.num_observables = self.circuit.num_observables

        self.qec_rounds = int(syndrome_cfg.get("rounds", 1))
        self.number_channel = int(syndrome_cfg.get("number_channel", 1))

        self.observable_flips = None
        self.syndrome_actual = None

        logger.info(
            f"Stim syndrome measurer ready: "
            f"{self.num_detectors} detectors (across {self.qec_rounds} QEC round(s)), "
            f"{self.num_observables} observables."
        )

    def measure_syndrome(self, error, decoder):
        """
        Detectors and observable flips of `error`, a DEM mechanism vector.

        The stim detector vector already encodes every QEC round of the
        circuit, so the output is always 2-D:

            syndrome:        [B, num_detectors]
            observable_flips:[B, num_observables]
        """
        device = error.device
        logger.info(f"Measuring stim syndrome for {error.shape[0]} shots.")

        if error.ndim != 2:
            raise ValueError(
                f"Stim syndromes are measured on a [batch, mechanisms] error, got shape "
                f"<{tuple(error.shape)}>; a circuit's detectors already cover every round."
            )
        if error.shape[1] != self._H.shape[1]:
            raise ValueError(
                f"This circuit has <{self._H.shape[1]}> error mechanisms, got an error of "
                f"width <{error.shape[1]}>; both must come from the same circuit."
            )

        # float rather than int: CUDA has no integer matmul. The parity is still exact,
        # since a row sum counts at most the circuit's fault mechanisms and float32
        # represents every integer below 2^24 exactly
        e = error.to(dtype=torch.float32)
        H = self._H.to(device=device, dtype=torch.float32)
        L = self._L.to(device=device, dtype=torch.float32)

        syndrome = ((e @ H.t()) % 2).to(torch.int64)
        self.observable_flips = ((e @ L.t()) % 2).to(torch.uint8)
        self.syndrome_actual = syndrome

        logger.info("Stim syndrome measurement complete.")
        return syndrome
