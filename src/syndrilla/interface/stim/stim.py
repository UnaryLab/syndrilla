import torch
from loguru import logger

from syndrilla.utils import parse_device_dtype
from syndrilla.decoder import create_decoder
from syndrilla.error_model import create_error_model
from syndrilla.syndrome import create_syndrome
from syndrilla.logical_check import create_check
from syndrilla.matrix import load_matrices


def get_stim_circuit(circuit_str=None, circuit_cfg=None):
    """
    Return a stim.Circuit from an inline string or generation parameters.

    ``circuit_cfg`` is a dict passed to ``stim.Circuit.generated()``::

        circuit_cfg:
          code: surface_code:rotated_memory_x
          distance: 3
          rounds: 1
          after_clifford_depolarization: 0.01
          after_reset_flip_probability: 0.01
          before_measure_flip_probability: 0.01
          before_round_data_depolarization: 0.01
    """
    try:
        import stim
    except ImportError as e:
        raise ImportError("stim is required. Install with `pip install stim`.") from e

    if circuit_str is not None and not isinstance(circuit_str, dict):
        logger.info(f"Loading inline stim circuit.")
        return stim.Circuit(circuit_str)
    elif circuit_cfg is not None or isinstance(circuit_str, dict):
        cfg = circuit_cfg if circuit_cfg is not None else circuit_str
        code = cfg["code"]
        distance = int(cfg["distance"])
        rounds = int(cfg.get("rounds", 1))
        gen_kwargs = {}
        for noise_key in (
            "after_clifford_depolarization",
            "after_reset_flip_probability",
            "before_measure_flip_probability",
            "before_round_data_depolarization",
        ):
            if noise_key in cfg:
                gen_kwargs[noise_key] = float(cfg[noise_key])
        logger.info(
            f"Generating stim circuit: {code}, distance={distance}, rounds={rounds}."
        )
        return stim.Circuit.generated(
            code, distance=distance, rounds=rounds, **gen_kwargs
        )
    else:
        raise ValueError("get_stim_circuit requires 'circuit_str' or 'circuit_cfg'.")


class create:
    def __init__(self, interface_cfg, **kwargs) -> None:
        error_cfg = kwargs.get("error_cfg", {})
        syndrome_cfg = kwargs.get("syndrome_cfg", {})
        decoder_cfg = kwargs.get("decoder_cfg", {})
        # a build-time mode, passed on the way `main.py` passes it to create_decoder:
        # a training run supervises against the sampled error and never checks a
        # logical error, so the checker is not built
        training = kwargs.get("training", False)

        if decoder_cfg:
            device, dtype = parse_device_dtype(decoder_cfg)
            device_cfg = decoder_cfg.get(
                "device", {"device_type": device.type, "device_idx": device.index or 0}
            )
        else:
            device = torch.device("cpu")
            dtype = torch.float64
            device_cfg = {"device_type": "cpu", "device_idx": 0}

        number_channel = error_cfg.get(
            "number_channel", interface_cfg.get("number_channel", 1)
        )

        circuit_inline = interface_cfg.get("circuit", None)

        circuit_gen_cfg = None
        if isinstance(circuit_inline, dict):
            circuit_gen_cfg = circuit_inline
            circuit_inline = None
        elif "code" in interface_cfg:
            noise_keys = (
                "after_clifford_depolarization",
                "after_reset_flip_probability",
                "before_measure_flip_probability",
                "before_round_data_depolarization",
            )
            circuit_gen_cfg = {
                "code": interface_cfg["code"],
                "distance": interface_cfg["distance"],
                "rounds": int(syndrome_cfg.get("rounds", 1)),
            }
            for nk in noise_keys:
                if nk in error_cfg:
                    circuit_gen_cfg[nk] = error_cfg[nk]

            syn_meas_rate = syndrome_cfg.get("measurement_error_rate", None)
            if syn_meas_rate is not None:
                if isinstance(syn_meas_rate, (list, tuple)):
                    # on this path the measurement noise is a circuit generation
                    # parameter, and the sweep regenerates the circuit from <error.rate>
                    raise ValueError(
                        f"syndrome.measurement_error_rate=<{syn_meas_rate}> is a range, "
                        f"but on the stim path it sets the circuit's "
                        f"<before_measure_flip_probability>, which a swept <error.rate> "
                        f"already scales. Sweep <error.rate> instead and give this a "
                        f"scalar, or drop it."
                    )
                if "before_measure_flip_probability" in circuit_gen_cfg:
                    logger.warning(
                        f"syndrome.measurement_error_rate=<{syn_meas_rate}> overrides "
                        f"error.before_measure_flip_probability=<{circuit_gen_cfg['before_measure_flip_probability']}>."
                    )
                circuit_gen_cfg["before_measure_flip_probability"] = float(
                    syn_meas_rate
                )

        self.circuit = get_stim_circuit(
            circuit_str=circuit_inline, circuit_cfg=circuit_gen_cfg
        )

        logger.info(
            f"Stim interface: channels={number_channel}, device={device}, dtype={dtype}"
        )

        circuit_ref = {"circuit": str(self.circuit)}

        em_cfg = {
            "model": "stim_circuit",
            **circuit_ref,
            "number_channel": number_channel,
            "device": device_cfg,
        }
        if "rate" in error_cfg:
            em_cfg["rate"] = error_cfg["rate"]
        if "rate_points" in error_cfg:
            em_cfg["rate_points"] = error_cfg["rate_points"]
        # a swept <rate> regenerates the circuit per rate point, so the error model
        # needs the parameters it was generated from, not just the finished text
        if circuit_gen_cfg is not None:
            em_cfg["circuit_gen"] = circuit_gen_cfg
        self.error_model = create_error_model(cfg=em_cfg, training=training)

        rounds = syndrome_cfg.get("rounds", 1)

        syn_cfg = {"measure": "stim", "rounds": rounds, **circuit_ref}

        self.syndrome_generator = create_syndrome(cfg=syn_cfg, training=training)

        self.logical_check = (
            None if training else create_check(cfg={"check_type": "stim"})
        )

        matrix_base = {"file_type": "stim", **circuit_ref}
        self.matrix_cfg = {
            "parity_matrix_hx": {**matrix_base, "target": "check"},
            "parity_matrix_hz": {**matrix_base, "target": "check"},
            "logical_check_matrix": True,
            "logical_check_lx": {**matrix_base, "target": "observable"},
            "logical_check_lz": {**matrix_base, "target": "observable"},
        }
        self.matrix_bundle = load_matrices(self.matrix_cfg, device, dtype)

        if decoder_cfg and "algorithm" in decoder_cfg:
            self.decoders = create_decoder(
                cfg=decoder_cfg, bundle=self.matrix_bundle, training=training
            )
        else:
            self.decoders = None

        logger.info(f"Stim interface fully initialised.")
