"""
Stim-based error model.

Samples the fault mechanisms of a stim circuit's detector error model (DEM) and
provides the matching prior LLRs. The DEM's error instructions are the columns of the
`H` that ``matrix/stim`` hands the decoder, so a sampled mechanism vector `e` is a
ground-truth error *in the decoder's own coordinates*: ``syndrome/stim`` reads the
detectors straight off it as ``H @ e``, and a training loss can supervise against it.

Sampling is done with torch rather than stim's own DEM sampler so it runs off the
global torch RNG. That is what `MetricState`'s per-phase reseeding drives, so a training
run's error stream stays reproducible and a resumed run replays it. The two are
equivalent in distribution: a DEM's mechanisms are independent Bernoulli draws, and
stim's sampler XORs exactly the same supports together.

YAML config::

    error:
      model: stim_circuit
      circuit: <inline stim circuit string>
      number_channel: 1
      device:
        device_type: cpu

Training may sweep the noise instead of pinning it, mirroring `bsc`::

      rate: [0.001, 0.01]   # a range makes this a training-only model
      rate_points: 9

A range needs the circuit's *generation* parameters (``circuit_gen``), which the stim
interface passes through, since each rate point is a freshly generated circuit whose
DEM priors are re-extracted. Every configured noise key is scaled together so the
circuit's noise profile is preserved rather than flattened.
"""

import torch
from loguru import logger

from syndrilla.utils import dataset, is_rate_range, build_rate_sweep
from syndrilla.interface.stim.stim import get_stim_circuit


# Every noise knob `stim.Circuit.generated` accepts. Kept in one place so the sweep
# scales exactly the keys the circuit was built with and invents none.
NOISE_KEYS = (
    "after_clifford_depolarization",
    "after_reset_flip_probability",
    "before_measure_flip_probability",
    "before_round_data_depolarization",
)

# stim rejects a probability outside [0, 1] and a 0 makes the prior LLR infinite, so a
# scaled rate is held inside a range every noise instruction accepts.
_MIN_RATE = 1e-9
_MAX_RATE = 0.5


def _dem_of(circuit):
    """The undecomposed DEM, matching how `matrix/stim` builds `H`."""
    return circuit.detector_error_model(decompose_errors=False)


def _error_priors(dem):
    """Per-mechanism flip probability, one per column of `H`, in DEM order."""
    return [inst.args_copy()[0] for inst in dem.flattened() if inst.type == "error"]


def _error_supports(dem):
    """What each mechanism flips: `(detectors, observables)` per column, in DEM order.

    Two DEMs sharing this list share an `H` and an `L` exactly, which is what lets a
    noise sweep reuse one matrix bundle across every rate point.
    """
    supports = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dets, obs = [], []
        for tgt in inst.targets_copy():
            if tgt.is_relative_detector_id():
                dets.append(tgt.val)
            elif tgt.is_logical_observable_id():
                obs.append(tgt.val)
        supports.append((tuple(sorted(dets)), tuple(sorted(obs))))
    return supports


class create:
    def __init__(self, error_model_cfg, **kwargs) -> None:
        circuit_str = error_model_cfg.get("circuit", None)
        circuit = get_stim_circuit(circuit_str=circuit_str)

        # device
        device_cfg = error_model_cfg.get("device", {})
        self.device = device_cfg.get(
            "device_type",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        if self.device not in {
            "cuda",
            "cpu",
            torch.device("cuda"),
            torch.device("cpu"),
        }:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            device_idx = device_cfg.get("device_idx", 0)
            if device_idx >= torch.cuda.device_count():
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device(f"cuda:{device_idx}")

        self.number_channel = error_model_cfg.get("number_channel", 1)

        # extract DEM priors
        self.dem = _dem_of(circuit)
        priors = _error_priors(self.dem)
        self.num_errors = len(priors)
        self.priors = torch.tensor(priors, dtype=torch.float64)
        self._supports = _error_supports(self.dem)

        self.dtype = torch.float64
        self.rounds = 1
        # written by inject_error, read by get_llr
        self._shot_prior = None

        rate_cfg = error_model_cfg.get("rate", None)
        self.rate_is_range = is_rate_range(rate_cfg)
        if self.rate_is_range:
            self.rate = list(rate_cfg)
            self._build_rate_sweep(error_model_cfg, kwargs.get("training", False))
        else:
            # a scalar <rate> is the run's reported physical error rate, not a knob:
            # the circuit's own noise parameters are what actually generate the errors
            self.rate = float(rate_cfg) if rate_cfg is not None else self._avg_rate()
            self.rates = None
            self._prior_table = None

        logger.info(
            f"Stim error model ready: <{self.num_errors}> DEM mechanisms, "
            f"rate <{self.rate}>."
        )

    # ------------------------------------------------------------------
    # rates
    # ------------------------------------------------------------------
    @property
    def _prior_llr(self):
        """This circuit's per-mechanism prior LLR, at its own noise level.

        `get_llr` reports the rate each *shot* drew, which a sweep varies; this is the
        base circuit's, which is what a caller comparing against the DEM wants.
        """
        prior = self.priors.clamp(min=1e-12, max=1 - 1e-12)
        return torch.log((1 - prior) / prior)

    def _avg_rate(self) -> float:
        """Mean mechanism probability, the scalar a result file records."""
        return float(self.priors.mean()) if self.num_errors else 0.0

    def _build_rate_sweep(self, error_model_cfg, training):
        """Precompute one prior vector per rate point, all sharing this circuit's `H`.

        Generating a circuit per rate point is the only way to get *physical* priors at
        that noise level; scaling this circuit's priors directly would not, since a DEM
        probability is a combination of several noise instructions. Reusing one `H`
        across the points is checked rather than assumed: a rate point whose DEM has a
        different mechanism set is rejected, because the decoder is built from the base
        circuit's matrix and would silently be fed columns that no longer mean the same
        thing.
        """
        if self.number_channel != 1:
            raise ValueError(
                f"A swept <rate> needs <number_channel> 1, got <{self.number_channel}>."
            )
        # the range itself is validated the same way every swept rate in the framework
        # is, so a malformed one reads the same here as it does under <bsc>
        self.rates = build_rate_sweep(
            self.rate, error_model_cfg, training, "stim_circuit"
        )
        rate_points = self.rates.numel()

        gen_cfg = error_model_cfg.get("circuit_gen", None)
        if not gen_cfg:
            raise ValueError(
                "A <rate> range regenerates the circuit at each rate point, so it needs "
                "the circuit generation parameters (code, distance, rounds, noise), not "
                "an inline circuit string. Configure the stim interface with <code> and "
                "<distance> so they are known."
            )
        configured = {k: float(gen_cfg[k]) for k in NOISE_KEYS if k in gen_cfg}
        if not configured:
            raise ValueError(
                "A <rate> range scales the circuit's noise parameters, but none of "
                f'<{", ".join(NOISE_KEYS)}> is configured, so there is nothing to scale.'
            )
        # the scale is anchored on the mean configured rate, so a circuit whose knobs
        # were deliberately set to different values keeps their ratios; when they are
        # all equal, which is the usual case, a rate point simply sets every one to it
        reference = sum(configured.values()) / len(configured)

        table = []
        for rate in self.rates.tolist():
            point_cfg = dict(gen_cfg)
            scale = rate / reference
            for key, value in configured.items():
                point_cfg[key] = min(max(value * scale, _MIN_RATE), _MAX_RATE)
            dem = _dem_of(get_stim_circuit(circuit_cfg=point_cfg))
            if _error_supports(dem) != self._supports:
                raise ValueError(
                    f"Rate point <{rate}> gives a detector error model with a different "
                    f"set of error mechanisms than the base circuit "
                    f"(<{len(_error_supports(dem))}> vs <{self.num_errors}> columns, or "
                    f"a different support), so one H cannot cover the sweep. Narrow the "
                    f"<rate> range."
                )
            table.append(_error_priors(dem))
        self._prior_table = torch.tensor(table, dtype=torch.float64)
        logger.info(
            f"Stim rate sweep ready: <{rate_points}> points over <{self.rate}>, "
            f"sharing one H of <{self.num_errors}> columns."
        )

    # ------------------------------------------------------------------
    # error model interface
    # ------------------------------------------------------------------
    def inject_error(self, codeword, batch_size: int = 0):
        """Sample a DEM mechanism vector per shot, as `codeword` XOR the drawn faults.

        The result is the ground-truth error over the columns of `H`, so training reads
        its target from here and `syndrome/stim` derives the detectors from the same
        vector. Both halves therefore describe one draw, which is what a decode run's
        deferred-sample queue needs too: re-measuring a stored error reproduces its own
        syndrome rather than an unrelated fresh one.
        """
        logger.info(f"Injecting error.")
        if self.rounds > 1:
            raise ValueError(
                f"Error model <stim_circuit> needs <rounds> 1, got <{self.rounds}>; a "
                f"stim circuit's detectors already cover every QEC round."
            )
        codeword = codeword.to(self.device)
        if batch_size == 0:
            batch_size = codeword.size(0)
        self.dtype = codeword.dtype

        if codeword.size(-1) != self.num_errors:
            raise ValueError(
                f"Error model <stim_circuit> samples <{self.num_errors}> DEM mechanisms, "
                f"got a codeword of width <{codeword.size(-1)}>. The matrix and the error "
                f"model have to come from the same circuit."
            )

        shots = codeword.size(0)
        if self.rate_is_range:
            # one rate per shot, so a single run covers the whole noise curve
            idx = torch.randint(self.rates.numel(), (shots,), device=codeword.device)
            prior = self._prior_table.to(device=codeword.device, dtype=codeword.dtype)[
                idx
            ]
        else:
            prior = self.priors.to(device=codeword.device, dtype=codeword.dtype)
            prior = prior.unsqueeze(0).expand(shots, -1)
        self._shot_prior = prior

        random_values = torch.rand_like(codeword)
        error = torch.where(random_values < prior, 1 - codeword, codeword)

        dataloader = torch.utils.data.DataLoader(
            dataset(error, self.get_llr(error), torch.arange(0, shots)),
            batch_size=batch_size,
            shuffle=False,
        )
        logger.info(f"Injection complete.")
        return error, dataloader

    def get_llr(self, error):
        """Per-mechanism prior LLR, `log((1 - p) / p)`, at the rate each shot drew."""
        prior = self._shot_prior
        if prior is None:
            prior = self.priors.to(device=error.device, dtype=error.dtype)
            prior = prior.unsqueeze(0).expand(error.shape[0], -1)
        prior = prior.clamp(min=1e-12, max=1 - 1e-12)
        return torch.log((1 - prior) / prior).contiguous()
