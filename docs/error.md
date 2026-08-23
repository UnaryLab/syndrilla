# Error module

The error YAML file defines the noise parameters used to inject errors on data qubits.
Syndrilla currently supports three error models: a Binary Symmetric Channel (BSC) and a Depolarizing channel for the alist-based workflow, plus a Stim circuit-level error model for the circuit-level workflow.
For `bsc` and `stim_circuit`, the `number_channel` field determines whether errors are injected as a single bit-flip stream (1-channel) or as two streams (2-channel) used by BP4-style decoders. `depol` is always 2-channel and ignores the key.
Every model takes its `rate` either as a scalar or, when training, as a swept range; see §2.

## 1. Binary symmetric channel
The BSC model independently flips each data qubit with probability `rate`.
It supports both 1-channel and 2-channel modes.
An example BSC error configuration file is provided in ```bsc.error.yaml```:

```
error:
  model: bsc
  number_channel: 1
  device:
    device_type: cpu
    device_idx: 0
  rate: 0.1
```

The following table details the configuration parameters used in the BSC error YAML file.
| Key                         | Description                                                                                          | Example          |
|-----------------------------|------------------------------------------------------------------------------------------------------|------------------|
| `error.model`               | Type of error model applied to data qubits                                                           | `bsc`            |
| `error.number_channel`      | Number of error channels (1 for pure bit-flip, 2 for independent X/Z channels)                       | `1` or `2`       |
| `error.device.device_type`  | Type of the device where the error injection will happen                                             | `cpu` or `cuda`  |
| `error.device.device_idx`   | Index of the device where the error injection will happen. Only used when `device_type = cuda`.      | `0`              |
| `error.rate`                | Physical bit-flip probability applied to each data qubit. A training run may sweep it, see §2         | `0.1`            |

In 1-channel mode, each qubit is flipped independently with probability `rate`, and the LLR prior is `log((1 - rate) / rate)`.
In 2-channel mode, X and Z streams are sampled independently with the same `rate`, and the Pauli priors passed to the decoder are `(p_I, p_X, p_Y, p_Z) = (1 - rate^2 - 2*rate, rate*(1-rate), rate*rate, rate*(1-rate))`.

The BSC model has no `rounds` field of its own — `error_model.rounds` is set from the syndrome config's `rounds` (see [syndrome.md](syndrome.md) §2). When `rounds > 1`, errors become **cumulative** across rounds rather than i.i.d. per round: each round independently flips qubits at `rate`, and round `t`'s error is the parity-sum of all flips up to `t`, so a flip persists until another flip on the same qubit clears it. The model then emits a per-round tensor `[B, rounds, N]` consumed directly by the phenomenological syndrome measurer. With `rounds == 1` it emits the usual `[B, N]`. This applies to 1-channel mode only: the 2-channel path ignores `rounds` and always emits `[B, 2, N]`.

## 2. Swept rates for training
Training a decoder against a single physical error rate gives a decoder that only holds at that rate. Any model's `rate` may instead be given as a `[lower, upper]` range with `rate_points`: the range is split into that many evenly spaced levels at construction, and every shot in the batch draws its own, so each row is flipped at its own rate and carries the matching prior. One training run then covers a stretch of the curve rather than a single point on it.

A range is the **training-only** form. A decode run records one physical error rate per result file, so building any model from a range without training refuses with an error naming the model and the key. Nothing about the scalar form changes: an existing decode configuration keeps sampling exactly the numbers it sampled before, and a model built from a scalar carries no sweep state at all.

The form is the same wherever a rate is configured, so what is written here also applies to `depol` (§3), to `stim_circuit` (§4, which regenerates its circuit per level) and to the phenomenological measurer's `measurement_error_rate` (see [syndrome.md](syndrome.md) §2).

An example configuration is provided in ```bsc_train.error.yaml```:

```
error:
  model: bsc
  number_channel: 1
  device:
    device_type: cpu
    device_idx: 0
  rate: [0.01, 0.20]
  rate_points: 9
```

The following table details the configuration parameters that differ from the scalar form above.
| Key                         | Description                                                                                          | Example          |
|-----------------------------|------------------------------------------------------------------------------------------------------|------------------|
| `error.rate`                | Physical error probability as a `[lower, upper]` range, with `0 < lower <= upper < 1`                | `[0.01, 0.20]`   |
| `error.rate_points`         | Number of evenly spaced levels in that range, one drawn per shot. Positive integer, required.        | `9`              |

The priors of a swept shot are the scalar ones evaluated at the level that shot drew: `log((1 - r) / r)` for a 1-channel BSC, constant along that shot's qubits, and the same Pauli formulas as the scalar form for a 2-channel model. With `rounds > 1` a shot keeps its level across its own rounds, so the flips stay cumulative in exactly the way §1 describes.

`syndrilla -t` is what turns the form on. It is passed as a build-time mode to `create_error_model` and `create_syndrome`, not as a key of the YAML, so a configuration file cannot switch itself into training.

## 3. Depolarizing error model
The depolarizing model draws a single uniform value per qubit and thresholds it twice: the first channel flips when the draw is below `2*rate/3`, the second when it is below `rate/3`. The two channels are therefore nested rather than independent, and the probability that a qubit carries any error is `2*rate/3`. The Pauli priors handed to the decoder are computed separately, from `rate` (see below), so they follow the textbook `rate/3`-per-Pauli form; the sampler and the priors do not describe the same channel.
This model is always 2-channel.
An example depolarizing error configuration file is provided in ```depol.error.yaml```:

```
error:
  model: depol
  device:
    device_type: cpu
    device_idx: 0
  rate: 0.05
```

The following table details the configuration parameters used in the depolarizing error YAML file.
| Key                         | Description                                                                                     | Example          |
|-----------------------------|-------------------------------------------------------------------------------------------------|------------------|
| `error.model`               | Type of error model applied to data qubits                                                      | `depol`          |
| `error.device.device_type`  | Type of the device where the error injection will happen                                        | `cpu` or `cuda`  |
| `error.device.device_idx`   | Index of the device where the error injection will happen. Only used when `device_type = cuda`.| `0`              |
| `error.rate`                | Total single-qubit depolarizing probability; split equally as X/Y/Z each at `rate/3`. A training run may sweep it, see §2 | `0.05`           |

The Pauli priors passed to the decoder are `(p_I, p_X, p_Y, p_Z) = (1 - rate, rate/3, rate/3, rate/3)`.

`rate` takes the training-only range of §2 as well, in ```depol_train.error.yaml```: every shot draws its own level, thresholds both of its channels against it, and carries the Pauli priors of that level. A decode run against a range is refused.

## 4. Stim circuit-level error model
The stim error model samples a stim circuit's detector error model (DEM) and derives the matching per-error-mechanism LLR priors. The DEM's error instructions are the columns of the `H` the decoder is built from, so a drawn mechanism vector is a ground-truth error in the decoder's own coordinates; the stim syndrome measurer reads the detectors and observable flips off that same vector (see [syndrome.md](syndrome.md) §3 and [interface.md](interface.md)).
An example error configuration file using the stim circuit-level model is provided in ```stim_generated.error.yaml```:

```
error:
  model: stim_circuit
  after_clifford_depolarization: 0.1
  after_reset_flip_probability: 0.1
  before_round_data_depolarization: 0.1
```

The following table details the configuration parameters used in the stim error YAML file.
| Key                                      | Description                                                                 | Example          |
|------------------------------------------|-----------------------------------------------------------------------------|------------------|
| `error.model`                            | Type of error model applied to the stim circuit                             | `stim_circuit`   |
| `error.after_clifford_depolarization`    | Depolarizing noise applied after each Clifford gate                         | `0.1`            |
| `error.after_reset_flip_probability`     | Bit-flip noise applied after each reset operation                           | `0.1`            |
| `error.before_round_data_depolarization` | Depolarizing noise applied to data qubits before each syndrome round        | `0.1`            |
| `error.before_measure_flip_probability`  | (optional) Bit-flip noise applied before each measurement. Not in the shipped example; `syndrome.measurement_error_rate` overrides it, with a warning | `0.1` |

The four noise rates are read by the **interface** when it builds the circuit, not by this error model, which reads only `circuit`, `device`, `number_channel` and an optional `rate`. A stim error YAML is therefore usable only together with `-i`; pointing `-e` at one without `-i` fails. The rates apply only when the circuit is generated from `interface.code`/`interface.distance`: a circuit supplied through `interface.circuit` carries its own noise and these rates are ignored.

For each error mechanism `i` in the DEM with probability `p_i`, the LLR prior passed to the decoder is `log((1 - p_i) / p_i)`, and `inject_error` flips mechanism `i` with probability `p_i`. The draw uses torch rather than stim's own DEM sampler so it follows the global torch RNG, which is what a training run's per-phase reseeding drives; the two are equivalent, since a DEM's mechanisms are independent Bernoulli draws either way.

`rate` doubles as the training-only sweep: given as a `[lower, upper]` range with `rate_points`, every shot draws its own noise level and its own priors, so one run covers a stretch of the curve rather than a single point. A range needs the circuit's generation parameters, which `-i` passes through, since each point regenerates the circuit. A decode run against a range is refused, the same way and by the same shared check `bsc` uses. See [interface.md](interface.md) §3.
