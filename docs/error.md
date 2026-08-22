# Error module

The error YAML file defines the noise parameters used to inject errors on data qubits.
Syndrilla currently supports four error models: a Binary Symmetric Channel (BSC), a swept-rate BSC used only for training, and a Depolarizing channel for the alist-based workflow, plus a Stim circuit-level error model for the circuit-level workflow.
For `bsc` and `stim_circuit`, the `number_channel` field determines whether errors are injected as a single bit-flip stream (1-channel) or as two streams (2-channel) used by BP4-style decoders. `bsc_train` requires 1, and `depol` is always 2-channel and ignores the key.

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
| `error.rate`                | Physical bit-flip probability applied to each data qubit                                             | `0.1`            |

In 1-channel mode, each qubit is flipped independently with probability `rate`, and the LLR prior is `log((1 - rate) / rate)`.
In 2-channel mode, X and Z streams are sampled independently with the same `rate`, and the Pauli priors passed to the decoder are `(p_I, p_X, p_Y, p_Z) = (1 - rate^2 - 2*rate, rate*(1-rate), rate*rate, rate*(1-rate))`.

The BSC model has no `rounds` field of its own — `error_model.rounds` is set from the syndrome config's `rounds` (see [syndrome.md](syndrome.md) §2). When `rounds > 1`, errors become **cumulative** across rounds rather than i.i.d. per round: each round independently flips qubits at `rate`, and round `t`'s error is the parity-sum of all flips up to `t`, so a flip persists until another flip on the same qubit clears it. The model then emits a per-round tensor `[B, rounds, N]` consumed directly by the phenomenological syndrome measurer. With `rounds == 1` it emits the usual `[B, N]`. This applies to 1-channel mode only: the 2-channel path ignores `rounds` and always emits `[B, 2, N]`.

## 2. Binary symmetric channel with a swept rate
The `bsc_train` model is `bsc` with one difference: `rate` is a `[lower, upper]` range instead of a scalar. The range is split into `rate_points` evenly spaced rates at construction, and every shot in the batch draws its own, so each row is flipped at its own rate and carries the matching LLR prior. Training a decoder against a single rate gives a model that only holds at that rate; sweeping lets one training run cover the whole curve.

This model exists for training (`syndrilla -t`) and is separate from `bsc` on purpose: `bsc` keeps its scalar `rate` and its decode behaviour unchanged, so no existing configuration is affected. Decode runs report one physical error rate per run and should keep using `bsc`.

An example configuration is provided in ```bsc_train.error.yaml```:

```
error:
  model: bsc_train
  number_channel: 1
  device:
    device_type: cpu
    device_idx: 0
  rate: [0.01, 0.20]
  rate_points: 9
```

The following table details the configuration parameters that differ from the BSC model above.
| Key                         | Description                                                                                          | Example          |
|-----------------------------|------------------------------------------------------------------------------------------------------|------------------|
| `error.model`               | Type of error model applied to data qubits                                                           | `bsc_train`      |
| `error.rate`                | Physical bit-flip probability as a `[lower, upper]` range, with `0 < lower <= upper < 1`             | `[0.01, 0.20]`   |
| `error.rate_points`         | Number of evenly spaced rates in that range, one drawn per shot. Positive integer, required.         | `9`              |

The LLR prior of a shot drawn at rate `r` is `log((1 - r) / r)`, constant along that shot's qubits. Sweeping only makes sense for a single bit-flip stream measured once, so `number_channel` other than `1`, a scalar `rate`, or `rounds > 1` are each rejected with an error rather than silently ignored.

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
| `error.rate`                | Total single-qubit depolarizing probability; split equally as X/Y/Z each at `rate/3`            | `0.05`           |

The Pauli priors passed to the decoder are `(p_I, p_X, p_Y, p_Z) = (1 - rate, rate/3, rate/3, rate/3)`.

## 4. Stim circuit-level error model
The stim error model derives per-error-mechanism LLR priors from a stim circuit's detector error model (DEM); error sampling itself is handled by the stim syndrome measurer (see [syndrome.md](syndrome.md) §3 and [interface.md](interface.md)).
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

For each error mechanism `i` in the DEM with probability `p_i`, the LLR prior passed to the decoder is `log((1 - p_i) / p_i)`. The actual sample errors are produced by the stim syndrome measurer, so `inject_error` returns dummy zero errors paired with the DEM-derived LLRs.
