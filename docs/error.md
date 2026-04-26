# Error module

The error YAML file defines the noise parameters used to inject errors on data qubits.
Syndrilla currently supports three error models: a Binary Symmetric Channel (BSC) and a Depolarizing channel for the alist-based workflow, and a Stim circuit-level error model for the circuit-level workflow.
The `number_channel` field determines whether errors are injected as a single bit-flip stream (1-channel) or as separate X/Z streams (2-channel) used by BP4-style decoders.

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
  rate: 0.05
```

The following table details the configuration parameters used in the BSC error YAML file.
| Key                         | Description                                                                                          | Example          |
|-----------------------------|------------------------------------------------------------------------------------------------------|------------------|
| `error.model`               | Type of error model applied to data qubits                                                           | `bsc`            |
| `error.number_channel`      | Number of error channels (1 for pure bit-flip, 2 for independent X/Z channels)                       | `1` or `2`       |
| `error.device.device_type`  | Type of the device where the error injection will happen                                             | `cpu` or `cuda`  |
| `error.device.device_idx`   | Index of the device where the error injection will happen. Only used when `device_type = cuda`.      | `0`              |
| `error.rate`                | Physical bit-flip probability applied to each data qubit                                             | `0.05`           |

In 1-channel mode, each qubit is flipped independently with probability `rate`, and the LLR prior is `log((1 - rate) / rate)`.
In 2-channel mode, X and Z streams are sampled independently with the same `rate`, and the Pauli priors passed to the decoder are `(p_I, p_X, p_Y, p_Z) = (1 - rate^2 - 2*rate, rate*(1-rate), rate*rate, rate*(1-rate))`.

## 2. Deplorization error model
The depolarizing model assigns each Pauli error type (X, Y, Z) an equal probability of `rate/3`, so the total single-qubit error probability is `rate`.
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

## 3. Stim circuit-level error model
The stim error model derives per-error-mechanism LLR priors from a stim circuit's detector error model (DEM); error sampling itself is handled by the stim syndrome measurer (see [syndrome.md](syndrome.md) §3 and [interface.md](interface.md)).
An example error configuration file using the stim circuit-level model is provided in ```stim_generated.error.yaml```:

```
error:
  model: stim_circuit
  after_clifford_depolarization: 0.01
  after_reset_flip_probability: 0.01
  before_measure_flip_probability: 0.01
  before_round_data_depolarization: 0.01
```

The following table details the configuration parameters used in the stim error YAML file.
| Key                                      | Description                                                                 | Example          |
|------------------------------------------|-----------------------------------------------------------------------------|------------------|
| `error.model`                            | Type of error model applied to the stim circuit                             | `stim_circuit`   |
| `error.after_clifford_depolarization`    | Depolarizing noise applied after each Clifford gate                         | `0.01`           |
| `error.after_reset_flip_probability`     | Bit-flip noise applied after each reset operation                           | `0.01`           |
| `error.before_measure_flip_probability`  | Bit-flip noise applied before each measurement                              | `0.01`           |
| `error.before_round_data_depolarization` | Depolarizing noise applied to data qubits before each syndrome round        | `0.01`           |

For each error mechanism `i` in the DEM with probability `p_i`, the LLR prior passed to the decoder is `log((1 - p_i) / p_i)`. The actual sample errors are produced by the stim syndrome measurer, so `inject_error` returns dummy zero errors paired with the DEM-derived LLRs.
