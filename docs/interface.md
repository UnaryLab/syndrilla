# Stim Interface

The stim interface enables circuit-level quantum error correction simulation by integrating with [Stim](https://github.com/quantumlib/Stim).
Instead of manually providing parity-check matrices, error rates, and syndrome measurement logic, the user defines a quantum circuit and the interface extracts everything automatically.

## Table of contents
- [Stim Interface](#stim-interface)
  - [Table of contents](#table-of-contents)
  - [Basic usage](#basic-usage)
    - [1. Run with command line arguments](#1-run-with-command-line-arguments)
    - [2. Input format and configurations](#2-input-format-and-configurations)
      - [2.1. Interface module](#21-interface-module)
      - [2.2. Decoder module](#22-decoder-module)
      - [2.3. Error module](#23-error-module)
      - [2.4. Syndrome module](#24-syndrome-module)
    - [3. Vote stage](#3-vote-stage)
    - [4. Phenomenological noise model](#4-phenomenological-noise-model)
  - [Pipeline flow](#pipeline-flow)

## Basic usage

### 1. Run with command line arguments
The stim workflow is driven by command-line arguments.
Below is an example command that runs a simulation using the Stim-generated surface code and BPOSD decoder:

```command
syndrilla -r=tests/test_outputs 
          -d=examples/stim/stim_generated.decoder.yaml 
          -i=examples/stim/stim_generated.interface.yaml 
          -e=examples/stim/stim_generated.error.yaml 
          -s=examples/stim/stim_generated.syndrome.yaml 
          -bs=1000 
          -te=100 
          -vs=syndrome
```

Following is a table for detailed explaination on each command line arguments:

| Argument | Description                                  | Example                                           |
|----------|----------------------------------------------|---------------------------------------------------|
| `-r`     | Path to store outputs                        | `-r=tests/test_outputs`                           |
| `-d`     | Path to decoder YAML file                    | `-d=examples/stim/stim_generated.decoder.yaml`    |
| `-i`     | Path to interface YAML file                  | `-i=examples/stim/stim_generated.interface.yaml`  |
| `-e`     | Path to error model YAML file                | `-e=examples/stim/stim_generated.error.yaml`      |
| `-s`     | Path to syndrome extraction YAML file        | `-s=examples/stim/stim_generated.syndrome.yaml`   |
| `-bs`    | Number of samples in each batch              | `-bs=1000`                                        |
| `-te`    | Total number of errors to stop decoding      | `-te=100`                                         |
| `-vs`    | Stage at which the majority vote is applied  | `-vs=syndrome`                                    |
| `-l`     | Level of logger                              | `-l=SUCCESS`                                      |

### 2. Input format and configurations
The stim workflow splits configuration across four modules: interface, decoder, error, and syndrome.
Each module has its own dedicated YAML configuration file.
The check and matrix YAML files used in the alist workflow are not needed here, since the stim circuit itself provides the parity-check matrix, logical observables, and LLR priors.

#### 2.1. Interface module
The interface YAML file defines the backend and the code structure.
The interface uses `code` and `distance` to generate the stim circuit and extract parity-check (H) and observable (L) matrices from its detector error model.
The number of syndrome measurement rounds in the generated circuit is taken from the syndrome module's `rounds` field (see section 2.4).
An example interface configuration file is provided in ```stim_generated.interface.yaml```:

```
interface:
  backend: stim
  code: surface_code:rotated_memory_x
  distance: 3
```

The following table details the configuration parameters used in the interface module YAML file.
| Key                   | Description                                                          | Example                          |
|-----------------------|----------------------------------------------------------------------|----------------------------------|
| `interface.backend`   | The quantum circuit simulator used                                   | `stim`                           |
| `interface.code`      | Stim code type passed to `stim.Circuit.generated()`                  | `surface_code:rotated_memory_x`  |
| `interface.distance`  | Code distance of the generated stim circuit                          | `3`                              |

The following table details all supported code types (from `stim.Circuit.generated`).

| Code type                            | Description                                          |
|--------------------------------------|------------------------------------------------------|
| `surface_code:rotated_memory_x`      | Rotated surface code, X memory experiment            |
| `surface_code:rotated_memory_z`      | Rotated surface code, Z memory experiment            |
| `surface_code:unrotated_memory_x`    | Unrotated surface code, X memory experiment          |
| `surface_code:unrotated_memory_z`    | Unrotated surface code, Z memory experiment          |
| `repetition_code:memory`             | Repetition code memory experiment                    |
| `color_code:memory_xyz`              | Color code memory experiment (requires `rounds >= 2`)|

#### 2.3. Error module
The error YAML file defines the noise parameters passed to `stim.Circuit.generated()` as circuit-level noise rates.
An example error configuration file using the stim circuit error model is provided in ```stim_generated.error.yaml```:

```
error:
  model: stim_circuit
  after_clifford_depolarization: 0.01
  after_reset_flip_probability: 0.01
  before_round_data_depolarization: 0.01
```

The following table details the configuration parameters used in the error module YAML file.
| Key                                      | Description                                                                 | Example          |
|------------------------------------------|-----------------------------------------------------------------------------|------------------|
| `error.model`                            | Type of error model applied to the stim circuit                             | `stim_circuit`   |
| `error.after_clifford_depolarization`    | Depolarizing noise applied after each Clifford gate                         | `0.01`           |
| `error.after_reset_flip_probability`     | Bit-flip noise applied after each reset operation                           | `0.01`           |
| `error.before_round_data_depolarization` | Depolarizing noise applied to data qubits before each syndrome round        | `0.01`           |

#### 2.4. Syndrome module
The syndrome YAML file defines the syndrome measurement settings.
`rounds` also dictates the number of QEC rounds used when the stim circuit is generated.
An example syndrome configuration file is provided in ```stim_generated.syndrome.yaml```:

```
syndrome:
  measure: stim
  rounds: 1
```

The following table details the configuration parameters used in the syndrome module YAML file.
| Key                 | Description                                                                                         | Example  |
|---------------------|-----------------------------------------------------------------------------------------------------|----------|
| `syndrome.measure`  | Model for syndrome measurement                                                                      | `stim`   |
| `syndrome.rounds` | Number of QEC rounds in the generated stim circuit and syndrome samples taken per error instance   | `1`      |

### 3. Vote stage
When `rounds > 1`, each round is an independent sample from the stim circuit.
The `-vs` flag controls where majority voting is applied across the rounds.

The following table details the options accepted by the `-vs` flag.
| `-vs` value  | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `syndrome`   | Vote on syndromes before decoding                                           |
| `decoder_0`  | Vote after the first decoder                                                |
| `decoder_1`  | Vote after the second decoder (e.g., after OSD)                             |
| `decoder`    | Vote after the last decoder                                                 |


## Pipeline flow

The stim circuit is generated jointly from all three YAMLs — `interface.yaml` supplies `code`/`distance`, `error.yaml` supplies the per-gate noise rates, and `syndrome.yaml` supplies `rounds` and (optionally) `measurement_error_rate`, which is forwarded to stim as `before_measure_flip_probability`.

```
interface.yaml     →  backend (stim), code, distance
error.yaml         →  noise rates (after_clifford_depolarization, ...)
syndrome.yaml      →  rounds, measurement_error_rate
                                           ↓
                                  stim.Circuit.generated()
                                           ↓
                                    stim circuit
                                    ├── H matrix (detectors × errors)
                                    ├── L matrix (observables × errors)
                                    ├── LLR priors (per-error probabilities)
                                    └── syndrome sampler → [B, d, M] syndromes
                                           ↓
                                    decode → logical check → metrics
```
