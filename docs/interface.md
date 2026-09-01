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
      - [2.2. Error module](#22-error-module)
      - [2.3. Syndrome module](#23-syndrome-module)
    - [3. Training on the stim path](#3-training-on-the-stim-path)
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
| `-l`     | Level of logger, default `INFO`              | `-l=SUCCESS`                                      |
| `-ckpt`  | Path to a checkpoint YAML file to resume a decode run; a training run resumes with `-tckpt` instead, and passing `-ckpt` with `-t` is rejected rather than ignored. The stim path derives its physical error rate from the circuit's DEM, so the filename reflects that value rather than a rate you set | `-ckpt=<run dir>/result_phy_err_<rate>.yaml` |

`-i` derives the matrix and logical-check matrices from the circuit, so `-m` and `-c` are not used. It also makes `-e` and `-s` optional, leaving `-d` as the only required flag; supply them anyway when you want to set noise rates or rounds, since both are still read when present. Training (`-t`) works on the interface path too, and needs `-ls` for the objective, which the interface does not supply; see [decoder.md](decoder.md) for the training flags and [Training on the stim path](#3-training-on-the-stim-path) below.

### 2. Input format and configurations
The stim workflow splits configuration across four modules: interface, decoder, error, and syndrome.
Each module has its own dedicated YAML configuration file. The decoder YAML is unchanged by the stim path and is documented in [decoder.md](decoder.md); the sections below cover the three that differ.
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
| `interface.circuit`   | (optional) Inline stim circuit string, or a mapping of generation parameters, used instead of `code`/`distance` | `<stim circuit string>` |
| `interface.number_channel` | (optional) Fallback channel count when `error.number_channel` is absent | `1`                       |

The device and dtype of the run come from the **decoder** YAML, not from this file.

The following table details all supported code types (from `stim.Circuit.generated`).

| Code type                            | Description                                          |
|--------------------------------------|------------------------------------------------------|
| `surface_code:rotated_memory_x`      | Rotated surface code, X memory experiment            |
| `surface_code:rotated_memory_z`      | Rotated surface code, Z memory experiment            |
| `surface_code:unrotated_memory_x`    | Unrotated surface code, X memory experiment          |
| `surface_code:unrotated_memory_z`    | Unrotated surface code, Z memory experiment          |
| `repetition_code:memory`             | Repetition code memory experiment                    |
| `color_code:memory_xyz`              | Color code memory experiment (requires `rounds >= 2`)|

#### 2.2. Error module
The error YAML file defines the noise parameters passed to `stim.Circuit.generated()` as circuit-level noise rates.
An example error configuration file using the stim circuit error model is provided in ```stim_generated.error.yaml```:

```
error:
  model: stim_circuit
  after_clifford_depolarization: 0.1
  after_reset_flip_probability: 0.1
  before_round_data_depolarization: 0.1
```

The following table details the configuration parameters used in the error module YAML file.
| Key                                      | Description                                                                 | Example          |
|------------------------------------------|-----------------------------------------------------------------------------|------------------|
| `error.model`                            | Type of error model applied to the stim circuit                             | `stim_circuit`   |
| `error.after_clifford_depolarization`    | Depolarizing noise applied after each Clifford gate                         | `0.1`            |
| `error.after_reset_flip_probability`     | Bit-flip noise applied after each reset operation                           | `0.1`            |
| `error.before_round_data_depolarization` | Depolarizing noise applied to data qubits before each syndrome round        | `0.1`            |
| `error.before_measure_flip_probability`  | Bit-flip noise applied before each measurement. `syndrome.measurement_error_rate` overrides it, with a warning | `0.1` |

These four keys are consumed by the **interface** when it builds the circuit, not by the error model itself, so this YAML is only usable together with `-i`. They apply only when the circuit is generated from `interface.code`/`interface.distance`: if `interface.circuit` supplies the circuit instead, it carries its own noise and these rates, along with `syndrome.measurement_error_rate`, are ignored.

#### 2.3. Syndrome module
The syndrome YAML file defines the syndrome measurement settings.
`rounds` also dictates the number of QEC rounds used when the stim circuit is generated.
An example syndrome configuration file is provided in ```stim_generated.syndrome.yaml```:

```
syndrome:
  measure: stim
  rounds: 3
  measurement_error_rate: 0.1
```

The following table details the configuration parameters used in the syndrome module YAML file.
| Key                 | Description                                                                                         | Example  |
|---------------------|-----------------------------------------------------------------------------------------------------|----------|
| `syndrome.measure`  | Model for syndrome measurement                                                                      | `stim`   |
| `syndrome.rounds` | Number of QEC rounds baked into the generated stim circuit                                          | `3`      |
| `syndrome.measurement_error_rate` | Bit-flip noise before each measurement, forwarded to stim as `before_measure_flip_probability` | `0.1` |

The circuit's detectors already span every round, so the sampled syndrome carries **no** rounds axis: one shot per batch element, shaped `[B, num_detectors]`, whatever `rounds` is set to. The stim measurer therefore exposes the value as `qec_rounds` rather than `rounds`, and the error model's round count stays `1`.

### 3. Training on the stim path
`-t` drives a learned decoder from circuit-level data exactly as it does from a code's parity-check matrix, with `-i` in place of `-m`, and `-ls` naming the loss. The interface supplies the matrices, the noise and the measurer; it does not supply the objective.

```command
syndrilla -t -r=tests/test_outputs \
    -d=examples/stim/train_stim_saq.decoder.yaml \
    -i=examples/stim/stim_generated.interface.yaml \
    -e=examples/stim/stim_train.error.yaml \
    -s=examples/stim/stim_train.syndrome.yaml \
    -ls=examples/stim/logical_centric.loss.yaml \
    -bs=1024
```

**Choosing the loss.** `logical_centric` is the same module and the same weights the alist path uses; the stim path ships its own copy of the YAML only because this is where one of its terms is delicate. `L_Ent` takes the parity of the residual over the logical observable's support, which is a handful of qubits on a code but 36 error mechanisms on the shipped distance-3 circuit's detector error model. Written in the probability domain, as a product of one `+-1` Bernoulli mean per bit, that parity decays geometrically in the support's size: on this circuit the term reported a constant `ln 2 = 0.6931` for every epoch of a 40-epoch run, with a gradient of **exactly** zero, and since `L_Ent` is the only term that supervises the per-mechanism llr at all, that llr stayed at 0, which is itself what drove the product down. The term is computed in the log domain instead, its magnitude by the same max-log (min-sum) rule belief propagation's check node uses, so its gradient is O(1) on the least certain bit rather than exponentially small in the support.

Measured on the shipped configuration, comparing that fix against weighting the broken term to `0`: `L_Ent` now trains, `0.288` down to `0.018`, and the llr's logical parity agrees with the true logical class on `99.4%` of shots against `92.4%`. The end-to-end logical error rate is unchanged, `0.0081` against `0.0084` over 500 errors, which is within the sampling noise of those runs: `L_LC` already supervises the logical class directly, and CPND projects onto it, so the llr's parity is not what the final answer turns on. The term is left at `1.0` because it now does what it says, not because it moves the decoder.

**Where the supervision comes from.** A training step needs a ground-truth error in the decoder's own coordinates, and a detector error model provides one: its error instructions *are* the columns of `H`, so a sampled mechanism vector `e` is the error the decoder is trying to recover, `H @ e` is the syndrome it sees, and `L @ e` is the observable flip it is scored on. The error model draws `e` and the measurer reads the other two off it, so all three describe one shot. Sampling is done with torch rather than stim's own DEM sampler so it runs off the global torch RNG, which is what the per-phase reseeding behind the resume guarantee drives; the two are equivalent, since a DEM's mechanisms are independent Bernoulli draws either way.

**Sweeping the noise.** Training against one noise level gives a decoder that holds at that level. `rate` may instead be a `[lower, upper, points]` range, the same swept form `bsc` takes: every shot draws its own level, so one run covers a stretch of the curve. Each rate point regenerates the circuit with the configured noise keys scaled together, so a circuit whose knobs were deliberately set to different values keeps their ratios; when they are equal, which is usual, a rate point simply sets all of them to it. The points share one `H`, which is checked rather than assumed: a rate point whose DEM has a different mechanism set is rejected instead of being fed to a decoder built from the base circuit's matrix. A range is training-only, and a decode run against one is refused, because a result file records a single physical error rate.

| Key                 | Description                                                                                     | Example            |
|---------------------|-------------------------------------------------------------------------------------------------|--------------------|
| `error.rate`        | Scalar: the physical error rate a result file records, defaulting to the DEM's mean mechanism probability. `[lower, upper, points]`: a training-only sweep of the circuit's noise over that many evenly spaced levels | `[0.0005, 0.006, 9]` |

**Checkpoint names.** Weights trained here are named `<algorithm>_<check_type>_dem<detectors>x<mechanisms>`, not after a code distance: a DEM column is a circuit fault mechanism, so the column count carries no distance, and reading one off it would be a coincidence. The shipped distance-3 rotated circuit over 3 rounds gives 24 detectors and 221 mechanisms, so a `saq` run on `hx` writes `saq_hx_dem24x221_best.pt` and `saq_hx_dem24x221_last.pt`.

**What a decoder can assume.** A decoder built from a DEM is not looking at a code's Tanner graph: its checks are detectors in spacetime and its variables are fault mechanisms, so anything keyed to a code family, a distance or a qubit count does not apply. Decoders that measure such things off their matrix are told which kind they have through the loader's `is_circuit_dem` flag, rather than guessing from the shape.



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
                                    └── syndrome sampler → [B, M] syndromes
                                           ↓
                                    decode → logical check → metrics
```

 **Tanner-graph node convention:** In the H/L matrices the loader hands the decoder, **the stabilizer (detector) axis is the check-node axis of the Tanner graph, and the error-mechanism axis is the variable-node axis** — `H` is built as `[num_detectors, num_errors]`, the conventional `H[checks, variables]` layout used in classical LDPC literature. The matrix already arrives in the orientation the decoder expects, so no transpose is needed downstream.
