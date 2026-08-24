<p align="center">
  <img src="https://raw.githubusercontent.com/UnaryLab/syndrilla/main/images/logo.png" width="150" />
</p>


# Syndrilla
A PyTorch-based numerical simulator for decoders in quantum error correction.

## Table of contents
- [Syndrilla](#syndrilla)
  - [Table of contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
    - [Option 1: pip installation](#option-1-pip-installation)
    - [Option 2: source installation](#option-2-source-installation)
  - [Basic usage](#basic-usage)
    - [1. Run with command line arguments](#1-run-with-command-line-arguments)
      - [Training a learned decoder (`-t`)](#training-a-learned-decoder--t)
    - [2. Input format and configurations](#2-input-format-and-configurations)
      - [2.1. Error module](#21-error-module)
      - [2.2. Syndrome module](#22-syndrome-module)
      - [2.3. Matrix module](#23-matrix-module)
      - [2.4. Decoder module](#24-decoder-module)
      - [2.5. Logical check module](#25-logical-check-module)
      - [2.6. Interface module](#26-interface-module)
      - [2.7. Metric module](#27-metric-module)
    - [3. Output format and metrics](#3-output-format-and-metrics)
      - [3.1. Per-decoder metrics](#31-per-decoder-metrics)
      - [3.2. Final metrics](#32-final-metrics)
    - [4. Resume from checkpoint](#4-resume-from-checkpoint)
    - [5. Sweep configurations](#5-sweep-configurations)
  - [Simulation results](#simulation-results)
    - [Comparison across GPUs](#comparison-across-gpus)
    - [Comparison across data formats](#comparison-across-data-formats)
    - [Comparison across distances](#comparison-across-distances)
    - [Comparison across batch sizes and against CPU](#comparison-across-batch-sizes-and-against-cpu)
  - [Citation](#citation)
  - [Contribution](#contribution)
  - [License](#license)

## Features
1. High modularity: easily customizing your own **decoding algorithms** and **error models**
2. High compatibility: cross-platform simulation on CPUs, **GPUs**, and even AI accelerators
3. High performance: showing **10-20X** speedup on GPUs over CPUs
4. Community focus: support for standard **BPOSD**, **BP4**, **MWPM**, and **Union-Find** decoders
5. Flexible data format: support for **FP16/BF16/FP32/FP64** simulation
6. Hardware awareness: support for **quantization** simulation
7. Fine-grained measurement: support for a broad range of metrics, with **degeneracy errors** highlighted
8. Multi-purpose: allowing researching **new codes, new decoders, new error models**, and beyond
9. Circuit-level simulation: support Stim for **circuit-level modeling**, enabling fair and reproducible benchmarking across different decoders and noise models

## Installation
All provided installation methods allow running ```syndrilla``` in the command line and ```import syndrilla``` as a python module.

Make sure you have [Anaconda](https://www.anaconda.com/) installed before the steps below.

### Option 1: pip installation
1. ```git clone``` [this repo](https://github.com/UnaryLab/syndrilla) and ```cd``` to the repo dir.
2. ```conda env create -f environment.yaml```
   - The ```name: syndrilla``` in ```environment.yaml``` can be updated to a preferred one.
3. ```conda activate syndrilla```
4. ```pip install syndrilla```
5. Validate installation via ```syndrilla -h``` in the command line or ```import syndrilla``` in python code
   - If you want to validate the simulation results against BPOSD, you need to change python to version 3.10. Then install [BPOSD](https://github.com/quantumgizmos/bp_osd) and run ```python tests/validate_bposd.py```

### Option 2: source installation
This is the developer mode, where you can edit the source code with live changes reflected for simulation.
1. ```git clone``` [this repo](https://github.com/UnaryLab/syndrilla) and ```cd``` to the repo dir.
2. ```conda env create -f environment.yaml```
   - The ```name: syndrilla``` in ```environment.yaml``` can be updated to a preferred one.
3. ```conda activate syndrilla```
4. ```python3 -m pip install -e . --no-deps```
5. Validate installation via ```syndrilla -h``` in the command line or ```import syndrilla``` in python code

## Basic usage

### 1. Run with command line arguments
Syndrilla simulation can be done via command-line arguments.
Below is an example command that runs a simulation using the BPOSD decoder:

```command
syndrilla -r=tests/test_outputs 
          -d=examples/alist/bposd_hx.decoder.yaml 
          -e=examples/alist/bsc.error.yaml 
          -c=examples/alist/lx.check.yaml 
          -s=examples/alist/perfect.syndrome.yaml 
          -m=examples/alist/surface_10.matrix.yaml 
          -bs=10000 
          -te=1000
```

Following is a table for detailed explaination on each command line arguments:

| Argument | Description                                  | Example                                           |
|----------|----------------------------------------------|---------------------------------------------------|
| `-r`     | Path to store outputs                        | `-r=tests/test_outputs`                           |
| `-d`     | Path to decoder YAML file                    | `-d=examples/alist/bposd_hx.decoder.yaml`    |
| `-e`     | Path to error model YAML file                | `-e=examples/alist/bsc.error.yaml`                |
| `-c`     | Path to check matrix YAML file               | `-c=examples/alist/lx.check.yaml`                 |
| `-s`     | Path to syndrome extraction YAML file        | `-s=examples/alist/perfect.syndrome.yaml`         |
| `-m`     | Path to matrix YAML file                     | `-m=examples/alist/surface_10.matrix.yaml`        |
| `-i`     | Path to interface YAML file, replacing `-m`/`-e`/`-s`/`-c` | `-i=examples/stim/stim_generated.interface.yaml` |
| `-ckpt`  | Path to checkpoint YAML file to resume | `-ckpt=tests/test_outputs/result_phy_err_0.1.yaml` |
| `-bs`    | Number of samples in each batch             | `-bs=10000`                                       |
| `-te`    | Total number of errors to stop decoding      | `-te=1000`                                         |
| `-l`     | Level of logger                              | `-l=SUCCESS`                                      |
| `-t`     | Train the decoder instead of decoding        | `-t`                                              |
| `-ls`    | Path to loss YAML file                       | `-ls=examples/alist/logical_centric.loss.yaml`    |
| `-tckpt` | Path to a run's `*_last.pt`, to resume training | `-tckpt=tests/test_outputs/saq_hx_d5_last.pt` |

#### Training a learned decoder (`-t`)

Learned decoders (currently `saq`) need trained weights. `-t` trains the decoder given by `-d` and writes its two checkpoints, its epoch history, its result file and its log into `-r` (default `tests/test_outputs`, the same directory decode runs write to), all five named after the configuration that produced them (`<algorithm>_<check_type>_<distance>`), so two configurations in one run directory do not overwrite each other:

```command
syndrilla -t
          -r=tests/test_outputs
          -d=examples/alist/saq_hx_train.decoder.yaml
          -m=examples/alist/surface_5.matrix.yaml
          -e=examples/alist/bsc_train.error.yaml
          -s=examples/alist/perfect.syndrome.yaml
          -ls=examples/alist/logical_centric.loss.yaml
          -bs=256
```

Then point the decoder YAML's `config.checkpoint` key at `tests/test_outputs/saq_hx_d5.pt` and run the normal command above to evaluate.

**The run's results (`<stem>_result.yaml`).** `-t` writes `tests/test_outputs/saq_hx_d5_result.yaml` alongside the checkpoints, the `-t` counterpart of a decode run's `result_phy_err_<rate>.yaml` and the same two-part shape: a `train_full` summary block naming the run (algorithm, `model parameters` (the decoder's trainable weight count, not a seed: the run's seed is its own `seed` key), device, dtype, physical error rate in the form it was configured, a swept range's point count included, batch size, schedule, seed, best validation class error and the epoch it came from, what the run cost, and the path to the best checkpoint), then the run's curve stored by column under `epoch`:

```yaml
epoch:
  epoch: [1, 2, 3]
  learning rate: [0.0005, 0.00037525, 0.00012575]
  time (s): [0.2678, 0.0831, 0.0930]
  best: [true, true, true]
  train:
    loss: [3.1953, 1.6084, 1.4831]
    lc: [2.3369, 0.7794, 0.6716]
    lp: [0.8178, 0.7677, 0.7406]
    ent: [0.6949, 0.6755, 0.6634]
    class error: [0.4219, 0.4648, 0.3203]
  val:
    loss: [1.5102, 1.5085, 1.5355]
    lc: [0.6705, 0.6832, 0.7212]
    lp: [0.7600, 0.7537, 0.7613]
    ent: [0.6877, 0.6746, 0.6621]
    class error: [0.3828, 0.3594, 0.34375]
```

One value per epoch in every list, all of them index-aligned with the `epoch` list, so entry `i` of each belongs to epoch `epoch[i]`. A term is then one line to read and one list to plot. It holds the same numbers as `<stem>_history.json`, in the form the rest of the toolchain already reads results in. It is rewritten at every epoch boundary, so a run stopped part way still leaves behind the epochs it finished.

The summary also reports what the run cost, the way a decode result file does:

```yaml
  total time (s): 340.09              # wall clock of this invocation, setup included
  total epoch time (s): 338.71        # summed over the epochs, restored across a resume
  average time per epoch (s): 3.3871
  average time per batch (s): 0.015396
  average time per sample (s): 6.0142e-05
```

The averages divide the summed epoch time, not the wall clock: the wall clock carries the decoder build and the matrix load, and after a `-tckpt` resume it covers only the epochs since the resume, while the epoch times are restored with the history. A batch is a batch of either phase, so an epoch holds `batches_per_epoch + val_batches` of them.

One value per epoch in every column is fine for a hundred-epoch run and unwieldy for a very long one, so the schedule takes an optional `epochs_saved` that caps it:

```yaml
    train:
      epochs: 100000
      batches_per_epoch: 200
      val_batches: 20
      seed: 42
      epochs_saved: 50     # write the last 50 epochs, plus the best one
```

The result yaml and `<stem>_history.json` then carry the most recent `epochs_saved` epochs plus the run's best epoch wherever it fell, since that is the epoch `<stem>.pt` holds and the summary's `best epoch` names. The summary reports both numbers separately: `epochs` is the schedule and `epochs saved` the cap. `<stem>_last.pt` is never thinned, so a resumed run still restores the whole curve and the cap can be raised or lowered between runs. Leave the key out and every epoch is written.

**Resuming an interrupted run (`-tckpt`).** The `*_last.pt` file is rewritten at every epoch boundary and holds the whole training state: weights, Adam's moments, the cosine schedule's position, the epoch counter, the best score so far, the history, and the generator state. The error stream is reseeded at each phase boundary from the run's `seed`: the training phase gets the same seed every epoch, so training runs on a fixed set of batches, while validation gets a fresh one each epoch. Either way a batch's errors depend on where the run is, not on how it got there. Add `-tckpt` to the same command to continue from it, leaving every other flag as it was:

```command
syndrilla -t -tckpt=tests/test_outputs/saq_hx_d5_last.pt
          -r=tests/test_outputs
          ... the same -d, -m, -e, -s, -ls and -bs as above
```

The run continues exactly where it stopped, not from a warm start: a run interrupted after epoch 20 and resumed finishes with the same weights a 100-epoch run would have reached uninterrupted. Because that guarantee depends on the settings being unchanged, the `*_last.pt` file also stores a fingerprint of them (schedule, batch size, code shape, optimizer settings), and resuming with any of them changed fails with a message naming the field rather than silently producing a different run. The plain `<name>.pt` stays a bare `state_dict` for decoding and sharing; only `*_last.pt` carries the extra state, and both still load through the decoder YAML's `config.checkpoint` key.

Training needs `-d`, `-m`, `-e`, `-s` and `-ls`, takes its batch size from `-bs`, and writes into `-r`. It builds no logical check, so `-c` and `-te` are unused, and `-ckpt` resumes a *decode* run and should not be passed alongside `-t` (`-tckpt` is its training counterpart). `-i` replaces `-m`/`-e`/`-s` here as it does for decoding, so a learned decoder trains on circuit-level data from a stim circuit with the same command and `-ls` still naming the loss; see [Interface module](docs/interface.md) for what supervises it and how its noise is swept. Training hyperparameters come from three files: the decoder YAML holds the optimizer settings (`lr`, `weight_decay`, `min_lr`) under `config.optimizer` and the schedule (`epochs`, `batches_per_epoch`, `val_batches`, `seed`, and the optional `epochs_saved`) under `config.train`, the `-ls` YAML holds the loss weights (`lambda_lc`, `lambda_lp`, `lambda_ent`) under its `loss` key, and the error YAML holds the physical error rates, where a training run may give `rate` as a range (`rate: [0.01, 0.20, 9]`, the last value being the number of levels) so one run covers the whole curve rather than a single point; the phenomenological measurer sweeps its `measurement_error_rate` the same way. Errors and syndromes come from the same error model and syndrome measurer that decoding uses.

### 2. Input format and configurations
<table>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/UnaryLab/syndrilla/main/images/modules.png" width="600">
    </td>
  </tr>
</table>

Syndrilla virtualizes the full decoder pipeline of data encoding, syndrome measurement, and error decoding into modules: error, syndrome, matrix, decoder, logical check, interface, loss, and metric, as shown in the figure above.
All configurations are defined through YAML files. 
Each module requires its own dedicated YAML configuration file, with the exception of the metric module. The loss module is used only for training and is selected with `-ls`.

#### 2.1. Error module
The error YAML file defines all configuration parameters associated with the error model. 
It currently supports a 1-channel Binary Symmetric Channel (BSC) error model, 2-channel error models for both depolarizing noise and BSC, a training-only swept-rate BSC, and a stim circuit-level model.
An example error configuration file using the Binary Symmetric Channel (BSC) model is provided in ```bsc.error.yaml```:

```
error:
  model: bsc
  number_channel: 1
  device: 
    device_type: cpu
    device_idx: 0
  rate: 0.1
``` 

The following table details the configuration parameters used in the error YAML file.
| Key              | Description                                                   | Example                   |
|------------------|---------------------------------------------------------------|---------------------------|
| `error.model`     | Type of quantum error model applied to data qubits           | `bsc`, `depol` or `stim_circuit` |
| `error.number_channel`     | The number of error channel applied to quantum circuit           | `1` or `2`                     |
| `error.device.device_type`       | Type of the device where the error injection will happen                                       | `cpu` or `cuda`                                       |
| `error.device.device_idx`       | Index of the device where the error injection will happen. This option only works when `device_type = cuda`.                                                        | 0                           |
| `error.rate`      | Physical error rate applied to each data qubit. A training run may sweep it as a `[lower, upper, points]` range, one level drawn per shot | `0.05` or `[0.01, 0.20, 9]` |

The following table details all types of error model Syndrilla supports. (Using different error model may need different configuration format, which will be shown on [Error module](docs/error.md).)

| Error Model      | Number of channels                                         |Example                                            |
|------------------|------------------------------------------------------------|---------------------------------------------------|
|Binary Symmetric Channel (BSC)|Both 1 and 2                                    | bsc                                               |
|Depolarizing Channel |2                                                        | depol                                             |
|Stim circuit-level model|1                                                    | stim_circuit                                      |

#### 2.2. Syndrome module
The syndrome YAML file defines all configuration parameters associated with the syndrome measurement.
An example configuration file that assumes ideal (error-free) syndrome measurements is provided in ```perfect.syndrome.yaml```:

```
syndrome:
  measure: perfect
```

The following table details the configuration parameters used in the syndrome module YAML file. 
| Key              | Description                                                   | Example                   |
|------------------|---------------------------------------------------------------|---------------------------|
| `syndrome.measure`| Model for syndrome measurement                       | `perfect`, `phenomenological`, or `stim`                    |

The following table details all types of syndrome measurement Syndrilla supports. (Using different syndrome measurement model may need different configuration format, which will be shown on [Syndrome module](docs/syndrome.md).)

| Syndrome model            | Description                                                                                                | Example            |
|---------------------------|------------------------------------------------------------------------------------------------------------|--------------------|
| Perfect                   | Ideal (error-free) syndrome measurement: returns `H * e mod 2`                                             | `perfect`          |
| Phenomenological          | Replicates the true syndrome over `rounds` and flips each bit with probability `measurement_error_rate`, which a training run may sweep as a range | `phenomenological` |
| Stim                      | Circuit-level syndrome sampler driven by a stim circuit (used with the stim interface)                     | `stim`             |


#### 2.3. Matrix module
The matrix YAML file defines all configuration parameters associated with the matrix processing.
Syndrilla accepts matrix from:
1. [.alist](https://www.inference.org.uk/mackay/codes/alist.html) format introduced by David MacKay, Matthew Davey, and John Lafferty, which contains a sparse matrix.
2. [.npz](https://numpy.org/doc/2.1/reference/generated/numpy.savez.html) format from NumPy, which contains a sparse matrix.
3. .txt format containing a dense 2D matrix. Each row represents a check node of the H matrix, in which each 1 entry denotes a connecting variable node to that check node.

A decoder consumes a bundle of matrices (Hx, Hz, and optionally Lx and Lz). These are combined into a single matrix YAML file, where each entry can be referenced as a file path or inlined as a config dict. An example combined matrix configuration is provided in ```surface_10.matrix.yaml```, which is the file passed via `-m` in the command above:

```
matrix:
  parity_matrix_hx:
    file_type: alist
    path: examples/alist/surface/surface_10_hx.alist
  parity_matrix_hz:
    file_type: alist
    path: examples/alist/surface/surface_10_hz.alist
  logical_check_matrix: True
  logical_check_lx:
    file_type: alist
    path: examples/alist/surface/surface_10_lx.alist
  logical_check_lz:
    file_type: alist
    path: examples/alist/surface/surface_10_lz.alist
```

The following table details the configuration parameters used in the combined matrix YAML file.
| Key                           | Description                                                                                            | Example                                      |
|-------------------------------|--------------------------------------------------------------------------------------------------------|----------------------------------------------|
| `matrix.parity_matrix_hx`     | Matrix entry (path or inline dict) for the X-type parity-check matrix                                  | `examples/alist/surface/surface_10_hx.alist` |
| `matrix.parity_matrix_hz`     | Matrix entry (path or inline dict) for the Z-type parity-check matrix                                  | `examples/alist/surface/surface_10_hz.alist` |
| `matrix.logical_check_matrix` | Flag for whether logical-check matrices are provided; if `False`, they are computed from Hx/Hz via `compute_lz` | `True` or `False`                            |
| `matrix.logical_check_lx`     | Matrix entry for the X-type logical-check matrix (used when `logical_check_matrix = True`)             | `examples/alist/surface/surface_10_lx.alist` |
| `matrix.logical_check_lz`     | Matrix entry for the Z-type logical-check matrix (used when `logical_check_matrix = True`)             | `examples/alist/surface/surface_10_lz.alist` |

The following table details all matrix formats Syndrilla supports. (Using different matrix formats may need different configuration format, which will be shown on [Matrix module](docs/matrix.md).)

| Matrix format | Description                                                                                                       | Example |
|---------------|-------------------------------------------------------------------------------------------------------------------|---------|
| alist         | Sparse format from MacKay/Davey/Lafferty; lists nonzero neighbor indices per check node                            | `alist` |
| npz           | NumPy/SciPy compressed archive containing a sparse parity-check matrix                                             | `npz`   |
| txt           | Plain-text dense 2D matrix; each row is a check node with `1`s marking connected variable nodes                    | `txt`   |
| stim          | Built directly from a stim circuit's detector error model; selects either the `check` (H) or `observable` (L) matrix | `stim`  |


#### 2.4. Decoder module
The decoder YAML file defines all configuration parameters associated with the decoder.
An example decoder configuration file is provided in ```bposd_hx.decoder.yaml```:

```
decoder:
  algorithm: [bp_norm_min_sum, osd_0]
  check_type: hx
  dtype: float64
  device: 
    device_type: cuda
    device_idx: 0
  config:
    max_iter: 181
``` 

The following table details the configuration parameters used in the decoder module YAML file.
| Key                   | Description                                                                  | Example                                            |
|------------------------|-----------------------------------------------------------------------------|-----------------------------------------------------|
| `decoder.algorithm`    | List of decoding algorithms used                                            | `[bp_norm_min_sum, osd_0]`                         |
| `decoder.check_type`   | Type of parity-check matrix used                                            | `hx` or `hz`                                       |
| `decoder.device.device_type`       | Type of the device where the decoding will happen                                       | `cpu` or `cuda`                                       |
| `decoder.device.device_idx`       | Index of the device where the decoding will happen. This option only works when `device_type = cuda`.                                      | 0                           |
| `decoder.dtype`        | Data type for decoding computations                                         | `float32`, `float64`                              |
| `decoder.force_pytorch`| (optional) Run the plain PyTorch module even on a CUDA device                | `false`                                            |
| `decoder.rebatch_speedup`| (optional) Adaptive batch-shrinking cap; see [Decoder module](docs/decoder.md) | `{kl_eps: 0.001}`                                |
| `decoder.config`       | Algorithm-specific settings (e.g. `max_iter`, or a learned decoder's `checkpoint`). A mapping configures the first algorithm; a list gives one entry per entry of `decoder.algorithm` | `max_iter: 181`             |

The keys above the last one are framework-wide and apply to the whole block; anything only one algorithm understands (`max_iter`, quantization widths, relay_bp's leg schedule) goes under `decoder.config`. Written as a plain mapping, as above, it configures the first algorithm, so `max_iter` reaches `bp_norm_min_sum` and `osd_0`, which takes no settings of its own, runs on its defaults. Written as a list it is matched to `decoder.algorithm` by position, which is how a chain configures a stage other than its first. Keys left at the old top level are rejected with a message naming the block they moved into. See [Decoder module](docs/decoder.md) for the full rule.

When `decoder.device.device_type` is set to `cuda`, every decoder automatically uses its CUDA-kernel implementation if a CUDA-capable GPU is present and the kernel is available; otherwise it falls back to the PyTorch implementation. This covers every registered decoder except `saq`: the BP family plus `osd_0`, `mwpm`, and `union_find`. For `osd_0`, `mwpm`, and `union_find` the CUDA output is bit-for-bit identical to the CPU implementation. Non-NVIDIA accelerators (e.g. AMD ROCm, IBM), where the CUDA kernels do not compile, automatically use the PyTorch implementation. See [Decoder module](docs/decoder.md) for details.

The following table details the different types of decoding algorithms Syndrilla supports. (Using different decoder may need different configuration format, which will be shown on [Decoder module](docs/decoder.md).)

| Decoding Algorithm                | #Channel                                          | Example                                            | Reference         |
|-----------------------------------|-------------------------------------------------------------|----------------------------------------------------|---------------------|
|Min-Sum Belief Propagation  (Min-Sum BP)| 1                                                           | bp_norm_min_sum                                    | Factor Graphs and the Sum-Product Algorithm |
|Branch-Assisted Sign-Flipping Belief Propagation (BSFBP) | 1                                     | bp_branch_assisted                                 | Branch-Assisted Sign-Flipping Belief Propagation Decoding for Topological Quantum Codes Based on Hypergraph Product Structure |
|Ordered Statistics Decoding (OSD)  | 1                                                           | osd_0                                              | Soft-Decision Decoding of Linear Block Codes Based on Ordered Statistics |    
|Quaternary Belief Propagation (BP4)| 2                                                           | bp4                                                | Quaternary Neural Belief Propagation Decoding of Quantum LDPC Codes with Overcomplete Check Matrices|
|Relay Belief Propagation (Relay BP)| 1                                                           | relay_bp                                           | Relay BP: normalized min-sum over multiple legs with disordered per-variable memory (relay-bp crate, `trmue/relay`)|
|Belief Propagation with Syndrome Flipping (BP-SF)| 1                                               | bp_sf                                              | Fully Parallelized BP Decoding for Quantum LDPC Codes Can Outperform BP-OSD (Dies-Irae/BP-SF)|
|Quantized Min-Sum BP               | 1                                                           | bp_norm_min_sum_quant                              | Normalized min-sum BP with fixed-point quantized messages|
|Lottery BP                         | 1                                                           | bp_lottery                                         | Sobol/system-driven sign-flip perturbations on the BP messages|
|Quantized Lottery BP               | 1                                                           | bp_lottery_quant                                   | Lottery BP with fixed-point quantized messages|
|Lottery BP with a sign-flip policy | 1                                                           | bp_lottery_policy                                  | Lottery BP with a selectable sign-flip policy|
|SAQ (learned decoder)              | 1                                                           | saq                                                | SAQ: Stabilizer-Aware Quantum Error Correction Decoder (arXiv:2512.08914); trained with `-t`|
|Minimum-Weight Perfect Matching (MWPM)| 1                                                     | mwpm                                               | PyMatching v2 sparse-blossom (Higgott & Gidney); graphlike codes only|
|Union-Find (Delfosse-Nickerson)| 1                                                            | union_find                                         | Almost-linear-time decoding for topological codes (arXiv:1709.06218); graphlike codes only (surface and toric)|

#### 2.5. Logical check module
The check YAML file defines all configuration parameters associated with the computation of logical check error rates.
An example configuration file for computing the logical check error rate using the lx matrix is provided in ```lx.check.yaml```.

```
check:
  check_type: lx
```

The following table provides a detailed explanation of the configuration parameters used in the check module YAML file.
| Key              | Description                                                   | Example                   |
|------------------|---------------------------------------------------------------|---------------------------|
| `check.check_type`| Method used on logical check computation                     | `lx` or `lz`                     |

#### 2.6. Interface module
This module can be used with a quantum circuit simulator such as Stim to generate circuits that include various types of errors.
An example configuration file using Stim is provided in ```stim_generated.interface.yaml```. (Using interface will cause other modules having different format, which will be shown on [Interface module](docs/interface.md).)
```
interface:
  backend: stim
  code: surface_code:rotated_memory_x
  distance: 3
```

The following table provides a detailed explanation of the configuration parameters used in the interface module YAML file.
| Key              | Description                                                   | Example                   |
|------------------|---------------------------------------------------------------|---------------------------|
| `interface.backend`| The quantum circuit simulator is used            | `stim`                     |
| `interface.code`| Stim code family to generate, required unless `interface.circuit` is given | `surface_code:rotated_memory_x` |
| `interface.distance`| Code distance of the generated circuit, required unless `interface.circuit` is given | `3` |
| `interface.circuit`| (optional) Inline stim circuit string, or a mapping of generation parameters, used instead of `code`/`distance` | `<stim circuit string>` |
| `interface.number_channel`| (optional) Fallback channel count when `error.number_channel` is absent | `1` |

The device and dtype of an interface run come from the **decoder** YAML, not from this file. See [Interface module](docs/interface.md) for the full key list.
#### 2.7. Metric module
This module does not take any YAML file as inputs, it will report default metrics as output, which will be described in the output.

### 3. Output format and metrics
The result YAML file will be saved to the path specified by the ```-r``` option. 
In the example above, the result YAML file can be found in the ```tests/test_outputs``` folder.
This file includes both the metric results for each decoder and a summary of the full decoding.
Additionally, the result YAML file is updated every 100 batches, allowing Syndrilla to resume the simulation from the last checkpoint if the error budget was not reached in the previous run.

Example output of a run like the one above, abridged:

```
decoder_0:
  algorithm: bp_norm_min_sum
  decoder invoke rate: 1.00000000000000000e+00
  average iteration: 7.69916235294117968e+01
  iteration distribution: [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 6, 6, 8, 14, 131,
    131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131,
    131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131,
    131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131,
    131, 131, 131, 131, 131, 131, 131, 131, 131]
  total time (s): '1.10485250949859619e+01'
  average time per batch (s): '6.49913240881527104e-02'
  average time per sample (s): '6.49913240881526920e-05'
  average time per iteration (s): '8.44576808757733200e-07'
  hx:
    data qubit accuracy: 9.75190120246994030e-01
    data qubit correction accuracy: 6.41507008117403132e-01
    data frame error rate: 6.73282352941176554e-01
    syndrome frame error rate: 5.77482352941176336e-01
    logical error rate: 5.78029411764705681e-01
    converge failure rate: 5.47058823529412205e-04
    converge success rate: 4.21970588235294097e-01
decoder_1:
  algorithm: osd_0
  decoder invoke rate: 5.77482352941176336e-01
  average iteration: 1.77477071406242374e+02
  iteration distribution: [1, 170, 172, 172, 173, 173, 174, 174, 174, 175, 175, 175,
    175, 175, 175, 176, 176, 176, 176, 176, 176, 176, 176, 176, 177, 177, 177, 177,
    177, 177, 177, 177, 177, 177, 177, 178, 178, 178, 178, 178, 178, 178, 178, 178,
    178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 179, 179,
    179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179,
    179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179,
    179, 179, 179, 179, 179, 179, 179, 179, 179]
  total time (s): '6.03572950363159180e+01'
  average time per batch (s): '3.55042911978328934e-01'
  average time per sample (s): '3.55042911978328650e-04'
  average time per iteration (s): '2.00049854387823897e-06'
  hx:
    data qubit accuracy: 9.81503314917126946e-01
    data qubit correction accuracy: 6.86872349409486938e-01
    data frame error rate: 5.27194117647059146e-01
    syndrome frame error rate: 0.00000000000000000e+00
    logical error rate: 5.92352941176471030e-03
    converge failure rate: 5.92352941176471030e-03
    converge success rate: 9.94076470588235228e-01
decoder_full:
  batch size: 1000
  batch count: 170
  target error: 1000
  target error reached: 1007
  data type: torch.float64
  physical error rate: 5.00000000000000028e-02
  total time (s): '7.14058201313018799e+01'
  H matrix: /home/ya212494/code/syndrilla/examples/alist/surface/surface_10_hx.alist
  hx:
    logical error rate: 5.78029411764705681e-01
```

The block above is abridged: a real result file also carries `sample count` and `iteration count` for every decoder, and its numbers reflect whichever batch size and error rate the run actually used.

#### 3.1. Per-decoder metrics
Since Syndrilla supports a sequence of decoding algorithms, there are two types of output metrics: (1) per-decoder metrics for each individual decoder, and (2) final metrics after all decoders.

The following table provides a detailed explanation of the metrics in the output YAML file for per-decoder metrics:
| Metric                           | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| `algorithm`                      | Name of the decoding algorithm used (e.g., `bp_norm_min_sum`, `osd_0`)      |
| `data qubit accuracy`            | Ratio of correctly matched data qubits over all data qubits                 |
| `data qubit correction accuracy` | Ratio of correctly identified data qubit errors                               |
| `data frame error rate`          | Ratio of samples with any data qubit mismatched                                |
| `syndrome frame error rate`      | Ratio of samples with any syndrome mismatched                                  |
| `logical error rate`             | Ratio of samples that have a logical error                               |
| `converge failure rate`          | Ratio of samples that successfully converge with a logical error  |
| `converge success rate`          | Ratio of samples that successfully converge without a logical error |
| `decoder invoke rate`            | Ratio of samples for which the decoder is invoked                           |
| `average iteration`              | Average number of iterations per sample                                    |
| `sample count`                   | Total number of samples this decoder metered. Per-sample rates are accumulated weighted by this count (not by batch count), so they stay correct when batches differ in size — e.g. under the adaptive iteration speedup (`rebatch_speedup`), where a batch may meter only its converged samples. For equal-size batches it equals `batch count` × `batch size`. |
| `iteration distribution`         | Once the error budget is reached, the per-percentile iteration counts (101 values, 0–100% at 1% intervals); before then, the raw per-iteration histogram |
| `iteration count`                | Raw per-iteration histogram (samples stopping at each iteration index), always saved un-percentiled regardless of completion |
| `total time (s)`                 | Total time taken by the decoder in seconds                                  |
| `average time per batch (s)`     | Average time taken per batch in seconds                                     |
| `average time per sample (s)`    | Average time taken per sample in seconds                                    |
| `average time per iteration (s)` | Average time per iteration per sample in seconds                            |


#### 3.2. Final metrics
The following table provides a detailed explanation of the metrics in the output YAML file for final metrics:
| Metric                         | Description                                                    |
|--------------------------------|----------------------------------------------------------------|
| `H matrix`                     | Path to the parity-check matrix used                           |
| `batch size`                   | Number of samples in each batch                               |
| `batch count`                  | Total number of batches                                    |
| `target error`                 | Total number of errors to stop decoding                        |
| `target error reached`         | Actual number of logical errors observed                       |
| `data type`                    | Floating point data used                                       |
| `physical error rate`          | Physical error rate                                            |
| `logical error rate`           | Logical error rate across all samples             |
| `total time (s)`               | Total simulation time across all batches in seconds            |

*Note that the time metric here only considers the decoding time.*

To change the configuration of the simulator, user need to update the YAML files. 
For example, if you want to use a different physical error rate, you need to find the input error YAML (e.g., ```examples/alist/bsc.error.yaml```) and update the ```rate``` field.

### 4. Resume from checkpoint
If previous run is terminated by accident, the simulation can resume by setting ```-ckpt``` to the checkpoint YAML file, the results of a previous run (e.g., ```tests/test_outputs/result_phy_err_0.1.yaml```). The checkpoint's physical error rate has to match the one in the error YAML, or the run is rejected.

```command
syndrilla -r=tests/test_outputs 
          -d=examples/alist/bposd_hx.decoder.yaml 
          -m=examples/alist/surface_10.matrix.yaml 
          -e=examples/alist/bsc.error.yaml 
          -c=examples/alist/lx.check.yaml 
          -s=examples/alist/perfect.syndrome.yaml 
          -bs=10000 
          -te=1000
          -ckpt=tests/test_outputs/result_phy_err_0.1.yaml
```

### 5. Sweep configurations
Syndrilla also allows sweeping configurations during simulation, which is done in the ```zoo``` folder.
To generate all the configurations in the zoo directory, user can use the ```generate_sweeping_configs.py``` script. 

```command
python zoo/script/generate_sweeping_configs.py 
```

The configurations to sweep are specified in the ```sweeping_configs.yaml``` file.
It allows specifying decoder (decoder algorithm), code (code type), probability (physical error rate), check_type (check type), distance (code distance), and dtype (data type).
Below is an example:

```
decoder: [bposd_quant]
code: [surface]
probability: [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
check_type: [hx]
distance: [3, 5, 7, 9, 11, 13]
dtype: ['float32']
```

This file lives at ```zoo/script/sweeping_configs.yaml```; it ships with the wider alternatives commented out above each line.

*Note that currently supported data format includes ['bfloat16', 'float16', 'float32', 'float64'].*

Once all configurations are prepared, you can see the corresponding folders in the ```zoo```, and you can now sweep the simulation using the ```run_sweeping.py``` script. 
This command will generate a corresponding result YAML file within each configuration folder.
Moreover, if a result YAML file already exists and simulation is terminated by accident, running the script again will, by default, automatically resume from the checkpoint, where the simulated is terminated.

```command
python zoo/script/run_sweeping.py -r=zoo/bposd_sweeping/ -d=bposd
```

There are command line arguments to control the script, allowing you to specify the configuration path, select the decoder, define batch sizes, and adjust logging verbosity.
| Argument | Description                                  | Example                                           |
|----------|----------------------------------------------|---------------------------------------------------|
| `-r`     | Path to configuration folder                 | `-r=zoo/bposd_sweeping/`                          |
| `-d`     | Decoder algorithm to run                     | `-d=bposd`                                        |
| `-bs`    | Number of samples run each batch             | `-bs=10000`                                       |
| `-st`    | Syndrome type used for the sweep             | `-st=perfect`                                     |
| `-l`     | Level of logger                              | `-l=SUCCESS`                                      |

## Simulation results
We show some of the simulation results as below.
These results show the impact of data format, code distance, physical error rate, and hardware on logical error rate and runtime.

GPUs: AMD Insticnt MI210, NVIDIA A100, NVIDIA H200

CPU: Intel i9-13900K

### Comparison across GPUs
<table>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/UnaryLab/syndrilla/main/zoo/speedup/accuracy_gpu.png" width="240"><br>Accuracy
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/UnaryLab/syndrilla/main/zoo/speedup/time_gpu.png" width="240"><br>Time
    </td>
  </tr>
</table>


### Comparison across data formats
<table>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/UnaryLab/syndrilla/main/zoo/speedup/accuracy_data_format.png" width="240"><br>Accuracy
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/UnaryLab/syndrilla/main/zoo/speedup/time_data_format.png" width="240"><br>Time
    </td>
  </tr>
</table>


### Comparison across distances
<table>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/UnaryLab/syndrilla/main/zoo/speedup/accuracy_distance.png" width="240"><br>Accuracy
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/UnaryLab/syndrilla/main/zoo/speedup/time_distance.png" width="240"><br>Time
    </td>
  </tr>
</table>


### Comparison across batch sizes and against CPU
<table>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/UnaryLab/syndrilla/main/zoo/speedup/time_batch.png" width="240"><br>Time
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/UnaryLab/syndrilla/main/zoo/speedup/time_cpu_speedup.png" width="240"><br>Speedup over CPU
    </td>
  </tr>
</table>


## Citation
If you use Syndrilla in your research, please cite the following papers:

```bibtex
@article{2026_arxiv_lottery_bp,
	title={{Lottery BP: Unlocking Quantum Error Decoding at Scale}},
	author={Yanzhang Zhu and Chen-Yu Peng and Yun Hao Chen and Yeong-Luh Ueng and Di Wu},
	year={2026},
	eprint={2605.00038},
	archivePrefix={arXiv},
	url={https://arxiv.org/abs/2605.00038}
}
```
```bibtex
@article{2025_qce_syndrilla,
	title={{Syndrilla: Simulating Decoders for Quantum Error Correction using PyTorch}},
	author={Yanzhang Zhu and Chen-Yu Peng and Yun Hao Chen and Siyuan Niu and Yeong-Luh Ueng and Di Wu},
	booktitle={International Conference on Quantum Computing and Engineering},
	year={2025}
}
```

## Contribution
We warmly welcome contributions to Syndrilla — just open a pull request!

## License
Syndrilla is released under the MIT License. See [LICENSE](LICENSE) for the full text.
