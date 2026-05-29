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
    - [2. Input format and configurations](#2-input-format-and-configurations)
      - [2.1. Error module](#21-error-module)
      - [2.2. Syndrome module](#22-syndrome-module)
      - [2.3. Matrix module](#23-matrix-module)
      - [2.4. Decoder module](#24-decoder-module)
      - [2.5. Logical check module](#25-logical-check-module)
      - [2.6. Interface module](#26-interface-module)
      - [2.7. Vote module](#27-vote-module)
      - [2.8. Metric module](#28-metric-module)
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

## Features
1. High modularity: easily customizing your own **decoding algorithms** and **error models**
2. High compatibility: cross-platform simulation on CPUs, **GPUs**, and even AI accelerators
3. High performance: showing **10-20X** speedup on GPUs over CPUs
4. Community focus: support for standard **BPOSD** decoder and **BP4** decoder
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
   - The ```name: syndrilla``` in ```evironment.yaml``` can be updated to a preferred one.
3. ```conda activate syndrilla```
4. ```pip install syndrilla```
5. Validate installation via ```syndrilla -h``` in the command line or ```import syndrilla``` in python code
   - If you want to validate the simulation results against BPOSD, you need to change python to version 3.10. Then install [BPOSD](https://github.com/quantumgizmos/bp_osd) and run ```python tests/validate_bposd.py```

### Option 2: source installation
This is the developer mode, where you can edit the source code with live changes reflected for simulation.
1. ```git clone``` [this repo](https://github.com/UnaryLab/syndrilla) and ```cd``` to the repo dir.
2. ```conda env create -f environment.yaml```
   - The ```name: syndrilla``` in ```evironment.yaml``` can be updated to a preferred one.
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
| `-ckpt`  | Path to checkpoint YAML file to resume | `-ch=result_phy_err.yaml`                         |
| `-bs`    | Number of samples in each batch             | `-bs=10000`                                       |
| `-te`    | Total number of errors to stop decoding      | `-te=1000`                                         |
| `-l`     | Level of logger                              | `-l=SUCCESS`                                      |

### 2. Input format and configurations
<table>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/UnaryLab/syndrilla/main/images/modules.png" width="600">
    </td>
  </tr>
</table>

Syndrilla virtualizes the full decoder pipeline of data encoding, syndrome measurement, error decoding into five modules: error, syndrome, decoder, logical check, and metric, as shown in the figure above.
All configurations are defined through YAML files. 
Each module requires its own dedicated YAML configuration file, with the exception of the metric module.

#### 2.1. Error module
The error YAML file defines all configuration parameters associated with the error model. 
It currently supports a 1-channel Binary Symmetric Channel (BSC) error model, as well as 2-channel error models for both depolarizing noise and BSC.
An example error configuration file using the Binary Symmetric Channel (BSC) model is provided in ```bsc.error.yaml```:

```
error:
  model: bsc
  number_channel: 1
  device: 
    device_type: cpu
    device_idx: 0
  rate: 0.05
``` 

The following table details the configuration parameters used in the error YAML file.
| Key              | Description                                                   | Example                   |
|------------------|---------------------------------------------------------------|---------------------------|
| `error.model`     | Type of quantum error model applied to data qubits           | `bsc` or `depol`          |
| `error.number_channel`     | The number of error channel applied to quantum circuit           | `1` or `2`                     |
| `error.device.device_type`       | Type of the device where the error injection will happen                                       | `cpu` or `cuda`                                       |
| `error.device.device_idx`       | Index of the device where the error injection will happen. This option only works when `device_type = cuda`.                                                        | 0                           |
| `error.rate`      | Physical error rate                                          | `0.05`                    |

The following table details all types of error model Syndrilla supports. (Using different error model may need different configuration format, which will be shown on [Error module](docs/error.md).)

| Error Model      | Number of channels                                         |Example                                            |
|------------------|------------------------------------------------------------|---------------------------------------------------|
|Binary Symmetric Channel (BSC)|Both 1 and 2                                    | bsc                                               |
|Depolarizing Channel |2                                                        | depol                                             |

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
| Phenomenological          | Replicates the true syndrome over `rounds` and flips each bit with probability `measurement_error_rate`  | `phenomenological` |
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
  max_iter: 181
  dtype: float64
  device: 
    device_type: cuda
    device_idx: 0
``` 

The following table details the configuration parameters used in the decoder module YAML file.
| Key                   | Description                                                                  | Example                                            |
|------------------------|-----------------------------------------------------------------------------|-----------------------------------------------------|
| `decoder.algorithm`    | List of decoding algorithms used                                            | `[bp_norm_min_sum, osd_0]`                         |
| `decoder.check_type`   | Type of parity-check matrix used                                            | `hx` or `hz`                                       |
| `decoder.device.device_type`       | Type of the device where the decoding will happen                                       | `cpu` or `cuda`                                       |
| `decoder.device.device_idx`       | Index of the device where the decoding will happen. This option only works when `device_type = cuda`.                                      | 0                           |
| `decoder.max_iter`     | Maximum number of decoding iterations for iterative algorithms              | `181`                                              |
| `decoder.dtype`        | Data type for decoding computations                                         | `float32`, `float64`                              |

The following table details the different types of decoding algorithms Syndrilla supports. (Using different decoder may need different configuration format, which will be shown on [Decoder module](docs/decoder.md).)

| Error Model                       | #Channel                                          | Example                                            | Reference         |
|-----------------------------------|-------------------------------------------------------------|----------------------------------------------------|---------------------|
|Min-Sum Belief Propagation  (Min-Sum BP)| 1                                                           | bp_norm_min_sum                                    | Factor Graphs and the Sum-Product Algorithm |
|Branch-Assisted Sign-Flipping Belief Propagation (BSFBP) | 1                                     | bp_branch_assisted                                 | Branch-Assisted Sign-Flipping Belief Propagation Decoding for Topological Quantum Codes Based on Hypergraph Product Structure |
|Ordered Statistics Decoding (OSD)  | 1                                                           | osd_0                                              | Soft-Decision Decoding of Linear Block Codes Based on Ordered Statistics |    
|Quaternary Belief Propagation (BP4)| 2                                                           | bp4                                                | Quaternary Neural Belief Propagation Decoding of Quantum LDPC Codes with Overcomplete Check Matrices|

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
  device:
    device_type: cpu
    device_idx: 0
  dtype: float64
```

The following table provides a detailed explanation of the configuration parameters used in the check module YAML file.
| Key              | Description                                                   | Example                   |
|------------------|---------------------------------------------------------------|---------------------------|
| `interface.backend`| The quantum circuit simulator is used            | `stim`                     |
| `interface.device.device_type`| Type of the device where the output result will be processed           | `cpu` or `cude`                  |
| `interface.device.device_idx`| The indice of the device where the output result will be processed           | `0`                     |
| `interface.dtype`| The data type which the output result will be set as           | `0`                     |
#### 2.7. Vote module
This module does not take any YAML file as inputs, it will specified by ```-vs``` option. 
For example, ```decoder_0``` will do the majority voting on the output result of first decoder.

#### 2.8. Metric module
This module does not take any YAML file as inputs, it will report default metrics as output, which will be described in the output.

### 3. Output format and metrics
The result YAML file will be saved to the path specified by the ```-r``` option. 
In the example above, the result YAML file can be found in the ```tests/test_outputs``` folder.
This file includes both the metric results for each decoder and a summary of the full decoding.
Additionally, the result YAML file is updated every 100 batches, allowing Syndrilla to resume the simulation from the last checkpoint if the error budget was not reached in the previous run.

Example output of running above code:

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
| `distribution`                   | Distribution of iterations at 1% interval |
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
If previous run is terminated by accident, the simulation can resume by setting ```-ckpt``` to the checkpoint YAML file, the results of a previous run (e.g., ```tests/test_outputs=result_phy_err_0.01.yaml```).

```command
syndrilla -r=tests/test_outputs 
          -d=examples/alist/bposd_hx.decoder.yaml 
          -e=examples/alist/bsc.error.yaml 
          -c=examples/alist/lx.check.yaml 
          -s=examples/alist/perfect.syndrome.yaml 
          -bs=10000 
          -te=1000
          -ckpt=tests/test_outputs=result_phy_err_0.01.yaml
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
decoder: [bposd]
code: [surface, toric]
probability: [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
check_type: [hx, hz]
distance: [3, 5, 7, 9, 11, 13]
dtype: ['bfloat16', 'float16', 'float32', 'float64']
```

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
