# Decoder module

The decoder YAML file defines the decoding algorithm(s), the parity-check side they operate on, the iteration budget, and the device/dtype used during decoding.
A single YAML can specify one decoder or a chained list of decoders that run sequentially: if the first decoder fails to converge on a sample, the next decoder retries that sample.

When the syndrome carries a rounds dimension (`rounds > 1`), the decoder is wrapped in `RoundFlattenWrapper`, which transparently flattens `[B, d, ...]` into `[B*d, ...]` before the inner algorithm and reshapes outputs back. No per-decoder change is needed to support multi-round inputs.

Matrix entries (`parity_matrix_hx`, `parity_matrix_hz`, optional `logical_check_lx`/`logical_check_lz`) live in the matrix YAML loaded via the `-m` flag — see [matrix.md](matrix.md). The decoders below consume them through a pre-loaded `MatrixBundle`.

## 1. Common configuration
The following table details the configuration parameters shared by every decoder YAML file.
| Key                   | Description                                                                  | Example                                            |
|------------------------|-----------------------------------------------------------------------------|-----------------------------------------------------|
| `decoder.algorithm`    | List of decoding algorithms used                                            | `[bp_norm_min_sum, osd_0]`                         |
| `decoder.check_type`   | Type of parity-check matrix used                                            | `hx` or `hz`                                       |
| `decoder.device.device_type`       | Type of the device where the decoding will happen                                       | `cpu` or `cuda`                                       |
| `decoder.device.device_idx`       | Index of the device where the decoding will happen. This option only works when `device_type = cuda`.                                      | 0                           |
| `decoder.max_iter`     | Maximum number of decoding iterations for iterative algorithms              | `131`                                              |
| `decoder.dtype`        | Data type for decoding computations                                         | `float32`, `float64`                              |
| `decoder.force_pytorch`| (optional) Run the plain PyTorch module even on a CUDA device, skipping the fused-CUDA-kernel port | `false`                  |

**CUDA acceleration.** Every belief-propagation decoder ships a fused-CUDA-kernel implementation alongside its PyTorch module: `bp_norm_min_sum`, `bp_norm_min_sum_quant`, `bp_branch_assisted`, `bp_lottery`, `bp_lottery_quant`, `bp_lottery_policy`, `bp4`, `bp_sf`, and `relay_bp` (all algorithms except `osd_0`). There is **no** separate `*_cuda` algorithm name: set `device.device_type: cuda` and the kernel port (`<algo>/<algo>_cuda.py`) is selected automatically when a CUDA-capable GPU is present and the kernel builds.

The selection **falls back to PyTorch automatically**. If no CUDA GPU is available, or the `.cu` kernel fails to build or instantiate (nvcc missing, or a non-NVIDIA accelerator such as AMD ROCm or IBM, where the CUDA kernels do not compile), the plain `<algo>/<algo>.py` PyTorch module runs instead, on whatever device the config resolves to (the CUDA device under ROCm, otherwise CPU). The same fallback applies when an algorithm has no CUDA port. Set `force_pytorch: true` to force the PyTorch module even on an NVIDIA CUDA device.

## 2. Chained decoders
A list of algorithms runs each decoder in order; later decoders are only invoked on samples that the earlier ones did not converge on.
An example chained configuration is provided in ```bposd_hx.decoder.yaml```:

```
decoder:
  algorithm: [bp_norm_min_sum, osd_0]
  check_type: hx
  max_iter: 131
  dtype: float64
  device:
    device_type: cpu
    device_idx: 0
```

The decoder-extra fields below apply to whichever algorithm uses them; algorithms that do not consume a field simply ignore it.

## 3. Supported decoders
The following table lists every algorithm registered under `src/syndrilla/decoder/`. The per-decoder sections that follow only document fields *additional to* the common configuration in Section 1.

| Algorithm name              | #Channel | Description                                                                            | Reference                                                                                                                          |
|-----------------------------|----------|----------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| `bp_norm_min_sum`           | 1        | Normalized min-sum belief propagation                                                   | Factor Graphs and the Sum-Product Algorithm                                                                                        |
| `bp_norm_min_sum_quant`     | 1        | Normalized min-sum BP with fixed-point quantization                                | -                                                                                                                                  |
| `bp_branch_assisted`        | 1        | Branch-assisted sign-flipping BP (BSFBP)                                                 | Branch-Assisted Sign-Flipping Belief Propagation Decoding for Topological Quantum Codes Based on Hypergraph Product Structure      |
| `bp_sf`                     | 1        | Normalized min-sum BP with syndrome-flipping (SF) post-processing on the most-oscillating bits | Fully Parallelized BP Decoding for Quantum LDPC Codes Can Outperform BP-OSD (Dies-Irae/BP-SF)                                |
| `bp_lottery`                | 1        | Lottery BP                      | -                                                                                                                                  |
| `bp_lottery_quant`          | 1        | Lottery BP with fixed-point quantization                                         | -                                                                                                                                  |
| `bp_lottery_policy`         | 1        | Lottery BP with selectable sign-flip policy (paper's five-policy)               | -                                                                                                                                  |
| `bp4`                       | 2        | Quaternary BP (BP4) operating on the 2-channel Pauli prior                               | Quaternary Neural Belief Propagation Decoding of Quantum LDPC Codes with Overcomplete Check Matrices                               |
| `relay_bp`                  | 1        | Relay BP — normalized min-sum run over multiple "legs" with disordered per-variable memory, keeping the best converged solution | relay-bp crate (crates.io, `trmue/relay`)                                                  |
| `osd_0`                     | 1        | Order-0 Ordered Statistics Decoding               | Soft-Decision Decoding of Linear Block Codes Based on Ordered Statistics                                                            |

### 3.1. Decoders using only the common configuration
`bp_norm_min_sum` and `osd_0` introduce no algorithm-specific fields beyond Section 1.

- `bp_norm_min_sum` — normalized min-sum BP. Standalone example (`bp_hx.decoder.yaml`):

```
decoder:
  algorithm: bp_norm_min_sum
  check_type: hx
  max_iter: 181
  dtype: float64
  device:
    device_type: cuda
    device_idx: 0
```

- `osd_0` — order-0 Ordered Statistics Decoding. Almost always chained after a BP variant; runs only on samples the previous decoder did not converge on. There is no standalone OSD example, since it is configured inside a chained decoder YAML such as `bposd_hx.decoder.yaml` (see Section 2). `osd_0` does not iterate, so it ignores `max_iter`.

### 3.2. bp_norm_min_sum_quant
Normalized min-sum BP with fixed-point quantized messages. Example configuration (`bp_quant_hx.decoder.yaml`):

```
decoder:
  algorithm: bp_norm_min_sum_quant
  check_type: hx
  max_iter: 181
  dtype: float32
  int_width: 3
  frac_width: 4
  device:
    device_type: cuda
    device_idx: 0
```

| Key                       | Description                                                              | Example   |
|---------------------------|--------------------------------------------------------------------------|-----------|
| `decoder.int_width`       | Integer bit width of the fixed-point message representation              | `3`       |
| `decoder.frac_width`      | Fractional bit width of the fixed-point message representation           | `4`       |

`decoder.dtype` here applies outside the quantized accumulators.

### 3.3. bp_branch_assisted
Branch-assisted sign-flipping BP. Example configuration (`bsfbp_hx.decoder.yaml`):

```
decoder:
  algorithm: bp_branch_assisted
  check_type: hx
  max_iter: 181
  max_b_iter: 181
  dtype: float64
  device:
    device_type: cpu
    device_idx: 0
```

| Key                       | Description                                                              | Example   |
|---------------------------|--------------------------------------------------------------------------|-----------|
| `decoder.max_b_iter`      | Maximum branch (sign-flip) iterations per sample                         | `181`     |
| `decoder.random_machine`  | Random sampler used for branch perturbations: `sobol` or `system`        | `sobol`   |

### 3.4. bp_lottery
Lottery BP — Sobol/system-driven sign-flip perturbations on the BP messages. Example configuration (`lottery_bp_hx.decoder.yaml`):

```
decoder:
  algorithm: bp_lottery
  check_type: hx
  max_iter: 181
  dtype: float64
  random_machine: sobol
  device:
    device_type: cuda
    device_idx: 0
```

| Key                       | Description                                                              | Example   |
|---------------------------|--------------------------------------------------------------------------|-----------|
| `decoder.random_machine`  | Random sampler used to drive sign-flips: `sobol` or `system`             | `sobol`   |

### 3.5. bp_lottery_quant
Lottery BP with fixed-point quantized messages. Example configuration (`lottery_bp_quant_hx.decoder.yaml`):

```
decoder:
  algorithm: bp_lottery_quant
  check_type: hx
  max_iter: 181
  dtype: float64
  random_machine: sobol
  int_width: 3
  frac_width: 4
  device:
    device_type: cuda
    device_idx: 0
```

| Key                       | Description                                                              | Example   |
|---------------------------|--------------------------------------------------------------------------|-----------|
| `decoder.random_machine`  | Random sampler used to drive sign-flips: `sobol` or `system`             | `sobol`   |
| `decoder.int_width`       | Integer bit width of the fixed-point message representation              | `3`       |
| `decoder.frac_width`      | Fractional bit width of the fixed-point message representation           | `4`       |

`decoder.dtype` here applies outside the quantized accumulators.

### 3.6. bp_lottery_policy
Lottery BP with a selectable sign-flip policy. The policy names follow the paper's five-policy taxonomy plus two extras. Example configuration (`lottery_policy_hx.decoder.yaml`):

```
decoder:
  algorithm: bp_lottery_policy
  check_type: hx
  max_iter: 181
  dtype: float64
  random_machine: sobol
  sign_flip_policy: Proposed
  device:
    device_type: cpu
    device_idx: 0
```

| Key                       | Description                                                              | Example     |
|---------------------------|--------------------------------------------------------------------------|-------------|
| `decoder.random_machine`  | Random sampler used to drive sign-flips: `sobol` or `system`             | `sobol`     |
| `decoder.sign_flip_policy`| Sign-flip policy (see table below)                                        | `Proposed`  |

The accepted values for `sign_flip_policy`:

| Value                       | Source       | Candidate set                                                                              | Flip rule                          |
|-----------------------------|--------------|--------------------------------------------------------------------------------------------|------------------------------------|
| `global_optimal`            | Paper (1)    | All VNs tied for the maximum number of unsatisfied CNs                                     | Smallest \|LLR\|                   |
| `global_connectivity`       | Paper (2)    | All VNs tied for the maximum number of unsatisfied CNs                                     |  random                     |
| `local_random`              | Paper (3)    | All VNs neighboring one random CN                                                          |  random                     |
| `local_reliable`            | Paper (4)    | All VNs neighboring one random CN                                                          | Smallest \|LLR\|                   |
| `Proposed`                  | Paper (5)    | Among VNs neighboring one random CN, those tied for the maximum number of unsatisfied CNs  | Smallest \|LLR\|                   |
| `local_connectivity`        | Extra        | Among VNs neighboring one random CN, those tied for the maximum number of unsatisfied CNs  |  random                     |
| `global_weighted_random`    | Extra        | All VNs neighboring any CN that is connected to a VN tied for the most unsatisfied CNs     |  random                     |

### 3.7. bp4
Quaternary BP operating on the 2-channel Pauli prior (used with the depolarizing or 2-channel BSC error model). Example configuration (`bp4.decoder.yaml`):

```
decoder:
  algorithm: bp4
  max_iter: 181
  dtype: float64
  damping_factor: 0.1
  device:
    device_type: cuda
    device_idx: 0
```

| Key                       | Description                                                              | Example   |
|---------------------------|--------------------------------------------------------------------------|-----------|
| `decoder.damping_factor`  | Damping factor applied to BP4 messages between iterations                | `0.1`     |

`bp4` consumes both Hx and Hz from the matrix bundle directly; `check_type` is not used.

### 3.8. relay_bp
Relay BP (the `trmue/relay` crate's algorithm). It runs normalized min-sum over a sequence of "legs": leg 1 uses a constant memory strength `init_mem_strength`; each later (ensemble) leg resets the variable→check messages to the prior, carries the posterior forward (the "relay"), and applies random per-variable memory strengths drawn from `[center − width/2, center + width/2]`. Each converged leg yields a candidate solution; the lowest-weight valid one is kept. The leg ensemble stops once every sample has collected `solution` converged solutions (or after `legs` legs). Example configuration (`relay_bp_hx.decoder.yaml`):

```
decoder:
  algorithm: relay_bp
  check_type: hx
  legs: 20
  iteration_initial: 80
  iteration_count: 60
  solution: 5
  init_mem_strength: 0.35
  center: 0.21
  width: 0.9
  alpha: 0.0
  alpha_scaling: 1.0
  dtype: float64
  device:
    device_type: cuda
    device_idx: 0
```

| Key                          | Description                                                                                       | Example   |
|------------------------------|---------------------------------------------------------------------------------------------------|-----------|
| `decoder.legs`               | Number of relay legs (ensemble size; the crate's `num_sets`)                                      | `20`      |
| `decoder.iteration_initial`  | BP iterations in leg 1                                                                             | `80`      |
| `decoder.iteration_count`    | BP iterations in each later (ensemble) leg                                                         | `60`      |
| `decoder.solution`           | Converged solutions to collect before stopping (the crate's `stop_nconv`)                         | `5`       |
| `decoder.init_mem_strength`  | Leg-1 memory strength `gamma0`                                                                     | `0.35`    |
| `decoder.center`             | Center of the per-variable memory-strength interval for ensemble legs                             | `0.21`    |
| `decoder.width`              | Width of that interval (drawn from `[center − width/2, center + width/2]`)                         | `0.9`     |
| `decoder.alpha`              | Min-sum normalization: `0.0` → adaptive `1 − 2^(−i/alpha_scaling)`; `<0` → `1.0`; else constant   | `0.0`     |
| `decoder.alpha_scaling`      | Divisor in the adaptive `alpha` schedule                                                           | `1.0`     |

`relay_bp` uses `iteration_initial`/`iteration_count`/`legs` to bound its work, so it **ignores** the common `max_iter` field. The `center`/`width` defaults match the crate's `gamma_dist_interval = (−0.24, 0.66)`.

## 4. Adaptive iteration speedup (`rebatch_speedup`)
An opt-in, per-decoder block consumed by the iterative BP decoders `bp_norm_min_sum`, `bp_norm_min_sum_quant`, `bp4`, `bp_lottery`, `bp_lottery_quant`, `bp_lottery_policy`, and `relay_bp` (other algorithms, e.g. `bp_branch_assisted`, `bp_sf`, and `osd_0`, ignore it). It reduces decoding **time** by stopping a batch once a warm-up-learned fraction of samples has converged and deferring the unconverged tail to be re-decoded uncapped.

For these BP decoders the cap is **lossless**: every sample is still fully decoded, so the logical error rate is identical to a no-cap run. The deferred tail (`converge == 0`) is still re-decoded uncapped.

**Setup.** Add an `rebatch_speedup` block to the decoder YAML; omit it to disable the feature.

```
decoder:
  algorithm: bp_norm_min_sum
  check_type: hx
  max_iter: 181
  dtype: float64
  rebatch_speedup:
    kl_eps: 0.001
    kl_window: 2
    kl_min: 3
  device:
    device_type: cuda
    device_idx: 0
```

| Key                               | Description                                              | Example |
|-----------------------------------|----------------------------------------------------------|---------|
| `decoder.rebatch_speedup.kl_eps`     | Warm-up KL threshold (larger ⇒ shorter warm-up)          | `1.0`   |
| `decoder.rebatch_speedup.kl_window`  | Consecutive settled batches that end warm-up             | `1`     |
| `decoder.rebatch_speedup.kl_min`     | Minimum warm-up batches                                  | `2`     |
| `decoder.rebatch_speedup.candidates` | (optional) cap percentiles to consider; default `0..99`  | -       |

**Output.** When a decoder uses `rebatch_speedup`, its per-decoder block in the result YAML gains a `rebatch_speedup` entry reporting `warmup batches` (the number of warm-up batches the KL test consumed) and, once the cap is chosen, `chosen pct`. This entry is emitted **before** the `total time (s)` timing fields.


## 5. Decoder I/O contract
Every decoder consumes and returns an `io_dict` with the following entries.
| Key                | Direction | Description                                                                                              |
|--------------------|-----------|----------------------------------------------------------------------------------------------------------|
| `synd`             | in/out    | Syndrome tensor, shape `[B, M]` (1-channel) or `[B, 2, M]` (2-channel). When `rounds > 1`, an extra rounds dim is added at position 1. |
| `llr0`             | in        | Per-bit prior LLRs from the error model                                                                  |
| `H_matrix`         | in        | Dense parity-check matrix used by the decoder                                                            |
| `e_v`              | out       | Estimated error vector, same trailing shape as `llr0`                                                    |
| `llr`              | out       | Posterior per-bit LLRs after decoding                                                                    |
| `converge`         | out       | 0/1 per sample indicating whether the decoder reached a syndrome-consistent estimate                     |
| `iter`             | out       | Number of iterations actually used per sample                                                            |

When chained decoders are used, the next decoder reads the previous decoder's `synd`, `llr`, `e_v`, and `converge` and only runs on samples whose `converge` is 0.
