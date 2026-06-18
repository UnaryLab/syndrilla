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
| `bp_lottery`                | 1        | Lottery BP                      | -                                                                                                                                  |
| `bp_lottery_quant`          | 1        | Lottery BP with fixed-point quantization                                         | -                                                                                                                                  |
| `bp_lottery_policy`         | 1        | Lottery BP with selectable sign-flip policy (paper's five-policy)               | -                                                                                                                                  |
| `bp4`                       | 2        | Quaternary BP (BP4) operating on the 2-channel Pauli prior                               | Quaternary Neural Belief Propagation Decoding of Quantum LDPC Codes with Overcomplete Check Matrices                               |
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

## 4. Adaptive iteration speedup (`iter_speedup`)
An opt-in, per-decoder block currently consumed by `bp_norm_min_sum` (other algorithms ignore it). It reduces decoding **time** without changing results — every sample is still fully decoded, so the logical error rate is identical to a no-cap run. 

**Setup.** Add an `iter_speedup` block to the decoder YAML; omit it to disable the feature.

```
decoder:
  algorithm: bp_norm_min_sum
  check_type: hx
  max_iter: 181
  dtype: float64
  iter_speedup:
    kl_eps: 1.0
    kl_window: 1
    kl_min: 2
  device:
    device_type: cuda
    device_idx: 0
```

| Key                               | Description                                              | Example |
|-----------------------------------|----------------------------------------------------------|---------|
| `decoder.iter_speedup.kl_eps`     | Warm-up KL threshold (larger ⇒ shorter warm-up)          | `1.0`   |
| `decoder.iter_speedup.kl_window`  | Consecutive settled batches that end warm-up             | `1`     |
| `decoder.iter_speedup.kl_min`     | Minimum warm-up batches                                  | `2`     |
| `decoder.iter_speedup.candidates` | (optional) cap percentiles to consider; default `0..99`  | -       |


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
