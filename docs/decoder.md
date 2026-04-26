# Decoder module

The decoder YAML file defines the decoding algorithm(s), the parity-check side they operate on, the iteration budget, and the device/dtype used during decoding.
A single YAML can specify one decoder or a chained list of decoders that run sequentially: if the first decoder fails to converge on a sample, the next decoder retries that sample.

When the syndrome carries a rounds dimension (`d_rounds > 1`), the decoder is wrapped in `RoundFlattenWrapper`, which transparently flattens `[B, d, ...]` into `[B*d, ...]` before the inner algorithm and reshapes outputs back. No per-decoder change is needed to support multi-round inputs.

Matrix entries (`parity_matrix_hx`, `parity_matrix_hz`, optional `logical_check_lx`/`logical_check_lz`) live in the matrix YAML loaded via the `-m` flag — see [matrix.md](matrix.md). The decoders below consume them through a pre-loaded `MatrixBundle`.

## 1. Common configuration
The following table details the configuration parameters shared by every decoder YAML file.
| Key                              | Description                                                                                                                                  | Example                              |
|----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------|
| `decoder.algorithm`              | Decoding algorithm name, or a list of names for chained decoders                                                                              | `bp_norm_min_sum` or `[bp_norm_min_sum, osd_0]` |
| `decoder.check_type`             | Side of the stabilizer code used: `hx` consumes Hx and produces `e_v` correcting Z errors, `hz` consumes Hz and produces `e_v` correcting X errors | `hx` or `hz`                         |
| `decoder.max_iter`               | Maximum number of belief-propagation iterations per sample                                                                                    | `181`                                |
| `decoder.dtype`                  | Floating-point precision used during decoding                                                                                                 | `bfloat16`, `float16`, `float32`, `float64` |
| `decoder.device.device_type`     | Type of the device on which decoding runs                                                                                                     | `cpu` or `cuda`                      |
| `decoder.device.device_idx`      | Index of the device on which decoding runs. Only used when `device_type = cuda`.                                                              | `0`                                  |

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
The following table lists every algorithm registered under `src/syndrilla/decoder/`. Sections 3.1–3.8 give the per-decoder example YAML and full parameter table.

| Algorithm name              | #Channel | Description                                                                            | Reference                                                                                                                          |
|-----------------------------|----------|----------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| `bp_norm_min_sum`           | 1        | Normalized min-sum belief propagation                                                   | Factor Graphs and the Sum-Product Algorithm                                                                                        |
| `bp_norm_min_sum_quant`     | 1        | Normalized min-sum BP with fixed-point quantization                                | -                                                                                                                                  |
| `bp_branch_assisted`        | 1        | Branch-assisted sign-flipping BP (BSFBP)                                                 | Branch-Assisted Sign-Flipping Belief Propagation Decoding for Topological Quantum Codes Based on Hypergraph Product Structure      |
| `bp_lottery`                | 1        | Lottery BP, Sobol/system sign-flip perturbations on the BP messages                      | -                                                                                                                                  |
| `bp_lottery_quant`          | 1        | Lottery BP with fixed-point quantization                                         | -                                                                                                                                  |
| `bp_lottery_policy`         | 1        | Lottery BP with selectable sign-flip policy (paper's five-policy)               | -                                                                                                                                  |
| `bp4`                       | 2        | Quaternary BP (BP4) operating on the 2-channel Pauli prior                               | Quaternary Neural Belief Propagation Decoding of Quantum LDPC Codes with Overcomplete Check Matrices                               |
| `osd_0`                     | 1        | Order-0 Ordered Statistics Decoding, typically chained after a BP variant                | Soft-Decision Decoding of Linear Block Codes Based on Ordered Statistics                                                            |

### 3.1. bp_norm_min_sum
Normalized min-sum BP. Example configuration (`bp_hx.decoder.yaml`):

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

| Key                       | Description                                                              | Example   |
|---------------------------|--------------------------------------------------------------------------|-----------|
| `decoder.algorithm`       | Algorithm name                                                           | `bp_norm_min_sum` |
| `decoder.check_type`      | `hx` or `hz`                                                             | `hx`      |
| `decoder.max_iter`        | Maximum BP iterations per sample                                         | `181`     |
| `decoder.dtype`           | Floating-point precision                                                 | `float64` |
| `decoder.device.*`        | Device selection (see common config)                                     | -         |
| `decoder.trainable`       | Optional. Enables learnable damping/normalization parameters             | `False`   |

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
| `decoder.algorithm`       | Algorithm name                                                           | `bp_norm_min_sum_quant` |
| `decoder.check_type`      | `hx` or `hz`                                                             | `hx`      |
| `decoder.max_iter`        | Maximum BP iterations per sample                                         | `181`     |
| `decoder.dtype`           | Floating-point precision used outside the quantized accumulators         | `float32` |
| `decoder.int_width`       | Integer bit width of the fixed-point message representation              | `3`       |
| `decoder.frac_width`      | Fractional bit width of the fixed-point message representation           | `4`       |
| `decoder.device.*`        | Device selection (see common config)                                     | -         |

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
| `decoder.algorithm`       | Algorithm name                                                           | `bp_branch_assisted` |
| `decoder.check_type`      | `hx` or `hz`                                                             | `hx`      |
| `decoder.max_iter`        | Maximum BP iterations per sample                                         | `181`     |
| `decoder.max_b_iter`      | Maximum branch (sign-flip) iterations per sample                         | `181`     |
| `decoder.dtype`           | Floating-point precision                                                 | `float64` |
| `decoder.random_machine`  | Random sampler used for branch perturbations: `sobol` or `system`        | `sobol`   |
| `decoder.device.*`        | Device selection (see common config)                                     | -         |

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
| `decoder.algorithm`       | Algorithm name                                                           | `bp_lottery` |
| `decoder.check_type`      | `hx` or `hz`                                                             | `hx`      |
| `decoder.max_iter`        | Maximum BP iterations per sample                                         | `181`     |
| `decoder.dtype`           | Floating-point precision                                                 | `float64` |
| `decoder.random_machine`  | Random sampler used to drive sign-flips: `sobol` or `system`             | `sobol`   |
| `decoder.device.*`        | Device selection (see common config)                                     | -         |

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
| `decoder.algorithm`       | Algorithm name                                                           | `bp_lottery_quant` |
| `decoder.check_type`      | `hx` or `hz`                                                             | `hx`      |
| `decoder.max_iter`        | Maximum BP iterations per sample                                         | `181`     |
| `decoder.dtype`           | Floating-point precision used outside the quantized accumulators         | `float64` |
| `decoder.random_machine`  | Random sampler used to drive sign-flips: `sobol` or `system`             | `sobol`   |
| `decoder.int_width`       | Integer bit width of the fixed-point message representation              | `3`       |
| `decoder.frac_width`      | Fractional bit width of the fixed-point message representation           | `4`       |
| `decoder.device.*`        | Device selection (see common config)                                     | -         |

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
| `decoder.algorithm`       | Algorithm name                                                           | `bp_lottery_policy` |
| `decoder.check_type`      | `hx` or `hz`                                                             | `hx`        |
| `decoder.max_iter`        | Maximum BP iterations per sample                                         | `181`       |
| `decoder.dtype`           | Floating-point precision                                                 | `float64`   |
| `decoder.random_machine`  | Random sampler used to drive sign-flips: `sobol` or `system`             | `sobol`     |
| `decoder.sign_flip_policy`| Sign-flip policy (see table below)                                        | `Proposed`  |
| `decoder.device.*`        | Device selection (see common config)                                     | -           |

The accepted values for `sign_flip_policy`:

| Value                       | Description                                                                                                                                           |
|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Proposed`                  | Paper policy (5) — proposed two-tier local lottery: random unsat CN → max-unsat VN among its neighbors → min \|LLR\| tiebreak. Default.               |
| `global_optimal`            | Paper policy (1) — among all VNs with the most unsatisfied CNs, flip the one with minimum \|LLR\|. Upper-bound reference, impractical for hardware.   |
| `global_connectivity`       | Paper policy (2) — among all VNs with the most unsatisfied CNs, flip a random one. Demonstrates the value of reliability guidance.                     |
| `local_random`              | Paper policy (3) — for a random unsatisfied CN, flip a random neighboring VN. Lowest hardware complexity.                                              |
| `local_reliable`            | Paper policy (4) — for a random unsatisfied CN, flip its neighbor with the minimum \|LLR\|. No connectivity-based prioritization.                      |
| `local_connectivity`        | Extra. Local analogue of `global_connectivity`: random unsatisfied CN, then max-unsat neighbor, no \|LLR\| tiebreak.                                   |
| `global_weighted_random`    | Extra. Globally random VN selection weighted by per-VN unsatisfied-CN count.                                                                            |

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
| `decoder.algorithm`       | Algorithm name                                                           | `bp4`     |
| `decoder.max_iter`        | Maximum BP iterations per sample                                         | `181`     |
| `decoder.dtype`           | Floating-point precision                                                 | `float64` |
| `decoder.damping_factor`  | Damping factor applied to BP4 messages between iterations                | `0.1`     |
| `decoder.device.*`        | Device selection (see common config)                                     | -         |

`bp4` consumes both Hx and Hz from the matrix bundle directly; `check_type` is not used.

### 3.8. osd_0
Order-0 Ordered Statistics Decoding. Almost always chained after a BP variant; runs only on samples the previous decoder did not converge on. There is no standalone OSD example, since it is configured inside a chained decoder YAML such as `bposd_hx.decoder.yaml`:

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

| Key                       | Description                                                              | Example   |
|---------------------------|--------------------------------------------------------------------------|-----------|
| `decoder.algorithm`       | Algorithm name (typically the second entry of a chained list)            | `osd_0`   |
| `decoder.check_type`      | `hx` or `hz` — must match the BP that precedes it                        | `hx`      |
| `decoder.dtype`           | Floating-point precision                                                 | `float64` |
| `decoder.device.*`        | Device selection (see common config)                                     | -         |

`osd_0` does not iterate, so it ignores `max_iter`.

## 4. Decoder I/O contract
Every decoder consumes and returns an `io_dict` with the following entries.
| Key                | Direction | Description                                                                                              |
|--------------------|-----------|----------------------------------------------------------------------------------------------------------|
| `synd`             | in/out    | Syndrome tensor, shape `[B, M]` (1-channel) or `[B, 2, M]` (2-channel). When `d_rounds > 1`, an extra rounds dim is added at position 1. |
| `llr0`             | in        | Per-bit prior LLRs from the error model                                                                  |
| `H_matrix`         | in        | Dense parity-check matrix used by the decoder                                                            |
| `e_v`              | out       | Estimated error vector, same trailing shape as `llr0`                                                    |
| `llr`              | out       | Posterior per-bit LLRs after decoding                                                                    |
| `converge`         | out       | 0/1 per sample indicating whether the decoder reached a syndrome-consistent estimate                     |
| `iter`             | out       | Number of iterations actually used per sample                                                            |

When chained decoders are used, the next decoder reads the previous decoder's `synd`, `llr`, `e_v`, and `converge` and only runs on samples whose `converge` is 0.
