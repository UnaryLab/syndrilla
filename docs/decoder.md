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
| `decoder.dtype`        | Data type for decoding computations                                         | `float32`, `float64`                              |
| `decoder.force_pytorch`| (optional) Run the plain PyTorch module even on a CUDA device, skipping the fused-CUDA-kernel port | `false`                  |
| `decoder.rebatch_speedup`| (optional) Adaptive batch-shrinking cap (Section 4)                        | `{kl_eps: 0.001}`                                  |
| `decoder.config`       | Algorithm-specific settings, one entry per entry of `decoder.algorithm` (Section 1.1) | `{max_iter: 131}`         |

### 1.1. Algorithm-specific configuration (`decoder.config`)
The block is split by who reads it. The keys above are framework-wide: `main.py`, the loader, and every decoder alike consume them, so they sit at the top of `decoder`. Everything that only one algorithm understands — `max_iter`, the quantization widths, `sf`, relay_bp's leg schedule, saq's `model` / `cpnd` / `optimizer` / `train` blocks — lives under `config`:

```
decoder:
  algorithm: bp_norm_min_sum
  check_type: hx
  dtype: float64
  device:
    device_type: cuda
    device_idx: 0
  config:
    max_iter: 181
```

The one key almost every algorithm reads is `decoder.config.max_iter`, the per-sample iteration budget; it defaults to `50`, and the non-iterative decoders (`osd_0`, `mwpm`, `union_find`, `saq`) ignore it.

A `config` written as a plain mapping, as above, is the settings of the block's first (or only) algorithm. Written as a list it is **positional**: entry *i* belongs to `algorithm[i]`, which is what lets one chain give each stage its own settings (Section 2), including the same algorithm twice with different ones. Omitting `config` leaves every decoder on its defaults.

A key written in the wrong half is **rejected, not ignored**: `max_iter` left at the top level fails with a message naming `decoder.config`, and a framework-wide key such as `dtype` written inside `config` fails the same way in reverse. A decoder that quietly fell back to `max_iter: 50` instead of the configured `181` would still produce numbers, and they would look like results.

**CUDA acceleration.** Every registered decoder **except `saq`** ships a CUDA-kernel implementation alongside its PyTorch/NumPy module (`saq` is a plain PyTorch model and runs on whatever device the config selects): the belief-propagation family (`bp_norm_min_sum`, `bp_norm_min_sum_quant`, `bp_branch_assisted`, `bp_lottery`, `bp_lottery_quant`, `bp_lottery_policy`, `bp4`, `bp_sf`, `relay_bp`) plus `osd_0`, `mwpm`, and `union_find`. There is **no** separate `*_cuda` algorithm name: set `device.device_type: cuda` and the kernel port (`<algo>/<algo>_cuda.py`) is selected automatically when a CUDA-capable GPU is present and the kernel builds.

The kernels come in two flavors. The BP decoders use **fused per-iteration kernels** that vectorize the message-passing across the batch. The graph decoders `mwpm` and `union_find` are inherently sequential per shot, so their kernels parallelize over the **batch axis** (one CUDA thread decodes one shot), while `osd_0` runs one thread block per sample. For `osd_0`, `mwpm`, and `union_find` the CUDA output is **bit-for-bit identical** to the corresponding CPU implementation.

The selection **falls back to PyTorch automatically**. If no CUDA GPU is available, or the `.cu` kernel fails to build or instantiate (nvcc missing, or a non-NVIDIA accelerator such as AMD ROCm or IBM, where the CUDA kernels do not compile), the plain `<algo>/<algo>.py` PyTorch module runs instead, on whatever device the config resolves to (the CUDA device under ROCm, otherwise CPU). The same fallback applies when an algorithm has no CUDA port. Set `force_pytorch: true` to force the PyTorch module even on an NVIDIA CUDA device.

## 2. Chained decoders
A list of algorithms runs each decoder in order; later decoders are only invoked on samples that the earlier ones did not converge on.
An example chained configuration is provided in ```bposd_hx.decoder.yaml```:

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

A `config` written as a plain mapping, as above, is the **first** stage's settings, so `max_iter` reaches `bp_norm_min_sum` and `osd_0` runs on its defaults. That is what the shipped chains use, since only their first stage takes settings. To configure a later stage, write `config` as a list instead: entry *i* then belongs to `algorithm[i]`, the list may stop early (a chain ending in a stage that configures nothing needs no entry for it), and only *trailing* stages can be left out, so a stage that takes no settings ahead of one that does still needs its slot, written `- {}`. More entries than algorithms is an error rather than a silent drop. The shared keys at the top of the block — `check_type`, `dtype`, `device` — reach every stage.

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
| `mwpm`                      | 1        | Minimum-Weight Perfect Matching (sparse-blossom). Graphlike codes only (every qubit column touches ≤2 checks) | PyMatching v2 sparse-blossom (Higgott & Gidney); clean-room PyTorch/NumPy transformation                     |
| `union_find`                | 1        | Union-Find (Delfosse-Nickerson) cluster-growth + peeling decoder. Graphlike codes only (every qubit column touches at most 2 checks; weight-1 = open boundary, e.g. surface codes; weight-2 = toric) | Almost-linear-time decoding for topological codes (arXiv:1709.06218); port of chaeyeunpark/UnionFind         |
| `saq`                       | 1        | Learned dual-stream transformer decoder plus CPND constraint projection. Single feed-forward pass; toric and rotated surface codes only; needs trained weights | SAQ: Stabilizer-Aware Quantum Error Correction Decoder (arXiv:2512.08914); port of DavidZenati/SAQ-Decoder |

### 3.1. Decoders using only the common configuration
`bp_norm_min_sum` and `osd_0` introduce no algorithm-specific fields beyond Section 1 and `decoder.config.max_iter`.

- `bp_norm_min_sum` — normalized min-sum BP. Standalone example (`bp_hx.decoder.yaml`):

```
decoder:
  algorithm: bp_norm_min_sum
  check_type: hx
  dtype: float64
  device:
    device_type: cuda
    device_idx: 0
  config:
    max_iter: 181
```

- `osd_0` — order-0 Ordered Statistics Decoding. Almost always chained after a BP variant; runs only on samples the previous decoder did not converge on. There is no standalone OSD example, since it is configured inside a chained decoder YAML such as `bposd_hx.decoder.yaml` (see Section 2). `osd_0` does not iterate, so it ignores `max_iter`. On a `cuda` device it uses its CUDA kernel (`osd_0/osd_0_cuda.py`, one thread block per sample), whose correction is bit-for-bit identical to the PyTorch path; it falls back to PyTorch on CPU or when the kernel is unavailable.

### 3.2. bp_norm_min_sum_quant
Normalized min-sum BP with fixed-point quantized messages. Example configuration (`bp_quant_hx.decoder.yaml`):

```
decoder:
  algorithm: bp_norm_min_sum_quant
  check_type: hx
  dtype: float32
  device:
    device_type: cuda
    device_idx: 0
  config:
    max_iter: 181
    int_width: 3
    frac_width: 4
```

| Key                       | Description                                                              | Example   |
|---------------------------|--------------------------------------------------------------------------|-----------|
| `decoder.config.int_width`       | Integer bit width of the fixed-point message representation              | `3`       |
| `decoder.config.frac_width`      | Fractional bit width of the fixed-point message representation           | `4`       |

`decoder.dtype` here applies outside the quantized accumulators.

### 3.3. bp_branch_assisted
Branch-assisted sign-flipping BP. Example configuration (`bsfbp_hx.decoder.yaml`):

```
decoder:
  algorithm: bp_branch_assisted
  check_type: hx
  dtype: float64
  device:
    device_type: cuda
    device_idx: 0
  config:
    max_iter: 181
    max_b_iter: 181
```

| Key                       | Description                                                              | Example   |
|---------------------------|--------------------------------------------------------------------------|-----------|
| `decoder.config.max_b_iter`      | Maximum branch (sign-flip) iterations per sample                         | `181`     |
| `decoder.config.random_machine`  | Random sampler used for branch perturbations: `sobol` or `system`        | `sobol`   |

### 3.4. bp_lottery
Lottery BP — Sobol/system-driven sign-flip perturbations on the BP messages. Example configuration (`lottery_bp_hx.decoder.yaml`):

```
decoder:
  algorithm: bp_lottery
  check_type: hx
  dtype: float64
  device:
    device_type: cuda
    device_idx: 0
  config:
    max_iter: 181
    random_machine: sobol
```

| Key                       | Description                                                              | Example   |
|---------------------------|--------------------------------------------------------------------------|-----------|
| `decoder.config.random_machine`  | Random sampler used to drive sign-flips: `sobol` or `system`             | `sobol`   |

### 3.5. bp_lottery_quant
Lottery BP with fixed-point quantized messages. Example configuration (`lottery_bp_quant_hx.decoder.yaml`):

```
decoder:
  algorithm: bp_lottery_quant
  check_type: hx
  dtype: float64
  device:
    device_type: cuda
    device_idx: 0
  config:
    max_iter: 181
    random_machine: sobol
    int_width: 3
    frac_width: 4
```

| Key                       | Description                                                              | Example   |
|---------------------------|--------------------------------------------------------------------------|-----------|
| `decoder.config.random_machine`  | Random sampler used to drive sign-flips: `sobol` or `system`             | `sobol`   |
| `decoder.config.int_width`       | Integer bit width of the fixed-point message representation              | `3`       |
| `decoder.config.frac_width`      | Fractional bit width of the fixed-point message representation           | `4`       |

`decoder.dtype` here applies outside the quantized accumulators.

### 3.6. bp_lottery_policy
Lottery BP with a selectable sign-flip policy. The policy names follow the paper's five-policy taxonomy plus two extras. Example configuration (`lottery_policy_hx.decoder.yaml`):

```
decoder:
  algorithm: bp_lottery_policy
  check_type: hx
  dtype: float64
  device:
    device_type: cuda
    device_idx: 0
  config:
    max_iter: 181
    random_machine: sobol
    sign_flip_policy: Proposed
```

| Key                       | Description                                                              | Example     |
|---------------------------|--------------------------------------------------------------------------|-------------|
| `decoder.config.random_machine`  | Random sampler used to drive sign-flips: `sobol` or `system`             | `sobol`     |
| `decoder.config.sign_flip_policy`| Sign-flip policy (see table below)                                        | `Proposed`  |

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
  dtype: float64
  device:
    device_type: cuda
    device_idx: 0
  config:
    max_iter: 181
    damping_factor: 0.1
```

| Key                       | Description                                                              | Example   |
|---------------------------|--------------------------------------------------------------------------|-----------|
| `decoder.config.damping_factor`  | Damping factor applied to BP4 messages between iterations                | `0.1`     |

`bp4` consumes both Hx and Hz from the matrix bundle directly; `check_type` is not used.

### 3.8. relay_bp
Relay BP (the `trmue/relay` crate's algorithm). It runs normalized min-sum over a sequence of "legs": leg 1 uses a constant memory strength `init_mem_strength`; each later (ensemble) leg resets the variable→check messages to the prior, carries the posterior forward (the "relay"), and applies random per-variable memory strengths drawn from `[center − width/2, center + width/2]`. Each converged leg yields a candidate solution; the lowest-weight valid one is kept. The leg ensemble stops once every sample has collected `solution` converged solutions (or after `legs` legs). Example configuration (`relay_bp_hx.decoder.yaml`):

```
decoder:
  algorithm: relay_bp
  check_type: hx
  dtype: float64
  device:
    device_type: cuda
    device_idx: 0
  config:
    legs: 20
    iteration_initial: 80
    iteration_count: 60
    solution: 5
    init_mem_strength: 0.35
    center: 0.21
    width: 0.9
    alpha: 0.0
    alpha_scaling: 1.0
```

| Key                          | Description                                                                                       | Example   |
|------------------------------|---------------------------------------------------------------------------------------------------|-----------|
| `decoder.config.legs`               | Number of relay legs (ensemble size; the crate's `num_sets`)                                      | `20`      |
| `decoder.config.iteration_initial`  | BP iterations in leg 1                                                                             | `80`      |
| `decoder.config.iteration_count`    | BP iterations in each later (ensemble) leg                                                         | `60`      |
| `decoder.config.solution`           | Converged solutions to collect before stopping (the crate's `stop_nconv`)                         | `5`       |
| `decoder.config.init_mem_strength`  | Leg-1 memory strength `gamma0`                                                                     | `0.35`    |
| `decoder.config.center`             | Center of the per-variable memory-strength interval for ensemble legs                             | `0.21`    |
| `decoder.config.width`              | Width of that interval (drawn from `[center − width/2, center + width/2]`)                         | `0.9`     |
| `decoder.config.alpha`              | Min-sum normalization: `0.0` → adaptive `1 − 2^(−i/alpha_scaling)`; `<0` → `1.0`; else constant   | `0.0`     |
| `decoder.config.alpha_scaling`      | Divisor in the adaptive `alpha` schedule                                                           | `1.0`     |

`relay_bp` uses `iteration_initial`/`iteration_count`/`legs` to bound its work, so it **ignores** the common `max_iter` field. The `center`/`width` defaults match the crate's `gamma_dist_interval = (−0.24, 0.66)`.

### 3.9. mwpm
Minimum-Weight Perfect Matching via the sparse-blossom algorithm, a self-contained clean-room transformation of PyMatching v2 (it does not import `pymatching` or `networkx`). It is **graphlike-only**: every qubit column of `H` must touch at most two checks (a weight-1 column becomes a boundary edge, weight-2 a detector-detector edge; weight>2 raises an error). The matcher runs per shot and is non-iterative, so it ignores `max_iter`. The correction is **bit-for-bit identical** to PyMatching v2 (not merely equal-weight): the radix-heap LIFO tie-break and canonical neighbor order reproduce PyMatching's exact choice on degenerate syndromes. On a `cuda` device the CUDA port (`mwpm/mwpm_cuda.py`) decodes one shot per thread and matches the CPU output bit-for-bit, falling back to the CPU blossom for any shot the kernel cannot handle. Standalone example (`mwpm_hx.decoder.yaml`):

```
decoder:
  algorithm: [mwpm]
  check_type: hx
  dtype: float64
  device:
    device_type: cpu
```

`mwpm` adds two optional settings, both written under `decoder.config` like any other algorithm-specific key.

| Key                              | Description                                                              | Default   |
|----------------------------------|--------------------------------------------------------------------------|-----------|
| `decoder.config.num_workers`     | Worker processes used to decode a batch; `<= 1` keeps the sequential path | CPU count |
| `decoder.config.mp_min_batch`    | Smallest batch that is worth spreading across those workers               | `64`      |

It carries no real per-bit LLR, so it emits a sign-encoded soft output (`llr = 1 − 2·e_v`) and always reports `converge = 1`.

### 3.10. union_find
The Delfosse-Nickerson Union-Find decoder (arXiv:1709.06218): grow clusters around the syndrome defects, fuse them into a spanning forest, then peel to a correction. It is a PyTorch/NumPy port of `chaeyeunpark/UnionFind` and is **bit-for-bit identical** to that reference; matching its output on degenerate syndromes requires reproducing the C++ `tsl::robin_set` iteration order (a `_RobinTable` replica does this). It is **graphlike-only**: every qubit column of `H` must touch at most two checks. A weight-1 column becomes an open-boundary edge (planar/surface codes), a weight-2 column a detector-detector edge (toric codes), and weight>2 is rejected; a boundary vertex is added so open-boundary clusters absorb their unpaired defect. The decoder is non-iterative (ignores `max_iter`). On a `cuda` device the CUDA port (`union_find/union_find_cuda.py`) runs the **entire** decode, both the detector-graph build and the serial grow/fuse/peel, inside the extension (`cuda/union_find_kernel.cu` via `cuda/union_find_serial.cuh`, a line-for-line transliteration of `union_find.py`'s `decode_shot`). It decodes one shot per CUDA thread with its own scratch (the robin-hood iteration order is load-bearing, so each shot's decode is sequential; parallelism is across shots). It covers the **full graphlike domain**, both toric (weight-2 columns) and surface/open-boundary codes (weight-1 columns), and its output is **bit-for-bit identical** to the CPU `decode_shot` for every shot; on toric codes it additionally matches the C++ chaeyeunpark reference bit-for-bit. There is **no per-shot PyTorch fallback**: nothing in the CUDA module imports `union_find.py`. Large batches are split into fixed-memory-budget chunks (not fallbacks) so the per-launch scratch tensor stays bounded. Standalone example (`union_find_hx.decoder.yaml`):

```
decoder:
  algorithm: [union_find]
  check_type: hx
  dtype: float64
  device:
    device_type: cpu
```

`union_find` introduces no algorithm-specific fields beyond Section 1. Like `mwpm` it carries no real per-bit LLR, so it emits `llr = 1 − 2·e_v` and always reports `converge = 1`.

### 3.11. saq
SAQ (arXiv:2512.08914), a **learned** decoder: a syndrome is mapped to an error estimate in one feed-forward pass, so there is no message passing and no per-sample iteration count (`max_iter` is ignored, `iter` is always 1, `num_max_iter` is 1). The output heads emit a per-qubit posterior LLR (`llr`, positive ⇒ no error, the same sign convention as the BP decoders) and logical class logits. A final CPND stage (constraint-projected nullspace descent) projects the hard decision onto an operator that reproduces the measured syndrome exactly and then lightens it within the stabilizer coset; it is inference-only, on by default, and skipped during training.

Only **toric and surface codes** are supported, rotated or unrotated. The family is not configured: it is measured from the matrix, and a config still carrying the removed `code_type` key is rejected rather than ignored. Nothing reports a code distance, since `n` does not determine one, so run files are named by `n`. `llr0` is not consumed: SAQ conditions on the syndrome alone and learns the physical error rate from its training distribution. Standalone example (`saq_hx.decoder.yaml`):

```
decoder:
  algorithm: saq
  check_type: hx
  dtype: float32
  device:
    device_type: cpu
    device_idx: 0
  config:
    checkpoint: tests/test_outputs/saq_hx_n41_best.pt
    model:
      d_model: 128
      N_dec: 6
      h: 16
      dropout: 0.0
      no_mask: 0
    cpnd:
      enable: true
      passes: 1
    optimizer:
      lr: 5.0e-4
      weight_decay: 5.0e-8
      min_lr: 1.0e-6
    train:
      epochs: 100
      test_batches: 200
      validation_batches: 20
      error_random_seed: 42
      # optional; absent, the best epoch and the last are written out
      epochs_saved: 20
```

All four blocks live under `decoder.config`, grouped so each has exactly one reader: `model`, `cpnd` and `optimizer` are read by the decoder, `train` by `MetricState`. A key written at the top of the `decoder` block instead is rejected with a message naming where it belongs, rather than silently falling back to its default. `cpnd` is a block, so a bare `cpnd: false` is rejected.

| Key                          | Description                                                                    | Example            |
|------------------------------|--------------------------------------------------------------------------------|--------------------|
| `decoder.config.model.d_model`      | Token embedding width                                                           | `128`              |
| `decoder.config.model.N_dec`        | Number of transformer (SLTD) layers                                             | `6`                |
| `decoder.config.model.h`            | Number of attention heads (must divide `d_model`)                               | `16`               |
| `decoder.config.model.dropout`      | (optional) Dropout in attention and feed-forward blocks; training only          | `0.0`              |
| `decoder.config.model.no_mask`      | (optional) `>0` disables the topology attention mask (paper ablation)           | `0`                |
| `decoder.config.cpnd.enable`        | (optional) Run stage 4 (constraint projection + nullspace descent)              | `true`             |
| `decoder.config.cpnd.passes`        | (optional) Sweeps over the stabilizer basis in the descent                      | `1`                |
| `decoder.config.checkpoint`         | (optional) Trained weights to load; a decode run needs it, `-t` writes it       | `tests/test_outputs/saq_hx_n41_best.pt` |
| `decoder.config.optimizer.lr`       | (optional) Adam learning rate; training only                                    | `5.0e-4`           |
| `decoder.config.optimizer.weight_decay` | (optional) Adam weight decay; training only                                 | `5.0e-8`           |
| `decoder.config.optimizer.min_lr`   | (optional) Floor of the cosine schedule (`eta_min`); training only              | `1.0e-6`           |
| `decoder.config.train.*`            | (training only) The `-t` epoch schedule; see Training below                     | `epochs: 100`      |

`decoder.dtype` defaults to `float32` here rather than the BP decoders' `float64`, since float64 attention costs several times more for no accuracy gain. **Without `checkpoint` the decoder runs on randomly initialized weights and decodes at chance**, and warns when this happens.

**Training (`-t`).**

```
syndrilla -t -r=tests/test_outputs \
    -d=examples/alist/train_saq_hx.decoder.yaml \
    -m=examples/alist/surface_5.matrix.yaml \
    -e=examples/alist/bsc_train.error.yaml \
    -s=examples/alist/perfect.syndrome.yaml \
    -ls=examples/alist/logical_centric.loss.yaml \
    -bs=256
```

The settings are split by file: the decoder yaml holds the optimizer settings (`config.optimizer`) and the schedule (`config.train`), the `-ls` yaml holds the loss weights (`lambda_lc`, `lambda_lp`, `lambda_ent`) under its `loss` key, the error yaml holds the physical error rate, and `-bs` is the batch size. A training error yaml takes a `rate` range such as `[0.01, 0.20, 9]`, its last value the number of levels, and draws one rate per shot, so a single run covers the whole curve (`bsc_train.error.yaml`); the phenomenological measurer sweeps its `measurement_error_rate` the same way. A chain trains its **last** decoder, so `[bp_norm_min_sum, saq]` trains while `[saq, osd_0]` is rejected, naming where a trainable stage actually sits. The `-i` interface path trains too, with `-ls` still supplying the objective.

The run writes four files into `-r` (default `tests/test_outputs`, the same directory a decode run writes to), named after the configuration that produced them as `<algorithm>_<check_type>_<size>`, where `<size>` is `n<qubits>` or `dem<detectors>x<mechanisms>` for a circuit-level DEM run. Training the shipped hx config on `surface_5`, whose matrix has 41 columns, gives the stem `saq_hx_n41`:

| File | What it holds |
|---|---|
| `<stem>_best.pt` | bare `state_dict` of the best epoch: what `decoder.config.checkpoint` points at to decode |
| `<stem>_last.pt` | the whole run position (weights, optimizer, schedule, RNG, curve, fingerprint): what `-tckpt` resumes from |
| `<stem>_result.yaml` | a `train_full` summary of the run and what it cost, then the curve stored by column under `training result`, index-aligned with its `epoch` list. The run's best epoch and its last, widened to the last `epochs_saved` epochs when that key is set. What `-ckpt` is checked against on a resume |
| `<stem>_train.log` | one line per batch, one per epoch |

Training a second configuration into the same run directory adds files rather than overwriting the first. To evaluate the result, put `<stem>_best.pt` in `decoder.config.checkpoint` and run the normal decode CLI.

**Resuming (`-ckpt` and `-tckpt`).** `syndrilla -t -ckpt tests/test_outputs/<stem>_result.yaml -tckpt tests/test_outputs/<stem>_last.pt`, with every other flag left as it was, continues the run rather than warm-starting a new one: a run interrupted after epoch 3 and resumed ends with bit-identical weights and an identical curve to an uninterrupted run. Both flags are printed by the run that wrote them, and a training run takes the pair or neither: either one alone is refused naming the other, since the yaml says what the run was and the checkpoint says where it got to.

Everything is restored from the `*_last.pt`; the yaml is only ever read to be checked. Both halves are checked the same way, by `MetricState.train_validate_checkpoint`, which refuses a resume whose fingerprint differs on any field and names every one that moved, not just the first: `algo`, code shape (`n`, `m`, `k`), `dtype`, `device`, the optimizer settings, the schedule, `error_random_seed`, `-bs`, the physical error rate, the parity-check matrix file, and the loss and its weights. The `*_last.pt` states all of that and is held to all of it. `-ckpt`'s yaml is the run's report and states the part of it a person reads a finished run for (`algorithm`, `device`, `data type`, `physical error rate`, `batch size`, `epochs`, the two batch counts and `error random seed`), so it is held to those, which is what catches a pair naming two different runs. A field the checkpoint carries no value for is reported as `not recorded` rather than as one that happens to differ, which is what an older checkpoint looks like. A weights-only `<stem>_best.pt` has no fingerprint at all and is refused with a message pointing at `decoder.config.checkpoint` instead, and a decode run's `decoder_full` yaml under `-ckpt` is refused by the `train_full` block it lacks. `-tckpt` without `-t` is an error, as is combining it with a `decoder.config.checkpoint` key, since both would supply weights from different files. The flag is spelled `-tckpt`, not `-tc`: argparse splits `-tc` into `-t -c`, which would turn a typo into a silent training run.

### 3.12. bp_sf
Normalized min-sum BP followed by syndrome-flipping post-processing on the samples BP leaves unconverged: the most-oscillating bits become flip candidates, and combinations of them are sampled to look for one that satisfies the syndrome. Example configuration (`bp_sf_hx.decoder.yaml`):

```
decoder:
  algorithm: bp_sf
  check_type: hx
  dtype: float64
  device:
    device_type: cpu
    device_idx: 0
  config:
    max_iter: 181
    sf:
      topk: 20
      w_min: 0
      w_max: 2
      n_sample: 200
```

The SF stage is configured by a nested `sf` block under `decoder.config`.

| Key                              | Description                                                              | Default   |
|----------------------------------|--------------------------------------------------------------------------|-----------|
| `decoder.config.sf.topk`         | Most-oscillating bits used as flip candidates                            | `0`       |
| `decoder.config.sf.w_min`        | Minimum flip weight                                                      | `0`       |
| `decoder.config.sf.w_max`        | Maximum flip weight                                                      | `0`       |
| `decoder.config.sf.n_sample`     | Maximum combinations sampled per weight                                  | `0`       |

`w_max` below `w_min` disables SF with a warning, so the defaults leave the decoder as plain normalized min-sum BP.

## 4. Adaptive iteration speedup (`rebatch_speedup`)
An opt-in, per-decoder block consumed by the iterative BP decoders `bp_norm_min_sum`, `bp_norm_min_sum_quant`, `bp4`, `bp_lottery`, `bp_lottery_quant`, `bp_lottery_policy`, and `relay_bp` and by `bp_branch_assisted` on its CUDA path only (other algorithms, e.g. `bp_sf` and `osd_0`, ignore it). It reduces decoding **time** by stopping a batch once a warm-up-learned fraction of samples has converged and deferring the unconverged tail to be re-decoded uncapped.

For these BP decoders the cap is **lossless**: every sample is still fully decoded, so the logical error rate is identical to a no-cap run. The deferred tail (`converge == 0`) is still re-decoded uncapped.

**Setup.** Add an `rebatch_speedup` block to the decoder YAML; omit it to disable the feature.

```
decoder:
  algorithm: bp_norm_min_sum
  check_type: hx
  dtype: float64
  rebatch_speedup:
    kl_eps: 0.001
    kl_window: 2
    kl_min: 3
  device:
    device_type: cuda
    device_idx: 0
  config:
    max_iter: 181
```

| Key                               | Description                                              | Example | Default  |
|-----------------------------------|----------------------------------------------------------|---------|----------|
| `decoder.rebatch_speedup.kl_eps`     | Warm-up KL threshold (larger ⇒ shorter warm-up)          | `0.001` | `0.0001` |
| `decoder.rebatch_speedup.kl_window`  | Consecutive settled batches that end warm-up             | `2`     | `3`      |
| `decoder.rebatch_speedup.kl_min`     | Minimum warm-up batches                                  | `3`     | `3`      |
| `decoder.rebatch_speedup.candidates` | (optional) cap percentiles to consider                   | -       | `0..99`  |

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
