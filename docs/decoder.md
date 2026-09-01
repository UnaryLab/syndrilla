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
SAQ (arXiv:2512.08914), a **learned** decoder: a syndrome is mapped to an error estimate in one feed-forward pass, so there is no message passing and no per-sample iteration count (`max_iter` is ignored, `iter` is always 1, `num_max_iter` is 1). Two token streams are built from the syndrome (a syndrome stream, prefixed with a global token, and a logical stream seeded by a shallow MLP prior) and are updated by `N_dec` transformer layers with asymmetric attention: syndrome self-attention is restricted to the code topology (stabilizers that share a qubit, plus the global token), while the logical stream cross-attends the syndrome tokens without restriction so degenerate errors stay separable. Output heads emit a per-qubit posterior LLR (`llr`, positive ⇒ no error, same sign convention as the BP decoders) and logical class logits.

Stage 4 (CPND, constraint-projected nullspace descent) runs as inference-only post-processing, enabled by default. `project` maps the hard decision onto the operator that satisfies the measured syndrome *and* the predicted logical class exactly over GF(2), via a precomputed right inverse of `[H; L]`; `nullspace_descent` then walks a basis of `ker([H; L])`, every vector of which is a stabilizer, so both constraints survive, greedily accepting any move that lowers the LLR-weighted cost `Σ e_i·llr_i`. The GF(2) algebra is implemented against numpy, inlined in `saq/saq.py` alongside the model as the other decoders inline theirs, so no `galois` dependency is added.

With CPND on, `converge` is 0 only when the measured syndrome lies outside the image of `H`, which perfect measurement cannot produce but noisy syndrome extraction can; chaining `osd_0` after `saq` retries those. With `config.cpnd: {enable: false}` the raw hard decision is reported and `converge` is 1 wherever it happens to be syndrome-consistent. `cpnd` is a block, so a bare `cpnd: false` is rejected, and it belongs under `config` like every other saq setting.

Only **toric and surface codes** are supported, rotated or unrotated. The family is **not configured**: nothing saq does depends on being told which one it is, and both facts that identify it are measurable, so a key could only ever disagree with the matrix it describes. A toric lattice's stabilizers are linearly dependent, giving hx/hz exactly one redundant row, where a surface code's are full rank, and that rank deficiency is checked at construction. The code distance is never inferred: `n` alone does not fix one (25 qubits is a rotated code at distance 5 and an unrotated one at distance 4), so nothing in the decoder reports a distance and run files are named by `n` instead. A config still carrying the removed `code_type` key is rejected rather than ignored. CPND drops a maximal dependent subset of the check rows before building `[H; L]`, since the right inverse needs full row rank; this is the generic form of upstream's hardcoded `H[:-1]` for toric. `llr0` is not consumed: SAQ conditions on the syndrome alone and learns the physical error rate from its training distribution. Standalone example (`saq_hx.decoder.yaml`):

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

All four blocks live under `decoder.config`, since saq is the only reader of any of them, and they are grouped so each has exactly one reader in turn: `model`, `cpnd` and `optimizer` are read by the decoder, `train` by `MetricState`. A key written at the top of the `decoder` block instead is rejected with a message naming where it belongs, rather than silently falling back to its default.

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
| `decoder.config.train.*`            | (training only) The `-t` epoch schedule; see the Training section below         | `epochs: 100`      |

`decoder.dtype` defaults to `float32` here rather than the BP decoders' `float64`, since float64 attention costs several times more for no accuracy gain. **Without `checkpoint` the decoder runs on randomly initialized weights and decodes at chance**, and warns when this happens.

**Training.** Training is a loop, not a module of its own, and each of its five steps is owned by the module that owns the data it touches: the **batch** by the error model and syndrome measurer, the **forward** pass by the decoder, the **loss** by `src/syndrilla/loss/`, and **backward** and **update** by the decoder again. The remaining policy (schedule, model selection, checkpoint I/O) lives in `main.py`, behind the `-t` flag, as a branch inside the batch loop the decode path already uses:

```
syndrilla -t -r=tests/test_outputs \
    -d=examples/alist/saq_hx.decoder.yaml \
    -m=examples/alist/surface_5.matrix.yaml \
    -e=examples/alist/bsc_train.error.yaml \
    -s=examples/alist/perfect.syndrome.yaml \
    -ls=examples/alist/logical_centric.loss.yaml \
    -bs=256
```

Training hyperparameters are split by owner: the decoder yaml holds the optimizer settings under `config.optimizer` (`lr`, `weight_decay`, `min_lr`) and the schedule under `config.train` (`epochs`, `test_batches`, `validation_batches`, `error_random_seed`, and the optional `epochs_saved`), kept apart so each block has a single reader; the `-ls` yaml holds the loss weights (`lambda_lc`, `lambda_lp`, `lambda_ent`) under its `loss` key; the error yaml holds the physical error rates; and the batch size is the CLI's own `-bs`. The schedule sits in the decoder yaml rather than a training yaml of its own so a run is described by one file, with no second one to keep in step. `configure_optimizer(epochs)` builds Adam plus a `CosineAnnealingLR` from the decoder yaml's three keys and stores both on the decoder; the training loop in `main.py` then steps `decoder.optimizer` itself, inline, guarded by the mode `train_set_hyperparameter` put the decoder in. `train_fingerprint()` supplies the settings that `-tckpt` checks a resumed run against, and `train_state()` the optimizer, schedule and RNG it resumes from. `assert_trainable` (in `syndrilla.decoder`, next to the factory that builds them) requires all three of `train_fingerprint`, `configure_optimizer` and `train_state`; a decoder missing any of them is rejected with a clear error.

**Chain order.** A training run drives the **last** decoder in `algorithm`, because the loss is computed on the `io_dict` the chain finishes with and the trained decoder has to be the one that produced it. Earlier stages still run, untrained, ahead of it on every batch, and the schedule, optimizer settings and `checkpoint` are read from the last stage's `config` entry, as is the loss module's binding. So `[bp_norm_min_sum, saq]` trains, while `[saq, osd_0]` is rejected: a stage after the trained decoder would leave the run learning from output its gradient never passed through, and `osd_0`'s GF(2) solve is not differentiable in any case. A chain whose last stage cannot train is rejected with a message naming where a trainable one actually sits.

Training writes `<stem>_best.pt`, `<stem>_last.pt`, `<stem>_result.yaml` and `<stem>_train.log` into `-r`, which defaults to `tests/test_outputs`, the same directory a decode run writes its results to; put the checkpoint path in `decoder.config.checkpoint` and evaluate with the normal CLI. Checkpoint I/O is driven by the metric module, which owns the run directory and the run position; the decoder supplies its own half of the state and owns the file format.

**`<stem>_train.log`.** The run's own record, at two granularities. Every batch writes an indented line carrying the rate it ran at and the loss it produced, split into the bound loss's terms the same way the epoch line is, and each epoch closes with the summary of the batches above it:

```
  batch    1/5  epoch    1/2  train  lr=5.00e-04  loss=2.6372 (lc=1.7743 lp=0.8609 ent=0.6907)  err=0.7031
  batch    2/5  epoch    1/2  train  lr=5.00e-04  loss=4.1865 (lc=3.3204 lp=0.8143 ent=0.7032)  err=0.3281
  batch    3/5  epoch    1/2  train  lr=5.00e-04  loss=4.0172 (lc=3.1708 lp=0.7833 ent=0.6897)  err=0.4062
  batch    4/5  epoch    1/2  val    lr=5.00e-04  loss=2.0701 (lc=1.2145 lp=0.7947 ent=0.6966)  err=0.2812
  batch    5/5  epoch    1/2  val    lr=5.00e-04  loss=2.9120 (lc=2.0696 lp=0.7458 ent=0.6932)  err=0.4844
epoch    1/2  lr=5.00e-04  train_loss=3.6136 (lc=2.7552 lp=0.8195 ent=0.6945)  val_loss=2.4910  val_err=0.3828  0.3s  <- best
```

The batch position runs over the whole period, `test_batches` training batches followed by `validation_batches` validation ones, and the `train`/`val` column names which phase a batch was metered as, so the epoch line's two averages can be read straight back off the batches that produced them. The learning rate is constant across an epoch, since the cosine schedule steps at the epoch boundary; carrying it on every line keeps a batch's loss next to the rate that produced it. Batch lines go to this file only, never to stdout, which carries the epoch summaries alone: a run writes `epochs x (test_batches + validation_batches)` of them, so the shipped 100-epoch schedule leaves about 22,000 lines.

**`<stem>_result.yaml`.** The training counterpart of a decode run's `result_phy_err_<rate>.yaml`, and the same shape: a `train_full` block naming the run (algorithm, `model parameters` (the decoder's trainable weight count; the run's random seed is the separate `error random seed` key), device, dtype, error rate in the form it was configured, a swept range's point count included, batch size, schedule, `error random seed`, `best epoch index` and `best validation error`, what the run cost, and both checkpoint paths as `best checkpoint` and `last checkpoint`), then the results themselves. Two epochs are written, the run's best and its last, which is what a finished run is read for; they collapse to one entry when the last epoch is itself the best, and `epochs_saved` widens the tail. They are stored **by column**: under `training result` sit `epoch`, the list of epoch numbers, `learning rate`, `time (s)`, `best`, and a `training` and a `validation` block each holding a loss and a class error list named for its phase (`training loss` and `training error` under `training`, `validation loss` and `validation error` under `validation`), so a term read out of the file stays unambiguous once it no longer has its parent block for context. The abbreviated `train`/`val` remain the internal phase keys, which is what the in-memory history and `<stem>_last.pt` are still keyed by. Those two are what any trained decoder reports; the split of the total into `lc`, `lp` and `ent` belongs to the logical-centric loss rather than to a run, so it stays in the epoch line, and so in `<stem>_train.log`, and is not written here, where another model's run would otherwise be described in terms it never had. Every list is index-aligned with the epoch list, so `training['training loss'][i]` and `validation['validation error'][i]` belong to epoch `epoch[i]`, and a term is one line to read and one list to plot rather than something to collect out of a block per epoch. The epoch numbers are carried explicitly rather than implied by position, which is what keeps the columns meaningful once the file has been thinned to the best and the last. It is rewritten at every epoch boundary, so a run stopped part way still leaves behind what it did finish:

```yaml
train_full:
  algorithm: saq
  epochs: 12
  best epoch index: 7
  best validation error: 0.171875
  best checkpoint: tests/test_outputs/saq_hx_n41_best.pt
  last checkpoint: tests/test_outputs/saq_hx_n41_last.pt
training result:
  epoch: [7, 12]
  learning rate: [0.0002505, 9.5015e-06]
  time (s): [0.0388, 0.0454]
  best: [true, false]
  training:
    training loss: [1.1736, 1.1249]
    training error: [0.2578, 0.2266]
  validation:
    validation loss: [1.0618, 1.4991]
    validation error: [0.171875, 0.328125]
```

The summary's `best epoch index` and `best validation error` are computed over the whole run rather than over the two epochs written, so thinning the file never moves them.

**What the run cost.** The summary answers the question a decode file's timing block answers, with the epoch standing where the decoder's iteration stands:

```yaml
  total time (s): 340.09              # wall clock of this invocation, setup included
  total epoch time (s): 338.71        # summed over the epochs, restored across a resume
  average time per epoch (s): 3.3871
  average time per batch (s): 0.015396
  average time per sample (s): 6.0142e-05
```

The averages divide the summed *epoch* time rather than the wall clock, for two reasons. The wall clock includes building the decoder and loading the matrices, which is setup rather than training and would inflate a short run's per-epoch cost, and it is the time of this invocation alone, so a resumed run's wall clock covers only the epochs since the resume while its epoch times cover the whole run. A batch here is a batch of either phase, `test_batches + validation_batches` of them per epoch, since both phases run through the same channel and the epoch time is the two together. Each epoch's own time is a column of the curve, so a run that slowed down partway shows it.

**Widening the curve (`epochs_saved`).** The best and the last answer what a finished run is asked, but a run being watched, or one whose curve is being plotted, wants more of it, so `decoder.config.train.epochs_saved` sets how many trailing epochs `<stem>_result.yaml` carries: the last `epochs_saved` of them, still plus the run's best wherever it fell, since that is the epoch the saved checkpoint holds and `best epoch index` names. The summary's `epochs` still reports the whole schedule, and `epochs saved` the width itself, which appears only when the key is set. Only that file is thinned, never `<stem>_last.pt`, so a resumed run restores the complete curve and the width can be changed between runs at no cost. Leave the key out and the file carries the best epoch and the last.

**Checkpoint names.** Both files are named after the configuration that produced them, `<algorithm>_<check_type>_<size>`, so training a second configuration into the same run directory adds files rather than overwriting the first. `<size>` is `n<qubits>`, read straight off the parity-check matrix's column count, and `dem<detectors>x<mechanisms>` for a circuit-level DEM run. It is deliberately not the code distance: `n` does not determine one, since 25 qubits is both a rotated code at distance 5 and an unrotated one at distance 4, so a distance here would be a guess that outlives the run that wrote it. An hx run on the shipped surface code, whose matrix has 41 columns, writes `saq_hx_n41_best.pt` and `saq_hx_n41_last.pt`.

**Resuming (`-tckpt`).** The two checkpoint files differ on purpose. `<stem>_best.pt` is a bare `state_dict`: it is what `decoder.config.checkpoint` points at, and decoding has no use for anything else. `<stem>_last.pt`, rewritten at every epoch boundary, additionally carries what a *run* needs to continue, split by the same ownership rule the loop follows:

| Saved | Owner | Why it is needed |
|---|---|---|
| `state_dict`, `optimizer`, `scheduler`, `rng` | the decoder, via `train_state()` | Adam's per-parameter moments and the cosine schedule's position decide the next step's size and direction; which generators exist is a question about the decoder's own device, so the decoder answers it |
| `epoch`, `best`, `history` | `MetricState.train_compute_avg()` | where the run had got to, and which epoch was best |
| `rng` | `main.py` | the error stream is drawn from the global generators, so a resumed run must continue the same sequence of errors |
| `fingerprint` | the decoder's `train_fingerprint()`, merged by `_train_fingerprint()` | the settings the resumed run must still agree with: the decoder states the model half (`algo`, `n`, `m`, `k`, `lr`, `weight_decay`, `min_lr`), the metrics add the schedule half (`epochs`, `test_batches`, `validation_batches`, `error_random_seed`, `batch_size`) |

`syndrilla -t -tckpt tests/test_outputs/<stem>_last.pt` with otherwise unchanged flags continues the run rather than warm-starting a new one: `MetricState.train_load_checkpoint` restores the decoder and the metrics after `configure_optimizer` has built an optimizer to restore Adam's moments into, `num_batches` is derived there from the epoch counter, so the two cannot drift, and the generators are reseeded for the epoch about to run. That last step is what makes the error stream reproducible: `MetricState.train_set_hyperparameter` reseeds at each phase boundary rather than letting the stream run on, so what a batch draws depends on where the run is and not on the draws before it. The two phases are seeded differently on purpose. Training reseeds to the same value every epoch, so the model trains on one fixed set of batches instead of fresh noise each time round; validation reseeds to an epoch-dependent value well clear of it, so each round of validation is new data and never replays the training set. `tests/test_saq.py::test_resume_cli_finishes_an_interrupted_run` pins the guarantee end to end: a run interrupted after epoch 3 and resumed ends with bit-identical weights and an identical history to an uninterrupted four-epoch run.

That guarantee only holds while the settings are unchanged, so `MetricState.train_validate_checkpoint` refuses a checkpoint whose `algo`, schedule, `error_random_seed`, `-bs`, code shape (`n`, `m`, `k`) or optimizer settings differ, naming the changed field. A weights-only checkpoint has no fingerprint and is refused with a message pointing at `decoder.config.checkpoint` instead. `-tckpt` without `-t` is an error, as is combining it with a `decoder.config.checkpoint` key, since both would supply weights from different files. The flag is spelled `-tckpt`, not `-tc`: argparse splits `-tc` into `-t -c`, which would turn a typo into a silent training run.

The loop hands each finished batch to `MetricState.train_update_metric`, which accumulates it under its phase, closes the epoch through `train_compute_avg` if the batch ended one, and opens the next batch through `train_set_hyperparameter`. What stays in `main()` is what the loop genuinely owns -- run the decoder, read the loss off it, step the optimizer -- while everything keyed by where the run has got to happens in one call, in an order the loop cannot get wrong.

One ordering detail is load-bearing. `MetricState.train_compute_avg` steps the schedule, appends to `history`, advances `epoch`, then writes the checkpoints and the result yaml, then prints the epoch line, then refreshes `lr`. The schedule step lives there because `train_compute_avg` is the only place that knows an epoch just ended, so no caller can step the cosine schedule twice or not at all. Saving last means the checkpoint describes the epoch to run *next* rather than the one just finished; printing last means a visible epoch line implies its checkpoint is already on disk, which is what makes an interrupted run recoverable at exactly that boundary. Refreshing `lr` last is what lets the loop drop its own copy: the finished epoch ran at the rate `MetricState` has carried since the previous call, and what the decoder reports after the step belongs to the epoch about to start.

Training runs after Step 3 of `main()` and reuses the decoder, matrix bundle, error model, and syndrome measurer already built there, so the logical check is never constructed. The metric state is still built, but nothing in the training branch reports from it. Errors come from `error_model.inject_error` and syndromes from `syndrome_generator.measure_syndrome`, making the training channel literally the same code as the evaluation channel. Nothing checks up front whether a multi-round or multi-channel batch is acceptable: what stops such a run is the mismatch itself. `RoundFlattenWrapper` unfolds `e_v` / `synd` / `llr` / `converge` / `iter` back over the rounds dimension but not `logical_logits` / `logical_prior`, so training `saq` on one hands the loss an llr at `[B, d, n]` against a logical head at `[B*d, 2^k]` and it raises on the row counts; a second channel is read as a second round for the same reason. A decoder that consumed the rounds dimension itself would train on either. An error model takes a `rate` range such as `[0.01, 0.20, 9]`, its last value the number of levels, and draws one physical error rate per shot, so a single run covers the whole curve (`bsc_train.error.yaml`). The range is a *mode* of the ordinary model rather than a model of its own: `training` is passed to `create_error_model` and `create_syndrome` the way it is passed to `create_decoder`, a scalar `rate` samples exactly what it sampled before, and a range outside training is refused by the module that reads it. The phenomenological measurer sweeps its `measurement_error_rate` the same way, so a run can vary the data noise and the measurement noise together. The `-i` interface path trains too, with `-ls` still supplying the objective: a stim circuit's detector error model gives a ground-truth error over the columns of the `H` the decoder is built from, so the same loop reads its target from `inject_error` there as it does here (see [interface.md](interface.md) §3), and its `rate` takes a range in the same way. A decoder built from a DEM has detectors for checks and fault mechanisms for variables, so nothing keyed to a code family, distance or qubit count applies to it; the matrix loader flags which kind it is through `is_circuit_dem` rather than leaving decoders to guess from the shape. CPND is off during training, decided by the decoder itself: `create_decoder(..., training=True)` passes the *mode*, and saq turns its own inference-only stage off and skips its GF(2) precompute entirely. `main.py` never rewrites the decoder config, so it does not have to know CPND exists. CPND is not differentiable, cannot affect the loss (which reads `llr` / `logical_logits` / `logical_prior`, all pre-CPND), and would add a Python loop per step.

The loss lives in `src/syndrilla/loss/logical_centric/`, selected by the `-ls` yaml's `function` key, the way the other module factories dispatch on theirs. `terms(io_dict, e)` returns the three unweighted terms `(L_LC, L_LP, L_Ent)` and `combine` applies the configured lambdas, so a loop that logs the components computes them once rather than twice. It reads the ground-truth error plus the `llr`, `logical_logits`, and `logical_prior` entries `forward` writes, so it is coupled to that output contract rather than to any one decoder's internals.

A loss also states how its total is broken down, through a `term_names` attribute: one name per value `terms` returns, in that order, `("lc", "lp", "ent")` here. That is what the metric half meters and logs a run by, so the names of one loss's terms live in that loss and nowhere else, and a second loss is metered by declaring its own rather than by editing the metric module. A loss whose total has no parts worth logging declares `()` and its runs are metered on the total alone. The count is checked at every batch: values land in the slot their position picks, so a loss returning one term more or fewer than it named would file each under the next name and report a run in terms it never computed. `MetricState.train_bind_loss`, called from `train_resume_checkpoint` before the first batch, is what reads the attribute and sizes the accumulators; `total` and `class_err` are the run's own two and a loss may not reuse either name.

`L_Ent` is the one term whose conditioning depends on the code it runs against. It scores the parity of the residual `e XOR prediction` over each logical operator, so that degenerate solutions differing by a stabilizer are not penalised, and it is the only term that supervises the per-bit `llr` at all. Taken in the probability domain, as a product of one `+-1` Bernoulli mean per bit in the operator's support, that parity decays geometrically in the support's size: fine over a handful of qubits, worthless over the 36 error mechanisms a circuit-level detector error model puts in one, where both the value and its gradient underflow and the term reports a constant `ln 2`. It is computed in the log domain instead, on the residual's llr, which is the decoder's own llr with the sign flipped wherever the true error is 1. The sign of the parity stays exact; the magnitude uses the max-log (min-sum) rule, the same approximation belief propagation's check node makes, whose gradient is O(1) on the least certain bit in the support. `_diff_GF2_mul` is kept alongside it as the plain statement of the quantity being approximated, and the tests check the two against each other.

`forward` is one method call per block of the paper's Figure 1, in the figure's order, so the decoding pass reads as the architecture is drawn. The split is naming rather than reuse: unlike the BP decoders' `v2c` / `vn_update` / `cn_update` / `c2v`, none of these is half of a message-passing loop a caller might drive separately, and each has exactly one call site. Only stage 4 is conditional, on `use_cpnd`, and it is skipped entirely during training.

| Method                             | Stage                                                                                                                                              |
|------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `initial_embedding_layer(s±)`      | Stage 1: `MLP` (shallow MLP over the syndrome → initial logical class logits), `learnable_embed_S` / `learnable_embed_L` scaling, `global_tok` prepended; syndrome vector → dual token streams (SAQ's analogue of `v2c`) |
| `SAQ_decoder_layer(SN, LN)`        | Stage 2: `N_dec` layers of topology-masked syndrome self-attention followed by unrestricted syndrome → logical cross-attention reading the *updated* SN, plus the `SN_norm2` / `LN_norm2` re-normalization at `idx == N_dec // 2` |
| `output_layer(SN, LN)`             | Stage 3: `SN_norm` / `LN_norm` → `proj_e` / `proj_l` → `out_fc_*`; token states → per-qubit LLR and logical logits (analogue of `c2v`)                |
| `project(e, s, out_L)`             | Stage 4a: project the hard decision onto the exactly-feasible operator                                                                                |
| `nullspace_descent(e0, llr)`       | Stage 4b: greedily lighten it within the stabilizer coset                                                                                             |
| `syndrome_estimation(e)`           | Shared helper: `H @ e` over GF(2)                                                                                                                     |

`test_forward_matches_the_paper_pipeline` is what pins the wiring: it rebuilds the streams from the learned parameters, drives the layers with the two masks by hand, and applies the heads separately, without calling the stage methods, then requires the result to equal `forward`'s `logical_prior` / `llr` / `logical_logits`. A mis-plumbed mask, a dropped global token, a missed mid-depth norm, or a logical stream reading the previous layer's SN all fail it.

The decoding stages above are distinct from training: the gradient step is the optimizer API described earlier in this section, taken by the loop rather than by the decoder, and is not a step of the decoding pass. The scalar it acts on comes from the loss module's `combine`, not from the decoder.

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
