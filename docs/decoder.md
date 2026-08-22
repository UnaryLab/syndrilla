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

`config` is **positional**: entry *i* belongs to `algorithm[i]`, which is what lets one chain give each stage its own settings (Section 2), including the same algorithm twice with different ones. A block naming a single algorithm may write the mapping directly, as above, instead of a one-element list. Omitting `config` leaves every decoder on its defaults.

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
    - max_iter: 181  # bp_norm_min_sum
```

Each stage reads its own `config` entry, matched by position, so `max_iter` above reaches `bp_norm_min_sum` and not `osd_0`. The list may stop early: `osd_0` configures nothing, so it gets no entry and runs on its defaults. Only *trailing* stages can be left out — position is what binds an entry to an algorithm, so a stage that takes no settings ahead of one that does still needs its slot, written `- {}`. More entries than algorithms is an error rather than a silent drop. The shared keys at the top of the block — `check_type`, `dtype`, `device` — reach every stage.

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

Stage 3 (CPND, constraint-projected nullspace descent) runs as inference-only post-processing, enabled by default. `project` maps the hard decision onto the operator that satisfies the measured syndrome *and* the predicted logical class exactly over GF(2), via a precomputed right inverse of `[H; L]`; `nullspace_descent` then walks a basis of `ker([H; L])`, every vector of which is a stabilizer, so both constraints survive, greedily accepting any move that lowers the LLR-weighted cost `Σ e_i·llr_i`. The GF(2) algebra is implemented against numpy, inlined in `saq/saq.py` alongside the model as the other decoders inline theirs, so no `galois` dependency is added.

With CPND on, `converge` is 0 only when the measured syndrome lies outside the image of `H`, which perfect measurement cannot produce but noisy syndrome extraction can; chaining `osd_0` after `saq` retries those. With `config.cpnd: {enable: false}` the raw hard decision is reported and `converge` is 1 wherever it happens to be syndrome-consistent. `cpnd` is a block, so a bare `cpnd: false` is rejected, and it belongs under `config` like every other saq setting.

Only **toric and surface codes** are supported, rotated or unrotated. The family is **not configured**: nothing saq does depends on being told which one it is, and both facts that identify it are measurable, so a key could only ever disagree with the matrix it describes. A toric lattice's stabilizers are linearly dependent, giving hx/hz exactly one redundant row, where a surface code's are full rank; and the qubit count solves the relation its family fixes (`n = 2d^2` toric, `d^2` rotated, `d^2 + (d-1)^2` unrotated). At construction a deficiency outside `{0, 1}`, or a qubit count fitting no known relation, is warned about. A config still carrying the removed `code_type` key is rejected rather than ignored. CPND drops a maximal dependent subset of the check rows before building `[H; L]`, since the right inverse needs full row rank; this is the generic form of upstream's hardcoded `H[:-1]` for toric. `llr0` is not consumed: SAQ conditions on the syndrome alone and learns the physical error rate from its training distribution. Standalone example (`saq_hx.decoder.yaml`):

```
decoder:
  algorithm: saq
  check_type: hx
  dtype: float32
  device:
    device_type: cpu
    device_idx: 0
  config:
    checkpoint: runs/saq_hx/saq_hx_d5.pt
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
      batches_per_epoch: 200
      val_batches: 20
      seed: 42
```

All four blocks live under `decoder.config`, since saq is the only reader of any of them, and they are grouped so each has exactly one reader in turn: `model`, `cpnd` and `optimizer` are read by the decoder, `train` by `TrainMetrics`. A key written at the top of the `decoder` block instead is rejected with a message naming where it belongs, rather than silently falling back to its default.

| Key                          | Description                                                                    | Example            |
|------------------------------|--------------------------------------------------------------------------------|--------------------|
| `decoder.config.model.d_model`      | Token embedding width                                                           | `128`              |
| `decoder.config.model.N_dec`        | Number of transformer (SLTD) layers                                             | `6`                |
| `decoder.config.model.h`            | Number of attention heads (must divide `d_model`)                               | `16`               |
| `decoder.config.model.dropout`      | (optional) Dropout in attention and feed-forward blocks; training only          | `0.0`              |
| `decoder.config.model.no_mask`      | (optional) `>0` disables the topology attention mask (paper ablation)           | `0`                |
| `decoder.config.cpnd.enable`        | (optional) Run stage 3 (constraint projection + nullspace descent)              | `true`             |
| `decoder.config.cpnd.passes`        | (optional) Sweeps over the stabilizer basis in the descent                      | `1`                |
| `decoder.config.checkpoint`         | (optional) Trained weights to load; a decode run needs it, `-t` writes it       | `runs/saq_hx/saq_hx_d5.pt` |
| `decoder.config.optimizer.lr`       | (optional) Adam learning rate; training only                                    | `5.0e-4`           |
| `decoder.config.optimizer.weight_decay` | (optional) Adam weight decay; training only                                 | `5.0e-8`           |
| `decoder.config.optimizer.min_lr`   | (optional) Floor of the cosine schedule (`eta_min`); training only              | `1.0e-6`           |
| `decoder.config.train.*`            | (training only) The `-t` epoch schedule; see the Training section below         | `epochs: 100`      |

`decoder.dtype` defaults to `float32` here rather than the BP decoders' `float64`, since float64 attention costs several times more for no accuracy gain. **Without `checkpoint` the decoder runs on randomly initialized weights and decodes at chance**, and warns when this happens.

**Training.** Training is a loop, not a module of its own, and each of its five steps is owned by the module that owns the data it touches: the **batch** by the error model and syndrome measurer, the **forward** pass by the decoder, the **loss** by `src/syndrilla/loss/`, and **backward** and **update** by the decoder again. The remaining policy (schedule, model selection, checkpoint I/O) lives in `main.py`, behind the `-t` flag, as a branch inside the batch loop the decode path already uses:

```
syndrilla -t -r=runs/saq_surface_5 \
    -d=examples/alist/saq_hx.decoder.yaml \
    -m=examples/alist/surface_5.matrix.yaml \
    -e=examples/alist/bsc_train.error.yaml \
    -s=examples/alist/perfect.syndrome.yaml \
    -ls=examples/alist/logical_centric.loss.yaml \
    -bs=256
```

Training hyperparameters are split by owner: the decoder yaml holds the optimizer settings under `config.optimizer` (`lr`, `weight_decay`, `min_lr`) and the schedule under `config.train` (`epochs`, `batches_per_epoch`, `val_batches`, `seed`), kept apart so each block has a single reader; the `-ls` yaml holds the loss weights (`lambda_lc`, `lambda_lp`, `lambda_ent`) under its `loss` key; the error yaml holds the physical error rates; and the batch size is the CLI's own `-bs`. The schedule sits in the decoder yaml rather than a training yaml of its own so a run is described by one file, with no second one to keep in step. `configure_optimizer(epochs)` builds Adam plus a `CosineAnnealingLR` from the decoder yaml's three keys and stores both on the decoder; `backward(loss)` and `update()` then apply the gradient. `check_train_batch(rounds, number_channel)` lets the decoder refuse a batch shape it cannot learn from, so that constraint stays with the decoder rather than being hardcoded into the loop, and `train_fingerprint()` supplies the settings that `-tckpt` checks a resumed run against. `assert_trainable` (in `syndrilla.decoder`, next to the factory that builds them) requires all five of `check_train_batch`, `train_fingerprint`, `configure_optimizer`, `backward` and `update`; a decoder missing any of them is rejected with a clear error.

**Chain order.** A training run drives the **last** decoder in `algorithm`, because the loss is computed on the `io_dict` the chain finishes with and the trained decoder has to be the one that produced it. Earlier stages still run, untrained, ahead of it on every batch, and the schedule, optimizer settings and `checkpoint` are read from the last stage's `config` entry, as is the loss module's binding. So `[bp_norm_min_sum, saq]` trains, while `[saq, osd_0]` is rejected: a stage after the trained decoder would leave the run learning from output its gradient never passed through, and `osd_0`'s GF(2) solve is not differentiable in any case. A chain whose last stage cannot train is rejected with a message naming where a trainable one actually sits.

Training writes `<stem>.pt` and `<stem>_last.pt` into `-r`; put the path in `decoder.config.checkpoint` and evaluate with the normal CLI. Checkpoint I/O is driven by the metric module, which owns the run directory and the run position; the decoder supplies its own half of the state and owns the file format.

**Checkpoint names.** Both files are named after the configuration that produced them, `<algorithm>_<check_type>_<size>`, so training a second configuration into the same run directory adds files rather than overwriting the first. `<size>` is the code distance, solved back out of the qubit count through the relations above; a count matching none of them yields `n<count>` instead, since a wrong distance in a filename outlives the run that wrote it. A count satisfying two relations, `n = 25` being a rotated code at distance 5 and an unrotated one at distance 4, takes the first in a fixed order, so the same matrix always yields the same name. A distance-5 hx run on the shipped surface code writes `saq_hx_d5.pt` and `saq_hx_d5_last.pt`.

**Resuming (`-tckpt`).** The two checkpoint files differ on purpose. `<stem>.pt` is a bare `state_dict`: it is what `decoder.config.checkpoint` points at, and decoding has no use for anything else. `<stem>_last.pt`, rewritten at every epoch boundary, additionally carries what a *run* needs to continue, split by the same ownership rule the loop follows:

| Saved | Owner | Why it is needed |
|---|---|---|
| `state_dict`, `optimizer`, `scheduler`, `rng` | the decoder, via `train_state()` | Adam's per-parameter moments and the cosine schedule's position decide the next step's size and direction; which generators exist is a question about the decoder's own device, so the decoder answers it |
| `epoch`, `best`, `history` | `TrainMetrics.train_state()` | where the run had got to, and which epoch was best |
| `rng` | `main.py` | the error stream is drawn from the global generators, so a resumed run must continue the same sequence of errors |
| `fingerprint` | the decoder's `train_fingerprint()`, merged by `TrainMetrics.fingerprint()` | the settings the resumed run must still agree with: the decoder states the model half (`algo`, `n`, `m`, `k`, `lr`, `weight_decay`, `min_lr`), the metrics add the schedule half (`epochs`, `batches_per_epoch`, `val_batches`, `seed`, `batch_size`) |

`syndrilla -t -tckpt runs/<run>/<stem>_last.pt` with otherwise unchanged flags continues the run rather than warm-starting a new one: `load_train_state` restores the decoder and the metrics after `configure_optimizer` has built an optimizer to restore Adam's moments into, `num_batches` is set from `TrainMetrics.batches_done` (derived from the epoch counter, so the two cannot drift), and the generators are reseeded for the epoch about to run. That last step is what makes the error stream reproducible: `TrainMetrics.begin_batch` reseeds at each phase boundary rather than letting the stream run on, so what a batch draws depends on where the run is and not on the draws before it. The two phases are seeded differently on purpose. Training reseeds to the same value every epoch, so the model trains on one fixed set of batches instead of fresh noise each time round; validation reseeds to an epoch-dependent value well clear of it, so each round of validation is new data and never replays the training set. `tests/test_saq.py::test_resume_cli_finishes_an_interrupted_run` pins the guarantee end to end: a run interrupted after epoch 3 and resumed ends with bit-identical weights and an identical history to an uninterrupted four-epoch run.

That guarantee only holds while the settings are unchanged, so `TrainMetrics.validate_checkpoint` refuses a checkpoint whose `algo`, schedule, `seed`, `-bs`, code shape (`n`, `m`, `k`) or optimizer settings differ, naming the changed field. A weights-only checkpoint has no fingerprint and is refused with a message pointing at `decoder.config.checkpoint` instead. `-tckpt` without `-t` is an error, as is combining it with a `decoder.config.checkpoint` key, since both would supply weights from different files. The flag is spelled `-tckpt`, not `-tc`: argparse splits `-tc` into `-t -c`, which would turn a typo into a silent training run.

One ordering detail is load-bearing. `TrainMetrics.record_epoch` appends to `history`, advances `epoch`, then calls the save callback, then prints the epoch line. Saving last means the checkpoint describes the epoch to run *next* rather than the one just finished; printing last means a visible epoch line implies its checkpoint is already on disk, which is what makes an interrupted run recoverable at exactly that boundary.

Training runs after Step 3 of `main()` and reuses the decoder, matrix bundle, error model, and syndrome measurer already built there, so the logical check is never constructed. The metric state is still built, but nothing in the training branch reports from it. Errors come from `error_model.inject_error` and syndromes from `syndrome_generator.measure_syndrome`, making the training channel literally the same code as the evaluation channel. Whether a multi-round or multi-channel batch is acceptable is asked of the decoder through `check_train_batch`, not decided by `main()`: `saq` refuses both, because `RoundFlattenWrapper` unfolds `e_v` / `synd` / `llr` / `converge` / `iter` back over the rounds dimension but not `logical_logits` / `logical_prior`, so the loss would be handed an llr at `[B, d, n]` against a logical head at `[B*d, 2^k]`; a second channel is read as a second round for the same reason. A decoder that consumed the rounds dimension itself would be free to accept one. The training-only `bsc_train` error model takes a `rate` range such as `[0.01, 0.20]` with `rate_points: 9` and draws one physical error rate per shot, so a single model covers the whole curve (`bsc_train.error.yaml`). It is a separate model rather than an option on `bsc`, which keeps its scalar `rate` so no existing decode configuration changes. The `-i` interface path is not supported for training. CPND is off during training, decided by the decoder itself: `create_decoder(..., training=True)` passes the *mode*, and saq turns its own inference-only stage off and skips its GF(2) precompute entirely. `main.py` never rewrites the decoder config, so it does not have to know CPND exists. CPND is not differentiable, cannot affect the loss (which reads `llr` / `logical_logits` / `logical_prior`, all pre-CPND), and would add a Python loop per step.

The loss lives in `src/syndrilla/loss/logical_centric/`, selected by the `-ls` yaml's `function` key, the way the other module factories dispatch on theirs. `terms(io_dict, e)` returns the three unweighted terms `(L_LC, L_LP, L_Ent)` and `combine` applies the configured lambdas, so a loop that logs the components computes them once rather than twice. It reads the ground-truth error plus the `llr`, `logical_logits`, and `logical_prior` entries `forward` writes, so it is coupled to that output contract rather than to any one decoder's internals.

Beyond the common stage helpers `hard_decision` and `syndrome_estimation`, the decoding pass is split into named stages, mirroring how the BP decoders expose `v2c` / `vn_update` / `cn_update` / `c2v`:

| Method                    | Stage                                                                                       |
|---------------------------|---------------------------------------------------------------------------------------------|
| `logical_prior(s)`        | Shallow MLP over the syndrome → initial logical class logits                                  |
| `build_streams(s, prior)` | Format conversion: syndrome vector → dual token streams (SAQ's analogue of `v2c`)             |
| `sn_update(SN, idx)`      | Syndrome-stream self-attention under the topology mask                                        |
| `ln_update(LN, SN, idx)`  | Reverse (syndrome → logical) unrestricted cross-attention                                     |
| `layer_update(SN, LN, idx)`| One full SLTD layer: `sn_update` then `ln_update`, plus the mid-depth re-normalization         |
| `head_update(SN, LN)`     | Output heads: token states → per-qubit LLR and logical logits (SAQ's analogue of `c2v`)       |
| `project(e, s, out_L)`    | Stage 3a: project the hard decision onto the exactly-feasible operator                        |
| `nullspace_descent(e0, llr)` | Stage 3b: greedily lighten it within the stabilizer coset                                  |

The decoding stages above are distinct from the training stages: `backward(loss)` and `update()` are the optimizer API described earlier in this section, not steps of the decoding pass. The scalar they act on comes from the loss module's `combine`, not from the decoder.

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
