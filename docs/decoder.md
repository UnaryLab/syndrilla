# Decoder module

The decoding YAML file defines the decoding algorithm(s), the parity-check side they operate on, the iteration budget, and the device/dtype used during decoding.
A single YAML can specify one decoder or a chained list of decoders that run sequentially: if the first decoder fails to converge on a sample, the next decoder retries that sample.

When the syndrome carries a rounds dimension (`rounds > 1`), the decoder is wrapped in `RoundFlattenWrapper`, which transparently flattens `[B, d, ...]` into `[B*d, ...]` before the inner algorithm and reshapes outputs back. No per-decoder change is needed to support multi-round inputs.

Matrix entries (`parity_matrix_hx`, `parity_matrix_hz`, optional `logical_check_lx`/`logical_check_lz`) live in the matrix YAML loaded via the `-m` flag — see [matrix.md](matrix.md). The decoders below consume them through a pre-loaded `MatrixBundle`.

## 1. Common configuration
The following table details the configuration parameters shared by every decoding YAML file.
| Key                   | Description                                                                  | Example                                            |
|------------------------|-----------------------------------------------------------------------------|-----------------------------------------------------|
| `decoding.algorithm`    | List of decoding algorithms used                                            | `[bp_norm_min_sum, osd_0]`                         |
| `decoding.check_type`   | Type of parity-check matrix used                                            | `hx` or `hz`                                       |
| `decoding.device.device_type`       | Type of the device where the decoding will happen. Without it, `cuda` if available, else `cpu`. A device that is unknown or unavailable falls back the same way, with a warning. `mps` (Apple GPU) runs the PyTorch module and needs `dtype` other than `float64`; `mps` with `float64` falls back to `cpu`, with a warning. | `cpu`, `cuda` or `mps`                                       |
| `decoding.device.device_idx`       | Index of the device where the decoding will happen. This option only works when `device_type = cuda`.                                      | 0                           |
| `decoding.dtype`        | Data type for decoding computations                                         | `float32`, `float64`                              |
| `decoding.force_pytorch`| (optional) Run the plain PyTorch module even on a CUDA device, skipping the fused-CUDA-kernel port | `false`                  |
| `decoding.rebatch_opt`  | (optional) Iteration cap on or off, default `true`; `false` means no iteration cap (Section 4) | `false`                                  |
| `decoding.rebatch_opt_params`| (optional) Block overriding the cap's defaults `kl_eps`, `kl_window`, `kl_min`, `candidates`, `min_pct`, `min_speedup` (Section 4) | `{kl_eps: 0.001}`                                  |
| `decoding.config`       | Algorithm-specific settings, one entry per entry of `decoding.algorithm` (Section 1.1) | `{max_iter: 131}`         |

### 1.1. Algorithm-specific configuration (`decoding.config`)
The block is split by who reads it. The keys above are framework-wide: `main.py`, the loader, and every decoder alike consume them, so they sit at the top of `decoder`. Everything that only one algorithm understands — `max_iter`, the quantization widths, `sf`, relay_bp's leg schedule, saq's `model` and `cpnd` blocks — lives under `config`:

```
decoding:
  algorithm: bp_norm_min_sum
  check_type: hx
  dtype: float64
  device:
    device_type: cuda
    device_idx: 0
  config:
    max_iter: 181
```

The one key almost every algorithm reads is `decoding.config.max_iter`, the per-sample iteration budget; it defaults to `50`, and the non-iterative decoders (`osd_0`, `mwpm`, `union_find`, `saq`) ignore it.

A `config` written as a plain mapping, as above, is the settings of the block's first (or only) algorithm. Written as a list it is **positional**: entry *i* belongs to `algorithm[i]`, which is what lets one chain give each stage its own settings (Section 2), including the same algorithm twice with different ones. Omitting `config` leaves every decoder on its defaults.

A key written in the wrong half is **rejected, not ignored**: `max_iter` left at the top level fails with a message naming `decoding.config`, and a framework-wide key such as `dtype` written inside `config` fails the same way in reverse. A decoder that quietly fell back to `max_iter: 50` instead of the configured `181` would still produce numbers, and they would look like results.

**CUDA acceleration.** Every registered decoder **except `saq`** ships a CUDA-kernel implementation alongside its PyTorch/NumPy module (`saq` is a plain PyTorch model and runs on whatever device the config selects): the belief-propagation family (`bp_norm_min_sum`, `bp_norm_min_sum_quant`, `bp_branch_assisted`, `bp_lottery`, `bp_lottery_quant`, `bp_lottery_policy`, `bp4`, `bp_sf`, `relay_bp`) plus `osd_0`, `mwpm`, and `union_find`. There is **no** separate `*_cuda` algorithm name: set `device.device_type: cuda` and the kernel port (`<algo>/<algo>_cuda.py`) is selected automatically when a CUDA-capable GPU is present and the kernel builds.

The kernels come in two flavors. The BP decoders use **fused per-iteration kernels** that vectorize the message-passing across the batch. The graph decoders `mwpm` and `union_find` are inherently sequential per shot, so their kernels parallelize over the **batch axis** (one CUDA thread decodes one shot), while `osd_0` runs one thread block per sample. For `osd_0`, `mwpm`, and `union_find` the CUDA output is **bit-for-bit identical** to the corresponding CPU implementation.

The selection **falls back to PyTorch automatically**. If no CUDA GPU is available, or the `.cu` kernel fails to build or instantiate (nvcc missing, or a non-NVIDIA accelerator such as AMD ROCm or IBM, where the CUDA kernels do not compile), the plain `<algo>/<algo>.py` PyTorch module runs instead, on whatever device the config resolves to (the CUDA device under ROCm, otherwise CPU). The same fallback applies when an algorithm has no CUDA port. Set `force_pytorch: true` to force the PyTorch module even on an NVIDIA CUDA device. The PyTorch `bp_norm_min_sum` and the PyTorch decoders built on it (`bp_sf`, `bp_lottery`, `bp_lottery_policy`, `bp_norm_min_sum_quant`, `bp_lottery_quant`, `bp_sum_prod`), plus `relay_bp`, `bp_branch_assisted` and `bp4`, compile their iteration body with `torch.compile` on CUDA by default (decoder config key `compile`, default `true`); set `compile: false` to run them eager. Each decoder instance keeps its own compile cache. On CPU, or without torch.compile and Triton, the key is ignored and the eager path runs.

## 2. Chained decoders
A list of algorithms runs each decoder in order; later decoders are only invoked on samples that the earlier ones did not converge on.
An example chained configuration is provided in ```bposd_hx.decoding.yaml```:

```
decoding:
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

### 3.1. bp_norm_min_sum and osd_0
`osd_0` adds one optional key beyond Section 1, `decoding.config.workspace_bytes` (default 4 GiB), described in its entry below. `bp_norm_min_sum`, the PyTorch decoders built on it, `relay_bp`, `bp_branch_assisted` and `bp4` add one optional key, `decoding.config.compile` (default `true`), which applies only to their PyTorch path on a CUDA device; see Section 1.1.

- `bp_norm_min_sum` — normalized min-sum BP. Standalone example (`bp_hx.decoding.yaml`):

```
decoding:
  algorithm: bp_norm_min_sum
  check_type: hx
  dtype: float64
  device:
    device_type: cuda
    device_idx: 0
  config:
    max_iter: 181
```

- `osd_0` — order-0 Ordered Statistics Decoding. Almost always chained after a BP variant; runs only on samples the previous decoder did not converge on. There is no standalone OSD example, since it is configured inside a chained decoding YAML such as `bposd_hx.decoding.yaml` (see Section 2). `osd_0` does not iterate, so it ignores `max_iter`. It scans the columns in ascending LLR order and keeps each column that is independent of those already kept. For a syndrome in the column space of `H` it stops once the syndrome lies in their span, and the correction is the unique solution on the kept columns. A syndrome outside the column space never stops early: the scan runs to the end of the order and the correction solves the system on the pivot rows only. It sorts and scans the batch in chunks of samples whose working memory fits `workspace_bytes` (default 4 GiB), set in the `osd_0` entry of `decoding.config`, the same key `osd_0_cuda` reads. The budget counts `12*(M+1)*U + 16*D*U + 64*(M+1) + 48*U + 60*N` bytes per sample, with `U = ceil((M+1)/64)` words and `D` the largest column weight of `H` rounded up to a power of two: the bit-packed row transform (`8*(M+1)*U`) and its row-update temporaries (`4*(M+1)*U`), the gathered column words, the per-row vectors (a row-transform column, its nonzero indices, the pivot rows and columns), the packed per-sample vectors, and the sort, column order and estimate. The inputs (`llr`, `synd`), the `[B, N]` outputs and the retry stages' copies of the inputs are outside the budget. Its `iter` output is the previous decoder's `iter` (zeros when there is none), with `N`, the number of columns of `H`, on the samples OSD decodes; `converge` is 1 for every sample. On a `cuda` device it uses its CUDA kernel (`osd_0/osd_0_cuda.py`, one thread block per sample), whose correction is bit-for-bit identical to the PyTorch path; it falls back to PyTorch on CPU or when the kernel is unavailable.

### 3.2. bp_norm_min_sum_quant
Normalized min-sum BP with fixed-point quantized messages. Example configuration (`bp_quant_hx.decoding.yaml`):

```
decoding:
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
| `decoding.config.int_width`       | Integer bit width of the fixed-point message representation              | `3`       |
| `decoding.config.frac_width`      | Fractional bit width of the fixed-point message representation           | `4`       |

`decoding.dtype` here applies outside the quantized accumulators.

### 3.3. bp_branch_assisted
Branch-assisted sign-flipping BP. Example configuration (`bsfbp_hx.decoding.yaml`):

```
decoding:
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
| `decoding.config.max_b_iter`      | Maximum branch (sign-flip) iterations per sample                         | `181`     |
| `decoding.config.random_machine`  | Random sampler used for branch perturbations: `sobol` or `system`        | `sobol`   |

### 3.4. bp_lottery
Lottery BP — Sobol/system-driven sign-flip perturbations on the BP messages. Example configuration (`lottery_bp_hx.decoding.yaml`):

```
decoding:
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
| `decoding.config.random_machine`  | Random sampler used to drive sign-flips: `sobol` or `system`             | `sobol`   |
| `decoding.config.flip_start_iter` | Sign-flips start after this iteration (first flip at iteration `flip_start_iter + 1`) | `4`       |
| `decoding.config.flip_interval`   | Iterations between sign-flips; a value that is not an int `>= 1` (a float, a bool or a string too) raises `ValueError` | `1`       |

A sign-flip happens at the end of iteration `i` only when `i > flip_start_iter` and `(i - flip_start_iter - 1) % flip_interval == 0`, so the first flip is at iteration `flip_start_iter + 1` and the next ones every `flip_interval` iterations; on the other iterations no random value is drawn.

### 3.5. bp_lottery_quant
Lottery BP with fixed-point quantized messages. Example configuration (`lottery_bp_quant_hx.decoding.yaml`):

```
decoding:
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
| `decoding.config.random_machine`  | Random sampler used to drive sign-flips: `sobol` or `system`             | `sobol`   |
| `decoding.config.flip_start_iter` | Sign-flips start after this iteration (first flip at iteration `flip_start_iter + 1`) | `4`       |
| `decoding.config.flip_interval`   | Iterations between sign-flips, as in `bp_lottery`                        | `1`       |
| `decoding.config.int_width`       | Integer bit width of the fixed-point message representation              | `3`       |
| `decoding.config.frac_width`      | Fractional bit width of the fixed-point message representation           | `4`       |

`decoding.dtype` here applies outside the quantized accumulators.

### 3.6. bp_lottery_policy
Lottery BP with a selectable sign-flip policy. The policy names follow the paper's five-policy taxonomy plus two extras. Example configuration (`lottery_policy_hx.decoding.yaml`):

```
decoding:
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
| `decoding.config.random_machine`  | Random sampler used to drive sign-flips: `sobol` or `system`             | `sobol`     |
| `decoding.config.sign_flip_policy`| Sign-flip policy (see table below)                                        | `Proposed`  |

`bp_lottery_policy` does not read `flip_interval`: it makes a sign-flip at the end of every iteration.

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
Quaternary BP operating on the 2-channel Pauli prior (used with the depolarizing or 2-channel BSC error model). Example configuration (`bp4.decoding.yaml`):

```
decoding:
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
| `decoding.config.damping_factor`  | Damping factor applied to BP4 messages between iterations                | `0.1`     |

`bp4` consumes both Hx and Hz from the matrix bundle directly; `check_type` is not used.

### 3.8. relay_bp
Relay BP (the `trmue/relay` crate's algorithm). It runs normalized min-sum over a sequence of "legs": leg 1 uses a constant memory strength `init_mem_strength`; each later (ensemble) leg resets the variable→check messages to the prior, carries the posterior forward (the "relay"), and applies random per-variable memory strengths drawn from `[center − width/2, center + width/2]`. Each converged leg yields a candidate solution; the lowest-weight valid one is kept. The leg ensemble stops once every sample has collected `solution` converged solutions (or after `legs` legs). Example configuration (`relay_bp_hx.decoding.yaml`):

```
decoding:
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
| `decoding.config.legs`               | Number of relay legs (ensemble size; the crate's `num_sets`)                                      | `20`      |
| `decoding.config.iteration_initial`  | BP iterations in leg 1                                                                             | `80`      |
| `decoding.config.iteration_count`    | BP iterations in each later (ensemble) leg                                                         | `60`      |
| `decoding.config.solution`           | Converged solutions to collect before stopping (the crate's `stop_nconv`)                         | `5`       |
| `decoding.config.init_mem_strength`  | Leg-1 memory strength `gamma0`                                                                     | `0.35`    |
| `decoding.config.center`             | Center of the per-variable memory-strength interval for ensemble legs                             | `0.21`    |
| `decoding.config.width`              | Width of that interval (drawn from `[center − width/2, center + width/2]`)                         | `0.9`     |
| `decoding.config.alpha`              | Min-sum normalization: `0.0` → adaptive `1 − 2^(−i/alpha_scaling)`; `<0` → `1.0`; else constant   | `0.0`     |
| `decoding.config.alpha_scaling`      | Divisor in the adaptive `alpha` schedule                                                           | `1.0`     |

`relay_bp` uses `iteration_initial`/`iteration_count`/`legs` to bound its work, so it **ignores** the common `max_iter` field. The `center`/`width` defaults match the crate's `gamma_dist_interval = (−0.24, 0.66)`.

### 3.9. mwpm
Minimum-Weight Perfect Matching via the sparse-blossom algorithm, a self-contained clean-room transformation of PyMatching v2 (it does not import `pymatching` or `networkx`). It is **graphlike-only**: every qubit column of `H` must touch at most two checks (a weight-1 column becomes a boundary edge, weight-2 a detector-detector edge; weight>2 raises an error). The matcher runs per shot and is non-iterative, so it ignores `max_iter`. The correction is **bit-for-bit identical** to PyMatching v2 (not merely equal-weight): the radix-heap LIFO tie-break and canonical neighbor order reproduce PyMatching's exact choice on degenerate syndromes. On a `cuda` device the CUDA port (`mwpm/mwpm_cuda.py`) decodes one shot per thread and matches the CPU output bit-for-bit, falling back to the CPU blossom for any shot the kernel cannot handle. Standalone example (`mwpm_hx.decoding.yaml`):

```
decoding:
  algorithm: [mwpm]
  check_type: hx
  dtype: float64
  device:
    device_type: cpu
```

`mwpm` adds two optional settings, both written under `decoding.config` like any other algorithm-specific key.

| Key                              | Description                                                              | Default   |
|----------------------------------|--------------------------------------------------------------------------|-----------|
| `decoding.config.num_workers`     | Worker processes used to decode a batch; `<= 1` keeps the sequential path | CPU count |
| `decoding.config.mp_min_batch`    | Smallest batch that is worth spreading across those workers               | `64`      |

It carries no real per-bit LLR, so it emits a sign-encoded soft output (`llr = 1 − 2·e_v`) and always reports `converge = 1`.

### 3.10. union_find
The Delfosse-Nickerson Union-Find decoder (arXiv:1709.06218): grow clusters around the syndrome defects, fuse them into a spanning forest, then peel to a correction. It is a PyTorch/NumPy port of `chaeyeunpark/UnionFind` and is **bit-for-bit identical** to that reference; matching its output on degenerate syndromes requires reproducing the C++ `tsl::robin_set` iteration order (a `_RobinTable` replica does this). It is **graphlike-only**: every qubit column of `H` must touch at most two checks. A weight-1 column becomes an open-boundary edge (planar/surface codes), a weight-2 column a detector-detector edge (toric codes), and weight>2 is rejected; a boundary vertex is added so open-boundary clusters absorb their unpaired defect. The decoder is non-iterative (ignores `max_iter`). On a `cuda` device the CUDA port (`union_find/union_find_cuda.py`) runs the **entire** decode, both the detector-graph build and the serial grow/fuse/peel, inside the extension (`cuda/union_find_kernel.cu` via `cuda/union_find_serial.cuh`, a line-for-line transliteration of `union_find.py`'s `decode_shot`). It decodes one shot per CUDA thread with its own scratch (the robin-hood iteration order is load-bearing, so each shot's decode is sequential; parallelism is across shots). It covers the **full graphlike domain**, both toric (weight-2 columns) and surface/open-boundary codes (weight-1 columns), and its output is **bit-for-bit identical** to the CPU `decode_shot` for every shot; on toric codes it additionally matches the C++ chaeyeunpark reference bit-for-bit. There is **no per-shot PyTorch fallback**: nothing in the CUDA module imports `union_find.py`. Large batches are split into fixed-memory-budget chunks (not fallbacks) so the per-launch scratch tensor stays bounded. Standalone example (`union_find_hx.decoding.yaml`):

```
decoding:
  algorithm: [union_find]
  check_type: hx
  dtype: float64
  device:
    device_type: cpu
```

`union_find` introduces no algorithm-specific fields beyond Section 1. Like `mwpm` it carries no real per-bit LLR, so it emits `llr = 1 − 2·e_v` and always reports `converge = 1`.

### 3.11. saq
SAQ (arXiv:2512.08914), a **learned** decoder: one feed-forward pass from syndrome to error estimate, so `max_iter` is ignored and `iter` is always 1. The heads emit a per-qubit posterior `llr` and logical class logits; a final CPND stage projects the hard decision onto a syndrome-consistent operator and lightens it within the stabilizer coset, inference-only and skipped during training.

It takes toric and surface codes, rotated or unrotated, and circuit-level detector error models from the stim interface ([interface.md](interface.md)); which one is measured from the matrix, never configured. No code distance is reported, so run files are stemmed `<algorithm>_<check_type>_n<qubits>`, or `dem<detectors>x<mechanisms>` for a DEM. Example (`saq_hx.decoding.yaml`):

```
decoding:
  algorithm: saq
  check_type: hx
  dtype: float32
  device:
    device_type: cpu
    device_idx: 0
  config:
    model:
      d_model: 128
      N_dec: 6
      h: 16
      dropout: 0.0
      no_mask: 0
    cpnd:
      enable: true
      passes: 1
    checkpoint: examples/alist/saq_hx_n41_best.pt
```

`checkpoint` is the only key that tells a fitted architecture from an unfitted one, so it ships as two yamls that differ in that one line: `train_saq_hx.decoding.yaml` leaves it out and is the one a training run fits from random weights ([trainer.md](trainer.md)), and `saq_hx.decoding.yaml` is the file above, the same model pointed at the weights that run produced. Each mode is held to its own file. A decode run is **refused** if a learned decoder in the chain names no weights, since a decoder left at its initialization decodes at chance and the result file reports that like any other logical error rate. Editing the architecture means editing both files, or the checkpoint no longer fits the yaml naming it; a mismatch is refused when the weights are loaded rather than decoded around. The stim path ships the same pair, `train_stim_saq.decoding.yaml` and `stim_saq.decoding.yaml` ([interface.md](interface.md)).

| Key                          | Description                                                                    | Example            |
|------------------------------|--------------------------------------------------------------------------------|--------------------|
| `decoding.config.checkpoint`         | Trained weights; required by every decode run, written by a training run ([trainer.md](trainer.md)) | `examples/alist/saq_hx_n41_best.pt` |
| `decoding.config.model.d_model`      | Token embedding width                                                      | `128`              |
| `decoding.config.model.N_dec`        | Number of transformer (SLTD) layers                                        | `6`                |
| `decoding.config.model.h`            | Number of attention heads (must divide `d_model`)                          | `16`               |
| `decoding.config.model.dropout`      | (optional) Dropout; training only                                          | `0.0`              |
| `decoding.config.model.no_mask`      | (optional) `>0` disables the topology attention mask (paper ablation)      | `0`                |
| `decoding.config.cpnd.enable`        | (optional) Run the CPND stage; a block, so a bare `cpnd: false` is rejected | `true`            |
| `decoding.config.cpnd.passes`        | (optional) Sweeps over the stabilizer basis in the descent                 | `1`                |

Each block has one reader, the decoder itself; a key written at the top of `decoder` is rejected naming where it belongs. The optimizer and the epoch schedule are not here: they configure a run rather than the model, so they belong to the trainer module and live in the training yaml `-tr` names; an `optimizer` or `train` block left in the decoding yaml is rejected pointing there. `dtype` defaults to `float32`, since float64 attention costs several times more for no accuracy gain, and **a decode run naming no `checkpoint` is refused rather than metering random weights**.

The weights this architecture needs are produced by a training run, which the trainer module owns end to end: the objective, the optimizer, the epoch schedule, the run's outputs, and how an interrupted run is resumed. See [trainer.md](trainer.md).

### 3.12. bp_sf
Normalized min-sum BP followed by syndrome-flipping post-processing on the samples BP leaves unconverged: the most-oscillating bits become flip candidates, and combinations of them are sampled to look for one that satisfies the syndrome. Example configuration (`bp_sf_hx.decoding.yaml`):

```
decoding:
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

The SF stage is configured by a nested `sf` block under `decoding.config`.

| Key                              | Description                                                              | Default   |
|----------------------------------|--------------------------------------------------------------------------|-----------|
| `decoding.config.sf.topk`         | Most-oscillating bits used as flip candidates                            | `0`       |
| `decoding.config.sf.w_min`        | Minimum flip weight                                                      | `0`       |
| `decoding.config.sf.w_max`        | Maximum flip weight                                                      | `0`       |
| `decoding.config.sf.n_sample`     | Maximum combinations sampled per weight                                  | `0`       |

`w_max` below `w_min` disables SF with a warning, so the defaults leave the decoder as plain normalized min-sum BP.

## 4. Adaptive iteration speedup (`rebatch_opt`, `rebatch_opt_params`)
The iteration cap stops a batch once a warm-up-learned fraction of its samples has converged and defers the unconverged rest to be decoded again uncapped. It reduces decoding **time**.

The group key `rebatch_opt` (boolean, default `true`, Section 5.1) turns the cap on and off. The `rebatch_opt_params` block only overrides the cap's default parameters:
- `rebatch_opt: true` without a `rebatch_opt_params` block runs the cap with the default parameters in the table below.
- `rebatch_opt: true` with the block runs the cap with the block's values, and the defaults for the keys it leaves out.
- `rebatch_opt: false` runs without the cap, even when the block is present.

`rebatch_opt` goes in a stage's `decoding.config` entry or at the top level of the `decoding` block, like the other group keys (Section 5.3). The key `rebatch_speedup` is not accepted: a config that has it raises a `ValueError` that names `rebatch_opt` and `rebatch_opt_params`.

**Decoders.** The cap is read by `bp_norm_min_sum` (both paths) and the decoders built on its loop: `bp_norm_min_sum_quant`, `bp_sum_prod`, `bp_lottery`, `bp_lottery_quant`, `bp_lottery_policy`; by `bp4` and `relay_bp` (both paths); and by `bp_branch_assisted` on its CUDA path only. `bp_sf`, `osd_0`, `mwpm`, `union_find` and `saq` have no cap. `main.py` keeps the cap only on the first decoder of a chain; BP stages after the first never cap.

**Single round only.** The cap applies when the syndrome has no rounds dimension. With a syndrome generator whose `rounds` is above 1 (the phenomenological measurer), `main.py` logs a warning that the cap supports one round only, and the run never caps. A stim circuit counts as one round, since its detectors already cover every QEC round.

**Warm-up.** For each batch the decoder records the histogram of stop iterations and pools it with the earlier batches. From the second batch on it computes the KL divergence between the pooled histogram with and without the new batch (Laplace smoothed) and logs `batch <n>: KL=<kl> streak=<s>/<kl_window>`. A batch counts as settled when at least `kl_min` batches have been seen and the KL is below `kl_eps`; any other batch resets the streak. After `kl_window` settled batches in a row, the decoder picks the percentile `P` from `candidates` with the best projected speedup (ties keep the higher percentile) and logs `warm-up done after <n> batches: cap p<P> (stop each batch at <P>% converged), projected speedup <x>x`. Only percentiles from `min_pct` to `floor(100 * (1 - f)) - 1` are considered, where `f` is the pooled fraction of samples that stopped at `max_iter` without converging, so `P` stays below the share of samples that converge. When no candidate is in that range, or the best projected speedup is below `min_speedup`, the decoder declines the cap: warm-up ends, every later batch runs uncapped, and it logs `warm-up done after <n> batches: cap off, <reason>`. With the defaults the earliest end of warm-up is batch 5: batches 1 and 2 cannot count as settled (`kl_min: 3`), and batches 3, 4 and 5 make the streak of 3. How many batches it takes beyond that depends on how fast the pooled histogram stops moving.

**Capped batches.** After warm-up each batch stops once `P` percent of its samples have converged. `bp_norm_min_sum` (both paths) marks the samples to decode again in `io["defer"]`, a boolean mask over the batch: the unconverged samples of a batch the cap stopped before `max_iter`. A batch that reaches `max_iter` before `P` percent have converged has run the full iteration count, so it defers nothing and its results are kept as they are. `bp4`, `relay_bp` and `bp_branch_assisted` do not set `io["defer"]`; for them `main.py` defers every unconverged sample of a batch in which the cap was active. `osd_0` (both paths) skips the rows set in `io["defer"]`. `main.py` keeps the results of the other samples, puts the deferred ones in a queue, and decodes them again in later batches with the cap bypassed, so every sample is still fully decoded. A deferred sample reuses the syndrome, initial LLRs and observable flips measured with its first batch, so its result is the same as in a run without the cap. The exception is the `bp_lottery` family (`bp_lottery`, `bp_lottery_policy`, `bp_lottery_quant`) when the sign flip draws from the global RNG (`random_machine: system`, or the `local_random` policy): decoding a deferred sample again in another batch draws new random values, so capped and uncapped runs are not bit-equal. With the default Sobol draws the value depends only on the iteration index, so the result does not change. With `-te` the queue is decoded once it is predicted to hold the remaining errors; with `-tb` once it holds a full batch, and `-tb` counts the first-pass batches only, not the batches of deferred samples. After the run's budget is spent the loop keeps going until the queue is empty.

```
decoding:
  algorithm: bp_norm_min_sum
  check_type: hx
  dtype: float64
  rebatch_opt_params:
    kl_eps: 0.001
    kl_window: 2
    kl_min: 3
  device:
    device_type: cuda
    device_idx: 0
  config:
    max_iter: 181
```

| Key                                    | Description                                              | Example | Default  |
|----------------------------------------|----------------------------------------------------------|---------|----------|
| `decoding.rebatch_opt_params.kl_eps` | Warm-up KL threshold (larger ⇒ shorter warm-up)          | `0.001` | `0.0001` |
| `decoding.rebatch_opt_params.kl_window` | Consecutive settled batches that end warm-up             | `2`     | `3`      |
| `decoding.rebatch_opt_params.kl_min` | Minimum warm-up batches                                  | `3`     | `3`      |
| `decoding.rebatch_opt_params.candidates` | (optional) cap percentiles to consider                   | -       | `0..99`  |
| `decoding.rebatch_opt_params.min_pct` | Lowest cap percentile the chooser considers              | `60`    | `50`     |
| `decoding.rebatch_opt_params.min_speedup` | Projected speedup below which the cap is declined        | `1.2`   | `1.1`    |

**Output.** When a decoder runs with the cap, its per-decoder block in the result YAML gains a `rebatch_opt` entry reporting `warmup batches` (the number of warm-up batches the KL test consumed) and, once a cap is chosen, `chosen pct`; a declined cap reports `warmup batches` only. This entry is emitted **before** the `total time (s)` timing fields.


## 5. Optimization knobs
Every optimization in `bp_norm_min_sum` (both paths) and in `osd_0` (both paths), the H storage, and the iteration cap can be turned off by a decoder config key. Turning one off runs the plain implementation of the same algorithm, so the outputs stay the same; the exceptions are noted below. The decoders built on `bp_norm_min_sum` read only some of the knobs (see Section 5.3). Every boolean knob is `true` when its optimization is on (`persistent_kernel` also takes `auto`, its default); `edge_layout` is the one string knob, and `host_check_every`, `compact_frac` and `workspace_bytes` are numbers. The knobs are for ablation studies and debugging; the defaults are the fastest settings.

### 5.1. Group keys
Six boolean group keys, each default `true`, turn off a whole group at once. Setting a group key to `false` sets each of its members to the off value listed below. A decoder ignores the members it does not read.

| Key           | Idea                                                       | Members, with off value |
|---------------|------------------------------------------------------------|-------------------------|
| `pruning_opt` | Stop or skip work once its result is known                 | `host_check_every: 0`, `skip_converged: false`, `compact_frac: 0`, `osd_early_stop: false`, `osd_prefix_scan: false`, `osd_solve_by_pivots: false`, `osd_skip_converged: false` |
| `fusion_opt`  | Fewer kernel launches and fewer host round trips           | `persistent_kernel: false`, `compile: false`, `fuse_vn: false` |
| `mapping_opt` | Map the message passing onto the GPU's execution model     | `cn_sign_parity: false`, `warp_per_check: false`, `f64_int_compare: false`, `edge_layout: padded` |
| `gather_opt`  | Gather-based sums in a fixed order, same result on every run | `c2v_gather: false`, `vn_gather: false` |
| `memory_opt`  | Allocate only what is live, and pack it                    | `reuse_buffers: false`, `osd_column_scan: false`, `osd_packed_transform: false`, `workspace_bytes: 1 << 60`, `sparse_h: false` |
| `rebatch_opt` | Stop a batch at a learned percentile and defer the slow rest | `rebatch_opt: false` turns the iteration cap off; no other members (Section 4) |

The group map is `GROUP_OFF` in `src/syndrilla/decoder/knobs.py`. `RebatchSpeedup.from_cfg` (`src/syndrilla/decoder/decoder.py`) reads `rebatch_opt`.

### 5.2. Individual knobs
Paths: "BP CUDA" is `bp_norm_min_sum/bp_norm_min_sum_cuda.py`, "BP PyTorch" is `bp_norm_min_sum/bp_norm_min_sum.py`, "OSD" is both `osd_0/osd_0.py` and `osd_0/osd_0_cuda.py`, "H" is the matrix bundle (`matrix/matrix.py`) and the stim syndrome measurer (`syndrome/stim/stim.py`). "per-step path only" means the knob acts on the BP CUDA per-step path and the persistent path never reads it.

`pruning_opt`

| Key                   | Path(s)    | Default | Off behavior |
|-----------------------|------------|---------|--------------|
| `host_check_every`    | BP CUDA    | `8`     | Per-step path checks on the host for early exit every this many iterations; `0` never checks, so the batch runs to `max_iter`. With the rebatch cap active the check runs every iteration. |
| `skip_converged`      | BP CUDA    | `true`  | `false`: the kernels run every sample to the end of the batch, and each converged sample returns its iterate from its convergence iteration. |
| `compact_frac`        | BP PyTorch | `0.75` (at most `0.75`) | Compact the per-iteration state to the unconverged samples once fewer than this fraction of rows are unconverged; `0` never compacts. |
| `osd_early_stop`      | OSD        | `true`  | `false`: the scan does not stop once the syndrome is spanned. PyTorch scans the whole order; CUDA stops at full rank. The outputs are equal. |
| `osd_prefix_scan`     | OSD        | `true`  | `false`: one stable sort over the whole order, instead of a first scan over the `N/16` least reliable columns with retries on `4N/16` and then all columns. The prefix applies only when `N >= 16384` and `osd_column_scan` is on. |
| `osd_solve_by_pivots` | OSD        | `true`  | `false`: back substitution over `M` pivot slots (PyTorch) or elimination over `rank(H)` columns (CUDA), instead of the largest pivot count found. |
| `osd_skip_converged`  | OSD        | `true`  | `false`: decode every sample and keep the result only where `converge` is 0. |

`fusion_opt`

| Key                 | Path(s)    | Default | Off behavior |
|---------------------|------------|---------|--------------|
| `persistent_kernel` | BP CUDA    | `auto`  | `auto` runs the persistent kernel (one launch for all iterations) when `M <= 120`, or when `M <= 1320` and the batch has at least 4 samples per SM; `true` runs it whenever it can; `false` launches once per iteration step. `force_per_step: true` is the same as `persistent_kernel: false` when `persistent_kernel` is not set. |
| `compile`           | BP PyTorch | `true`  | `false`: run the iteration body eager instead of through `torch.compile`. Applies on a CUDA device only. |
| `fuse_vn`           | BP CUDA    | `true`  | `false`: a separate variable-node kernel (`vn_update_csr`) fills an `a_v2c` buffer `[B, nnz + 1]` (CSR) or `[B, M * D + 1]` (padded) before each check update, instead of the variable-node message computed inside the check-node kernel. Per-step path only. |

`mapping_opt`

| Key               | Path(s)    | Default | Off behavior |
|-------------------|------------|---------|--------------|
| `cn_sign_parity`  | BP PyTorch | `true`  | `false`: the check-node update takes the product of signs and `topk(2)`, instead of a sign-bit parity and min/amin. Ignored by a decoder that overrides `cn_update`. |
| `warp_per_check`  | BP CUDA    | `true`  | `false`: one thread per check, a serial scan in `k` order, instead of a warp per check with shuffle reductions; same minima and tie order as the warp merge. Per-step path only. |
| `f64_int_compare` | BP CUDA    | `true`  | `false`: the double check-node compares (sign test, absolute value and minimum match) use float compares instead of integer ops on the bit patterns. Both paths; no effect on other dtypes. |
| `edge_layout`     | BP CUDA    | `csr`   | `padded`: check rows stored as padded `[M, D]` rows whose dummy edges have column index `N` and are skipped, with edge buffers `[B, M * D + 1]`, instead of CSR. Runs on both the per-step and the persistent path. |

`gather_opt`

| Key             | Path(s)    | Default | Off behavior |
|-----------------|------------|---------|--------------|
| `c2v_gather`    | BP PyTorch | `true`  | `false`: sum the check-to-variable messages with `index_add_` and run eager. On CUDA the atomic adds have no fixed order, so the LLRs can differ run to run in the last bits; on CPU the result is bit-identical. Ignored by a decoder that overrides `c2v`. |
| `vn_gather`     | BP CUDA    | `true`  | `false`: the variable-node update sums its messages with one atomic add per edge. A baseline for ablation studies, not an optimization the decoder uses. Per-step path only; does not force `fuse_vn` off. The atomic add order is not fixed, so `llr` varies run to run. At stim d=9, `e_v`, `iter` and `converge` match the default, and `llr` is within rtol `1e-6` on converged rows and rtol `1e-4` / atol `1e-4` on unconverged rows. At stim d >= 11 the drift can change `e_v` and `iter`, and converged-row `llr` can differ beyond rtol `1e-6`. |

`rebatch_opt`

| Key           | Path(s) | Default | Off behavior |
|---------------|---------|---------|--------------|
| `rebatch_opt` | `bp_norm_min_sum` (both paths), `bp_norm_min_sum_quant`, `bp_sum_prod`, the `bp_lottery` family, `bp4`, `relay_bp`, `bp_branch_assisted` CUDA | `true` | `false`: no iteration cap, so every batch runs until all its samples converge or reach `max_iter`, and nothing is deferred. The `rebatch_opt_params` block, if present, is ignored. Single-round runs only (Section 4). |

`memory_opt`

| Key                    | Path(s)     | Default | Off behavior |
|------------------------|-------------|---------|--------------|
| `reuse_buffers`        | BP PyTorch  | `true`  | `false`: allocate the eager work buffers every iteration and compact by plain indexing. |
| `osd_column_scan`      | OSD         | `true`  | `false`: CUDA eliminates over all `N` columns of the order without the column-local scan; PyTorch runs a dense Gauss-Jordan on `H` augmented with `s` over the whole order. |
| `osd_packed_transform` | OSD PyTorch | `true`  | `false`: keep the row transform as a dense bool matrix instead of bit-packed and transposed. |
| `workspace_bytes`      | OSD         | `4 << 30` (4 GiB) | Byte budget per chunk of samples (see Section 3.1). The group off value `1 << 60` puts the whole batch in one chunk. |
| `sparse_h`             | H           | `true`  | Set outside the `decoding` block (see Section 5.3), read by the matrix bundle and the stim syndrome measurer. `true` keeps H as a coalesced bool sparse COO tensor and computes the syndrome with a sparse matmul; `false`: `MatrixBundle.select()` returns a dense int64 H and the syndrome uses a dense matmul. The bundle builds each check type's H once and `select()` returns that same tensor on every call. The outputs are bit-identical. `union_find`, `mwpm` and `saq` read the sparse H from the matrix loader. |

### 5.3. Coupling rules
- `osd_early_stop: false` forces `osd_prefix_scan` off.
- `skip_converged: false`, `warp_per_check: false`, `fuse_vn: false`, `vn_gather: false`, or `persistent_kernel: false` (or `force_per_step: true`) force the BP CUDA per-step path, so the persistent kernel does not run. `edge_layout: padded` does not. The per-step path also runs for a subclass that overrides `_iter_hook` (the `bp_sf` main pass and the `bp_lottery` family), when no persistent block fits on an SM, and when the rebatch cap is active and the batch has more samples than can be co-resident.
- The quantized subclasses (`bp_norm_min_sum_quant`, `bp_lottery_quant`) run the default kernels and ignore `warp_per_check`, `fuse_vn`, `f64_int_compare`, `edge_layout` and `vn_gather`, but `warp_per_check: false`, `fuse_vn: false` and `vn_gather: false` still force their per-step path.
- `bp_sf` runs its persistent retries on the default kernels. `relay_bp` and `bp_branch_assisted` on CUDA ignore the BP CUDA kernel knobs.
- `c2v_gather: false` or `cn_sign_parity: false` force `compile` off.
- `rebatch_opt` is the switch for the iteration cap; the `rebatch_opt_params` block only overrides its default parameters (Section 4). On a syndrome with more than one round `main.py` runs uncapped whatever `rebatch_opt` says.
- An individual key set explicitly wins over its group key: `pruning_opt: false` with `osd_skip_converged: true` turns off every other `pruning_opt` member.
- All knobs and group keys except the H storage go in a stage's `decoding.config` entry, or at the top level of the `decoding` block, where they reach every stage of a chain. A key in a stage's entry wins over the same key at the top level.
- H storage (`sparse_h`, or `memory_opt: false` for its `sparse_h: false` member) is read only outside the `decoding` block: from the `interface` block with `-i` (forwarded to both the stim syndrome measurer and the matrix bundle), else from the `matrix` block (the H the decoders get) and the stim `syndrome` block (the syndrome matmul). `memory_opt: false` in the `decoding` block alone turns off the other `memory_opt` members and leaves H sparse.

```
decoding:
  algorithm: [bp_norm_min_sum, osd_0]
  check_type: hx
  dtype: float64
  memory_opt: false        # every stage
  config:
    - {max_iter: 181, compile: false}
    - {osd_early_stop: false}
```

```
interface:
  backend: stim
  code: surface_code:rotated_memory_x
  distance: 11
  sparse_h: false          # dense H for the syndrome and the decoders
```

## 6. Decoder I/O contract
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
