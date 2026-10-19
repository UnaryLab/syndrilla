# Decoder module

The decoding YAML file defines the decoding algorithm(s), the parity-check side they operate on, the iteration budget, and the device/dtype used during decoding.
A single YAML can specify one decoder or a chained list of decoders that run sequentially: if the first decoder fails to converge on a sample, the next decoder retries that sample.
Every decoder rejects unknown configuration keys with `ValueError`; each stage of a chain uses its own legal keys.

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

The one key almost every algorithm reads is `decoding.config.max_iter`, the per-sample iteration budget; it defaults to `50`, and the non-iterative decoders `osd_0`, `mwpm`, `mwpm_gpu` and `union_find` reject it. `saq` accepts it but runs once.

A `config` written as a plain mapping, as above, is the settings of the block's first (or only) algorithm. Written as a list it is **positional**: entry *i* belongs to `algorithm[i]`, which is what lets one chain give each stage its own settings (Section 2), including the same algorithm twice with different ones. Omitting `config` leaves every decoder on its defaults.

A key written in the wrong half is **rejected, not ignored**: `max_iter` left at the top level fails with a message naming `decoding.config`, and a framework-wide key such as `dtype` written inside `config` fails the same way in reverse. A decoder that quietly fell back to `max_iter: 50` instead of the configured `181` would still produce numbers, and they would look like results.

**CUDA acceleration.** Every registered decoder **except `saq` and `mwpm`** ships a CUDA-kernel implementation alongside its PyTorch/NumPy module (`saq` is a plain PyTorch model and runs on whatever device the config selects): the belief-propagation family (`bp_symbreak`, `bp_norm_min_sum`, `bp_norm_min_sum_quant`, `bp_branch_assisted`, `bp_lottery`, `bp_lottery_quant`, `bp_lottery_policy`, `bp_ots`, `bp_gd`, `bp4`, `bp_sf`, `bp_sf_lottery`, `bp_lottery_pre`, `relay_bp`, `bp_norm_min_sum_layered`, `bp_lottery_layered`) plus `osd_0`, `mwpm_gpu`, and `union_find`. `mwpm` has no CUDA kernel: on any device other than `cpu` it logs an info line, decodes on the host CPU with PyMatching and returns its tensors on the configured device (Section 3.9). There is **no** separate `*_cuda` algorithm name: set `device.device_type: cuda` and the kernel port (`<algo>/<algo>_cuda.py`) is selected automatically.

The kernels come in two flavors. The BP decoders use **fused per-iteration kernels** that vectorize the message-passing across the batch. The graph decoders `mwpm_gpu` and `union_find` are inherently sequential per shot, so their kernels parallelize over the **batch axis** (one CUDA thread decodes one shot), while `osd_0` runs one thread block per sample. For `osd_0`, `mwpm_gpu`, and `union_find` the CUDA output is **bit-for-bit identical** to the corresponding CPU implementation.

The selection **does not fall back to PyTorch on failure**. On a `cuda` device a decoder with a CUDA module loads it, and that module raises `RuntimeError` when no CUDA GPU is present and fails when its kernel does not build (nvcc missing, or a non-NVIDIA accelerator such as AMD ROCm or IBM). The plain `<algo>/<algo>.py` PyTorch module runs only for a decoder without a CUDA module or with `force_pytorch: true`, on whatever device the config resolves to. Set `force_pytorch: true` on ROCm, or to force the PyTorch module even on an NVIDIA CUDA device. The PyTorch `bp_norm_min_sum` and the PyTorch decoders built on it (`bp_sf`, `bp_sf_lottery`, `bp_lottery_pre`, `bp_lottery`, `bp_lottery_policy`, `bp_norm_min_sum_quant`, `bp_lottery_quant`, `bp_sum_prod`, `bp_ots`, `bp_gd`), plus `relay_bp`, `bp_branch_assisted` and `bp4`, compile their iteration body with `torch.compile` on CUDA by default (decoder config key `compile`, default `true`); set `compile: false` to run them eager. Each decoder instance keeps its own compile cache. On CPU, or without torch.compile and Triton, the key is ignored and the eager path runs.

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
| `bp_symbreak`              | 1        | Normalized min-sum BP with per-shot symmetry-breaking check splits | [SymBreak, arXiv:2412.02885](https://arxiv.org/abs/2412.02885) |
| `bp_norm_min_sum`           | 1        | Normalized min-sum belief propagation                                                   | Factor Graphs and the Sum-Product Algorithm                                                                                        |
| `bp_norm_min_sum_layered`        | 1        | Normalized min-sum BP on a layered (serial) schedule: checks are updated layer by layer, each layer reading the newest posteriors | A Reduced Complexity Decoder Architecture via Layered Decoding of LDPC Codes |
| `bp_lottery_layered`        | 1        | Lottery BP on the layered (serial) schedule of `bp_norm_min_sum_layered` | - |
| `bp_norm_min_sum_quant`     | 1        | Normalized min-sum BP with fixed-point quantization                                | -                                                                                                                                  |
| `bp_branch_assisted`        | 1        | Branch-assisted sign-flipping BP (BSFBP)                                                 | Branch-Assisted Sign-Flipping Belief Propagation Decoding for Topological Quantum Codes Based on Hypergraph Product Structure      |
| `bp_sf`                     | 1        | Normalized min-sum BP with syndrome-flipping (SF) post-processing on the most-oscillating bits | Fully Parallelized BP Decoding for Quantum LDPC Codes Can Outperform BP-OSD (Dies-Irae/BP-SF)                                |
| `bp_sf_lottery`             | 1        | `bp_sf` with every BP pass (main pass and SF retries) replaced by lottery BP | -                                                                                                                                  |
| `bp_lottery_pre`            | 1        | Lottery BP with Predict-and-Reduce (PRE) post-processing: top-ranked single-bit syndrome flips decoded as parallel sub-lotteries | Informed Dynamic Scheduling for QLDPC Codes (Huang and Ueng, Quantum 2025) |
| `bp_lottery`                | 1        | Lottery BP                      | -                                                                                                                                  |
| `bp_lottery_quant`          | 1        | Lottery BP with fixed-point quantization                                         | -                                                                                                                                  |
| `bp_lottery_policy`         | 1        | Lottery BP with selectable sign-flip policy (paper's five-policy)               | -                                                                                                                                  |
| `bp_ots`                    | 1        | Sum-product BP with oscillator and least-reliable prior switches; PyTorch and CUDA | Chytas et al., Algorithm 1 (DOI:10.1109/LCOMM.2024.3356312) |
| `bp_gd`                     | 1        | Sum-product BP with guided decimation: one variable's prior fixed to a large magnitude at the end of each round | Belief Propagation Decoding of Quantum LDPC Codes with Guided Decimation (Yao et al., arXiv:2312.10950) |
| `bp4`                       | 2        | Quaternary BP (BP4) operating on the 2-channel Pauli prior                               | Quaternary Neural Belief Propagation Decoding of Quantum LDPC Codes with Overcomplete Check Matrices                               |
| `relay_bp`                  | 1        | Relay BP — normalized min-sum run over multiple "legs" with disordered per-variable memory, keeping the best converged solution | relay-bp crate (crates.io, `trmue/relay`)                                                  |
| `osd_0`                     | 1        | Order-0 Ordered Statistics Decoding               | Soft-Decision Decoding of Linear Block Codes Based on Ordered Statistics                                                            |
| `mwpm`                      | 1        | Minimum-Weight Perfect Matching with PyMatching v2 (dependency `pymatching`). Graphlike codes only (every qubit column touches ≤2 checks) | PyMatching v2 sparse-blossom (Higgott & Gidney); CPU decode on every device                            |
| `mwpm_gpu`                  | 1        | Minimum-Weight Perfect Matching, native sparse-blossom port with a CUDA kernel. Graphlike codes only. Up to 2 orders of magnitude slower than `mwpm` in every measured case; not recommended (Section 3.10) | PyMatching v2 sparse-blossom (Higgott & Gidney); clean-room PyTorch/NumPy transformation, bit-exact to PyMatching in all three `weights` modes |
| `union_find`                | 1        | Union-Find (Delfosse-Nickerson) cluster-growth + peeling decoder. Graphlike codes only (every qubit column touches at most 2 checks; weight-1 = open boundary, e.g. surface codes; weight-2 = toric) | Almost-linear-time decoding for topological codes (arXiv:1709.06218); port of chaeyeunpark/UnionFind         |
| `saq`                       | 1        | Learned dual-stream transformer decoder plus CPND constraint projection. Single feed-forward pass; toric and rotated surface codes only; needs trained weights | SAQ: Stabilizer-Aware Quantum Error Correction Decoder (arXiv:2512.08914); port of DavidZenati/SAQ-Decoder |

### 3.1. bp_norm_min_sum and osd_0
`decoding.config.early_termination` defaults to `true`; `false` runs every sample for `max_iter` iterations in `bp_norm_min_sum`, `bp_norm_min_sum_layered`, `bp_lottery_layered`, `bp_norm_min_sum_quant`, `bp_lottery`, `bp_lottery_quant`, `bp_lottery_policy`, `bp_sf`, `bp_sf_lottery`, and `bp_lottery_pre` on their PyTorch and CUDA kernel paths and returns its final estimate, with convergence checked at the end.
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

- `osd_0`: order-0 Ordered Statistics Decoding. Almost always chained after a BP variant; runs only on samples the previous decoder did not converge on. There is no standalone OSD example, since it is configured inside a chained decoding YAML such as `bposd_hx.decoding.yaml` (see Section 2). `osd_0` has no `max_iter` key. It scans the columns in ascending LLR order and keeps each column that is independent of those already kept. For a syndrome in the column space of `H` it stops once the syndrome lies in their span, and the correction is the unique solution on the kept columns. A syndrome outside the column space never stops early: the scan runs to the end of the order and the correction solves the system on the pivot rows only. It sorts and scans the batch in chunks of samples whose working memory fits `workspace_bytes` (default 4 GiB), set in the `osd_0` entry of `decoding.config`, the same key `osd_0_cuda` reads. The budget counts `12*(M+1)*U + 16*D*U + 64*(M+1) + 48*U + 60*N` bytes per sample, with `U = ceil((M+1)/64)` words and `D` the largest column weight of `H` rounded up to a power of two: the bit-packed row transform (`8*(M+1)*U`) and its row-update temporaries (`4*(M+1)*U`), the gathered column words, the per-row vectors (a row-transform column, its nonzero indices, the pivot rows and columns), the packed per-sample vectors, and the sort, column order and estimate. The inputs (`llr`, `synd`), the `[B, N]` outputs and the retry stages' copies of the inputs are outside the budget. Its `iter` output is the previous decoder's `iter` (zeros when there is none), with `N`, the number of columns of `H`, on the samples OSD decodes; `converge` is 1 for every sample. On a `cuda` device it uses its CUDA kernel (`osd_0/osd_0_cuda.py`, one thread block per sample), whose correction is bit-for-bit identical to the PyTorch path; it falls back to PyTorch on CPU or when the kernel is unavailable.

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
Lottery BP uses Sobol or system draws for sign-flip perturbations on the BP messages. Example configuration (`bp_lottery_hx.decoding.yaml`):

```
decoding:
  algorithm: bp_lottery
  check_type: hx
  dtype: float64
  device:
    device_type: cuda
    device_idx: 0
  config:
    max_iter: 121
```

| Key                       | Description                                                              | Example   |
|---------------------------|--------------------------------------------------------------------------|-----------|
| `decoding.config.random_machine`  | Random sampler used to drive sign-flips: `sobol` or `system`             | `sobol`   |

A sign-flip happens at the end of every iteration in a `lottery` phase, before the final iteration. No flip happens at `max_iter`, so the returned `llr` of a non-converged shot is the BP posterior and `llr <= 0` equals `e_v` on every row.

A single flip at iteration `t` uses `flip_phase_iters: [t-1, t]` and `flip_mode: [off, lottery, off]`.

**Flip pick knobs.** The pick keys select the variable whose posterior is negated. Each flip picks one variable per row. A row with no unsatisfied check gets no flip. The pick has two steps. The first step selects a candidate set of variables with `cn_pick`. The second step selects variables from that set with `vn_pick`.

| Key                                  | Description                                                              | Default      |
|--------------------------------------|--------------------------------------------------------------------------|--------------|
| `decoding.config.cn_pick`            | Candidate set: `rand_unsat`, `max_err_unsat`, `all_unsat` or `all`       | `rand_unsat` |
| `decoding.config.cn_pick_tie_break`  | Breaks `max_err_unsat` ties with the sampler draw; must be a bool        | `true`       |
| `decoding.config.vn_pick`            | Variable pick in the candidate set: `max_unsat` or `random` | `max_unsat`  |
| `decoding.config.vn_pick_tie_break`  | Enables the second sort key of `vn_pick`; must be a bool                 | `true`       |

The `cn_pick` values:
- `rand_unsat` uses the sampler draw to select one unsatisfied check uniformly. The candidates are the variables on that check.
- `max_err_unsat` selects the unsatisfied check with the most ones in the current hard decision. The candidates are the variables on that check. With `cn_pick_tie_break: true`, the sampler draw selects uniformly among tied checks. With `false`, the tie goes to the lowest check index.
- `all_unsat` uses no single check. The candidates are all variables on at least one unsatisfied check.
- `all` uses no syndrome filter. The candidates are all variables.

`cn_pick_tie_break` has an effect only with `max_err_unsat`.

The `vn_pick` values:
- `max_unsat` selects the candidate with the most unsatisfied checks. With `vn_pick_tie_break: true`, ties go to the smallest `|LLR|`, then to the lowest index. With `false`, ties go to the lowest index.
- `random` selects one candidate uniformly. It ignores `vn_pick_tie_break`. With `sobol`, it uses a second Sobol stream, so the check draws stay the same as with the other `vn_pick` values. With `system`, it draws one more value per row from the global RNG.


Zero posterior candidates are skipped when a nonzero candidate exists in the pool. An all-zero pool falls back to its original candidates.

A flip sets the picked variable's posterior to `-LLR`. Invalid pick values or a tie-break key that is not a bool raise `ValueError`.

The default setting (`rand_unsat`, `max_unsat`, both tie-breaks `true`) picks an unsatisfied check from the sampler draw, then on that check the variable with the most unsatisfied checks, with ties broken by the smallest `|LLR|`, and negates its posterior. Every lottery path accepts these keys with the same meaning: the PyTorch module, the CUDA module, `bp_lottery_quant` (Section 3.5), `bp_sf_lottery` (Section 3.14) and `bp_lottery_layered` (Section 3.17).

**Phase controls.** `bp_lottery` supports phase controls in its PyTorch and CUDA paths.

| Key | Description | Default |
|-----|-------------|---------|
| `decoding.config.flip_phase_iters` | Nondecreasing list of up to two nonnegative iteration boundaries. Each boundary is the last iteration of its phase. Equal values give an empty phase. | `[40]` when both phase keys are absent |
| `decoding.config.flip_mode` | `lottery`, `pre` or `off`. | `[lottery, pre]` when both phase keys are absent |

When neither `flip_phase_iters` nor `flip_mode` is set, the decoder uses lottery through iteration 40, then serial PRE with `pre_max_lambda: 16` and `pre_max_iter: 5`. This default also applies to `bp_lottery_quant`, `bp_lottery_layered`, `bp_lottery_pre`, and `bp_sf_lottery`. An explicit `flip_mode`, as a scalar or list, preserves that mode without an implicit PRE phase. Explicit phase boundaries also suppress the default schedule. With boundaries alone, every phase uses lottery. Use `flip_mode: lottery` for a lottery-only decode.

With `max_iter > 40`, a default `bp_lottery`, `bp_lottery_quant`, `bp_lottery_layered`, or `bp_lottery_pre` runs without a rebatch cap because the PRE phase disables it.

With `max_iter <= 40`, the default PRE phase is empty. It performs no initial check and keeps the lottery rebatch-cap behavior.

`cn_pick`, `vn_pick`, and `flip_mode` accept a scalar for all phases or a list with one entry per phase. A list of any other length raises `ValueError`. The first flip of a lottery phase follows its starting boundary.

`off` disables new flips in that phase.

**PRE phase.** `flip_mode: pre` keeps the ordinary lottery flip at each iteration. It also ranks the current residual by node degree minus twice the number of unsatisfied checks. Ties go to the lowest node index. Estimated bits and previously tried candidates are excluded. At each PRE window check, a single-bit syndrome match in the top pool fixes the shot before the normal convergence check. Otherwise the next candidate starts with its prior and posterior at the negative channel magnitude. The prior stays changed through that window, then returns to its channel value. The posterior is not clamped. Messages carry between windows.

| Key under `decoding.config` | Meaning | Default |
|---|---|---|
| `pre_max_lambda` | Positive PRE pool size and maximum number of serial candidates | `16` |
| `pre_max_iter` | Positive update count per serial candidate | `5` |

These two keys require a `pre` phase. Ranking and the first single-bit check occur at the phase boundary, before the first PRE update. A phase at the start of the decode begins at its first check. Each later window reranks and skips tried nodes. Only complete windows within the main budget run; a phase with no full window still performs its initial single-bit check. After its candidate budget, ordinary lottery continues. Quantized and layered lottery inherit the PRE phase. The separate policy decoder uses its own pick rules and does not support this phase.

**CPU and GPU results.** On one device the PyTorch module and the CUDA kernels give the same shots. Between CPU and GPU the aggregate rates agree, but individual shots can differ. The cause is floating-point summation order, which differs between CPU and GPU. With the default knobs, the sign-flip picks one variable by the largest number of unsatisfied checks, with ties broken by the smallest `|LLR|`, and such ties are frequent. A difference in the last bit of an LLR can therefore pick a different variable, and that shot's trajectory diverges from there. It holds at every `dtype`. `bp_norm_min_sum` makes no such pick and gives the same shots on CPU and GPU; `bp_sf` can differ (Section 3.13).

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
| `decoding.config.int_width`       | Integer bit width of the fixed-point message representation              | `3`       |
| `decoding.config.frac_width`      | Fractional bit width of the fixed-point message representation           | `4`       |

`decoding.dtype` here applies outside the quantized accumulators.

The flip pick knobs (`cn_pick`, `cn_pick_tie_break`, `vn_pick`, `vn_pick_tie_break`) are those of `bp_lottery` (Section 3.4).

CPU and GPU runs can differ on individual shots, as for `bp_lottery` (Section 3.4).

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

`bp_lottery_policy` makes a sign-flip at the end of every iteration except the last (`i == max_iter`), so the returned `llr` of a non-converged shot has no negated entry. CPU and GPU runs can differ on individual shots, as for `bp_lottery` (Section 3.4).

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
Minimum-Weight Perfect Matching with PyMatching v2 (`mwpm/mwpm.py`; dependency `pymatching>=2,<3`, `pyproject.toml:30`). It is **graphlike-only**: every qubit column of `H` must touch at most two checks (a weight-1 column is a boundary edge, weight-2 a detector-detector edge); a heavier column raises `ValueError` at create, "Column {c} has weight {d} > 2; H is not graphlike, so the MWPM decoder does not apply." (`mwpm.py:59-66`). The matcher is `pymatching.Matching.from_check_matrix` on the sparse `H` as a scipy CSC matrix (`mwpm.py:55-58`, `:82`), so the fault ids are the columns of `H` and the correction `e_v` is a column vector like every other decoder's. It is non-iterative and ignores `max_iter`. The default configuration, `weights: prior` with no quantization, is the weighting of `pymatching.Matching.from_detector_error_model`: on the decomposed stim DEM of `stim_mwpm.interface.yaml` the default decoder and PyMatching built from the DEM predict the same observables on every shot, 10000 shots at distance 5 and p 0.001 and 20000 shots at distance 9 and p 0.01 (`tests/test_mwpm.py:287-312`). There is no CUDA kernel: on a `cuda` device, or any device other than `cpu`, the decoder logs one info line at create, "mwpm decodes on the host CPU with PyMatching and returns tensors on the configured device." (`mwpm.py:31-33`), moves the syndrome and the LLRs to the host, decodes in NumPy and returns `e_v`, `llr`, `converge` and `iter` on the configured device (`mwpm.py:126`, `:134`, `:160-166`). Standalone example (`mwpm_hx.decoding.yaml`):

```
decoding:
  algorithm: [mwpm]
  check_type: hx
  dtype: float64
  device:
    device_type: cpu
  config:
    - weights: uniform
```

Edge weights are floats passed to PyMatching as they are: with `weights: prior` the edge of qubit `v` gets weight `|llr0_v|`, and PyMatching maps the float weights to its internal integer grid itself (the limit `2^24 - 1` at `mwpm.py:11` is PyMatching's weight limit; the mapping is PyMatching's, not this repo's); with `weights: posterior` the same formula reads the previous stage's `llr` (belief-matching, Higgott et al. 2023, arXiv:2203.04948), so the stage must follow a BP stage in `algorithm`. In both modes a qubit with `llr_v <= 0` is preflipped: its `H` column is added to the syndrome before matching, it is matched with weight `|llr_v|` (0 for `llr_v = 0`), and it is flipped back into `e_v` after matching (`mwpm.py:137-141`, `:158-159`). A missing input, a wrong shape, a complex tensor or a non-finite value raises `ValueError` (`mwpm.py:131-136`); a weight above PyMatching's limit of `2^24 - 1` raises "Edge weights exceed the PyMatching backend limit of 16777215." (`mwpm.py:78-81`). In `uniform` mode one `Matching` is built at create with PyMatching's default weights and decodes every batch with `decode_batch` (`mwpm.py:67`, `:144`); in `prior` mode the batch is grouped by unique weight row and one `Matching` is built per group per forward (`mwpm.py:147-153`); in `posterior` mode one `Matching` is built per shot per forward (`mwpm.py:154-157`). In `prior` and `uniform` mode the matcher runs on every shot, BP-converged ones too, and reports `converge = 1`; in `posterior` mode `skip_converged` defaults to `true` (`mwpm.py:24`; the default per mode holds on `mwpm_gpu` too, `mwpm_gpu.py:1710`), so the converged shots pass through as the previous stage returned them, and `skip_converged: false` matches them too. `mwpm_gpu` (Section 3.10) solves the same problem without the `pymatching` dependency and normalizes the float weights the way PyMatching does; its corrections match `mwpm` bit for bit in all three modes (`tests/test_mwpm.py:89-108`). Chain example (`bp_mwpm_hx.decoding.yaml`):

```
decoding:
  algorithm: [bp_norm_min_sum, mwpm]
  check_type: hx
  dtype: float64
  device:
    device_type: cpu
  config:
    - max_iter: 50
    - weights: posterior
```

The optional settings, all written under `decoding.config` like any other algorithm-specific key (`mwpm.py:21-28`):

| Key                              | Description                                                              | Default   |
|----------------------------------|--------------------------------------------------------------------------|-----------|
| `decoding.config.weights`         | `prior` (float edge weights `\|llr0\|` from the input `llr0`, the log-prior of each column, so no BP stage is needed; the PyMatching `from_detector_error_model` weighting), `posterior` (the same formula on the previous stage's `llr`, see above) or `uniform` (PyMatching's default weights, no `llr` input needed); any other value raises `ValueError` | `prior`   |
| `decoding.config.skip_converged`  | `true`: in `posterior` mode, only the shots the previous stage left unconverged are matched; a converged shot keeps the previous stage's `e_v`, `llr`, `converge` and `iter` (its BP values), while a matched shot gets the matcher's `e_v`, its sign-encoded `llr` and `converge = 1` (`mwpm.py:87-119`). `true` with `weights: uniform` or `prior` raises `ValueError`, as does a value other than `true` or `false` (`mwpm.py:24-28`, `mwpm_gpu.py:1707-1714`). Separate from the BP CUDA pruning knob of the same name (Section 5) | `true` with `posterior`, `false` otherwise |


The three `weights` modes, as PyMatching conditions:

| Mode      | Weight per column                                                                 | PyMatching equivalent                                                                                   | When to use |
|-----------|-----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|-------------|
| `prior`   | `\|llr0\|`, the log-prior `log((1-p)/p)` of each column from the error model, as a float | `Matching.from_detector_error_model(dem)`, and `Matching.from_check_matrix(H, weights=log((1-p)/p))` | The default: circuit-level or code-capacity decoding with the error model's priors, the PyMatching baseline |
| `uniform` | every column 1                                                                    | `Matching.from_check_matrix(H)` with no weights                                                         | The plain MWPM baseline with no prior information |
| `posterior` | `\|llr\|` of the previous BP stage as a float (read from the io_dict key `llr`), with the preflip on `llr <= 0` | none                                                                                                    | Belief matching after a BP stage; the only mode that allows `skip_converged`, and converged shots pass through untouched unless `skip_converged: false` |

The default is `weights: prior`; with no `config` block the decoder matches on the error model's priors like `Matching.from_detector_error_model`.

It carries no real per-bit LLR, so for every shot it matches it emits a sign-encoded soft output (`llr = 1 − 2·e_v`) and reports `converge = 1`; `iter` is the defect count of the matched (preflip-adjusted) syndrome, at least 1; the shots `skip_converged` keeps carry the previous stage's values.

### 3.10. mwpm_gpu
**Not recommended: use `mwpm` instead.** `mwpm_gpu` is up to 2 orders of magnitude slower than `mwpm` (PyMatching) in every measured case, so do not use it.

The native sparse-blossom port: a self-contained clean-room transformation of PyMatching v2 (it imports neither `pymatching` nor `networkx`) with a CUDA kernel (`mwpm_gpu/mwpm_gpu.py`, `mwpm_gpu/mwpm_gpu_cuda.py`, `cuda/mwpm_gpu_kernel.cu`). It solves the same problem as `mwpm` (Section 3.9) with the same `weights` modes, the same graphlike-only rule (a column of weight above 2 raises `ValueError`, `mwpm_gpu.py:68-72`), the same preflip on `llr <= 0`, and the same `llr`, `converge` and `iter` outputs, and its corrections match `mwpm` bit for bit in `uniform`, `posterior` and `prior` mode on the CPU path and on the CUDA path (`tests/test_mwpm.py:89-108`, `test_200_shot_native_matches_pymatching`: a random graphlike `H` with parallel columns, 200 shots, normal LLRs). The radix-heap LIFO tie-break and the canonical neighbour order (boundary edges first, then the other edges sorted by column in `uniform` mode and in insertion order in the float modes, `mwpm_gpu.py:1448-1458`, `:1514-1516`) reproduce PyMatching's choice on degenerate syndromes (`RadixHeapQueue`, `mwpm_gpu.py:116`, LIFO pop at `:157`). The matcher runs per shot and is non-iterative, so it ignores `max_iter`. Standalone example (`mwpm_gpu_hx.decoding.yaml`, the `mwpm_hx` example with `algorithm: [mwpm_gpu]` and `config` written as a mapping):

```
decoding:
  algorithm: [mwpm_gpu]
  check_type: hx
  dtype: float64
  device:
    device_type: cpu
  config:
    weights: uniform
```

Weights are normalized the way PyMatching normalizes them: with `weights: posterior` or `prior` the float weight of qubit `v` is `|llr_v|` (`mwpm_gpu.py:1870`); after the parallel columns of each node pair, and the repeated boundary columns of each node, collapse to the lowest-weight column (ties to the lowest column index, `mwpm_gpu.py:1427-1445`, `:1493`), the retained weights of a shot are left as they are when they are all integers, and are otherwise scaled so that the largest one maps to `2^24 - 1`; each scaled weight is rounded half away from zero and doubled, so the matcher works on even integers, a zero weight stays zero, and the reported matching weight is scaled back (`mwpm_gpu.py:1494-1505`; the kernel path does the same per shot in `mwpm_gpu_cuda.py:163-170`). A shot whose weights are all zero keeps them at zero. In `uniform` mode every edge weighs 2 with the reported weight scaled back to unit edges (`mwpm_gpu.py:386`, `:1479-1481`). `posterior` mode reads the key `llr` of its input, so it must follow a BP stage in `algorithm`; `prior` mode reads `llr0`; a missing input, a wrong shape, a complex tensor or a non-finite value raises `ValueError` (`mwpm_gpu.py:1864-1869`), and a weight above `2^24 - 1` raises "Edge weights exceed the PyMatching backend limit of 16777215." (`mwpm_gpu.py:1484-1489`, `mwpm_gpu_cuda.py:199-200`). On a `cpu` device the matcher runs per shot, or over the `num_workers` process pool when `num_workers > 1` and the batch is at least `mp_min_batch` (`mwpm_gpu.py:1766`); in `prior` mode the shots are grouped by unique weight row and one matcher per row is built in each worker and rebuilt per batch (`mwpm_gpu.py:1673-1681`, `:1831-1849`). On a `cuda` device `mwpm_gpu_cuda.py` decides the path once at create from `N`, the number of `H` columns (`mwpm_gpu_cuda.py:105`): for `N <= 64` the kernel decodes one shot per thread and returns the correction as a bit mask; for `64 < N <= 256` the kernel returns the match edges and the host rebuilds each shot's shortest paths on the CPU, over the `num_workers` pool or serially by the same rule as the `cpu` device (`mwpm_gpu_cuda.py:230-259`); a shot the kernel cannot handle (arena cap, iteration cap, or a correction that fails the syndrome check) is redone on the CPU blossom with a warning (`mwpm_gpu_cuda.py:260-272`). For `N > 256` the decoder logs a warning at create and the whole batch runs the serial CPU matcher in the main process, so `num_workers` is unused on that path (`mwpm_gpu_cuda.py:107-117`, `:273-278`); the "decoder ready (CUDA blossom kernel, ...)" line is logged on that path too (`mwpm_gpu_cuda.py:133`). The kernel output matches the CPU port bit-for-bit in `uniform` and `posterior` mode (`tests/test_mwpm_gpu.py`, `test_cuda_matches_cpu_surface10`). The kernel is compiled by nvcc on first use in each Python environment and cached afterwards: at create in `uniform` mode, on the first forward in `posterior` and `prior` mode (`mwpm_gpu_cuda.py:101`, `:212-213`). Chain example (`bp_mwpm_hx.decoding.yaml` with `mwpm_gpu` in place of `mwpm`):

```
decoding:
  algorithm: [bp_norm_min_sum, mwpm_gpu]
  check_type: hx
  dtype: float64
  device:
    device_type: cpu
  config:
    - max_iter: 50
    - weights: posterior
```

The optional settings, all written under `decoding.config` like any other algorithm-specific key (`mwpm_gpu.py:1707-1714`, `:1746-1747`):

| Key                              | Description                                                              | Default   |
|----------------------------------|--------------------------------------------------------------------------|-----------|
| `decoding.config.num_workers`     | Worker processes used to decode a batch; `<= 1` keeps the sequential path. The pool is created on the first forward and kept for the decoder's lifetime; it starts with `fork` when the parent has not initialised CUDA and with `forkserver` after CUDA init; in the `forkserver` case a user script must guard its entry with `if __name__ == "__main__":`, or the pooled decode fails with `BrokenProcessPool` (one retry with a fresh pool, then the error is raised) | CPU count |
| `decoding.config.mp_min_batch`    | Smallest batch that is worth spreading across those workers               | `64`      |
| `decoding.config.weights`         | `prior` (per-shot float weights `\|llr0\|` from the input `llr0`, the log-prior of each column, so no BP stage is needed, normalized as above; the PyMatching `from_detector_error_model` weighting, the same default as `mwpm`), `posterior` (the same on the previous stage's `llr`) or `uniform` (every edge weight 1); any other value raises `ValueError` | `prior`   |
| `decoding.config.skip_converged`  | `true`: in `posterior` mode, only the shots the previous stage left unconverged are matched; a converged shot keeps the previous stage's `e_v`, `llr`, `converge` and `iter` (its BP values), while a matched shot gets the matcher's `e_v`, its sign-encoded `llr` and `converge = 1`. `true` with `weights: uniform` or `prior` raises `ValueError`, as does a value other than `true` or `false` (`mwpm.py:24-28`, `mwpm_gpu.py:1707-1714`). Separate from the BP CUDA pruning knob of the same name (Section 5) | `true` with `posterior`, `false` otherwise |

It carries no real per-bit LLR, so for every shot it matches it emits a sign-encoded soft output (`llr = 1 − 2·e_v`) and reports `converge = 1`; `iter` is the defect count of the matched (preflip-adjusted) syndrome, at least 1; the shots `skip_converged` keeps carry the previous stage's values.

### 3.11. union_find
The Delfosse-Nickerson Union-Find decoder (arXiv:1709.06218): grow clusters around the syndrome defects, fuse them into a spanning forest, then peel to a correction. It is a PyTorch/NumPy port of `chaeyeunpark/UnionFind` and is **bit-for-bit identical** to that reference; matching its output on degenerate syndromes requires reproducing the C++ `tsl::robin_set` iteration order (a `_RobinTable` replica does this). It is **graphlike-only**: every qubit column of `H` must touch at most two checks. A weight-1 column becomes an open-boundary edge (planar/surface codes), a weight-2 column a detector-detector edge (toric codes), and weight>2 is rejected; a boundary vertex is added so open-boundary clusters absorb their unpaired defect. The decoder is non-iterative (ignores `max_iter`). On a `cuda` device the CUDA port (`union_find/union_find_cuda.py`) runs the **entire** decode, both the detector-graph build and the serial grow/fuse/peel, inside the extension (`cuda/union_find_kernel.cu` via `cuda/union_find_serial.cuh`, a line-for-line transliteration of `union_find.py`'s `decode_shot`). It decodes one shot per CUDA thread with its own scratch (the robin-hood iteration order is load-bearing, so each shot's decode is sequential; parallelism is across shots). It covers the **full graphlike domain**, both toric (weight-2 columns) and surface/open-boundary codes (weight-1 columns), and its output is **bit-for-bit identical** to the CPU `decode_shot` for every shot; on toric codes it additionally matches the C++ chaeyeunpark reference bit-for-bit. There is **no per-shot PyTorch fallback**: nothing in the CUDA module imports `union_find.py`. Large batches are split into fixed-memory-budget chunks (not fallbacks) so the per-launch scratch tensor stays bounded. Standalone example (`union_find_hx.decoding.yaml`):

```
decoding:
  algorithm: [union_find]
  check_type: hx
  dtype: float64
  device:
    device_type: cpu
```

`union_find` introduces no algorithm-specific fields beyond Section 1. Like `mwpm` and `mwpm_gpu` it carries no real per-bit LLR, so it emits `llr = 1 − 2·e_v` and always reports `converge = 1`.

### 3.12. saq
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

### 3.13. bp_sf
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
    max_iter: 100
    sf:
      topk: 50
      w_min: 1
      w_max: 10
      n_sample: 10
```

The SF stage is configured by a nested `sf` block under `decoding.config`.

| Key                              | Description                                                              | Default   |
|----------------------------------|--------------------------------------------------------------------------|-----------|
| `decoding.config.sf.topk`         | Most-oscillating bits used as flip candidates                            | `0`       |
| `decoding.config.sf.w_min`        | Minimum flip weight                                                      | `0`       |
| `decoding.config.sf.w_max`        | Maximum flip weight                                                      | `0`       |
| `decoding.config.sf.n_sample`     | Maximum combinations sampled per weight                                  | `0`       |

`w_max` below `w_min` disables SF with a warning, so the defaults leave the decoder as plain normalized min-sum BP. With SF on, a missing or zero `topk` becomes `50` with a warning.

SF follows the reference implementation's parallel path. The flip candidates of an unconverged sample are its `topk` most-oscillating bits; among bits with equal counts the higher bit index is kept. Each sample draws its own combinations, by weight ascending from `w_min` to `w_max`. All trials of all unconverged samples decode at once, each from the channel prior with the full `max_iter`. The returned estimate is the trial that converges earliest, that is with the fewest iterations, with ties going to the lowest trial index. On a rescued sample `converge` is set to 1 and `iter` keeps the main-pass count. The SF iteration cost goes in two extra outputs, `iter_serial` and `iter_parallel` (Section 6).

CPU and GPU runs can differ on individual shots here too. The candidate pick is the same on both, but on GPU the retries run on the persistent kernel by default, so a borderline trial can converge on one device and not on the other. `iter` matches between CPU and GPU; `iter_serial` and `iter_parallel` of the affected shots do not.

### 3.14. bp_sf_lottery
`bp_sf` with lottery BP in place of normalized min-sum BP. The main pass is lottery BP that also counts how often each bit's hard decision flips. On the samples it leaves unconverged, the most-oscillating bits become flip candidates, as in `bp_sf`, and every SF retry is a lottery BP run too. Example configuration (`bp_sf_lottery_hx.decoding.yaml`):

```
decoding:
  algorithm: bp_sf_lottery
  check_type: hx
  dtype: float64
  device:
    device_type: cuda
    device_idx: 0
  config:
    max_iter: 100
    random_machine: sobol
    sf:
      topk: 50
      w_min: 1
      w_max: 10
      n_sample: 10
```

The lottery keys sit directly under `decoding.config`, and the SF keys in the nested `sf` block, as in `bp_sf`. The flip pick knobs of `bp_lottery` (Section 3.4) also apply, to the main pass and to every SF retry.

| Key                               | Description                                                              | Default   |
|-----------------------------------|--------------------------------------------------------------------------|-----------|
| `decoding.config.random_machine`  | Random sampler used to drive sign-flips: `sobol` or `system`, as in `bp_lottery` | `sobol`   |
| `decoding.config.sf.topk`         | Most-oscillating bits used as flip candidates                            | `0`       |
| `decoding.config.sf.w_min`        | Minimum flip weight                                                      | `0`       |
| `decoding.config.sf.w_max`        | Maximum flip weight                                                      | `0`       |
| `decoding.config.sf.n_sample`     | Maximum combinations sampled per weight                                  | `0`       |

The SF keys follow the `bp_sf` rules: `w_max` below `w_min` disables SF with a warning, and a missing or zero `topk` with SF on becomes `50` with a warning. With SF disabled the decoder returns the `bp_lottery` result. With `random_machine: sobol` every retry replays the same Sobol sequence as the main pass; with `system` each retry draws new values from the global RNG. Candidates, combinations and the returned trial follow the `bp_sf` rules: all trials decode at once and the earliest to converge is returned. On a sample an SF retry recovers, `converge` is set to 1 and `iter` keeps the main-pass count; `iter_serial` and `iter_parallel` carry the SF iteration cost (Section 6). The decoder has no rebatch cap (Section 4). The retries run on the per-step loop on GPU, not on the persistent kernel, so the only cross-device difference is the lottery pick: CPU and GPU runs can differ on individual shots, as for `bp_lottery` (Section 3.4); this also changes `iter_serial` and `iter_parallel` of the affected shots.

### 3.15. bp_ots
Sum-product BP with the oscillating-trapping-set prior switches of Dimitris Chytas, Michele Pacenti, Nithin Raveendran, Mark F. Flanagan and Bane Vasic, "Enhanced Message-Passing Decoding of Degenerate Quantum Codes Utilizing Trapping Set Dynamics," IEEE Communications Letters, 2024, Algorithm 1 ([DOI](https://doi.org/10.1109/LCOMM.2024.3356312), [author manuscript](https://par.nsf.gov/servlets/purl/10493558)). The paper's check-node rule is sum-product, not min-sum. The PyTorch module uses the existing probability-domain sum-product BP loop. The CUDA module reuses the base modular BP per-step loop, variable-node updates, hard decisions, syndrome checks and convergence kernels. It replaces only the check-node arithmetic with probability-domain sum-product and adds the OTS prior switch. `force_pytorch: true` uses the PyTorch loop and switch instead.

Each variable has a counter of hard-decision changes, starting from the channel hard decision. On an unconverged sample, every `interval` completed iterations before the last iteration, select the variable with the largest positive counter. Ties select the smallest absolute posterior LLR, then the lowest index. Also select the least-reliable variable, by absolute posterior LLR and then index. Restore every channel prior, replace these selected priors with `-strength`, and clear only the oscillator's counter. Keep the check-to-variable messages. The next BP update uses these changed priors. If every counter is zero, bias only the least-reliable variable. This resolves the undefined oscillator selection in Algorithm 1's empty-set case. The shared hard-decision rule treats a zero posterior as bit one.

| Key | Meaning | Default |
|-----|---------|---------|
| `max_iter` | Positive main-pass iteration limit | `50` |
| `interval` | Positive number of iterations between prior switches | `10` |
| `strength` | Finite positive magnitude of each selected negative prior | `10.0` |

The paper uses period 9 for surface and toric codes and period 7 for bicycle codes. It gives no numeric value for the positive strength constant. The configurable defaults above retain this implementation's settings. Both float32 and float64 are supported. Early termination and row compaction use the shared BP loop. The inherited rebatch cap is honored, including deferred rows and cap bypass. Example: [bp_ots_hx.decoding.yaml](../examples/alist/bp_ots_hx.decoding.yaml).

All six optimization groups use the shared knob resolver. An explicit member overrides its group. On PyTorch, `compact_frac`, `reuse_buffers` and CUDA `compile` control the inherited loop. On native CUDA, `host_check_every`, `skip_converged`, `warp_per_check`, `fuse_vn`, `edge_layout` and `vn_gather` control the base kernels and loop. Both paths honor `rebatch_opt` and `rebatch_opt_params`.

The following members have no effect on OTS in the stated path. They are not separate OTS tuning controls.

| Member | Path | Reason |
|--------|------|--------|
| `persistent_kernel` | Both | Native CUDA needs the interval hook and always uses the per-step loop. PyTorch does not call a native persistent kernel. |
| `cn_sign_parity`, `f64_int_compare` | Both | These choose min-sum sign/minimum comparisons. SPA uses probability products and floating-point clamps instead. |
| `c2v_gather` | PyTorch | SPA supplies its fixed-order gather and sum; the base alternative does not replace this override. |
| `host_check_every`, `skip_converged`, `warp_per_check`, `fuse_vn`, `edge_layout`, `vn_gather` | PyTorch | These select native CUDA kernels. The tensor loop uses its own convergence checks, padded gather and row compaction. |
| `compile`, `compact_frac`, `reuse_buffers`, `c2v_gather` | Native CUDA | These select tensor-loop compilation, compaction and buffers. The modular CUDA loop masks converged rows and uses its own buffers. |
| `compile` | CPU PyTorch | The shared BP compiler path is CUDA-only. CPU runs eager. |
| `vn_gather` | OTS switch only | Prior replacement recomputes the posterior in fixed check-slot order from retained messages. Ordinary native BP updates still honor this member. |
| `osd_*`, `workspace_bytes` | Both | OTS has no OSD stage or OSD workspace. |

`sparse_h` is a matrix-loader option, as for the other BP decoders. It changes how the upstream matrix bundle is stored; OTS receives the resulting bundle and builds the same edge tables.

### 3.16. bp_norm_min_sum_layered
Normalized min-sum BP with a layered, also called serial, schedule. At construction the checks are split into layers by a greedy first-fit pass in check order: a check joins the first layer that shares no variable with it. One iteration is one sweep over the layers in order. Each layer computes its check-to-variable messages from the current posteriors, so it sees the updates of every earlier layer in the same iteration, and writes its new messages back into the posteriors before the next layer runs. The flooding schedule of `bp_norm_min_sum` instead updates every check from the posteriors of the previous iteration. Example configuration (`bp_norm_min_sum_layered_hx.decoding.yaml`):

```
decoding:
  algorithm: bp_norm_min_sum_layered
  check_type: hx
  dtype: float64
  device:
    device_type: cuda
    device_idx: 0
  config:
    max_iter: 181
```

**Relation to `bp_norm_min_sum`.** The arithmetic is the same: the same min-sum check update, the same per-iteration normalization factor and no damping. Only the update order differs. The decoder adds no key of its own and takes the `bp_norm_min_sum` keys: `max_iter`, `early_termination`, the rebatch cap (Section 4), `compact_frac` (`pruning_opt` group) and `reuse_buffers` (`memory_opt` group) (Section 5). None of the `fusion_opt` knobs has an effect. `compile` is ignored and the PyTorch module always runs eager, because each layer must write the posteriors in place before the next layer reads them. `persistent_kernel` and `fuse_vn` have no effect because the CUDA module subclasses the PyTorch layered decoder, not the `bp_norm_min_sum` CUDA module. The knobs that select the check-node and gather variants of the flooding update have no effect either, since the layered sweep does not use those code paths.

**CUDA port.** On a `cuda` device the CUDA module runs one kernel launch per layer per iteration, plus the convergence check per iteration; it raises `ValueError` on any other device. It runs at `float32` and `float64`. Its hard decisions, convergence flags and iteration counts match the PyTorch module, and its `llr` agrees to within floating-point rounding.

**Hook contract.** The decoder calls `_iter_hook` and `_exit_hook` with the same arguments and at the same points in the iteration as `bp_norm_min_sum`, so a subclass written against those hooks on `bp_norm_min_sum` runs on the layered schedule unchanged. Its outputs are the `bp_norm_min_sum` outputs (Section 6).

### 3.17. bp_lottery_layered
Lottery BP (Section 3.4) on the layered schedule of `bp_norm_min_sum_layered` (Section 3.16). The decoder combines the two: the sign-flip logic comes from `bp_lottery` through the same iteration hook, and the message passing is the layered sweep. Example configuration (`bp_lottery_layered_hx.decoding.yaml`):

```
decoding:
  algorithm: bp_lottery_layered
  check_type: hx
  dtype: float64
  device:
    device_type: cuda
    device_idx: 0
  config:
    max_iter: 181
    random_machine: sobol
```

The keys, defaults and validation are those of `bp_lottery`:

| Key                               | Description                                                              | Default   |
|-----------------------------------|--------------------------------------------------------------------------|-----------|
| `decoding.config.random_machine`  | Random sampler used to drive sign-flips: `sobol` or `system`; any other value falls back to `sobol` with a warning | `sobol`   |

**The flip.** The flip schedule and the pick are those of `bp_lottery`: one posterior is negated at the end of each iteration in a lottery phase, before `max_iter`. The pick follows the knobs of Section 3.4. The sampler draw is one quasi-random value per flip iteration, shared by all rows, with `sobol`, and one value per row from the global RNG with `system`. The flip is transient. The next layered sweep runs on the flipped posterior, and after that sweep the flip is undone while the message updates of the sweep are kept. The undo comes before the hard decision, so a flip lasts one iteration, as in flooding lottery BP, and the channel prior stays fixed. No flip happens at iteration `max_iter`, so the returned `llr` of a non-converged shot holds no negated entry.

**Other keys.** The `bp_norm_min_sum_layered` rules apply: `max_iter`, `early_termination`, the rebatch cap (Section 4), `compact_frac` and `reuse_buffers` (Section 5) are supported; `compile` is ignored and the PyTorch module always runs eager; the `fusion_opt` knobs and the check-node and gather variants of the flooding update have no effect. Pending flips follow their samples when the batch is compacted.

**CUDA port.** On a `cuda` device the CUDA module runs the `bp_norm_min_sum_layered` layer kernel, one launch per layer per iteration; the flip and its undo run as tensor operations on the device. It raises `ValueError` on any other device. It runs at `float32` and `float64`. Its hard decisions, convergence flags, iteration counts and random state match the PyTorch module, and its `llr` agrees to within floating-point rounding. With `random_machine: system` the decoder draws its random values the way the `bp_lottery` PyTorch module does, which is not the draw order of the `bp_lottery` CUDA module, so the two decoders do not share a random stream.

### 3.18. bp_lottery_pre

This decoder adds optional parallel PRE retries to the lottery main pass. The serial PRE phase is `flip_mode: pre`, as in Section 3.4. With no `pre_parallel_*` key, this decoder runs the same main pass as `bp_lottery`.

```yaml
decoding:
  algorithm: bp_lottery_pre
  config:
    max_iter: 121
    pre_parallel_max_lambda: 32
```

Any of the following keys enables parallel retries. These keys are accepted only by `bp_lottery_pre`.

| Key under `decoding.config` | Meaning | Default |
|---|---|---|
| `pre_parallel_max_lambda` | Positive number of residual-ranked trials per open shot | `32` |
| `pre_parallel_max_iter` | Positive update budget per trial | Main `max_iter` |
| `pre_parallel_start_iter` | Main-pass check at which open shots leave the main pass | PRE budget end, capped at main `max_iter`; otherwise main `max_iter` |

The inferred parallel start is the end of the last complete PRE window within its phase, limited by `pre_max_lambda` and the phase boundary. With the default serial schedule and `max_iter: 121`, this is iteration 120. A nonempty phase with no complete window ends its PRE work at its initial check. With no nonempty PRE phase, the parallel start is the main budget. An explicit start must be within the main budget and at or after the PRE work ends.

The parallel phase ranks again from the final hard decision. It uses the residual syndrome, excludes estimated bits, and breaks equal PRE scores by node index. A candidate column equal to the residual fixes the shot at zero extra iterations. Each remaining candidate starts a sub-lottery from the channel prior with that column toggled in its target syndrome. A converged trial adds the candidate bit back to its estimate and reverses the corresponding posterior sign. The posterior sign matches the final hard decision, including a small positive crossing when a corrected zero needs bit zero.

With early termination on, the first converging trial wins and its siblings stop. Ties go to the lowest rank. With early termination off, trials run their full budgets. Main-pass successes remain unchanged. The returned `iter` is the main-pass count. `iter_parallel` adds the winner's updates, or the full trial budget if all fail. `iter_serial` uses the lowest rank observed at the sibling stop. A slower lower rank is unobserved, so this is an upper bound on the serial model. Active PRE work disables the rebatch cap. CUDA uses the per-step lottery kernels and the same sibling stop.

### 3.19. bp_symbreak

This study decoder implements Algorithm 1 and the generic split in Sec. IV-C of
[SymBreak, Yin et al.](https://arxiv.org/abs/2412.02885). It uses normalized min-sum
BP and the same global normalization schedule as `bp_norm_min_sum`. The paper
does not fix this BP variant. It does not use the BB-specific split in Sec. VI.
Example: [bp_symbreak_hx.decoding.yaml](../examples/alist/bp_symbreak_hx.decoding.yaml).

| Key under `decoding.config` | Meaning | Default |
|---|---|---|
| `bp_iters` | Positive number of BP updates between split attempts | `10` |
| `k_max` | Positive stagnation monitor limit | `3` |
| `max_splits` | Nonnegative per-shot split limit | `10` |
| `assign` | Syndrome assignment: `bp` or `syndrome` | `bp` |

`max_iter` is the total BP budget, including all split stages. A partial final
block uses the remaining budget. Convergence is checked against the original
matrix only at block ends, including the final partial block. A converged shot's
iteration count records that block end. At each complete block, mismatch
stagnation increments K; a decrease decrements positive K. The first block
establishes the reference.
K reaching `k_max` stops that shot. `early_termination: false` disables convergence
and K stops and returns the state at `max_iter`. A zero mismatch skips splitting.
The decoder returns the common fields plus `splits`, a per-shot count. CPU and
CUDA support `float32` and `float64`. Set `device.device_type: cuda` to select the
CUDA kernels. Set `force_pytorch: true` to run the torch graph loop on that device.
Example: [bp_symbreak_cuda_hx.decoding.yaml](../examples/alist/bp_symbreak_cuda_hx.decoding.yaml).

The torch loop uses the base NMS check-node update. The CUDA port runs BP and
original-matrix syndrome checks in a kernel with one block per shot. A second
kernel updates K and selects, assigns and moves split edges. Its check graphs
are per shot; the base CUDA NMS kernels require a shared graph and cannot use
this edge layout. The global beta schedule, split rules, channel resets, K stop
and total BP budget match the torch decoder.

Optimization groups use the shared NMS defaults. Explicit member keys override
their group values. The graph loops support these paths:

| Group | Torch path | CUDA kernel path |
|---|---|---|
| `pruning_opt` | `compact_frac` reduces BP work to active rows and retains shot ids | `compact_frac` packs active rows and their graph state |
| `fusion_opt` | No effect: the graph loop uses its own BP step and split/cap barriers | No effect: the per-shot BP kernel is fused and split/cap barriers remain |
| `mapping_opt` | `cn_sign_parity` selects the base parity or sign-product check update | `cn_sign_parity` selects parity or sign-product arithmetic |
| `gather_opt` | `c2v_gather` selects ordered incident-edge gather or `scatter_add_` | `vn_gather` selects ordered gather or ordered scatter; `c2v_gather` supplies its default |
| `memory_opt` | `reuse_buffers` reuses BP work buffers within the batch | `reuse_buffers` retains posterior/message buffers; false copies them each update |
| `rebatch_opt` | Enables or disables the adaptive iteration cap | Same cap, with the threshold checked at block ends |

The gather path rebuilds torch incident-edge tables after a split block. The
CUDA kernel changes only moved edge ids and keeps each variable's incident ids
in edge order. Its scatter path also sums in edge order. Check-reliability sums use slot order on both devices. CPU sums and the default
CUDA gather path are deterministic. Torch CUDA `c2v_gather: false` uses atomics;
addition order can change soft values and correction choices near a tie.

`compile`, `persistent_kernel` and `fuse_vn` have no effect on these graph loops.
`warp_per_check`, `f64_int_compare` and `edge_layout` have no effect because the
CUDA port uses a fixed per-shot padded mapping. `host_check_every` and
`skip_converged` have no effect: block boundaries determine syndrome checks and
freeze stopped shots. `vn_gather` applies only to the kernel port. The torch
loop always runs eager. These limits also apply when a group sets those members.

Each shot starts with its own padded graph. Each original check can split once.
Both halves become ineligible for further splitting, so graph growth is bounded
by the original check count. The least reliable opposite check uses mean absolute
posterior LLR. Ties use row id.
Among its eligible overlapping checks, the decoder selects the least reliable
check, with row id as the tie rule. A valid split needs a positive even overlap.
It skips the split when the selected opposite check has no valid candidate.
Qubit ties use qubit id. Odd check sizes use floor(size / 2); if insufficient
non-overlap qubits exist in BP-guided mode, the first side keeps the smaller
valid odd-overlap set.
These choices fill details left open by the paper.

The BP assignment takes the parity of the first side's hard decisions. The other
syndrome is the original syndrome XOR that parity. Syndrome-guided mode selects
a balanced qubit partition that satisfies the near-half odd-overlap rule and
has a side whose other adjacent checks all have zero syndrome. It chooses the first
valid partition in qubit-index order, assigns zero to that side and assigns the
original syndrome to the other side. An empty set of other neighbors satisfies
this condition. If no valid partition exists for the selected check, the split
is skipped and the stagnation monitor continues. Moved edge messages
restart from their channel LLR; other edge messages persist. Unary split checks
use finite message bounds, sqrt(finfo.max) / (edge_slots + 1), to avoid inf - inf.
Split exhaustion continues BP until a stop condition. These numerical and message
choices are explicit in the module docstring.

### 3.20. bp_gd
Sum-product BP with guided decimation (GD), from Hanwen Yao, Waleed Abu Laban, Christian Haeger, Alexandre Graell i Amat and Henry D. Pfister, "Belief Propagation Decoding of Quantum LDPC Codes with Guided Decimation", arXiv:2312.10950, Algorithm 1. The decoder runs BP in rounds of `round_iters` iterations. The check-to-variable messages start once and persist across rounds. At the end of each round, on each unconverged sample, the decoder decimates one variable. It picks the undecimated variable with the largest absolute posterior, with ties to the lowest index. It sets the channel prior of that variable to `+llr_max` if the posterior is strictly positive, and to `-llr_max` if not. The next round starts from the kept messages and the changed priors. A decimated prior stays fixed in later rounds. It is a finite value, not a hard pin. Example configuration:

```
decoding:
  algorithm: bp_gd
  check_type: hx
  dtype: float64
  device:
    device_type: cpu
  config:
    max_iter: 181
    round_iters: 10
    llr_max: 25
```

| Key | Meaning | Default |
|-----|---------|---------|
| `max_iter` | Total BP iteration budget; a positive int, at least `round_iters` | `50` |
| `round_iters` | BP iterations per round; a positive int | `10` |
| `llr_max` | Magnitude of a decimated prior; finite and positive | `25` |
| `saturation` | Variable-to-check message magnitude limit from eq. 23; finite and positive | `25` |
| `max_rounds` | Largest number of rounds; a positive int | number of variables |
| `order` | Decimation order: `reliability` (largest absolute posterior) or `random` (a uniform pick among the undecimated variables) | `reliability` |
| `random_machine` | Random source for `order: random`: `sobol` (unscrambled, one draw per round shared by all samples) or `system` (the device RNG) | `sobol` |

A value of the wrong type, or outside these ranges, raises `ValueError`. `dtype` must be `float32` or `float64`.

**Budget.** The number of rounds is `min(max_rounds, n, floor(max_iter / round_iters))`, where `n` is the number of variables. The default `max_iter` gives five rounds at the default `round_iters`. The decoder runs only complete rounds, so iterations of `max_iter` beyond the last complete round are not used. No decimation follows the last round. The paper's budget is `max_iter = n * round_iters`. With `max_iter = n`, use `round_iters: 1` to get the paper's number of rounds.

**Early termination.** With `early_termination: true` the decoder checks the syndrome after every iteration, as BP in the paper does. With `early_termination: false` it checks only at the end of each round. A sample that matches the syndrome there stops, with no further rounds.

**Outputs.** The decoder returns the standard outputs (Section 6) and one extra output, `rounds`: the number of decimations done on each sample. `iter` counts BP iterations.

**Precision.** With the same random draws, float64 results are the same on the CPU and CUDA paths. With float32, results can differ between backends on unconverged samples: a small difference in the posterior can change the decimated variable or the sign of its prior.

On a `cuda` device the CUDA module runs the sum-product check update with the `bp_ots` CUDA kernels, and the decimation runs as tensor operations between iterations. `force_pytorch: true` selects the PyTorch module. The CPU and CUDA `system` random streams are not the same.

## 4. Adaptive iteration speedup (`rebatch_opt`, `rebatch_opt_params`)
The iteration cap stops a batch once a warm-up-learned fraction of its samples has converged and defers the unconverged rest to be decoded again uncapped. It reduces decoding **time**.

The group key `rebatch_opt` (boolean, default `true`, Section 5.1) turns the cap on and off. The `rebatch_opt_params` block only overrides the cap's default parameters:
- `rebatch_opt: true` without a `rebatch_opt_params` block runs the cap with the default parameters in the table below.
- `rebatch_opt: true` with the block runs the cap with the block's values, and the defaults for the keys it leaves out.
- `rebatch_opt: false` runs without the cap, even when the block is present.

`rebatch_opt` goes in a stage's `decoding.config` entry or at the top level of the `decoding` block, like the other group keys (Section 5.3). The key `rebatch_speedup` is not accepted: a config that has it raises a `ValueError` that names `rebatch_opt` and `rebatch_opt_params`.

**Decoders.** The cap is read by `bp_norm_min_sum` (both paths), `bp_norm_min_sum_quant`, `bp_sum_prod`, `bp_ots` and `bp_lottery_policy`; by `bp_lottery`, `bp_lottery_quant` and `bp_lottery_layered` only with no PRE phase; by `bp_lottery_pre` with no PRE phase and no `pre_parallel_*` key; by `bp4` and `relay_bp` (both paths); and by `bp_branch_assisted` on its CUDA path only. `bp_sf`, `bp_sf_lottery`, `osd_0`, `mwpm`, `mwpm_gpu`, `union_find` and `saq` have no cap. Thus `bp_sf_lottery` also has no cap during a PRE phase. `main.py` keeps the cap only on the first decoder of a chain; BP stages after the first never cap.

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
| `pruning_opt` | Stop or skip work once its result is known                 | `host_check_every: 0`, `skip_converged: false`, `compact_frac: 0`, `early_stop: false`, `prefix_scan: false`, `solve_by_pivots: false` |
| `fusion_opt`  | Fewer kernel launches and fewer host round trips           | `persistent_kernel: false`, `compile: false`, `fuse_vn: false`, `fused: false` |
| `mapping_opt` | Map the message passing onto the GPU's execution model     | `cn_sign_parity: false`, `warp_per_check: false`, `f64_int_compare: false`, `edge_layout: padded` |
| `gather_opt`  | Gather-based sums in a fixed order, same result on every run | `c2v_gather: false`, `vn_gather: false` |
| `memory_opt`  | Allocate only what is live, and pack it                    | `reuse_buffers: false`, `column_scan: false`, `packed_transform: false`, `workspace_bytes: 1 << 60`, `sparse_h: false` |
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
| `early_stop`      | OSD        | `true`  | `false`: the scan does not stop once the syndrome is spanned. PyTorch scans the whole order; CUDA stops at full rank. The outputs are equal. |
| `prefix_scan`     | OSD        | `true`  | `false`: one stable sort over the whole order, instead of a first scan over the `N/16` least reliable columns with retries on `4N/16` and then all columns. The prefix applies only when `N >= 16384` and `column_scan` is on. |
| `solve_by_pivots` | OSD        | `true`  | `false`: back substitution over `M` pivot slots (PyTorch) or elimination over `rank(H)` columns (CUDA), instead of the largest pivot count found. |
| `skip_converged`  | OSD        | `true`  | `false`: decode every sample and keep the result only where `converge` is 0. |

`fusion_opt`

| Key                 | Path(s)    | Default | Off behavior |
|---------------------|------------|---------|--------------|
| `persistent_kernel` | BP CUDA    | `auto`  | `auto` runs the persistent kernel (one launch for all iterations) when `M <= 120`, or when `M <= 1320` and the batch has at least 4 samples per SM; `true` runs it whenever it can; `false` launches once per iteration step. |
| `fused`         | OSD CUDA   | `true`  | `false`: use the modular elimination kernels instead of fused elimination. An explicit value overrides `fusion_opt`. |
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
| `rebatch_opt` | `bp_norm_min_sum` (both paths), `bp_norm_min_sum_quant`, `bp_sum_prod`, `bp_ots`, the `bp_lottery` family, `bp4`, `relay_bp`, `bp_branch_assisted` CUDA | `true` | `false`: no iteration cap, so every batch runs until all its samples converge or reach `max_iter`, and nothing is deferred. The `rebatch_opt_params` block, if present, is ignored. Single-round runs only (Section 4). |

`memory_opt`

| Key                    | Path(s)     | Default | Off behavior |
|------------------------|-------------|---------|--------------|
| `reuse_buffers`        | BP PyTorch  | `true`  | `false`: allocate the eager work buffers every iteration and compact by plain indexing. |
| `column_scan`      | OSD         | `true`  | `false`: CUDA eliminates over all `N` columns of the order without the column-local scan; PyTorch runs a dense Gauss-Jordan on `H` augmented with `s` over the whole order. |
| `packed_transform` | OSD PyTorch | `true`  | `false`: keep the row transform as a dense bool matrix instead of bit-packed and transposed. |
| `workspace_bytes`      | OSD         | `4 << 30` (4 GiB) | Byte budget per chunk of samples (see Section 3.1). The group off value `1 << 60` puts the whole batch in one chunk. |
| `sparse_h`             | H           | `true`  | Set outside the `decoding` block (see Section 5.3), read by the matrix bundle and the stim syndrome measurer. `true` keeps H as a coalesced bool sparse COO tensor and computes the syndrome with a sparse matmul; `false`: `MatrixBundle.select()` returns a dense int64 H and the syndrome uses a dense matmul. The bundle builds each check type's H once and `select()` returns that same tensor on every call. The outputs are bit-identical. `union_find`, `mwpm`, `mwpm_gpu` and `saq` read the sparse H from the matrix loader. |

### 5.3. Coupling rules
- A top-level `skip_converged` setting reaches both BP and OSD. Set it in a stage's config entry to control that stage alone.
- `early_stop: false` forces `prefix_scan` off.
- `skip_converged: false`, `warp_per_check: false`, `fuse_vn: false`, `vn_gather: false`, or `persistent_kernel: false` force the BP CUDA per-step path, so the persistent kernel does not run. `edge_layout: padded` does not. The per-step path also runs for a subclass that overrides `_iter_hook` (the `bp_sf` main pass, the `bp_lottery` family, and both the main pass and the SF retries of `bp_sf_lottery`), when no persistent block fits on an SM, and when the rebatch cap is active and the batch has more samples than can be co-resident.
- The quantized subclasses (`bp_norm_min_sum_quant`, `bp_lottery_quant`) run the default kernels and ignore `warp_per_check`, `fuse_vn`, `f64_int_compare`, `edge_layout` and `vn_gather`, but `warp_per_check: false`, `fuse_vn: false` and `vn_gather: false` still force their per-step path.
- `bp_sf` runs its persistent retries on the default kernels. `bp_sf_lottery` never uses the persistent kernel, so `persistent_kernel` has no effect on it. `relay_bp` and `bp_branch_assisted` on CUDA ignore the BP CUDA kernel knobs.
- `c2v_gather: false` or `cn_sign_parity: false` force `compile` off.
- `rebatch_opt` is the switch for the iteration cap; the `rebatch_opt_params` block only overrides its default parameters (Section 4). On a syndrome with more than one round `main.py` runs uncapped whatever `rebatch_opt` says.
- An individual key set explicitly wins over its group key: `pruning_opt: false` with `skip_converged: true` turns off every other `pruning_opt` member.
- All knobs and group keys except the H storage go in a stage's `decoding.config` entry, or at the top level of the `decoding` block. Top-level member knobs reach the stages that accept them. All stages accept the group keys; a group has no effect where none of its members apply. A key in a stage's entry wins over the same key at the top level. A top-level key that no stage accepts raises `ValueError`.
- H storage (`sparse_h`, or `memory_opt: false` for its `sparse_h: false` member) is read only outside the `decoding` block: from the `interface` block with `-i` (forwarded to both the stim syndrome measurer and the matrix bundle), else from the `matrix` block (the H the decoders get) and the stim `syndrome` block (the syndrome matmul). `memory_opt: false` in the `decoding` block alone turns off the other `memory_opt` members and leaves H sparse.

```
decoding:
  algorithm: [bp_norm_min_sum, osd_0]
  check_type: hx
  dtype: float64
  memory_opt: false        # every stage
  config:
    - {max_iter: 181, compile: false}
    - {early_stop: false}
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
| `iter`             | out       | Number of iterations actually used per sample. For `bp_sf`, `bp_sf_lottery` and `bp_lottery_pre`, the main-pass count only |
| `iter_serial`      | out, optional | `bp_sf`, `bp_sf_lottery` and `bp_lottery_pre` only. Shape `[B]`, cumulative iterations under a serial execution model (below) |
| `iter_parallel`    | out, optional | `bp_sf`, `bp_sf_lottery` and `bp_lottery_pre` only. Shape `[B]`, two-stage iterations when all trials run at once (below) |
| `rounds`           | out, optional | `bp_gd` only. Shape `[B]`, number of decimations done per sample |

**SF iteration counts.** All SF trials decode at once, and the returned estimate is the trial that converges earliest (fewest iterations, ties to the lowest trial index). `iter_parallel` counts that two-stage run. `iter_serial` follows the serial model the BP-SF paper reports: trials are counted as if decoded one after another in trial order, up to the first that converges. That trial can differ from the returned one.
- A sample that converges in the main pass keeps its `iter` in both outputs.
- A sample SF rescues: `iter_parallel` is the main pass plus the returned trial's own iterations. `iter_serial` is the main pass, plus `max_iter` for every trial before the first converging trial in trial order, plus that trial's own iterations.
- A sample every trial fails: `iter_parallel` is the main pass plus one full trial. `iter_serial` is the main pass plus `max_iter` for every trial.
- With SF off, or with no unconverged sample, both equal `iter`.

**PRE iteration counts.** `bp_lottery_pre` uses the same two outputs, with sub-lotteries as trials, `pre_parallel_max_iter` as their budget, and rank order as trial order.
- A sample that converges in the main pass, or that the single-bit check fixes, keeps its `iter` in both outputs.
- A sample a sub-lottery rescues: `iter_parallel` is the main pass plus the winner's own iterations. `iter_serial` is the main pass, plus `pre_parallel_max_iter` for every candidate ranked before the highest-ranked converged sub-lottery, plus that sub-lottery's own iterations.
- A sample every sub-lottery fails: `iter_parallel` is the main pass plus `pre_parallel_max_iter`. `iter_serial` is the main pass plus `pre_parallel_max_iter` for every sub-lottery.
- Without a `pre_parallel_*` key, or with no unconverged sample, both equal `iter`.

With `early_termination: true` the sibling stop hides a higher-ranked sub-lottery that would converge after the winner. `iter_serial` then counts that sub-lottery as failed, so it is an upper bound on the serial model.

Only a decoder that sets these keys gets the extra result fields. For each of `serial` and `parallel` the result YAML gains `average iteration <mode>`, `iteration distribution <mode>` and `iteration count <mode>`, built like `average iteration`, `iteration distribution` and `iteration count`. Their histograms run to the largest observed value, not to `max_iter`. In a chain each stage reports its own values. Pooling across the workers of a `syndrilla-parallel` run or sweep point sums these histograms element-wise and weights the means by the samples each entry recorded. A run resumed from a checkpoint written before these fields existed warns and reports them over the new samples only.

When chained decoders are used, the next decoder reads the previous decoder's `synd`, `llr`, `e_v`, and `converge` and only runs on samples whose `converge` is 0.
