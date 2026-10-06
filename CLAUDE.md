# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Install (conda env `syndrilla`; run project Python as `conda run -n syndrilla ...`):

```
conda env create -f environment.yaml
conda activate syndrilla
python3 -m pip install -e . --no-deps
```

Tests (pytest, flat `tests/test_*.py`, reference tensors in `tests/data/*.pt`; no lint or format config):

```
conda run -n syndrilla python -m pytest                       # all
conda run -n syndrilla python -m pytest tests/test_bp4.py -q   # one file
conda run -n syndrilla python -m pytest tests/test_bp.py::<name>
```

GPU tests skip without CUDA. Many tests are parametrized over backend cpu / pytorch / cuda. The first CUDA use in a process JIT-compiles the kernels with nvcc (tens of seconds).

Decode run (every YAML has a header key, `decoding:`, `matrix:` and so on):

```
syndrilla -r=tests/test_outputs -d=examples/alist/bposd_hx.decoding.yaml -e=examples/alist/bsc.error.yaml -c=examples/alist/lx.check.yaml -s=examples/alist/perfect.syndrome.yaml -m=examples/alist/surface_10.matrix.yaml -bs=10000 -te=1000
```

`-i <interface yaml>` (stim path, `examples/stim/`) replaces `-e -c -s -m`. `-te` target errors or `-tb` target batches (wins) stop the run. `-t` trains the last stage and needs `-tr`. Sweeps: `syndrilla-parallel sweep-gen` then `sweep`, run from the repo root (templates are relative paths).

## Architecture

- **Run flow** (`src/syndrilla/main.py`): matrix bundle, decoder chain, error model, syndrome measurer, logical check. Per batch: sample errors, measure syndrome, build `io_dict = {synd, llr0, H_matrix}`, run each stage, logical check, metric writes the result YAML and checkpoint.
- **Plugin loading** (`utils/utils.py: call_func_from_cfg`): a module named `x` lives at `<kind>/x/x.py` and exposes `create(cfg, **kwargs)`. No registry: a new decoder, error model, syndrome, matrix, or logical check is a new directory with that layout.
- **CUDA dispatch** (`decoder/decoder.py: _create_one_decoder`): on a cuda device without `force_pytorch`, `<algo>/<algo>_cuda.py` is used when it exists, else the torch module. Kernels are `decoder/cuda/*.cu`, loaded with `torch.utils.cpp_extension.load`. Some decoders are CUDA only.
- **Chains**: `algorithm: [a, b]` makes one stage per entry. `config` as a mapping applies to the first stage; as a list it matches by position. Later stages run only on samples with `converge == 0`. Shared keys (`algorithm`, `check_type`, `dtype`, `device`, `force_pytorch`, `rebatch_opt_params`) must stay at the top level and raise if placed inside `config`; `optimizer` and `train` live in the `-tr` YAML.
- **io_dict contract** (docs/decoder.md section 6): inputs `synd`, `llr0`, `H_matrix`; outputs `e_v`, `llr`, `converge`, `iter`. `bp4` is the only 2-channel decoder.
- **H storage**: `sparse_h` (default true) keeps H as a coalesced bool sparse COO tensor; it is set in the `matrix`, stim `syndrome`, or `interface` block, never in `decoding`.
- **Device** (`utils/utils.py: parse_device_dtype`): omitted device means cuda if available; bad dtype falls back to float64; mps with float64 falls back to cpu.
- **Optimization groups** (`decoder/knobs.py`): six bool groups, all on by default: `pruning_opt`, `fusion_opt`, `mapping_opt`, `gather_opt`, `memory_opt`, `rebatch_opt`. An explicit knob wins over its group. `rebatch_opt` is the learned iteration cap; the old key `rebatch_speedup` raises.
- **Docs**: `docs/` has one file per module (decoder, error, syndrome, matrix, interface, trainer) and design notes (optimizations, parallel, bp_interventions). `zoo/` holds tracked sweep/plot tooling; `zoo/study/` holds ignored local research scripts, notes and results. `reports/` holds dated tech reports.

## Config keys

- A config key that maps to an argument of an external tool (stim, PyMatching, ldpc, torch) keeps that tool's argument name verbatim, for example `decompose_errors`, not `decompose`.

## Git and releases

- Features go on topic branches merged to `main` by PR.
- The version lives only in `pyproject.toml`. Every PR to `main` ends with a version bump in its own "Bump to vX.Y.Z" commit, added without being asked. The default bump is the patch number (+0.0.1); a larger bump only when asked.

## Docs rules (docs/*.md, README.md)

- State capability and relative performance only, as a rounded magnitude (for example "up to 2 orders of magnitude slower"). No measured numbers.
- No test conditions: no code name, error rate, distance, batch size, or shot count.
- No links to logs, run directories, or artifacts.
- No code references: no file names, line numbers, or test file names.
- Known bugs and defects go in code comments and runtime warnings, never in the docs.
