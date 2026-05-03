# Matrix module

The matrix YAML file defines how parity-check and logical-check matrices are loaded.
Each matrix entry specifies a `file_type` and the source of the matrix data.
Syndrilla currently supports three formats: `alist`, `npz`, `txt`, and  directly load from `stim`.
Matrix entries can either live in their own dedicated YAML file (referenced by path from a decoder) or be inlined as a config dict inside a larger YAML file.

## 1. alist format
The [.alist](https://www.inference.org.uk/mackay/codes/alist.html) format (David MacKay, Matthew Davey, John Lafferty) stores a sparse matrix as a list of nonzero neighbor indices per check node.
An example matrix configuration using the alist format is provided in ```hx.matrix.yaml```:

```
matrix:
  file_type: alist
  path: examples/alist/surface/surface_10_hx.alist
```

The following table details the configuration parameters used in the alist matrix YAML file.
| Key                 | Description                                                   | Example                                     |
|---------------------|---------------------------------------------------------------|---------------------------------------------|
| `matrix.file_type`  | Format of the parity-check matrix file                        | `alist`                                     |
| `matrix.path`       | Path to the parity-check matrix file                          | `examples/alist/surface/surface_10_hx.alist`|

## 2. npz format
The [.npz](https://numpy.org/doc/2.1/reference/generated/numpy.savez.html) format from NumPy stores a SciPy sparse matrix in a compressed archive.
An example matrix configuration using the npz format is provided in ```examples/npz/hx.matrix.yaml```:

```
matrix:
  file_type: npz
  path: examples/npz/h_matrix/pk_x_p3_delta8.npz
```

The following table details the configuration parameters used in the npz matrix YAML file.
| Key                 | Description                                                   | Example                                       |
|---------------------|---------------------------------------------------------------|-----------------------------------------------|
| `matrix.file_type`  | Format of the parity-check matrix file                        | `npz`                                         |
| `matrix.path`       | Path to the SciPy-saved sparse matrix archive                 | `examples/npz/h_matrix/pk_x_p3_delta8.npz`    |

## 3. txt format
The .txt format stores a dense 2D matrix in plain text, where each row represents a check node and each `1` entry denotes a connection to the corresponding variable node.
An example matrix configuration using the txt format is provided in ```examples/txt/hx.matrix.yaml```:

```
matrix:
  file_type: txt
  path: examples/txt/h_matrix/hgp_(4,7)-[[400,16,6]]_hx.txt
```

The following table details the configuration parameters used in the txt matrix YAML file.
| Key                 | Description                                                   | Example                                              |
|---------------------|---------------------------------------------------------------|------------------------------------------------------|
| `matrix.file_type`  | Format of the parity-check matrix file                        | `txt`                                                |
| `matrix.path`       | Path to the dense-text parity-check matrix file               | `examples/txt/h_matrix/hgp_(4,7)-[[400,16,6]]_hx.txt`|

## 4. Bundling matrices for a decoder
Decoders consume a bundle of matrices (Hx, Hz, optionally Lx and Lz).
These can be referenced either as file paths or inlined as config dicts.
An example combined matrix configuration is provided in ```surface_10.matrix.yaml```:

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
| Key                              | Description                                                                                              | Example                                     |
|----------------------------------|----------------------------------------------------------------------------------------------------------|---------------------------------------------|
| `matrix.parity_matrix_hx`        | Matrix entry (path or inline dict) for the X-type parity-check matrix                                    | `examples/alist/hx.matrix.yaml`             |
| `matrix.parity_matrix_hz`        | Matrix entry (path or inline dict) for the Z-type parity-check matrix                                    | `examples/alist/hz.matrix.yaml`             |
| `matrix.logical_check_matrix`    | Whether logical-check matrices are provided; if `False`, they are computed from Hx/Hz via `compute_lz`   | `True` or `False`                           |
| `matrix.logical_check_lx`        | Matrix entry for the X-type logical-check matrix (used when `logical_check_matrix = True`)               | `examples/alist/lx.matrix.yaml`             |
| `matrix.logical_check_lz`        | Matrix entry for the Z-type logical-check matrix (used when `logical_check_matrix = True`)               | `examples/alist/lz.matrix.yaml`             |

## 5. stim format
The stim matrix loader builds a GF(2) matrix directly from a stim circuit's detector error model (DEM), so the user does not need to pre-extract parity-check or observable matrices.
The `target` field selects which matrix to expose:
- `check`: the detector parity-check matrix H (detectors x error mechanisms).
- `observable`: the logical observable matrix L (observables x error mechanisms).

In the resulting matrix, **the stabilizer (detector) axis is treated as the variable-node axis of the Tanner graph, and the error-mechanism axis as the check-node axis** — i.e. the loader hands the decoder the matrix in the orientation it expects, so no transpose is needed downstream. This is opposite to the conventional `H[checks, variables]` layout used in classical LDPC literature; keep it in mind when porting matrices in/out of stim.

An example inline matrix configuration using the stim format is:

```
matrix:
  file_type: stim
  circuit: <inline stim circuit string>
  target: check
```

The following table details the configuration parameters used in the stim matrix YAML file.
| Key                 | Description                                                                                              | Example                  |
|---------------------|----------------------------------------------------------------------------------------------------------|--------------------------|
| `matrix.file_type`  | Format identifier for the stim DEM loader                                                                | `stim`                   |
| `matrix.circuit`    | Inline stim circuit string from which the DEM is extracted                                               | `<stim circuit string>`  |
| `matrix.target`     | Which DEM-derived matrix to expose: `check` (H) or `observable` (L)                                      | `check` or `observable`  |

In the stim workflow, the interface module constructs the stim circuit and passes it to the matrix loader automatically, so user-facing stim YAML configuration is typically handled through the interface module (see [interface.md](interface.md)).