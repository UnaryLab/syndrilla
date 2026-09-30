import numpy as np
import scipy.sparse as sp
from loguru import logger

from syndrilla.interface.stim.stim import get_stim_circuit
from syndrilla.matrix.matrix import STIM_CIRCUIT_CACHE, dense_to_index_format


def _binary_csr(rows, cols, shape):
    """int64 CSR matrix over GF(2): a (row, col) pair given an even number of times cancels, as in Stim."""
    m = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.int64), (rows, cols)), shape=shape
    )
    m.sum_duplicates()
    m.data %= 2
    m.eliminate_zeros()
    return m


def _build_dem_matrices(circuit):
    """Extract (H, obs_mat, priors) from a stim circuit, H and obs_mat as scipy CSR.
    Cached by circuit content (string form), since id() can be recycled by CPython when a Circuit is freed,
    causing a fresh Circuit to alias an unrelated cached entry.
    """
    key = str(circuit)
    if key in STIM_CIRCUIT_CACHE:
        return STIM_CIRCUIT_CACHE[key]

    dem = circuit.detector_error_model(decompose_errors=False)
    num_detectors = dem.num_detectors
    num_observables = dem.num_observables

    h_rows, h_cols = [], []
    o_rows, o_cols = [], []
    priors = []
    err_idx = 0
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        priors.append(inst.args_copy()[0])
        for tgt in inst.targets_copy():
            if tgt.is_relative_detector_id():
                h_rows.append(tgt.val)
                h_cols.append(err_idx)
            elif tgt.is_logical_observable_id():
                o_rows.append(tgt.val)
                o_cols.append(err_idx)
        err_idx += 1
    num_errors = err_idx

    H = _binary_csr(h_rows, h_cols, (num_detectors, num_errors))
    obs_mat = _binary_csr(o_rows, o_cols, (num_observables, num_errors))

    result = (H, obs_mat, np.asarray(priors, dtype=np.float64))
    STIM_CIRCUIT_CACHE[key] = result
    logger.info(
        f"DEM extraction complete: "
        f"detectors={num_detectors}, observables={num_observables}, errors={num_errors}"
    )
    return result


class create:
    """
    Stim DEM matrix loader. Conforms to the same interface as alist/npz/txt
    loaders: exposes `path`, `get_index()`, and `get_dense()`.
    """

    # Read by decoders that mean something different on a circuit-level DEM than on a
    # code's parity-check matrix: here a column is a circuit fault mechanism, not a
    # qubit, so a code family and a distance cannot be measured off the shape.
    is_circuit_dem = True

    def __init__(self, matrix_cfg, **kwargs) -> None:
        self.device = kwargs["device"]

        circuit_str = matrix_cfg.get("circuit", None)
        circuit = get_stim_circuit(circuit_str=circuit_str)
        self.path = "<inline-stim-circuit>"

        self.target = matrix_cfg.get("target", "check").lower()
        if self.target not in ("check", "observable"):
            raise ValueError(
                f"stim matrix loader 'target' must be 'check' or 'observable', got <{self.target}>."
            )

        H, obs_mat, priors = _build_dem_matrices(circuit)
        self._matrix = H if self.target == "check" else obs_mat
        self.priors = priors

    def get_index(self):
        logger.info(f"Building index for stim {self.target} matrix from <{self.path}>.")
        return dense_to_index_format(self._matrix, self.device)

    def get_dense(self):
        return self._matrix.toarray()
