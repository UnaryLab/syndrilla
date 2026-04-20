"""
Stim DEM matrix loader.

A syndrilla matrix file_type that builds its dense GF(2) matrix directly from a
stim circuit's detector error model:

    matrix:
      file_type: stim
      circuit: <inline stim circuit string>
      target: check                        # 'check' (H) or 'observable' (L)
"""

import numpy as np
from loguru import logger

from syndrilla.interface.stim.stim import get_stim_circuit
from syndrilla.matrix.matrix import dense_to_index_format, STIM_CIRCUIT_CACHE


def _build_dem_matrices(circuit):
    """Extract (H, obs_mat, priors) from a stim circuit. Cached by circuit id."""
    key = id(circuit)
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
        if inst.type != 'error':
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

    H = np.zeros((num_detectors, num_errors), dtype=np.int64)
    if h_rows:
        H[h_rows, h_cols] = 1
    obs_mat = np.zeros((num_observables, num_errors), dtype=np.int64)
    if o_rows:
        obs_mat[o_rows, o_cols] = 1

    result = (H, obs_mat, np.asarray(priors, dtype=np.float64))
    STIM_CIRCUIT_CACHE[key] = result
    logger.info(
        f'DEM extraction complete: '
        f'detectors={num_detectors}, observables={num_observables}, errors={num_errors}'
    )
    return result


class create():
    """
    Stim DEM matrix loader. Conforms to the same interface as alist/npz/txt
    loaders: exposes `path`, `get_index()`, and `get_dense()`.
    """

    def __init__(self, matrix_cfg, **kwargs) -> None:
        self.device = kwargs['device']

        circuit_str = matrix_cfg.get('circuit', None)
        circuit = get_stim_circuit(circuit_str=circuit_str)
        self.path = '<inline-stim-circuit>'

        self.target = matrix_cfg.get('target', 'check').lower()
        if self.target not in ('check', 'observable'):
            raise ValueError(
                f"stim matrix loader 'target' must be 'check' or 'observable', got <{self.target}>."
            )

        H, obs_mat, priors = _build_dem_matrices(circuit)
        self._dense_np = H if self.target == 'check' else obs_mat
        self.priors = priors

    def get_index(self):
        logger.info(f'Building index for stim {self.target} matrix from <{self.path}>.')
        return dense_to_index_format(self._dense_np, self.device)

    def get_dense(self):
        return self._dense_np
