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


def _build_decomposed_dem(circuit):
    """Graphlike component columns and their correlated original DEM mechanisms."""
    key = (str(circuit), True)
    if key in STIM_CIRCUIT_CACHE:
        return STIM_CIRCUIT_CACHE[key]
    try:
        dem = circuit.detector_error_model(decompose_errors=True)
    except ValueError as exc:
        raise ValueError(f"decompose=True requires a graphlike Stim DEM: {exc}") from exc

    component_ids, priors, mechanism_priors, mechanism_components = {}, [], [], []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        probability = inst.args_copy()[0]
        mechanism_priors.append(probability)
        # Repeated components in one mechanism cancel; only distinct mechanisms
        # contribute independent Bernoulli variables to a component's prior.
        parity = {}
        dets, obs = set(), set()
        for target in [*inst.targets_copy(), None]:
            if target is None or target.is_separator():
                if len(dets) > 2:
                    raise ValueError("decompose=True requires each DEM component to have at most two detectors.")
                if dets or obs:
                    support = (tuple(sorted(dets)), tuple(sorted(obs)))
                    parity[support] = not parity.get(support, False)
                dets, obs = set(), set()
            elif target.is_relative_detector_id():
                dets.symmetric_difference_update((target.val,))
            elif target.is_logical_observable_id():
                obs.symmetric_difference_update((target.val,))
        indices = []
        for support, present in parity.items():
            if not present:
                continue
            if support not in component_ids:
                component_ids[support] = len(priors)
                priors.append(0.0)
            index = component_ids[support]
            indices.append(index)
            prior = priors[index]
            priors[index] = prior * (1 - probability) + (1 - prior) * probability
        mechanism_components.append(tuple(indices))

    h_rows, h_cols, o_rows, o_cols = [], [], [], []
    for (dets, obs), index in component_ids.items():
        h_rows.extend(dets)
        h_cols.extend([index] * len(dets))
        o_rows.extend(obs)
        o_cols.extend([index] * len(obs))
    result = dict(
        dem=dem,
        H=_binary_csr(h_rows, h_cols, (dem.num_detectors, len(priors))),
        obs_mat=_binary_csr(o_rows, o_cols, (dem.num_observables, len(priors))),
        priors=np.asarray(priors, dtype=np.float64),
        mechanism_priors=np.asarray(mechanism_priors, dtype=np.float64),
        components=tuple(component_ids),
        mechanism_components=tuple(mechanism_components),
    )
    STIM_CIRCUIT_CACHE[key] = result
    return result


def _build_dem_matrices(circuit, decompose=False):
    """Extract (H, obs_mat, priors) from a stim circuit, H and obs_mat as scipy CSR.
    Cached by circuit content (string form) and decomposition flag, since id() can be recycled by CPython when a Circuit is freed,
    causing a fresh Circuit to alias an unrelated cached entry.
    """
    if not isinstance(decompose, bool):
        raise ValueError("decompose must be a bool.")
    if decompose:
        data = _build_decomposed_dem(circuit)
        return data["H"], data["obs_mat"], data["priors"]
    key = (str(circuit), False)
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
    # code's parity-check matrix: here a column is a circuit fault mechanism or decomposed component, not a
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

        H, obs_mat, priors = _build_dem_matrices(circuit, matrix_cfg.get("decompose", False))
        self._matrix = H if self.target == "check" else obs_mat
        self.priors = priors

    def get_index(self):
        logger.info(f"Building index for stim {self.target} matrix from <{self.path}>.")
        return dense_to_index_format(self._matrix, self.device)

    def get_dense(self):
        return self._matrix.toarray()
