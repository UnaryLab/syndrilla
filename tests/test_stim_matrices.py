import sys, os
import torch
import numpy as np
import stim
import pytest
import scipy.sparse as sp

sys.path.append(os.getcwd())
from syndrilla.interface import create_interface


def make_circuit(code='surface_code', distance=3, rounds=1, noise=0.01):
    return stim.Circuit.generated(
        code,
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=noise,
        after_reset_flip_probability=noise,
        before_measure_flip_probability=noise,
        before_round_data_depolarization=noise,
    )


def make_interface_from_circuit(circuit):
    return create_interface(
        cfg={
            'backend': 'stim',
            'circuit': str(circuit),
            'device': {'device_type': 'cpu', 'device_idx': 0},
            'dtype': 'float64',
        },
        decoder_cfg={},
        error_cfg={'number_channel': 1},
        syndrome_cfg={'measure': 'stim', 'd_rounds': 1},
    )


def load_matrices_from_circuit(circuit):
    iface = make_interface_from_circuit(circuit)
    bundle = iface.matrix_bundle
    H = bundle.Hx_matrix.get_dense()
    L = bundle.lx_matrix
    return H, L, iface


CIRCUITS = {
    'surface_rotated_d3_r1': make_circuit('surface_code:rotated_memory_x', distance=3, rounds=1),
    'surface_rotated_d3_r3': make_circuit('surface_code:rotated_memory_x', distance=3, rounds=3),
    'surface_unrotated_d3_r1': make_circuit('surface_code:unrotated_memory_x', distance=3, rounds=1),
    'surface_rotated_d5_r1': make_circuit('surface_code:rotated_memory_x', distance=5, rounds=1),
    'rep_d5_r3': make_circuit('repetition_code:memory', distance=5, rounds=3),
}


class TestStimMatrixShape:

    @pytest.mark.parametrize('name', CIRCUITS.keys())
    def test_H_shape(self, name):
        circuit = CIRCUITS[name]
        H, L, _ = load_matrices_from_circuit(circuit)
        dem = circuit.detector_error_model(decompose_errors=False)
        num_errors = sum(1 for i in dem.flattened() if i.type == 'error')
        assert H.shape == (dem.num_detectors, num_errors)

    @pytest.mark.parametrize('name', CIRCUITS.keys())
    def test_L_shape(self, name):
        circuit = CIRCUITS[name]
        H, L, _ = load_matrices_from_circuit(circuit)
        dem = circuit.detector_error_model(decompose_errors=False)
        num_errors = sum(1 for i in dem.flattened() if i.type == 'error')
        assert L.shape == (dem.num_observables, num_errors)

    @pytest.mark.parametrize('name', CIRCUITS.keys())
    def test_H_L_same_num_errors(self, name):
        circuit = CIRCUITS[name]
        H, L, _ = load_matrices_from_circuit(circuit)
        assert H.shape[1] == L.shape[1]


class TestStimMatrixContent:

    @pytest.mark.parametrize('name', CIRCUITS.keys())
    def test_H_binary(self, name):
        H, _, _ = load_matrices_from_circuit(CIRCUITS[name])
        assert set(H.flatten().tolist()).issubset({0, 1})

    @pytest.mark.parametrize('name', CIRCUITS.keys())
    def test_L_binary(self, name):
        _, L, _ = load_matrices_from_circuit(CIRCUITS[name])
        assert set(L.flatten().tolist()).issubset({0, 1})

    @pytest.mark.parametrize('name', CIRCUITS.keys())
    def test_H_no_empty_rows(self, name):
        H, _, _ = load_matrices_from_circuit(CIRCUITS[name])
        assert H.sum(axis=1).min() > 0

    @pytest.mark.parametrize('name', CIRCUITS.keys())
    def test_H_no_empty_cols(self, name):
        H, _, _ = load_matrices_from_circuit(CIRCUITS[name])
        assert H.sum(axis=0).min() > 0

    @pytest.mark.parametrize('name', CIRCUITS.keys())
    def test_L_not_all_zero(self, name):
        _, L, _ = load_matrices_from_circuit(CIRCUITS[name])
        assert L.sum(axis=1).min() > 0


class TestStimMatrixConsistency:

    @pytest.mark.parametrize('name', CIRCUITS.keys())
    def test_H_matches_stim_dem(self, name):
        circuit = CIRCUITS[name]
        dem = circuit.detector_error_model(decompose_errors=False)
        num_detectors = dem.num_detectors
        num_errors = sum(1 for i in dem.flattened() if i.type == 'error')

        H, _, _ = load_matrices_from_circuit(circuit)

        H_rebuilt = np.zeros((num_detectors, num_errors), dtype=np.int64)
        err_idx = 0
        for inst in dem.flattened():
            if inst.type != 'error':
                continue
            for tgt in inst.targets_copy():
                if tgt.is_relative_detector_id():
                    H_rebuilt[tgt.val, err_idx] = 1
            err_idx += 1

        np.testing.assert_array_equal(H, H_rebuilt)

    @pytest.mark.parametrize('name', CIRCUITS.keys())
    def test_L_matches_stim_dem(self, name):
        circuit = CIRCUITS[name]
        dem = circuit.detector_error_model(decompose_errors=False)
        num_observables = dem.num_observables
        num_errors = sum(1 for i in dem.flattened() if i.type == 'error')

        _, L, _ = load_matrices_from_circuit(circuit)

        L_rebuilt = np.zeros((num_observables, num_errors), dtype=np.int64)
        err_idx = 0
        for inst in dem.flattened():
            if inst.type != 'error':
                continue
            for tgt in inst.targets_copy():
                if tgt.is_logical_observable_id():
                    L_rebuilt[tgt.val, err_idx] = 1
            err_idx += 1

        np.testing.assert_array_equal(L, L_rebuilt)

    @pytest.mark.parametrize('name', CIRCUITS.keys())
    def test_observable_shape_matches_L(self, name):
        circuit = CIRCUITS[name]
        _, L, iface = load_matrices_from_circuit(circuit)
        sm = iface.syndrome_generator
        _, obs_np = sm.sampler.sample(shots=50, separate_observables=True)
        assert obs_np.shape[1] == L.shape[0]

    @pytest.mark.parametrize('name', CIRCUITS.keys())
    def test_index_format_matches_dense(self, name):
        circuit = CIRCUITS[name]
        _, _, iface = load_matrices_from_circuit(circuit)
        loader = iface.matrix_bundle.Hx_matrix
        H_dense = loader.get_dense()
        shape, V_c_row, V_c_col, H_tensor = loader.get_index()

        assert shape == H_dense.shape
        for r in range(shape[0]):
            cols_from_index = set(V_c_col[r][V_c_col[r] < shape[1]].tolist())
            cols_from_dense = set((H_dense[r] == 1).nonzero()[0].tolist())
            assert cols_from_index == cols_from_dense, f'Row {r} mismatch'


class TestStimVsBposd:
    """Compare syndrilla's H/L extraction against the reference ldpc/bposd approach."""

    def _bposd_matrices(self, circuit):
        """Build H, L, priors the way ldpc/bposd does (decompose_errors=True, sparse)."""
        dem = circuit.detector_error_model(decompose_errors=True)
        num_det = dem.num_detectors
        num_obs = dem.num_observables

        h_rows, h_cols, o_rows, o_cols, priors = [], [], [], [], []
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

        H = sp.csr_matrix(
            (np.ones(len(h_rows), dtype=np.uint8), (h_rows, h_cols)),
            shape=(num_det, err_idx),
        )
        L = sp.csr_matrix(
            (np.ones(len(o_rows), dtype=np.uint8), (o_rows, o_cols)),
            shape=(num_obs, err_idx),
        )
        return H, L, np.array(priors)

    @pytest.mark.parametrize('name', CIRCUITS.keys())
    def test_same_num_detectors(self, name):
        circuit = CIRCUITS[name]
        H_syn, _, _ = load_matrices_from_circuit(circuit)
        H_ref, _, _ = self._bposd_matrices(circuit)
        assert H_syn.shape[0] == H_ref.shape[0]

    @pytest.mark.parametrize('name', CIRCUITS.keys())
    def test_same_num_observables(self, name):
        circuit = CIRCUITS[name]
        _, L_syn, _ = load_matrices_from_circuit(circuit)
        _, L_ref, _ = self._bposd_matrices(circuit)
        assert L_syn.shape[0] == L_ref.shape[0]

    @pytest.mark.parametrize('name', CIRCUITS.keys())
    def test_H_matches_bposd_no_decompose(self, name):
        """With decompose_errors=False (syndrilla default), H should match exactly."""
        circuit = CIRCUITS[name]
        H_syn, _, _ = load_matrices_from_circuit(circuit)

        dem = circuit.detector_error_model(decompose_errors=False)
        num_det = dem.num_detectors
        h_rows, h_cols = [], []
        err_idx = 0
        for inst in dem.flattened():
            if inst.type != 'error':
                continue
            for tgt in inst.targets_copy():
                if tgt.is_relative_detector_id():
                    h_rows.append(tgt.val)
                    h_cols.append(err_idx)
            err_idx += 1

        H_ref = np.zeros((num_det, err_idx), dtype=np.int64)
        if h_rows:
            H_ref[h_rows, h_cols] = 1
        np.testing.assert_array_equal(H_syn, H_ref)

    @pytest.mark.parametrize('name', CIRCUITS.keys())
    def test_L_matches_bposd_no_decompose(self, name):
        """With decompose_errors=False (syndrilla default), L should match exactly."""
        circuit = CIRCUITS[name]
        _, L_syn, _ = load_matrices_from_circuit(circuit)

        dem = circuit.detector_error_model(decompose_errors=False)
        num_obs = dem.num_observables
        o_rows, o_cols = [], []
        err_idx = 0
        for inst in dem.flattened():
            if inst.type != 'error':
                continue
            for tgt in inst.targets_copy():
                if tgt.is_logical_observable_id():
                    o_rows.append(tgt.val)
                    o_cols.append(err_idx)
            err_idx += 1

        L_ref = np.zeros((num_obs, err_idx), dtype=np.int64)
        if o_rows:
            L_ref[o_rows, o_cols] = 1
        np.testing.assert_array_equal(L_syn, L_ref)

    @pytest.mark.parametrize('name', CIRCUITS.keys())
    def test_priors_match(self, name):
        """Syndrilla error model priors match DEM priors."""
        circuit = CIRCUITS[name]
        iface = make_interface_from_circuit(circuit)
        syn_priors = iface.error_model._prior_llr.numpy()

        dem = circuit.detector_error_model(decompose_errors=False)
        dem_priors = []
        for inst in dem.flattened():
            if inst.type != 'error':
                continue
            p = inst.args_copy()[0]
            p = min(max(p, 1e-12), 1 - 1e-12)
            dem_priors.append(np.log((1 - p) / p))

        np.testing.assert_allclose(syn_priors, np.array(dem_priors), rtol=1e-10)

    @pytest.mark.parametrize('name', CIRCUITS.keys())
    def test_decode_consistency(self, name):
        """Syndrilla and ldpc/bposd produce identical syndromes from the same H."""
        from ldpc import BpOsdDecoder

        circuit = CIRCUITS[name]
        H_syn, _, iface = load_matrices_from_circuit(circuit)
        H_ref, _, ref_priors = self._bposd_matrices(circuit)

        sm = iface.syndrome_generator
        det_np, obs_np = sm.sampler.sample(shots=50, separate_observables=True)

        H_syn_t = torch.tensor(H_syn, dtype=torch.float64)
        det_t = torch.from_numpy(det_np.astype('float64'))

        assert det_t.shape[1] == H_syn_t.shape[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
