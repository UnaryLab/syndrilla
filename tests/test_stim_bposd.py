import os
import sys

import numpy as np
import scipy.sparse as sp
import torch
from ldpc import BpOsdDecoder

sys.path.append(os.getcwd())

from syndrilla.interface import create_interface


def build_interface(distance=3, rounds=1, p=0.05, max_iter=100):
    return create_interface(
        cfg={
            'backend': 'stim',
            'code': 'surface_code:rotated_memory_z',
            'distance': distance,
            'device': {'device_type': 'cpu', 'device_idx': 0},
            'dtype': 'float64',
        },
        error_cfg={
            'model': 'stim_circuit',
            'before_measure_flip_probability': p,
            'number_channel': 1,
        },
        syndrome_cfg={
            'measure': 'stim',
            'rounds': rounds,
        },
        decoding_cfg={
            'algorithm': ['bp_norm_min_sum', 'osd_0'],
            'check_type': 'hx',
            'dtype': 'float64',
            'device': {'device_type': 'cpu', 'device_idx': 0},
            'config': [{'max_iter': max_iter}],  # osd_0 configures nothing
        },
    )


def decode_syndrilla(iface, n_shots, batch_size=500):
    error_model = iface.error_model
    syndrome_generator = iface.syndrome_generator
    bundle = iface.matrix_bundle
    decoders = iface.decoders

    check_type = 'hx'
    shape, _, _, _ = bundle.Hx_matrix.get_index()
    H_matrix = bundle.select(check_type)[3]
    dtype = decoders[0].dtype

    all_syndromes = []
    all_obs_flips = []
    all_e_v = []

    shots_done = 0
    while shots_done < n_shots:
        b = min(batch_size, n_shots - shots_done)
        zero_qubits = torch.zeros([b, shape[1]], dtype=dtype)
        _, error_dataloader = error_model.inject_error(zero_qubits, b)

        for err, llr, _ in error_dataloader:
            synd = syndrome_generator.measure_syndrome(err, decoders[0])
            obs_flips = syndrome_generator.observable_flips

            all_syndromes.append(synd.cpu().numpy())
            if obs_flips is not None:
                all_obs_flips.append(obs_flips.cpu().numpy())

            io_dict = {'synd': synd, 'llr0': llr, 'H_matrix': H_matrix}
            for decoder in decoders:
                io_dict = decoder(io_dict)

            all_e_v.append(io_dict['e_v'].cpu().numpy().astype(np.uint8))
            shots_done += b

    syndromes = np.concatenate(all_syndromes, axis=0)[:n_shots]
    obs_flips = np.concatenate(all_obs_flips, axis=0)[:n_shots] if all_obs_flips else None
    predictions = np.concatenate(all_e_v, axis=0)[:n_shots]

    return syndromes, obs_flips, predictions


def decode_reference(H, priors, det_data, max_iter=100):
    bp_osd = BpOsdDecoder(
        H,
        error_channel=priors.tolist(),
        bp_method='minimum_sum',
        ms_scaling_factor=1.0,
        max_iter=max_iter,
        schedule='parallel',
        osd_method='OSD_0',
    )

    n_shots = det_data.shape[0]
    n_errors = H.shape[1]
    out = np.zeros((n_shots, n_errors), dtype=np.uint8)
    for i in range(n_shots):
        out[i] = bp_osd.decode(det_data[i].astype(np.uint8))
    return out


def test_stim_bposd(distance=3, rounds=1, p=0.05,
                    n_shots=100000, max_iter=100):
    print('\n=== Stim BPOSD comparison ===')
    print(f'  surface_code rotated_memory_z, distance={distance}, rounds={rounds}, p={p}')

    # ---- 1. create interface ----
    iface = build_interface(distance=distance, rounds=rounds, p=p, max_iter=max_iter)

    H_dense = iface.matrix_bundle.Hx_matrix.get_dense()
    L_dense = iface.matrix_bundle.lx_matrix
    print(f'  num_detectors      = {H_dense.shape[0]}')
    print(f'  num_error_mechs    = {H_dense.shape[1]}')
    print(f'  num_observables    = {L_dense.shape[0]}')

    # ---- 2. build priors from error model ----
    priors_llr = iface.error_model._prior_llr.numpy()
    priors = 1.0 / (1.0 + np.exp(priors_llr))

    H_sparse = sp.csr_matrix(H_dense.astype(np.uint8))

    # ---- 3. syndrilla decode (through interface) ----
    syndromes, obs_flips, est_preds = decode_syndrilla(iface, n_shots)
    print(f'  shots              = {n_shots}')

    # ---- 4. reference decode (same syndromes) ----
    ref_preds = decode_reference(H_sparse, priors, syndromes, max_iter=max_iter)

    # ---- 5. compute LERs ----
    obs_data = obs_flips.astype(np.uint8)

    syn_obs_pred = (est_preds @ L_dense.T) % 2
    ref_obs_pred = (ref_preds @ L_dense.T) % 2

    syn_logical_errors = int(np.any(syn_obs_pred != obs_data, axis=1).sum())
    ref_logical_errors = int(np.any(ref_obs_pred != obs_data, axis=1).sum())
    syn_ler = syn_logical_errors / n_shots
    ref_ler = ref_logical_errors / n_shots

    raw_match = int((est_preds == ref_preds).all(axis=1).sum())
    obs_match = int((syn_obs_pred == ref_obs_pred).all(axis=1).sum())

    print(f'\n  syndrilla logical errors = {syn_logical_errors:5d} / {n_shots}  (LER = {syn_ler:.4f})')
    print(f'  reference logical errors = {ref_logical_errors:5d} / {n_shots}  (LER = {ref_ler:.4f})')
    print(f'  raw error-vector match   = {raw_match} / {n_shots}')
    print(f'  observable-pred  match   = {obs_match} / {n_shots}')


if __name__ == '__main__':
    test_stim_bposd(distance=3, rounds=3, p=0.001, n_shots=1000000)
