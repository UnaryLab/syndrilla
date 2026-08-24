import os
import sys

import numpy as np
import torch

sys.path.append(os.getcwd())

from syndrilla.error_model import create_error_model
from syndrilla.matrix import load_matrices
from syndrilla.utils import get_path, parse_device_dtype, read_yaml


def test_error_model():
    decoder_yaml = "examples/txt/bp_hx.decoder.yaml"
    matrix_yaml = "examples/txt/hgp.matrix.yaml"

    decoder_cfg = read_yaml(get_path(decoder_yaml))["decoder"]
    matrix_cfg = read_yaml(get_path(matrix_yaml))["matrix"]
    device, dtype = parse_device_dtype(decoder_cfg)
    bundle = load_matrices(matrix_cfg, device, dtype)

    shape, _, _, _ = bundle.Hx_matrix.get_index()
    sample_size = 10000
    batch_size = 1000
    benchmark = 0.0005

    error_yaml = 'examples/txt/bsc.error.yaml'
    error_model = create_error_model(yaml_path=error_yaml)
    configured_rate = read_yaml(get_path(error_yaml))['error']['rate']

    zero_qubits = torch.zeros([sample_size, shape[1]], dtype=dtype)
    error_vector, _ = error_model.inject_error(zero_qubits, batch_size)
    avg_error_rate = torch.mean(torch.sum(error_vector, 1) / shape[1])
    assert (
        abs(avg_error_rate - configured_rate) <= benchmark
    ), f'the difference is too high when p is {configured_rate}'

    for err_rate in np.linspace(0.01, 0.1, num=10):
        zero_qubits = torch.zeros([sample_size, shape[1]], dtype=dtype)
        error_model.rate = err_rate
        error_vector, _ = error_model.inject_error(zero_qubits, batch_size)
        avg_error_rate = torch.mean(torch.sum(error_vector, 1) / shape[1])
        assert abs(avg_error_rate - err_rate) <= benchmark, f'the difference is too high when p is {err_rate}'


if __name__ == '__main__':
    test_error_model()
