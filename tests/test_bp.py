import torch
import re, yaml
import sys, os, time
import subprocess
from loguru import logger

sys.path.append(os.getcwd())

from src.decoder import create_decoder
from src.error_model import create_error_model
from src.syndrome import create_syndrome
from src.metric import report_metric, compute_avg_metrics
from src.logical_check import create_check


def test_batch_alist_hx(target_error=1000, batch_size=1000):
    decoder_yaml = 'examples/alist/bp_hx.decoder.yaml'
    logical_check_yaml = 'examples/alist/lx.check.yaml'
    cmd = [
        'syndrilla',
        '-r=tests/test_outputs',
        f'-d={decoder_yaml}',
        '-e=examples/alist/bsc.error.yaml',
        f'-c={logical_check_yaml}',
        '-s=examples/alist/perfect.syndrome.yaml',
        f'-bs={batch_size}',
        f'-te={target_error}',
        '-se=examples/alist/save_error_llr.yaml'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print stdout and stderr
    print('STDOUT:\n', result.stdout)
    print('STDERR:\n', result.stderr)


def test_batch_alist_hz(target_error=1000, batch_size=1000):
    decoder_yaml = 'examples/alist/bp_hz.decoder.yaml'
    logical_check_yaml = 'examples/alist/lz.check.yaml'
    cmd = [
        'syndrilla',
        '-r=tests/test_outputs',
        f'-d={decoder_yaml}',
        '-e=examples/alist/bsc.error.yaml',
        f'-c={logical_check_yaml}',
        '-s=examples/alist/perfect.syndrome.yaml',
        f'-bs={batch_size}',
        f'-te={target_error}',
        '-se=examples/alist/save_error_llr.yaml'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print stdout and stderr
    print('STDOUT:\n', result.stdout)
    print('STDERR:\n', result.stderr)

 
def test_batch_txt_hx(target_error=1000, batch_size=1000):
    decoder_yaml = 'examples/txt/bp_hx.decoder.yaml'
    logical_check_yaml = 'examples/txt/lx.check.yaml'
    cmd = [
        'syndrilla',
        '-r=tests/test_outputs',
        f'-d={decoder_yaml}',
        '-e=examples/txt/bsc.error.yaml',
        f'-c={logical_check_yaml}',
        '-s=examples/txt/perfect.syndrome.yaml',
        f'-bs={batch_size}',
        f'-te={target_error}',
        '-se=examples/txt/save_error_llr.yaml'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print stdout and stderr
    print('STDOUT:\n', result.stdout)
    print('STDERR:\n', result.stderr)


def test_batch_txt_hz(target_error=1000, batch_size=1000):
    # create decoder
    decoder_yaml = 'examples/txt/bp_hz.decoder.yaml'
    logical_check_yaml = 'examples/txt/lz.check.yaml'
    cmd = [
        'syndrilla',
        '-r=tests/test_outputs',
        f'-d={decoder_yaml}',
        '-e=examples/txt/bsc.error.yaml',
        f'-c={logical_check_yaml}',
        '-s=examples/txt/perfect.syndrome.yaml',
        f'-bs={batch_size}',
        f'-te={target_error}',
        '-se=examples/txt/save_error_llr.yaml'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print stdout and stderr
    print('STDOUT:\n', result.stdout)
    print('STDERR:\n', result.stderr)


if __name__ == '__main__': 
    batch_size = 1000
    target_error = 100
    test_batch_txt_hx(target_error, batch_size)
    test_batch_txt_hz(target_error, batch_size)
    test_batch_alist_hx(target_error, batch_size)
    test_batch_alist_hz(target_error, batch_size)