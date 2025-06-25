import torch
import re
import sys, os, time
import numpy as np
import subprocess
from loguru import logger

sys.path.append(os.getcwd())

from src.decoder import create_decoder
from src.error_model import create_error_model
from src.syndrome import create_syndrome
from src.metric import report_metric
from src.logical_check import create_check


def test_batch_alist_hz(target_error=1000, batch_size=1000):    
    decoder_yaml = 'examples/txt/lottery_bp_hz.decoder.yaml'
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
        '-se=examples/alist/save_error_llr.yaml'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print stdout and stderr
    print('STDOUT:\n', result.stdout)
    print('STDERR:\n', result.stderr)


def test_batch_alist_hz_quant(target_error=1000, batch_size=1000):    
    decoder_yaml = 'examples/txt/lottery_bp_hz_qaunt.decoder.yaml'
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
        '-se=examples/alist/save_error_llr.yaml'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print stdout and stderr
    print('STDOUT:\n', result.stdout)
    print('STDERR:\n', result.stderr)


if __name__ == '__main__':
    batch_size = 100000
    target_error = 1000
    test_batch_alist_hz(target_error, batch_size)
    test_batch_alist_hz_quant(target_error, batch_size)