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


def test_batch_alist_hz(batch_size=1000, target_error=1000, save_error_llr=False):    
    decoder_yaml = 'examples/alist/lottery_bp_hz.decoder.yaml'
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
        f'-se={save_error_llr}'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print stdout and stderr
    print('STDOUT:\n', result.stdout)
    print('STDERR:\n', result.stderr)


def test_batch_alist_hz_quant(batch_size=1000, target_error=1000, save_error_llr=False):    
    decoder_yaml = 'examples/alist/lottery_bp_quant_hz.decoder.yaml'
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
        f'-se={save_error_llr}'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print stdout and stderr
    print('STDOUT:\n', result.stdout)
    print('STDERR:\n', result.stderr)


if __name__ == '__main__':
    batch_size = 100000
    target_error = 1000
    save_error_llr = False
    test_batch_alist_hz(batch_size, target_error, save_error_llr)
    test_batch_alist_hz_quant(batch_size, target_error, save_error_llr)