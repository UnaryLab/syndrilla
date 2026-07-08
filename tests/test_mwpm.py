import torch
import re, yaml
import sys, os, time
import subprocess
from loguru import logger

sys.path.append(os.getcwd())

from syndrilla.decoder import create_decoder
from syndrilla.error_model import create_error_model
from syndrilla.syndrome import create_syndrome
from syndrilla.metric import report_metric, compute_avg_metrics
from syndrilla.logical_check import create_check


def test_batch_alist_hx(batch_size=1000, target_error=1000):
    decoder_yaml = "examples/alist/mwpm_hx.decoder.yaml"
    logical_check_yaml = "examples/alist/lx.check.yaml"
    cmd = [
        "syndrilla",
        "-r=tests/test_outputs",
        f"-d={decoder_yaml}",
        "-e=examples/alist/bsc.error.yaml",
        f"-c={logical_check_yaml}",
        "-s=examples/alist/perfect.syndrome.yaml",
        f"-bs={batch_size}",
        f"-te={target_error}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print stdout and stderr
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)


if __name__ == "__main__":
    batch_size = 1000
    target_error = 1000
    test_batch_alist_hx(batch_size, target_error)
