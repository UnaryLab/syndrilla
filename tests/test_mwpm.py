import os
import subprocess
import sys


sys.path.append(os.getcwd())



def test_batch_alist_hx(batch_size=1000, target_error=1000):
    decoding_yaml = "examples/alist/mwpm_hx.decoding.yaml"
    logical_check_yaml = "examples/alist/lx.check.yaml"
    cmd = [
        "syndrilla",
        "-r=tests/test_outputs",
        f"-d={decoding_yaml}",
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
