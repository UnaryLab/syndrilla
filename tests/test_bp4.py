import subprocess
import sys
import os

sys.path.append(os.getcwd())


def test_bp4(batch_size=10000, target_error=1000):
    cmd = [
        'syndrilla',
        '-r=tests/test_outputs',
        '-d=examples/alist/bp4.decoder.yaml',
        '-e=examples/alist/depol.error.yaml',
        '-c=examples/alist/lx.check.yaml',
        '-s=examples/alist/perfect.syndrome.yaml',
        '-m=examples/alist//surface_10.matrix.yaml',
        f'-bs={batch_size}',
        f'-te={target_error}',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print stdout and stderr
    print('STDOUT:\n', result.stdout)
    print('STDERR:\n', result.stderr)



if __name__ == '__main__':
    test_bp4(batch_size=10000, target_error=1000)
