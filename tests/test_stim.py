import subprocess
import sys
import os

sys.path.append(os.getcwd())


def test_stim_bposd(batch_size=1000, target_error=100):
    cmd = [
        'syndrilla',
        '-r=tests/test_outputs',
        '-d=examples/stim/stim_generated.decoder.yaml',
        '-i=examples/stim/stim_generated.interface.yaml',
        '-e=examples/stim/stim_generated.error.yaml',
        '-s=examples/stim/stim_generated.syndrome.yaml',
        f'-bs={batch_size}',
        f'-te={target_error}',
    ]

    # Stream output live so progress and any error are visible immediately.
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f'syndrilla exited with code {result.returncode}: {cmd}')


if __name__ == '__main__':
    batch_size = 1000
    target_error = 100
    test_stim_bposd(batch_size, target_error)
