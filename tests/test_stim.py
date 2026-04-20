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

    result = subprocess.run(cmd, capture_output=True, text=True)

    print('STDOUT:\n', result.stdout)
    print('STDERR:\n', result.stderr)


def test_stim_bposd_majority_vote_syndrome(batch_size=1000, target_error=100):
    cmd = [
        'syndrilla',
        '-r=tests/test_outputs',
        '-d=examples/stim/stim_generated.decoder.yaml',
        '-i=examples/stim/stim_generated.interface.yaml',
        '-e=examples/stim/stim_generated.error.yaml',
        '-s=examples/stim/stim_generated.syndrome.yaml',
        f'-bs={batch_size}',
        f'-te={target_error}',
        '-vs=syndrome',
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    print('STDOUT:\n', result.stdout)
    print('STDERR:\n', result.stderr)


def test_stim_bposd_majority_vote_decoder_0(batch_size=1000, target_error=100):
    cmd = [
        'syndrilla',
        '-r=tests/test_outputs',
        '-d=examples/stim/stim_generated.decoder.yaml',
        '-i=examples/stim/stim_generated.interface.yaml',
        '-e=examples/stim/stim_generated.error.yaml',
        '-s=examples/stim/stim_generated.syndrome.yaml',
        f'-bs={batch_size}',
        f'-te={target_error}',
        '-vs=decoder_0',
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    print('STDOUT:\n', result.stdout)
    print('STDERR:\n', result.stderr)


def test_stim_bposd_majority_vote_decoder_last(batch_size=1000, target_error=100):
    cmd = [
        'syndrilla',
        '-r=tests/test_outputs',
        '-d=examples/stim/stim_generated.decoder.yaml',
        '-i=examples/stim/stim_generated.interface.yaml',
        '-e=examples/stim/stim_generated.error.yaml',
        '-s=examples/stim/stim_generated.syndrome.yaml',
        f'-bs={batch_size}',
        f'-te={target_error}',
        '-vs=decoder_1',
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    print('STDOUT:\n', result.stdout)
    print('STDERR:\n', result.stderr)


if __name__ == '__main__':
    batch_size = 1000
    target_error = 100
    test_stim_bposd(batch_size, target_error)
    test_stim_bposd_majority_vote_syndrome(batch_size, target_error)
    test_stim_bposd_majority_vote_decoder_0(batch_size, target_error)
    test_stim_bposd_majority_vote_decoder_last(batch_size, target_error)
