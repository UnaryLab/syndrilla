import yaml
import subprocess
import re
import csv
import os
import shutil


def modify_yaml(file_path, changes):
    """
    Load a YAML file, apply the given changes, and write them back.
    changes is a dict like:
        { 'decoder.int_width': 3, 'decoder.frac_width': 4 }
    """
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)

    # Apply changes, handling nested keys by splitting on '.'
    for dotted_key, value in changes.items():
        keys = dotted_key.split('.')
        d = config
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value

    with open(file_path, 'w') as f:
        yaml.safe_dump(config, f)


def main(batch_size, target_error, save_error_llr):
    # Paths to your YAML files (adjust to your actual paths)
    decoder_yaml = 'examples/alist/bp_hz.decoder.yaml'
    error_yaml   = 'examples/alist/bsc.error.yaml'

    # Define error rates
    error_rates = [0.01, 0.02, 0.03, 0.04, 0.05]

    # Loop over N in [4..10] => 'Int4' through 'Int10'
    # We assume 1 sign bit => int_width + frac_width = N - 1
    float_types = ['bfloat16', 'float16', 'float32', 'float64']
    for ftype in float_types:
        # Loop over error rates
        for rate in error_rates:
            print(f'\n=== dtype={ftype}, int_width={0}, frac_width={0}, physical_error_rate={rate} ===')

            # 1) Modify the decoder YAML
            modify_yaml(
                file_path=decoder_yaml,
                changes={
                    'decoder.dtype': ftype
                }
            )

                # 2) Modify the error YAML
            modify_yaml(
                file_path=error_yaml,
                changes={
                    'error.rate': rate
                }
            )

            # 3) Run syndrilla
            cmd = [
                'syndrilla',
                f'-r=zoo/bp_sweeping/bp_{ftype}_surface_10_hz/',
                f'-d={decoder_yaml}',
                f'-e={error_yaml}',
                '-c=examples/alist/lz.check.yaml',
                '-s=examples/alist/perfect.syndrome.yaml',
                f'-bs={batch_size}',
                f'-te={target_error}',
                f'-se={save_error_llr}'
            ]
            print('Command: ', ' '.join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)

            # Print stdout/stderr for debugging
            print('  --> STDOUT:\n', result.stdout)
            print('  --> STDERR:\n', result.stderr)


if __name__ == '__main__':
    batch_size = 100000
    target_error = 1000
    save_error_llr = False
    main(batch_size, target_error, save_error_llr)
