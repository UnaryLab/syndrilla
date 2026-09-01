import argparse
from pathlib import Path

import yaml


def parse_commandline_args():
    """
    parse command line inputs
    """
    parser = argparse.ArgumentParser(
        description='Test output for the decoder zoo.')
    parser.add_argument('-r', '--run_dir', type=str, default='examples/alist',
                        help = 'The run directory.')
    parser.add_argument('-i', '--int_width', type=int, default=3,
                        help = 'The integer bitwidth.')
    parser.add_argument('-f', '--frac_width', type=int, default=4,
                        help = 'The integer bitwidth.')
    parser.add_argument('-d', '--decoder', type=str, default='bposd_quant',
                        help = 'The decoder.')

    return parser.parse_args()


def main():
    args = parse_commandline_args()
    run_dir = Path(args.run_dir)

    # --- Iterate and update ---
    for file in run_dir.rglob('*quant*.decoder.yaml'):
        if file.is_file():
            print(f'Processing: {file}')

            # Load YAML
            with open(file, 'r') as f:
                data = yaml.safe_load(f)

            decoder_config = data['decoder'].setdefault('config', {})
            stage = decoder_config[0] if isinstance(decoder_config, list) else decoder_config
            stage['int_width'] = args.int_width
            stage['frac_width'] = args.frac_width

            # Write back
            with open(file, 'w') as f:
                yaml.safe_dump(data, f, sort_keys=False)

    sweeping_file = Path('zoo/script/sweeping_configs.yaml')
    with open(sweeping_file, 'r') as f:
        data = yaml.safe_load(f)

    print(f'Processing: {sweeping_file}')
    data['decoder'] = [args.decoder]

    # Write back
    with open(sweeping_file, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print('Update complete.')


if __name__ == '__main__':
    main()

