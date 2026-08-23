import argparse
import os
import subprocess
from pathlib import Path

from loguru import logger

decoder_list = ['bposd', 'bposd_quant', 'lottery_bp', 'lottery_bp_quant', 'lottery_bposd', 'lottery_bposd_quant']


def parse_commandline_args():
    """
    parse command line inputs
    """
    parser = argparse.ArgumentParser(
        description='Test code for the decoder zoo.')
    parser.add_argument('-r', '--run_dir', type=str, default=None,
                        help = 'The run directory.')
    parser.add_argument('-d', '--decoder', type=str, default='bposd',
                        help = 'The decoder.')
    parser.add_argument('-bs', '--batch_size', type=int, default=10000,
                        help = 'The number of samples run each batch.')
    parser.add_argument('-st', '--syndrome_type', type=str, default='perfect',
                        help = 'The syndrome type (perfect or phenomenological).')
    parser.add_argument('-l', '--log_level', type=str, default='SUCCESS',
                        help = 'The log level to record runtime statistics.')

    return parser.parse_args()


def main(target_error):
    args = parse_commandline_args()

    assert os.path.isdir(args.run_dir), logger.error(f'Illegal input run_dir: {args.run_dir}.')
    run_dir = args.run_dir
    assert args.decoder in decoder_list, logger.error(f'Illegal input decoder: {args.decoder}; legal values: {decoder_list}.')
    decoder = args.decoder
    batch_size = args.batch_size
    syndrome_type = args.syndrome_type
    log_level = args.log_level

    subdirs = sorted([d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))], reverse=True)

    for subdir in subdirs:
        print(f'\n=== folder {subdir} ===')

        # checking whether the decoder is hx or hz
        folder_path = os.path.join(run_dir, subdir)

        if 'hx' in subdir:
            decoder_yaml = os.path.join(folder_path, f'{decoder}_hx.decoder.yaml')
            check_yaml = os.path.join(folder_path, 'lx.check.yaml')
        else:
            decoder_yaml = os.path.join(folder_path, f'{decoder}_hz.decoder.yaml')
            check_yaml = os.path.join(folder_path, 'lz.check.yaml')

        error_yaml = os.path.join(folder_path, 'bsc.error.yaml')

        syndrome_yaml = os.path.join(folder_path, f'{syndrome_type}.syndrome.yaml')
        matrix_yaml = os.path.join(folder_path, 'matrix.yaml')

        # set command line arguement to load all yaml file from the folder
        cmd = [
            'syndrilla',
            f'-r={folder_path}/',
            f'-d={decoder_yaml}',
            f'-e={error_yaml}',
            f'-c={check_yaml}',
            f'-s={syndrome_yaml}',
            f'-m={matrix_yaml}',
            f'-bs={batch_size}',
            f'-te={target_error}',
            f'-l={log_level}'
        ]

        folder_path = Path(folder_path)
        results_files = list(folder_path.glob('result*'))

        if results_files:
            print('Results exist: ', results_files[0])
            cmd.append(f'-ckpt={results_files[0]}')
        else:
            print('Results do not exist.')

        print('Run command: ', ' '.join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        print('  --> STDOUT:\n', result.stdout)
        print('  --> STDERR:\n', result.stderr)


if __name__ == '__main__':
    main(target_error=1000)
