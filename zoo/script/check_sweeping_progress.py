import argparse
from pathlib import Path


def parse_commandline_args():
    """
    parse command line inputs
    """
    parser = argparse.ArgumentParser(
        description='Test output for the decoder zoo.')
    parser.add_argument('-r', '--run_dir', type=str, default=None,
                        help = 'The run directory.')
    parser.add_argument('-rs', '--run_dirs', type=str, default=None,
                        help = 'The run directories, a parent directory of multiple run directories.')

    return parser.parse_args()


def main():
    args = parse_commandline_args()

    if args.run_dir is not None:
        root_dir = Path(args.run_dir)

        total_sim = 0
        total_result = 0
        for subfolder in sorted(root_dir.iterdir()):
            if subfolder.is_dir():
                total_sim += 1
                results_files = list(subfolder.glob("result*"))

                if results_files:
                    print(f"{subfolder}: result ready.")
                    total_result += 1
                else:
                    print(f"{subfolder}: result NOT ready.")

        print(f"{total_result/total_sim*100:.2f} % simulations are done.")

    if args.run_dirs is not None:
        root_dirs = Path(args.run_dirs)

        for subfolder in sorted(root_dirs.iterdir()):
            total_sim = 0
            total_result = 0
            if Path(subfolder).is_dir():
                for subsubfolder in sorted(subfolder.iterdir()):
                    if subsubfolder.is_dir():
                        total_sim += 1
                        results_files = list(subsubfolder.glob("result*"))

                        if results_files:
                            total_result += 1

                print(f"{total_result/total_sim*100:.2f} % simulations are done in {subfolder}.")


if __name__ == '__main__':
    main()

