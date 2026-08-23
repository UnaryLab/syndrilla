import argparse
from collections import OrderedDict
from pathlib import Path

import yaml
from yamlordereddictloader import SafeLoader


def parse_commandline_args():
    """
    parse command line inputs
    """
    parser = argparse.ArgumentParser(
        description='Plot output for the decoder zoo.')
    parser.add_argument('-r', '--run_dir', type=str, default=None,
                        help = 'The run directory.')

    return parser.parse_args()


def parse_dirname_to_sorted_ordereddict(dirname):
    parts = dirname.split("_")
    output = sorted(parts)
    return output


def read_result_yaml_from_dir(directory):
    dir_path = Path(directory)

    # Find the first YAML file starting with "result"
    result_file = next(
        (f for f in dir_path.iterdir() if f.name.startswith("result") and f.suffix in {".yaml", ".yml"}),
        None
    )
    if result_file is None:
        raise FileNotFoundError(f"No YAML file starting with 'result' found in {directory}")

    # Load YAML content
    with open(result_file, "r") as f:
        yaml_data = yaml.load(f, Loader=SafeLoader)

    # Generate sorted OrderedDict from directory name
    sorted_key = parse_dirname_to_sorted_ordereddict(dir_path.name)

    return sorted_key, yaml_data


def load_results_dict(root_dir):
    root_dir = Path(root_dir)

    results_dict = OrderedDict()

    total_sim = 0
    total_result = 0
    for subfolder in root_dir.iterdir():
        if subfolder.is_dir():
            total_sim += 1
            results_files = list(subfolder.glob("result*"))

            if results_files:
                total_result += 1
                output_key, output_value = read_result_yaml_from_dir(subfolder)
                results_dict[str(output_key)] = output_value

    print(f"{total_result/total_sim*100:.2f} % simulations are done in {root_dir}")
    return results_dict


def is_substring(sub, full):
    """
    check if the sub string is a subtring of the full string.
    """
    it = iter(full)
    return all(char in it for char in sub)


def tag_to_str(tag: list):
    tag_sorted = sorted(tag)
    return str(tag_sorted)


def lookup_results_dict(input_dict: OrderedDict(), key_list: list):
    # query a dict with arbitrary key order in the list, until finding a value query path
    temp_dict = input_dict
    temp_key_list = key_list

    for _ in range(len(key_list)):
        for key in temp_key_list:
            temp_output = temp_dict.get(key, {})
            if (isinstance(temp_output, dict) or isinstance(temp_output, OrderedDict)) and (len(temp_output.keys()) > 0):
                # if a valid dict, remove this key from list and update to the next level dict
                temp_key_list.remove(key)
                temp_dict = temp_output
                continue
            elif (isinstance(temp_output, dict) or isinstance(temp_output, OrderedDict)) and (len(temp_output.keys()) == 0):
                # if an empty dict, move on to the next key trial
                pass
            else:
                # if not a dict, return result
                # it is possible the returned result is None, as the input_dict is empty
                output = float(temp_output)
                return output

    return None
