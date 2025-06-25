from src.utils import read_yaml, write_yaml
from collections import OrderedDict
import os, shutil, copy, itertools
from loguru import logger


def clean_dir(base_path: str):
    base_path = base_path if base_path.endswith('/') else base_path + '/'
    generated_path = f'{base_path}generated/'
    shutil.rmtree(generated_path, ignore_errors=True)


def generate_decoder(base_path: str, code, type, distance, float, decoder):
    """
        Generate decoder yaml file from examples
    """
    base_path = base_path if base_path.endswith('/') else base_path + '/'
    template_path = 'examples/alist/'
    generated_path = base_path

    configuration_dict = template_path + f'{decoder}_{type}.decoder.yaml'

    target_file = os.path.join(generated_path, f'{decoder}_{type}.decoder.yaml')

    # Read, modify, and write
    config = read_yaml(configuration_dict)
    if 'decoder' in config:
        config['decoder']['dtype'] = float
        config['decoder']['type'] = type
        if code == 'surface':
            config['decoder']['max_iter'] = distance*2*(distance-1)+1
        else:
            config['decoder']['max_iter'] = distance*2*(distance)
        config['decoder']['parity_matrix_hx'] = base_path + 'hx.matrix.yaml'
        config['decoder']['parity_matrix_hz'] = base_path + 'hz.matrix.yaml'
        config['decoder']['logical_check_matrix'] = True
        config['decoder']['logical_check_lx'] = base_path + 'lx.matrix.yaml'
        config['decoder']['logical_check_lz'] = base_path + 'lz.matrix.yaml'

    write_yaml(target_file, config)


def generate_matrix(base_path: str, code, distance):
    """
        Generate matrix yaml files from examples
    """
    base_path = base_path if base_path.endswith('/') else base_path + '/'
    template_path = 'examples/alist/'
    generated_path = base_path

    type_list = ['hx', 'hz', 'lx', 'lz']

    # create all hx, hz, lx, lz matrix yaml files 
    for type in type_list:
        configuration_dict = template_path + f'{type}.matrix.yaml'
        target_file = os.path.join(generated_path, f'{type}.matrix.yaml')

        # Read, modify, and write
        config = read_yaml(configuration_dict)
        if 'matrix' in config:
            config['matrix']['path'] = f'examples/alist/{code}/{code}_{distance}_{type}.alist'
        write_yaml(target_file, config)


def generate_error(base_path: str, probability):
    """
        Generate error yaml file from examples
    """
    base_path = base_path if base_path.endswith('/') else base_path + '/'
    template_path = 'examples/alist/'
    generated_path = base_path
    
    configuration_dict = template_path + 'bsc.error.yaml'
    target_file = os.path.join(generated_path, 'bsc.error.yaml')

    # Read, modify, and write
    config = read_yaml(configuration_dict)
    if 'error' in config:
        config['error']['rate'] = probability
    write_yaml(target_file, config)


def generate_syndrome(base_path: str):
    """
        Generate syndrome yaml file from examples
    """
    base_path = base_path if base_path.endswith('/') else base_path + '/'
    template_path = 'examples/alist/'
    generated_path = base_path
    
    configuration_dict = template_path + 'perfect.syndrome.yaml'
    target_file = os.path.join(generated_path, 'perfect.syndrome.yaml')

    # Read, and write
    config = read_yaml(configuration_dict)
    write_yaml(target_file, config)


def generate_checker(base_path: str, type):
    """
        Generate logical check yaml file from examples
    """
    base_path = base_path if base_path.endswith('/') else base_path + '/'
    template_path = 'examples/alist/'
    generated_path = base_path

    if type == 'hx':
        configuration_dict = template_path + 'lx.check.yaml'
        target_file = os.path.join(generated_path, 'lx.check.yaml')
    else:
        configuration_dict = template_path + 'lz.check.yaml'
        target_file = os.path.join(generated_path, 'lz.check.yaml')

    # Read, write
    config = read_yaml(configuration_dict)
    write_yaml(target_file, config)


def generate_checker(base_path: str, type):
    """
        Generate logical check yaml file from examples
    """
    base_path = base_path if base_path.endswith('/') else base_path + '/'
    template_path = 'examples/alist/'
    generated_path = base_path

    if type == 'hx':
        configuration_dict = template_path + 'lx.check.yaml'
        target_file = os.path.join(generated_path, 'lx.check.yaml')
    else:
        configuration_dict = template_path + 'lz.check.yaml'
        target_file = os.path.join(generated_path, 'lz.check.yaml')

    # Read, write
    config = read_yaml(configuration_dict)
    write_yaml(target_file, config)


def main():
    logger.remove()
    base_dir = 'zoo/'
    config_list = read_yaml(base_dir + 'script/code_configuration.yaml')
    # load each different code settings from code_configuration.yaml file
    for decoder in config_list['decoder']:
        for code in config_list['code']:
            for type in config_list['type']:
                for distance in config_list['distance']:
                    for float in config_list['float']:
                        for probability in config_list['probability']:
                            dir_name = f'{decoder}_sweeping/{code}_{type}_{distance}_{float}_{probability}'
                            base_path = os.path.join(base_dir, dir_name)
                            os.makedirs(base_path, exist_ok=True)
                            # clean up directionary
                            clean_dir(base_path)
                            if os.path.isdir(base_path):
                                # create each yaml files
                                generate_decoder(base_path, code, type, distance, float, decoder)
                                generate_syndrome(base_path)
                                generate_error(base_path, probability)
                                generate_matrix(base_path, code, distance)
                                generate_checker(base_path, type)
                                

if __name__ == '__main__':
    main()