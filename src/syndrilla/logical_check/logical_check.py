import os
from loguru import logger

from syndrilla.utils import call_func_from_yaml, call_func_from_cfg, get_path


def create_check(yaml_path: str=None, cfg: dict=None, **kwargs):
    """
    Create a logical check from a '.check.yaml' file or a config dict.
    """
    header = 'check'
    func_name = 'check_type'
    if cfg is not None:
        logger.info(f'Creating logical error check class from config dict.')
        output = call_func_from_cfg(cfg, header, func_name, os.path.dirname(__file__), **kwargs)
    else:
        logger.info(f'Creating logical error check class from <{get_path(yaml_path)}>.')
        output = call_func_from_yaml(yaml_path, header, func_name, os.path.dirname(__file__), **kwargs)
    logger.info(f'Creating logical error check class complete.')
    return output

