import os

from loguru import logger

from syndrilla.utils import call_func_from_yaml, call_func_from_cfg, get_path


def create_loss(yaml_path: str = None, cfg: dict = None, **kwargs):
    """
    Create a loss module from a '.loss.yaml' file or a config dict.

    The loss binds to the decoder it supervises, so pass it through:

        create_loss(yaml_path='logical_centric.loss.yaml', decoder=saq)
    """
    header = 'loss'
    func_name = 'function'
    if cfg is not None:
        logger.info(f'Creating loss module from config dict.')
        output = call_func_from_cfg(cfg, header, func_name, os.path.dirname(__file__), **kwargs)
    else:
        logger.info(f'Creating loss module from <{get_path(yaml_path)}>.')
        output = call_func_from_yaml(yaml_path, header, func_name, os.path.dirname(__file__), **kwargs)
    logger.info(f'Creating loss module complete.')
    return output
