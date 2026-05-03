import os

from loguru import logger

from syndrilla.utils import call_func_from_yaml, call_func_from_cfg, get_path, read_yaml


def create_interface(yaml_path: str = None, cfg: dict = None,
                     error_yaml: str = None, error_cfg: dict = None,
                     syndrome_yaml: str = None, syndrome_cfg: dict = None,
                     decoder_yaml: str = None, decoder_cfg: dict = None,
                     **kwargs):
    """
    Create an interface from a '.interface.yaml' file or a config dict.

    Error, syndrome, and decoder configs are passed through to the backend
    so it can use noise parameters for circuit generation, measurement
    settings, and inject matrix configs into the decoder.
    """
    if error_yaml is not None and error_cfg is None:
        full = read_yaml(get_path(error_yaml))
        error_cfg = full.get('error', {})
    if syndrome_yaml is not None and syndrome_cfg is None:
        full = read_yaml(get_path(syndrome_yaml))
        syndrome_cfg = full.get('syndrome', {})
    if decoder_yaml is not None and decoder_cfg is None:
        full = read_yaml(get_path(decoder_yaml))
        decoder_cfg = full.get('decoder', {})

    kwargs['error_cfg'] = error_cfg or {}
    kwargs['syndrome_cfg'] = syndrome_cfg or {}
    kwargs['decoder_cfg'] = decoder_cfg or {}

    header = 'interface'
    func_name = 'backend'
    if cfg is not None:
        logger.info(f'Creating interface from config dict.')
        output = call_func_from_cfg(cfg, header, func_name, os.path.dirname(__file__), **kwargs)
    else:
        logger.info(f'Creating interface from <{get_path(yaml_path)}>.')
        output = call_func_from_yaml(yaml_path, header, func_name, os.path.dirname(__file__), **kwargs)
    logger.info(f'Creating interface complete.')
    return output
