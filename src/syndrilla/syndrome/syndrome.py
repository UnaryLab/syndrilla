import os

from loguru import logger

from syndrilla.utils import call_func_from_cfg, call_func_from_yaml, get_path


def create_syndrome(yaml_path: str = None, cfg: dict = None, **kwargs):
    """
    Create a syndrome measurer from a '.syndrome.yaml' file or a config dict.

    `training` is a build-time *mode*, passed the way `create_error_model` takes it,
    not a key of the yaml. A measurer reads it to decide whether a swept
    `measurement_error_rate` is allowed: a range is the training-only form.
    """
    header = "syndrome"
    func_name = "measure"
    if cfg is not None:
        logger.info("Creating syndrome class from config dict.")
        output = call_func_from_cfg(
            cfg, header, func_name, os.path.dirname(__file__), **kwargs
        )
    else:
        logger.info(f"Creating syndrome class from <{get_path(yaml_path)}>.")
        output = call_func_from_yaml(
            yaml_path, header, func_name, os.path.dirname(__file__), **kwargs
        )
    logger.info("Creating syndrome class complete.")
    return output
