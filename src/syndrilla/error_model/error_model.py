import os

from loguru import logger

from syndrilla.utils import call_func_from_cfg, call_func_from_yaml, get_path


def create_error_model(yaml_path: str = None, cfg: dict = None, **kwargs):
    """
    Create an error model from a '.error.yaml' file or a config dict.

    `training` is a build-time *mode*, passed on the way `main.py` passes it to
    `create_decoder`, not a key of the yaml. A model reads it to decide whether a
    swept `rate` is allowed: a range is the training-only form, since a decode run
    records one physical error rate per result file.
    """
    header = "error"
    func_name = "model"
    if cfg is not None:
        logger.info("Creating error model class from config dict.")
        output = call_func_from_cfg(
            cfg, header, func_name, os.path.dirname(__file__), **kwargs
        )
    else:
        logger.info(f"Creating error model class from <{get_path(yaml_path)}>.")
        output = call_func_from_yaml(
            yaml_path, header, func_name, os.path.dirname(__file__), **kwargs
        )
    logger.info("Creating error model class complete.")
    return output
