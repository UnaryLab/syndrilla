import os
import yaml, copy
import torch
from loguru import logger

from syndrilla.utils import call_func_from_cfg, get_path, read_yaml, check_yaml_header


class RoundFlattenWrapper(torch.nn.Module):
    """
    Wraps a decoder to transparently handle a rounds dimension (always dim=1).

    Flattens [B, d, ...] → [B*d, ...] before the inner decoder and reshapes
    all outputs back to [B, d, ...] afterwards.  No-op when syndrome is 2D
    (1-channel, 1 round) or 3D with no rounds (2-channel, 1 round).

    Rounds dim convention: [B, d_rounds, (C), M]
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.decoder, name)

    def _needs_flatten(self, synd):
        """1ch decoder expects 2D, 2ch decoder expects 3D. Extra dim = rounds."""
        base_ndim = getattr(self.decoder, '_base_synd_ndim', 2)
        return synd.ndim > base_ndim

    def forward(self, io_dict):
        synd = io_dict['synd']
        if not self._needs_flatten(synd):
            return self.decoder(io_dict)

        B, d = synd.shape[0], synd.shape[1]
        Bd = B * d
        rest = synd.shape[2:]

        io_dict['synd'] = synd.reshape(Bd, *rest)

        llr0 = io_dict['llr0']
        llr0_rest = llr0.shape[1:]
        io_dict['llr0'] = llr0.unsqueeze(1).expand(B, d, *llr0_rest).reshape(Bd, *llr0_rest)

        for key in ('llr', 'converge', 'iter', 'e_v'):
            if key in io_dict and io_dict[key].shape[0] == B and io_dict[key].ndim >= 2:
                v = io_dict[key]
                io_dict[key] = v.reshape(Bd, *v.shape[2:])

        io_dict = self.decoder(io_dict)

        for key in ('e_v', 'synd', 'llr'):
            if key in io_dict and io_dict[key].shape[0] == Bd:
                v = io_dict[key]
                io_dict[key] = v.reshape(B, d, *v.shape[1:])
        for key in ('converge', 'iter'):
            if key in io_dict and io_dict[key].shape[0] == Bd:
                io_dict[key] = io_dict[key].reshape(B, d)
        io_dict['llr0'] = llr0

        return io_dict


def create_decoder(yaml_path: str=None, cfg: dict=None, **kwargs):
    """
    Create decoder(s) from a '.decoder.yaml' file or a config dict.

        create_decoder(yaml_path='bp_hx.decoder.yaml')
        create_decoder(cfg={'algorithm': 'bp_norm_min_sum', ...})
    """
    header = 'decoder'
    func_name = 'algorithm'

    if cfg is not None:
        dec_cfg = cfg
        logger.info(f'Creating decoder class from config dict.')
    else:
        logger.info(f'Creating decoder class from <{get_path(yaml_path)}>.')
        full_path = get_path(yaml_path)
        load_cfg = read_yaml(full_path)
        check_yaml_header(load_cfg, header, full_path)
        dec_cfg = load_cfg[header]

    # Read algorithm(s)
    algorithms = dec_cfg[func_name]
    if isinstance(algorithms, str):
        algorithms = [algorithms]  # wrap single decoder into a list

    MULTI_CHANNEL_DECODERS = {'bp4'}

    decoders = []
    for algo in algorithms:
        dec_cfg_copy = dec_cfg.copy()
        dec_cfg_copy[func_name] = algo
        decoder = call_func_from_cfg(dec_cfg_copy, header, func_name, os.path.dirname(__file__), **kwargs)
        decoder._base_synd_ndim = 3 if algo.lower() in MULTI_CHANNEL_DECODERS else 2
        decoders.append(RoundFlattenWrapper(decoder))

    return decoders
