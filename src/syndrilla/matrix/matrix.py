import os

import torch
from loguru import logger

from syndrilla.utils import call_func_from_yaml, call_func_from_cfg, get_path, compute_lz


# Shared circuit cache for the stim matrix loader. Lives in this module
# (instead of stim.py) because the factory dynamically re-imports loader
# modules per call, which would otherwise reset a per-loader cache.
STIM_CIRCUIT_CACHE: dict = {}


def dense_to_index_format(matrix_np, device):
    """
    Convert a dense GF(2) matrix into the (shape, V_c_row, V_c_col, matrix_tensor)
    tuple every syndrilla decoder consumes from a matrix loader's get_index().

    Same logic as the inlined version in alist.py / npz.py — extracted so new
    loaders (e.g. the stim DEM loader) don't have to copy ~25 lines of imperative
    code.
    """
    shape = matrix_np.shape
    matrix = torch.tensor(matrix_np, device=device)

    degree = torch.max(torch.sum(matrix, 1)).int().item()

    row_indices, indices = torch.where(matrix == 1)

    V_c_row = torch.full([shape[0], degree], -1, dtype=torch.long, device=device)
    V_c_col = torch.full([shape[0], degree], -1, dtype=torch.long, device=device)

    row = 0
    column = 0
    for i in range(indices.size()[0]):
        if row_indices[i] == row:
            V_c_row[row][column] = row
            V_c_col[row][column] = indices[i]
            column += 1
        else:
            while V_c_col[row][degree - 1] == -1:
                V_c_row[row][column] = row
                V_c_col[row][column] = shape[1]
                column += 1
            row += 1
            column = 0
            V_c_row[row][column] = row
            V_c_col[row][column] = indices[i]
            column += 1
    while V_c_col[row][degree - 1] == -1:
        V_c_row[row][column] = row
        V_c_col[row][column] = shape[1]
        column += 1

    return shape, V_c_row, V_c_col, matrix


def create_parity_matrix(yaml_path=None, cfg=None, **kwargs):
    """
    Create a matrix loader. Accepts either a yaml file path or a config dict.

        create_parity_matrix(yaml_path='check.matrix.yaml', device=...)
        create_parity_matrix(cfg={'file_type': 'stim', 'circuit': '...', 'target': 'check'}, device=...)
    """
    header = 'matrix'
    func_name = 'file_type'
    if cfg is not None:
        logger.info(f'Creating parity matrix class from config dict.')
        output = call_func_from_cfg(cfg, header, func_name, os.path.dirname(__file__), **kwargs)
    else:
        logger.info(f'Creating parity matrix class from <{get_path(yaml_path)}>.')
        output = call_func_from_yaml(yaml_path, header, func_name, os.path.dirname(__file__), **kwargs)
    logger.info(f'Complete.')
    return output


class MatrixBundle:
    """
    Container for the parity-check (Hx, Hz) and logical (lx, lz) matrices a
    decoder needs. Decoders should consume one of these instead of calling
    create_parity_matrix() and compute_lz() themselves.
    """
    __slots__ = ('Hx_matrix', 'Hz_matrix', 'lx_matrix', 'lz_matrix')

    def __init__(self, Hx_matrix, Hz_matrix, lx_matrix, lz_matrix):
        self.Hx_matrix = Hx_matrix
        self.Hz_matrix = Hz_matrix
        self.lx_matrix = lx_matrix
        self.lz_matrix = lz_matrix

    def select(self, check_type):
        """
        Return (H_shape, V_c_row, V_c_col, H_matrix) for the requested check
        type. Used by single-channel decoders that pick one of Hx/Hz.
        """
        m = self.Hx_matrix if check_type.lower() == 'hx' else self.Hz_matrix
        return m.get_index()

    def get_H_file_name(self, check_type, number_channel):
        if number_channel > 1:
            return [self.Hx_matrix.path, self.Hz_matrix.path]
        if check_type.lower() == 'hx':
            return self.Hx_matrix.path
        return self.Hz_matrix.path

    def get_l_matrix(self, check_type, number_channel):
        if number_channel > 1:
            lx = torch.as_tensor(self.lx_matrix)
            lz = torch.as_tensor(self.lz_matrix)
            return torch.stack((lx, lz), dim=1)
        if check_type.lower() == 'hx':
            return self.lx_matrix
        return self.lz_matrix

    def get_check_num(self, check_type, number_channel):
        if number_channel > 1:
            return 0
        if check_type.lower() == 'hx':
            return 0
        return 1


def _load_one_matrix(value, **kw):
    """Load a matrix from either a yaml path (str) or a config dict."""
    if isinstance(value, dict):
        return create_parity_matrix(cfg=value, **kw)
    return create_parity_matrix(yaml_path=value, **kw)


def load_matrices(matrix_cfg, device, dtype=None):
    """
    Load all matrices a decoder needs from a matrix config.

    Matrix entries ('parity_matrix_hx', etc.) can be either:
      - a yaml file path (str)  — loaded via create_parity_matrix(yaml_path=...)
      - a config dict           — loaded via create_parity_matrix(cfg=...)

    Returns a MatrixBundle. If logical_check_matrix is False/missing, the
    logical matrices are computed via compute_lz from Hx/Hz.
    """
    kw = {'device': device}
    if dtype is not None:
        kw['dtype'] = dtype

    logger.info(f'Loading hx parity check matrix.')
    Hx_matrix = _load_one_matrix(matrix_cfg['parity_matrix_hx'], **kw)

    logger.info(f'Loading hz parity check matrix.')
    Hz_matrix = _load_one_matrix(matrix_cfg['parity_matrix_hz'], **kw)

    logger.info(f'Loading lx and lz logical matrices.')
    if matrix_cfg.get('logical_check_matrix', False):
        lx_matrix = _load_one_matrix(matrix_cfg['logical_check_lx'], **kw).get_dense()
        lz_matrix = _load_one_matrix(matrix_cfg['logical_check_lz'], **kw).get_dense()
    else:
        lx_matrix = compute_lz(Hz_matrix.get_dense(), Hx_matrix.get_dense())
        lz_matrix = compute_lz(Hx_matrix.get_dense(), Hz_matrix.get_dense())

    return MatrixBundle(Hx_matrix, Hz_matrix, lx_matrix, lz_matrix)

