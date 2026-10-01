import os

import numpy as np
import scipy.sparse as sp
import torch
from loguru import logger

from syndrilla.utils import (
    call_func_from_cfg,
    call_func_from_yaml,
    compute_lz,
    get_path,
)

# Shared circuit cache for the stim matrix loader. Lives in this module
# (instead of stim.py) because the factory dynamically re-imports loader
# modules per call, which would otherwise reset a per-loader cache.
STIM_CIRCUIT_CACHE: dict = {}


def dense_to_index_format(matrix, device):
    """
    Convert a GF(2) matrix, a dense array or a scipy sparse matrix, into the
    (shape, V_c_row, V_c_col, matrix_tensor) tuple every syndrilla decoder consumes
    from a matrix loader's get_index(); every loader returns this tuple unchanged.
    Entries are summed and reduced mod 2, so a (row, col) pair listed an even number
    of times cancels.

    Row r of V_c_col lists the columns of row r's nonzeros in ascending order, padded
    to the largest row degree with the dummy column shape[1]; every entry of row r of
    V_c_row is r. matrix_tensor is the matrix as a coalesced sparse COO tensor with
    bool values.
    """
    csr = sp.csr_matrix(matrix, dtype=np.int64, copy=True)
    csr.sum_duplicates()  # also sorts each row's column indices
    csr.data %= 2
    csr.eliminate_zeros()
    shape = csr.shape

    counts = torch.from_numpy(np.diff(csr.indptr)).long()
    degree = int(counts.max())
    rows = torch.repeat_interleave(torch.arange(shape[0]), counts)
    pos = torch.arange(csr.nnz) - torch.from_numpy(csr.indptr[:-1]).long()[rows]

    V_c_col = torch.full([shape[0], degree], shape[1], dtype=torch.long)
    V_c_col[rows, pos] = torch.from_numpy(csr.indices).long()
    V_c_row = torch.arange(shape[0]).unsqueeze(1).expand(shape[0], degree).contiguous()

    coo = csr.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    matrix = torch.sparse_coo_tensor(
        indices, torch.ones(coo.nnz, dtype=torch.bool), shape, device=device
    ).coalesce()

    return shape, V_c_row.to(device), V_c_col.to(device), matrix


def create_parity_matrix(yaml_path=None, cfg=None, **kwargs):
    """
    Create a matrix loader. Accepts either a yaml file path or a config dict.

        create_parity_matrix(yaml_path='check.matrix.yaml', device=...)
        create_parity_matrix(cfg={'file_type': 'stim', 'circuit': '...', 'target': 'check'}, device=...)
    """
    header = 'matrix'
    func_name = 'file_type'
    if cfg is not None:
        logger.info('Creating parity matrix class from config dict.')
        output = call_func_from_cfg(cfg, header, func_name, os.path.dirname(__file__), **kwargs)
    else:
        logger.info(f'Creating parity matrix class from <{get_path(yaml_path)}>.')
        output = call_func_from_yaml(yaml_path, header, func_name, os.path.dirname(__file__), **kwargs)
    logger.info('Complete.')
    return output


class MatrixBundle:
    """
    Container for the parity-check (Hx, Hz) and logical (lx, lz) matrices a
    decoder needs. Decoders should consume one of these instead of calling
    create_parity_matrix() and compute_lz() themselves.
    """
    __slots__ = ('Hx_matrix', 'Hz_matrix', 'lx_matrix', 'lz_matrix', 'sparse_h', '_index')

    def __init__(self, Hx_matrix, Hz_matrix, lx_matrix, lz_matrix, sparse_h=True):
        self.Hx_matrix = Hx_matrix
        self.Hz_matrix = Hz_matrix
        self.lx_matrix = lx_matrix
        self.lz_matrix = lz_matrix
        self.sparse_h = sparse_h
        self._index = {}

    def select(self, check_type):
        """
        Return (H_shape, V_c_row, V_c_col, H_matrix) for the requested check
        type. Used by single-channel decoders that pick one of Hx/Hz.
        The tuple is built on the first call per check type and the same objects
        are returned after that: H_matrix is the coalesced bool sparse COO tensor
        dense_to_index_format builds when sparse_h is true, and its dense int64
        form when sparse_h is false.
        """
        key = 'hx' if check_type.lower() == 'hx' else 'hz'
        if key not in self._index:
            m = self.Hx_matrix if key == 'hx' else self.Hz_matrix
            shape, V_c_row, V_c_col, H_matrix = m.get_index()
            if not self.sparse_h:
                H_matrix = H_matrix.long().to_dense()
            self._index[key] = (shape, V_c_row, V_c_col, H_matrix)
        return self._index[key]

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
      - a yaml file path (str): loaded via create_parity_matrix(yaml_path=...)
      - a config dict:          loaded via create_parity_matrix(cfg=...)

    Returns a MatrixBundle. If logical_check_matrix is False/missing, the
    logical matrices are computed via compute_lz from Hx/Hz. The top-level
    knob sparse_h (default true) picks the H that select() returns: sparse COO,
    or dense int64 when false.
    """
    from syndrilla.decoder.knobs import knob

    kw = {'device': device}
    if dtype is not None:
        kw['dtype'] = dtype

    logger.info('Loading hx parity check matrix.')
    Hx_matrix = _load_one_matrix(matrix_cfg['parity_matrix_hx'], **kw)

    logger.info('Loading hz parity check matrix.')
    Hz_matrix = _load_one_matrix(matrix_cfg['parity_matrix_hz'], **kw)

    logger.info('Loading lx and lz logical matrices.')
    if matrix_cfg.get('logical_check_matrix', False):
        lx_matrix = _load_one_matrix(matrix_cfg['logical_check_lx'], **kw).get_dense()
        lz_matrix = _load_one_matrix(matrix_cfg['logical_check_lz'], **kw).get_dense()
    else:
        lx_matrix = compute_lz(Hz_matrix.get_dense(), Hx_matrix.get_dense())
        lz_matrix = compute_lz(Hx_matrix.get_dense(), Hz_matrix.get_dense())

    return MatrixBundle(
        Hx_matrix, Hz_matrix, lx_matrix, lz_matrix, knob(matrix_cfg, 'sparse_h', True)
    )

