import scipy as sp
from loguru import logger

from syndrilla.utils import get_path
from syndrilla.matrix.matrix import dense_to_index_format


class create():
    def __init__(self,
                 matrix_cfg,
                 **kwargs) -> None:
        self.device = kwargs['device']
        assert 'path' in matrix_cfg.keys(), logger.error('Missing key <path> in the configuration.')
        self.path = get_path(matrix_cfg['path'])


    def get_index(self):
        logger.info(f'Getting matrix indices from <{self.path}>.')

        # load a sparse matrix from a npz file
        out = dense_to_index_format(sp.sparse.load_npz(self.path), self.device)

        logger.info('Complete.')
        return out


    def get_dense(self):
        return sp.sparse.load_npz(self.path).toarray() % 2

