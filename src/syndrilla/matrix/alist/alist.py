import numpy as np
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

        # load a matrix from a alist file
        with open(self.path, 'r') as f:
            line = f.readline().strip()
            while not line:
                line = f.readline().strip()
            m, n = map(int, line.split())
            col_neighbors = []
            for i in range(3):
                f.readline()

            for _ in range(m):
                neighbors = list(map(int, f.readline().split()))
                neighbors = [r for r in neighbors if r != 0]
                col_neighbors.append(neighbors)

        matrix = np.zeros((m, n), dtype=int)

        # Use the column neighbor lists to fill the matrix.
        for j, neighbors in enumerate(col_neighbors):
            for r in neighbors:
                matrix[j, r-1] += 1

        out = dense_to_index_format(matrix, self.device)

        logger.info('Complete.')
        return out

    def get_dense(self):
        with open(self.path, 'r') as f:
            line = f.readline().strip()
            while not line:
                line = f.readline().strip()
            m, n = map(int, line.split())
            col_neighbors = []
            for i in range(3):
                f.readline()

            for _ in range(m):
                neighbors = list(map(int, f.readline().split()))
                neighbors = [r for r in neighbors if r != 0]
                col_neighbors.append(neighbors)

        matrix = np.zeros((m, n), dtype=int)

        # Use the column neighbor lists to fill the matrix.
        for j, neighbors in enumerate(col_neighbors):
            for r in neighbors:
                matrix[j, r-1] += 1

        return matrix % 2

