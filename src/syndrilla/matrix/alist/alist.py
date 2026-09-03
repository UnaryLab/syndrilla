import numpy as np
import torch
from loguru import logger

from syndrilla.utils import get_path


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
                matrix[j, r-1] = 1

        # ****************************************************************
        # don't change the code below
        # ****************************************************************
        shape = matrix.shape
        matrix = torch.tensor(matrix, device = self.device)

        degree = torch.max(torch.sum(matrix, 1)).int().item()

        row_indices, indices = torch.where(matrix == 1)

        V_c_row = torch.full([shape[0], degree], -1, dtype=torch.long, device=self.device)
        V_c_col = torch.full([shape[0], degree], -1, dtype=torch.long, device=self.device)

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

        logger.info('Complete.')
        return shape, V_c_row, V_c_col, matrix

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
                matrix[j, r-1] = 1

        return matrix

