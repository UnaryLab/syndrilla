import torch

from loguru import logger


class create():
    """
    This class creates a syndrome.
    """
    def __init__(self, syndrome_cfg, **kwargs) -> None:
        self.syndrome_cfg = syndrome_cfg
        self.syndrome_actual = None


    def measure_syndrome(self, error, decoder):
        logger.info(f'Measuring syndrome.')
        # for dummy column cases
        if error.ndim == 2:
            dummy_column = torch.zeros([error.shape[0],1], dtype=error.dtype, device=error.device)
            error = torch.cat((error, dummy_column), dim=1)

            v_c_col = decoder.V_c_col.to(error.device)
            syndrome = error[:, v_c_col].sum(dim = 2)
            syndrome = torch.where((syndrome%2) > 0, 1, 0)
            logger.info(f'Syndrome measurement complete.')
        else:
            dummy_column = torch.zeros([error.shape[0], error.shape[1], 1], dtype=error.dtype, device=error.device)
            error = torch.cat((error, dummy_column), dim=2)
            
            v_c_col = decoder.V_c_col.to(error.device)
            v_c_col_expanded = v_c_col.unsqueeze(0).expand(error.size(0), -1, -1, -1)  # [10000, 2, 9, 4]
            error_expanded = error.unsqueeze(2).expand(-1, -1, v_c_col.size(1), -1)  # [10000, 2, 9, 19]
            syndrome = torch.gather(error_expanded, dim=3, index=v_c_col_expanded).sum(dim=3)
            syndrome = torch.where((syndrome%2) > 0, 1, 0)
            logger.info(f'Syndrome measurement complete.')
        self.syndrome_actual = syndrome
        return syndrome
    
