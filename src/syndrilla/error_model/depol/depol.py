import torch

from loguru import logger

from syndrilla.utils import dataset


class create():
    """
    This class creates a bsc error model.
    """
    def __init__(self, 
                 error_model_cfg, 
                 **kwargs) -> None:
        assert 'rate' in error_model_cfg.keys(), logger.error(f'Missing key <rate> in the configuration.')
        self.rate = error_model_cfg['rate']

        device_cfg = error_model_cfg.get('device', {})
        self.device = device_cfg.get('device_type', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        if self.device not in {'cuda', 'cpu', torch.device('cuda'), torch.device('cpu')}:
            logger.warning(f'Invalid input device <{self.device}>, default to avaliable device in your machine.')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.device == 'cuda':
            device_idx = device_cfg.get('device_idx', 0)
            if device_idx >= torch.cuda.device_count():
                logger.warning(f'Invalid input device index <{device_idx}>, default to avaliable device in your machine.')
                self.device = torch.device(f'cuda:0')
            else:
                self.device = torch.device(f'cuda:{device_idx}')
        self.number_channel = 2


    def inject_error(self, codeword, batch_size:int=0):
        logger.info(f'Injecting error.')

        codeword = codeword.to(self.device)
        if batch_size == 0:
            batch_size = codeword.size(0)
        # random values in [0,1)
        random_values = torch.rand_like(codeword)
        self.dtype = codeword.dtype
        self.len = codeword.shape
        
        x_error = torch.where(random_values < 2*self.rate/3, 1 - codeword, codeword)
        y_error = torch.where(random_values < self.rate/3, 1 - codeword, codeword)
        error = torch.stack([x_error, y_error], dim=1)
        dataloader = torch.utils.data.DataLoader(dataset(error, self.get_llr(error), torch.arange(0, codeword.size(0))), batch_size=batch_size, shuffle=False)
        logger.info(f'Injection complete.')
        return error, dataloader
    

    def get_llr(self, error):
        p = self.rate
        # Probabilities for each Pauli event
        p_I = 1 - p
        p_X = p / 3
        p_Y = p / 3
        p_Z = p / 3

        # Stack probabilities per qubit
        probs = torch.tensor([p_I, p_X, p_Y, p_Z], device=self.device, dtype=self.dtype)
        llr = probs.view(4, 1).expand(4, error.shape[2]).unsqueeze(0).repeat(error.shape[0], 1, 1)
        # Optional: return log-probabilities for BP4 initialization
        # llr = -torch.log(llr + 1e-12)
        return llr
    
