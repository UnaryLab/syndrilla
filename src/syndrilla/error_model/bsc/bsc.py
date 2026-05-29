import torch, math

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
        # default to 1, and it will be set in main.py
        self.rounds = 1

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
        self.number_channel = error_model_cfg.get('number_channel', 1)


    def inject_error(self, codeword, batch_size:int=0):
        logger.info(f'Injecting error.')
        if self.number_channel == 1:
            codeword = codeword.to(self.device)
            if batch_size == 0:
                batch_size = codeword.size(0)
            self.dtype = codeword.dtype

            if self.rounds > 1:
                # [B, N] -> [B, rounds, N] with independent errors per round
                random_values = torch.rand(codeword.size(0), self.rounds, codeword.size(1),
                                           device=codeword.device, dtype=codeword.dtype)
                expanded = codeword.unsqueeze(1).expand(-1, self.rounds, -1)
                error = torch.where(random_values < self.rate, 1 - expanded, expanded)
            else:
                random_values = torch.rand_like(codeword)
                error = torch.where(random_values < self.rate, 1 - codeword, codeword)

            self.len = error.shape
            dataloader = torch.utils.data.DataLoader(dataset(error, self.get_llr(error), torch.arange(0, codeword.size(0))), batch_size=batch_size, shuffle=False)
            logger.info(f'Injection complete.')
        else:
            codeword = codeword.to(self.device)
            if batch_size == 0:
                batch_size = codeword.size(0)
            # random values in [0,1)
            random_values_x = torch.rand_like(codeword)
            random_values_z = torch.rand_like(codeword)
            self.dtype = codeword.dtype
            self.len = codeword.shape
            error_x = torch.where(random_values_x < self.rate, 1 - codeword, codeword)
            error_z = torch.where(random_values_z < self.rate, 1 - codeword, codeword)
            error = torch.stack((error_x, error_z), 1)
            dataloader = torch.utils.data.DataLoader(dataset(error, self.get_llr(error), torch.arange(0, codeword.size(0))), batch_size=batch_size, shuffle=False)
            logger.info(f'Injection complete.')
        return error, dataloader
    

    def get_llr(self, error):
        if self.number_channel == 1:
            llr =  torch.full(self.len, math.log((1 - self.rate)/self.rate), device=self.device, dtype=self.dtype)
        else: 
            p = self.rate
        # Probabilities for each Pauli event
            p_I = 1 - p * p - 2 * p
            p_X = p*(1-p)
            p_Y = p*p
            p_Z = p*(1-p)

            # Stack probabilities per qubit
            probs = torch.tensor([p_I, p_X, p_Y, p_Z], device=self.device, dtype=self.dtype)
            llr = probs.view(4, 1).expand(4, error.shape[2]).unsqueeze(0).repeat(error.shape[0], 1, 1)
        return llr
    
