import torch

from loguru import logger

from syndrilla.error_model.bsc.bsc import create as bsc
from syndrilla.utils import dataset


class create(bsc):
    """
    This class creates a bsc error model whose flip rate is drawn per shot.

    Training a decoder against a single physical error rate gives a model that only
    holds at that rate. This model sweeps instead: <rate> is a [lower, upper] range,
    split into <rate_points> evenly spaced rates, and every shot in the batch draws
    its own, so one training run covers the whole curve.

    Everything else is inherited from <bsc>, which keeps its scalar <rate> and is
    left untouched, so an existing configuration keeps decoding exactly as before.
    """
    def __init__(self,
                 error_model_cfg,
                 **kwargs) -> None:
        super().__init__(error_model_cfg, **kwargs)

        if not isinstance(self.rate, (list, tuple)):
            raise ValueError(f'Error model <bsc_train> needs <rate> as a [lower, upper] range, got <{self.rate}>. Use model <bsc> for a scalar rate.')
        if len(self.rate) != 2:
            raise ValueError(f'A <rate> range needs exactly 2 values [lower, upper], got <{self.rate}>.')
        lower, upper = self.rate
        if lower > upper:
            raise ValueError(f'A <rate> range needs lower <= upper, got <{lower}> > <{upper}>.')
        if not 0 < lower or not upper < 1:
            raise ValueError(f'A <rate> range needs 0 < lower and upper < 1, got <{self.rate}>.')
        if self.number_channel != 1:
            raise ValueError(f'Error model <bsc_train> needs <number_channel> 1, got <{self.number_channel}>.')
        if 'rate_points' not in error_model_cfg.keys():
            raise ValueError(f'Missing key <rate_points> in the configuration.')
        rate_points = error_model_cfg['rate_points']
        if not isinstance(rate_points, int) or rate_points < 1:
            raise ValueError(f'Key <rate_points> needs a positive integer, got <{rate_points}>.')
        self.rates = torch.linspace(lower, upper, rate_points, device=self.device, dtype=torch.float64)
        # main.py refuses a decode run against a swept rate by looking for this flag,
        # since a result file records one physical error rate for the whole run
        self.rate_is_range = True


    def inject_error(self, codeword, batch_size:int=0):
        logger.info(f'Injecting error.')
        if self.rounds > 1:
            raise ValueError(f'Error model <bsc_train> needs <rounds> 1, got <{self.rounds}>.')
        codeword = codeword.to(self.device)
        if batch_size == 0:
            batch_size = codeword.size(0)
        self.dtype = codeword.dtype

        # one rate per shot, so every row flips against its own rate
        idx = torch.randint(self.rates.numel(), (codeword.size(0), 1), device=codeword.device)
        self.shot_rate = self.rates.to(dtype=codeword.dtype)[idx]
        random_values = torch.rand_like(codeword)
        error = torch.where(random_values < self.shot_rate, 1 - codeword, codeword)

        self.len = error.shape
        dataloader = torch.utils.data.DataLoader(dataset(error, self.get_llr(error), torch.arange(0, codeword.size(0))), batch_size=batch_size, shuffle=False)
        logger.info(f'Injection complete.')
        return error, dataloader


    def get_llr(self, error):
        # <shot_rate> is written by inject_error, this method's only caller
        return torch.log((1 - self.shot_rate)/self.shot_rate).expand(self.len).contiguous()
