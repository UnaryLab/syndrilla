import torch
from loguru import logger

from syndrilla.utils import build_rate_sweep, dataset, draw_shot_rate, is_rate_range, parse_device_dtype


class create:
    """
    This class creates a depolarizing error model.

    <rate> is a scalar for a decode run. A training run may give it as a
    [lower, upper, points] range instead, exactly as <bsc> does: every shot draws its own
    level and carries the matching Pauli priors, so one run covers a stretch of the
    curve. A range outside training is refused.
    """

    def __init__(self, error_model_cfg, **kwargs) -> None:
        assert "rate" in error_model_cfg.keys(), logger.error(
            "Missing key <rate> in the configuration."
        )
        self.rate = error_model_cfg["rate"]

        self.device, _ = parse_device_dtype(error_model_cfg)
        self.number_channel = 2

        self.rate_is_range = is_rate_range(self.rate)
        self.rates = (
            build_rate_sweep(
                self.rate, error_model_cfg, kwargs.get("training", False), "depol"
            )
            if self.rate_is_range
            else None
        )
        self.shot_rate = None

    def inject_error(self, codeword, batch_size: int = 0):
        logger.info("Injecting error.")

        codeword = codeword.to(self.device)
        if batch_size == 0:
            batch_size = codeword.size(0)
        # random values in [0,1)
        random_values = torch.rand_like(codeword)
        self.dtype = codeword.dtype
        self.len = codeword.shape

        # a scalar is used as-is, so a decode run is unchanged; a swept rate gives every
        # shot its own level, broadcast over that shot's qubits
        rate = self.rate
        if self.rate_is_range:
            self.shot_rate = draw_shot_rate(
                self.rates, codeword.size(0), 2, codeword.device, codeword.dtype
            )
            rate = self.shot_rate

        x_error = torch.where(random_values < 2 * rate / 3, 1 - codeword, codeword)
        y_error = torch.where(random_values < rate / 3, 1 - codeword, codeword)
        error = torch.stack([x_error, y_error], dim=1)
        dataloader = torch.utils.data.DataLoader(
            dataset(error, self.get_llr(error), torch.arange(0, codeword.size(0))),
            batch_size=batch_size,
            shuffle=False,
        )
        logger.info("Injection complete.")
        return error, dataloader

    def get_llr(self, error):
        # <shot_rate> is written by inject_error, this method's only caller in a run
        p = self.shot_rate if self.rate_is_range else self.rate
        # Probabilities for each Pauli event
        p_I = 1 - p
        p_X = p / 3
        p_Y = p / 3
        p_Z = p / 3

        # Stack probabilities per qubit
        if self.rate_is_range:
            # one row of priors per shot, since each drew its own rate
            probs = torch.cat([p_I, p_X, p_Y, p_Z], dim=1)
            llr = probs.unsqueeze(2).expand(-1, -1, error.shape[2]).contiguous()
        else:
            probs = torch.tensor(
                [p_I, p_X, p_Y, p_Z], device=self.device, dtype=self.dtype
            )
            llr = (
                probs.view(4, 1)
                .expand(4, error.shape[2])
                .unsqueeze(0)
                .repeat(error.shape[0], 1, 1)
            )
        return llr
