import math

import torch
from loguru import logger

from syndrilla.utils import build_rate_sweep, dataset, draw_shot_rate, is_rate_range


class create:
    """
    This class creates a bsc error model.

    <rate> is a scalar for a decode run. A training run may give it as a
    [lower, upper, points] range instead: the range is split into that many evenly
    spaced levels and every shot in the batch draws its own, so one run covers a stretch
    of the curve rather than the single point a scalar pins. A range outside training is
    refused, since a result file records one physical error rate for the whole run.
    """

    def __init__(self, error_model_cfg, **kwargs) -> None:
        assert "rate" in error_model_cfg.keys(), logger.error(
            "Missing key <rate> in the configuration."
        )
        self.rate = error_model_cfg["rate"]
        # default to 1, and it will be set in main.py
        self.rounds = 1

        device_cfg = error_model_cfg.get("device", {})
        self.device = device_cfg.get(
            "device_type", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device not in {
            "cuda",
            "cpu",
            torch.device("cuda"),
            torch.device("cpu"),
        }:
            logger.warning(
                f"Invalid input device <{self.device}>, default to avaliable device in your machine."
            )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device == "cuda":
            device_idx = device_cfg.get("device_idx", 0)
            if device_idx >= torch.cuda.device_count():
                logger.warning(
                    f"Invalid input device index <{device_idx}>, default to avaliable device in your machine."
                )
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device(f"cuda:{device_idx}")
        self.number_channel = error_model_cfg.get("number_channel", 1)

        # `training` is a build-time mode, passed the way main.py passes it to
        # create_decoder, and it is what allows a swept <rate> at all
        self.rate_is_range = is_rate_range(self.rate)
        # the levels of a swept rate, and the level each shot of the last batch drew
        self.rates = (
            build_rate_sweep(
                self.rate, error_model_cfg, kwargs.get("training", False), "bsc"
            )
            if self.rate_is_range
            else None
        )
        self.shot_rate = None

    def _sample_rate(self, shots: int, ndim: int, device, dtype):
        """The flip probability to threshold against: one per shot when swept.

        A scalar is returned as-is, so a decode run draws exactly the random numbers it
        drew before. A range draws one level per shot, shaped to broadcast over the
        [shots, ...] tensor it is compared against.
        """
        if not self.rate_is_range:
            return self.rate
        self.shot_rate = draw_shot_rate(self.rates, shots, ndim, device, dtype)
        return self.shot_rate

    def inject_error(self, codeword, batch_size: int = 0):
        logger.info("Injecting error.")
        if self.number_channel == 1:
            codeword = codeword.to(self.device)
            if batch_size == 0:
                batch_size = codeword.size(0)
            self.dtype = codeword.dtype

            if self.rounds > 1:
                random_values = torch.rand(
                    codeword.size(0),
                    self.rounds,
                    codeword.size(1),
                    device=codeword.device,
                    dtype=codeword.dtype,
                )
                rate = self._sample_rate(
                    codeword.size(0), 3, codeword.device, codeword.dtype
                )
                flip = (random_values < rate).to(codeword.dtype)
                cumulative_flip = flip.cumsum(dim=1) % 2
                expanded = codeword.unsqueeze(1).expand(-1, self.rounds, -1)
                error = (expanded + cumulative_flip) % 2
            else:
                random_values = torch.rand_like(codeword)
                rate = self._sample_rate(
                    codeword.size(0), 2, codeword.device, codeword.dtype
                )
                error = torch.where(random_values < rate, 1 - codeword, codeword)

            self.len = error.shape
            dataloader = torch.utils.data.DataLoader(
                dataset(error, self.get_llr(error), torch.arange(0, codeword.size(0))),
                batch_size=batch_size,
                shuffle=False,
            )
            logger.info("Injection complete.")
        else:
            codeword = codeword.to(self.device)
            if batch_size == 0:
                batch_size = codeword.size(0)
            # random values in [0,1)
            random_values_x = torch.rand_like(codeword)
            random_values_z = torch.rand_like(codeword)
            self.dtype = codeword.dtype
            self.len = codeword.shape
            # both streams are drawn at the same level, so a swept shot keeps one rate
            rate = self._sample_rate(
                codeword.size(0), 2, codeword.device, codeword.dtype
            )
            error_x = torch.where(random_values_x < rate, 1 - codeword, codeword)
            error_z = torch.where(random_values_z < rate, 1 - codeword, codeword)
            error = torch.stack((error_x, error_z), 1)
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
        if self.number_channel == 1:
            if self.rate_is_range:
                llr = torch.log((1 - p) / p).expand(self.len).contiguous()
            else:
                llr = torch.full(
                    self.len,
                    math.log((1 - p) / p),
                    device=self.device,
                    dtype=self.dtype,
                )
        else:
            # Probabilities for each Pauli event
            p_I = 1 - p * p - 2 * p
            p_X = p * (1 - p)
            p_Y = p * p
            p_Z = p * (1 - p)

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
