import torch
from loguru import logger

from syndrilla.utils import build_rate_sweep, draw_shot_rate, is_rate_range


class create:
    def __init__(self, syndrome_cfg, **kwargs) -> None:
        self.rounds = int(syndrome_cfg.get("rounds", 1))
        rate_cfg = syndrome_cfg.get("measurement_error_rate", 0.0)
        self.rate_is_range = is_rate_range(rate_cfg)
        if self.rate_is_range:
            self.measurement_error_rate = list(rate_cfg)
            self.rates = build_rate_sweep(
                rate_cfg,
                syndrome_cfg,
                kwargs.get("training", False),
                "phenomenological",
                key="measurement_error_rate",
            )
        else:
            self.measurement_error_rate = float(rate_cfg)
            self.rates = None
        # the rate each shot of the last measured batch drew, read by adjust_llr0
        self.shot_rate = None
        self.observable_flips = None
        self.syndrome_actual = None

        logger.info(
            f"Phenomenological syndrome measurer ready: "
            f"{self.rounds} round(s), "
            f"measurement_error_rate={self.measurement_error_rate}."
        )

    def measure_syndrome(self, error, decoder):
        logger.info("Measuring syndrome.")

        if error.ndim == 2:
            dummy_column = torch.zeros(
                [error.shape[0], 1], dtype=error.dtype, device=error.device
            )
            error_ext = torch.cat((error, dummy_column), dim=1)

            v_c_col = decoder.V_c_col.to(error.device)
            syndrome = error_ext[:, v_c_col].sum(dim=2)
            syndrome = torch.where((syndrome % 2) > 0, 1, 0)

            self.syndrome_actual = syndrome
            logger.info("Syndrome measurement complete.")
            return self._apply_noise(syndrome)
        else:
            # [B, rounds, N] — error already carries per-round data
            dummy_column = torch.zeros(
                [error.shape[0], error.shape[1], 1],
                dtype=error.dtype,
                device=error.device,
            )
            error_ext = torch.cat((error, dummy_column), dim=2)

            v_c_col = decoder.V_c_col.to(error.device)
            syndrome = error_ext[:, :, v_c_col].sum(dim=3)
            syndrome = torch.where((syndrome % 2) > 0, 1, 0)

            self.syndrome_actual = syndrome
            logger.info("Syndrome measurement complete.")

            noisy = self._apply_noise(syndrome)
            logger.info(
                f"Phenomenological measurement complete: {self.rounds} rounds, output shape {list(noisy.shape)}."
            )
            return noisy

    def _apply_noise(self, syndrome):
        if self.rate_is_range:
            # one measurement error rate per shot, broadcast over that shot's bits
            self.shot_rate = draw_shot_rate(
                self.rates,
                syndrome.shape[0],
                syndrome.ndim,
                syndrome.device,
                torch.float32,
            )
            probs = self.shot_rate.expand(syndrome.shape)
        else:
            if self.measurement_error_rate <= 0:
                return syndrome
            probs = torch.full(
                syndrome.shape,
                self.measurement_error_rate,
                dtype=torch.float32,
                device=syndrome.device,
            )
        flip_mask = torch.bernoulli(probs).to(syndrome.dtype)
        return (syndrome + flip_mask) % 2

    def adjust_llr0(self, llr0):
        """Fold the measurement-error rate q into the per-data-qubit channel prior.

        A noisy syndrome bit looks like an apparent extra data flip, so the
        effective data-error probability is p_eff = p + q - 2*p*q.  The incoming
        prior encodes p (llr0 = log((1-p)/p)  =>  p = sigmoid(-llr0)); we inflate
        it to p_eff and rebuild the prior.  Elementwise, so it works for any llr0
        shape, and a zero rate leaves llr0 untouched.

        A swept rate folds in the rate each shot actually drew, taken from the batch
        this measurer just measured, so the prior matches that shot's own noise."""
        if self.rate_is_range:
            if self.shot_rate is None:
                # no batch measured yet, so there is no per-shot rate to fold in
                return llr0
            q = self.shot_rate.to(device=llr0.device, dtype=llr0.dtype)
        else:
            q = self.measurement_error_rate
            if q <= 0:
                return llr0
        p = torch.sigmoid(-llr0)
        p_eff = p + q - 2 * p * q
        return torch.log((1 - p_eff) / p_eff)
