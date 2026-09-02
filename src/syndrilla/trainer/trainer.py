import os

import torch
from loguru import logger

from syndrilla.utils import call_func_from_cfg, get_path, read_yaml

BLOCKS = ("loss", "optimizer", "schedule")

# what `Trainer.train_state` writes and `load_train_state` requires back
TRAIN_STATE_KEYS = ("state_dict", "optimizer", "scheduler")


def read_training_cfg(yaml_path: str):
    """Read a '.training.yaml' file's `training` block."""
    full_path = get_path(yaml_path)
    cfg = read_yaml(full_path)
    if "training" not in (cfg or {}):
        raise ValueError(
            f"Training yaml <{full_path}> has no `training` header. The objective, the "
            f"optimizer and the schedule all live under it; a file headed `loss:` "
            f"predates the move and carries only the first of the three."
        )
    return cfg["training"]


class Trainer:
    """The run that fits a decoder: the objective, the optimizer, the schedule."""

    def __init__(self, cfg: dict, loss) -> None:
        self.cfg = cfg
        self.loss = loss
        self.optimizer_cfg = cfg.get("optimizer") or {}
        self.schedule = cfg.get("schedule") or {}
        # all bound by `configure`, so a run that never trains carries none of them
        self.decoder = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.period = None
        self.test_batches = None
        self.seed = None
        # the phase the last opened batch is in; `begin_batch` is what decides it
        self.phase = "train"

    def configure(self, decoder, epochs, period, test_batches, seed):
        """Bind the decoder this run fits and build the optimizer and schedule for it."""
        self.decoder = decoder
        self.model = getattr(decoder, "decoder", decoder)
        self.optimizer, self.scheduler = self.loss.configure_optimizer(
            self.optimizer_cfg, self.model.parameters(), epochs
        )
        self.period = period
        self.test_batches = test_batches
        self.seed = seed

    def begin_batch(self, batch_index, epoch):
        """Open a batch: seed the phase it starts, and enter that phase."""
        position = batch_index % self.period
        if position == 0:
            # no epoch term: that is what makes each epoch train on the same batches
            torch.manual_seed(self.seed * 1_000_003)
        elif position == self.test_batches:
            # clear of the training seed so validation never replays its errors
            torch.manual_seed(self.seed * 1_000_003 + 9_999_991 + epoch)
        self.phase = "train" if position < self.test_batches else "val"
        if self.decoder is not None:
            # module mode and the grad switch are one decision: a validation batch that
            # still built a graph would train on the wrong set
            self.decoder.train(self.phase == "train")
            torch.set_grad_enabled(self.phase == "train")
        return self.phase

    def train_fingerprint(self):
        """The model half of a resume fingerprint; `MetricState` owns the run's half."""
        return {
            "algo": self.model.algo,
            "n": self.model.n,
            "m": self.model.m,
            "k": self.model.k,
            "dtype": str(self.model.dtype),
            "device": str(self.model.device),
        }

    def train_state(self):
        """Everything needed to resume this run, not just to decode from it."""
        rng = {"cpu": torch.get_rng_state()}
        if str(self.model.device).startswith("cuda"):
            rng["cuda"] = torch.cuda.get_rng_state_all()
        return {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "rng": rng,
        }

    def load_train_state(self, state):
        """Restore what `train_state` saved, onto this run and the model it fits."""
        missing = [key for key in TRAIN_STATE_KEYS if key not in state]
        if missing:
            raise ValueError(
                f"Training checkpoint is missing <{', '.join(missing)}>. It saved "
                f"weights only: it can be decoded from, but a run cannot be resumed."
            )
        self.model.load_state_dict(state["state_dict"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        rng = state.get("rng")
        if rng:
            # map_location puts every saved tensor on the model's device; both setters
            # want the state back on the CPU as a ByteTensor
            torch.set_rng_state(rng["cpu"].cpu())
            if "cuda" in rng and str(self.model.device).startswith("cuda"):
                torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])


def create_trainer(yaml_path: str = None, cfg: dict = None, **kwargs):
    """Create a trainer from a '.training.yaml' file or a config dict."""
    header = "training"
    if cfg is not None:
        logger.info("Creating trainer from config dict.")
        source = "dict"
    else:
        source = f"<{get_path(yaml_path)}>"
        logger.info(f"Creating trainer from {source}.")
        cfg = read_training_cfg(yaml_path)

    for block in BLOCKS:
        value = cfg.get(block)
        if value is not None and not isinstance(value, dict):
            raise ValueError(
                f"Training config {source} needs `{header}.{block}` to be a mapping, "
                f"got <{type(value).__name__}>."
            )
    loss_cfg = cfg.get("loss")
    if not loss_cfg:
        raise ValueError(
            f"Training config {source} has no `{header}.loss` block. Name the objective "
            f"there under `function`, with its own settings beside it."
        )

    loss = call_func_from_cfg(
        loss_cfg, header, "function", os.path.dirname(__file__), **kwargs
    )
    logger.info("Creating trainer complete.")
    return Trainer(cfg, loss)
