import subprocess

import torch
import yaml

from syndrilla.decoder import create_decoder
from syndrilla.decoder.decoder import SHARED_KEYS
from syndrilla.matrix import load_matrices
from syndrilla.utils import get_path, parse_device_dtype, read_yaml

TRAINING_DECODER_YAML = "examples/alist/train_saq_hx.decoding.yaml"
SURFACE_MATRIX_YAML = "examples/alist/surface_5.matrix.yaml"
CKPT_STEM = "saq_hx_n41"
BEST_PT = f"{CKPT_STEM}_best.pt"
LAST_PT = f"{CKPT_STEM}_last.pt"
RESULT_YAML = f"{CKPT_STEM}_result.yaml"
TRAIN_ERROR_YAML = "examples/alist/bsc_train.error.yaml"
TRAIN_SYNDROME_YAML = "examples/alist/perfect.syndrome.yaml"
TRAINING_YAML = "examples/alist/train_saq_hx.training.yaml"


def _plain(value):
    """Plain dicts/lists, so `yaml.safe_dump` can write a config `read_yaml` returned."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _make_decoder(matrix_yaml, training=False, **overrides):
    """Build a saq decoder from the example yaml, shrunk so tests stay fast."""
    cfg = read_yaml(get_path(TRAINING_DECODER_YAML))["decoding"]
    algo_cfg = cfg["config"]
    algo_cfg["model"] = dict(algo_cfg["model"], d_model=32, N_dec=2, h=4)
    for key, value in overrides.items():
        target = cfg if key in SHARED_KEYS else algo_cfg
        if isinstance(value, dict) and key in target:
            target[key] = dict(target[key], **value)
        else:
            target[key] = value
    bundle = load_matrices(
        read_yaml(get_path(matrix_yaml))["matrix"], *parse_device_dtype(cfg)
    )
    wrapper = create_decoder(cfg=cfg, bundle=bundle, training=training)[0]
    return wrapper, wrapper.decoder


def _random_shots(saq, batch_size=8, error_rate=0.05, seed=0):
    """Draw random errors and their exact syndromes for the decoder's check matrix."""
    torch.manual_seed(seed)
    H = saq.H_matrix.to(torch.float32)
    e = (torch.rand(batch_size, saq.n, device=saq.device) < error_rate).to(saq.dtype)
    synd = (e.to(torch.float32) @ H.t()) % 2
    return e, synd.to(saq.dtype), H


def _io_dict(saq, synd, H):
    """The io_dict a syndrilla run hands the decoder."""
    return {
        "synd": synd,
        "llr0": torch.full(
            (synd.size(0), saq.n), 2.9, dtype=saq.dtype, device=saq.device
        ),
        "H_matrix": H,
    }


def _make_loss(saq, **overrides):
    """Build the saq loss bound to the decoder, from the shipped training yaml."""
    from syndrilla.trainer import create_trainer

    cfg = _plain(read_yaml(get_path(TRAINING_YAML))["training"])
    cfg["loss"] = dict(cfg["loss"], **overrides)
    return create_trainer(cfg=cfg, decoder=saq).loss


def _optimizer_cfg():
    """The shipped `training.optimizer` block, what a `-t` run configures Adam from."""
    return _plain(read_yaml(get_path(TRAINING_YAML))["training"]["optimizer"])


def _training_algorithm():
    """The shipped `training.algorithm`, the name a `-t` run dispatches its trainer on."""
    return read_yaml(get_path(TRAINING_YAML))["training"]["algorithm"]


def _trainer_for(decoder, epochs, **overrides):
    """The trainer a `-t` run would fit `decoder` with."""
    from syndrilla.trainer import Trainer

    trainer = Trainer(
        {
            "algorithm": _training_algorithm(),
            "optimizer": dict(_optimizer_cfg(), **overrides),
        },
        _make_loss(decoder),
    )
    trainer.configure(decoder, epochs, 2, 1, 0)
    return trainer


def _run_cli(run_dir, *extra):
    """Run the installed `syndrilla` command, the way a user drives a run."""
    cmd = ["syndrilla", f"-r={run_dir}", *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


def _train_argv(training_yaml, *extra, decoding_yaml=TRAINING_DECODER_YAML):
    """The argv of a `-t` run on the shipped yamls."""
    return [
        "-t",
        f"-d={decoding_yaml}",
        f"-m={SURFACE_MATRIX_YAML}",
        f"-e={TRAIN_ERROR_YAML}",
        f"-s={TRAIN_SYNDROME_YAML}",
        f"-tr={training_yaml}",
        "-bs=16",
        *extra,
    ]


def _write_training_yaml(path, **overrides):
    """A copy of the shipped training yaml with a shrunk budget, so tests stay fast."""
    cfg = _plain(read_yaml(get_path(TRAINING_YAML))["training"])
    cfg["budget"] = dict(cfg["budget"], **overrides)
    path.write_text(yaml.safe_dump({"training": cfg}))
    return path


def _write_decode_yaml(path, checkpoint):
    """The shipped training yaml pointed at a checkpoint, for a decode run."""
    cfg = read_yaml(get_path(TRAINING_DECODER_YAML))["decoding"]
    cfg["config"]["checkpoint"] = str(checkpoint)
    path.write_text(yaml.safe_dump(_plain({"decoding": cfg})))
    return path


def _training_setup(**overrides):
    """A decoder and its loss, both built from the shipped yamls."""
    decoder, saq = _make_decoder(SURFACE_MATRIX_YAML, **overrides)
    return decoder, saq, _make_loss(saq)


def _train_step(run, seed):
    """One batch -> forward -> loss -> backward -> update, on a seeded batch."""
    decoder, saq, loss, trainer = run
    e, synd, H = _random_shots(saq, batch_size=8, seed=seed)
    out = decoder(_io_dict(saq, synd, H))
    loss.combine(*loss.terms(out, e)).backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)


def _fresh_run(epochs, seed=0):
    """A decoder seeded identically to every other `_fresh_run`, ready to train."""
    torch.manual_seed(seed)
    decoder, saq, loss = _training_setup()
    trainer = _trainer_for(saq, epochs)
    saq.train(True)
    torch.set_grad_enabled(True)
    return decoder, saq, loss, trainer


def test_overfits_a_fixed_batch():
    """The decoder's own training stages must drive the loss down on a fixed batch."""
    decoder, saq = _make_decoder(SURFACE_MATRIX_YAML)
    saq.train()
    loss = _make_loss(saq)
    e, synd, H = _random_shots(saq)

    optimizer = _trainer_for(saq, 40).optimizer
    assert all(p.grad is None for p in saq.parameters())

    losses = []
    for _ in range(40):
        out = decoder(_io_dict(saq, synd, H))
        total = loss(out, e)
        total.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(total.item())

    assert losses[-1] < losses[0]


def test_train_cli_produces_a_loadable_checkpoint(tmp_path):
    """`-t` must train and write a checkpoint the decoder's `checkpoint` key can load."""
    training_yaml = _write_training_yaml(
        tmp_path / "small.training.yaml", epochs=2, test_batches=4, validation_batches=2
    )
    result = _run_cli(
        tmp_path,
        "-t",
        f"-d={TRAINING_DECODER_YAML}",
        f"-m={SURFACE_MATRIX_YAML}",
        f"-e={TRAIN_ERROR_YAML}",
        f"-s={TRAIN_SYNDROME_YAML}",
        f"-tr={training_yaml}",
        "-bs=32",
    )
    assert result.returncode == 0, result.stderr

    best = tmp_path / BEST_PT
    assert best.is_file() and (tmp_path / LAST_PT).is_file()
    assert (tmp_path / RESULT_YAML).is_file()

    epochs = yaml.safe_load((tmp_path / RESULT_YAML).read_text())["training result"]
    numbers = epochs["epoch"]
    assert numbers[-1] == 2
    rates = epochs["learning rate"]
    shipped_lr = read_yaml(get_path(TRAINING_YAML))["training"]["optimizer"]["lr"]
    if numbers[0] == 1:
        assert rates[0] == shipped_lr and rates[-1] < rates[0]
    else:
        assert rates[-1] < shipped_lr

    assert not list(tmp_path.glob("result_phy_err_*.yaml"))
    assert list(tmp_path.glob("main-*.log")), "training wrote no toolchain log"

    def from_yaml(**overrides):
        """The shipped architecture, with `config` overrides applied."""
        cfg = read_yaml(get_path(TRAINING_DECODER_YAML))["decoding"]
        cfg["config"].update(overrides)
        bundle = load_matrices(
            read_yaml(get_path(SURFACE_MATRIX_YAML))["matrix"], *parse_device_dtype(cfg)
        )
        wrapper = create_decoder(cfg=cfg, bundle=bundle)[0]
        wrapper.eval()
        return wrapper, wrapper.decoder

    untrained_wrapper, untrained = from_yaml()
    trained_wrapper, trained = from_yaml(checkpoint=str(best))
    _, synd, H = _random_shots(untrained, batch_size=8)
    assert not torch.allclose(
        trained_wrapper(_io_dict(trained, synd, H))["llr"],
        untrained_wrapper(_io_dict(untrained, synd, H))["llr"],
    )


def test_train_then_decode_cli(tmp_path, batch_size=200, target_error=20):
    """Train with `syndrilla -t`, then decode with `syndrilla`, both through the CLI."""
    train_dir = tmp_path / "train"
    training_yaml = _write_training_yaml(
        tmp_path / "small.training.yaml", epochs=2, test_batches=4, validation_batches=2
    )
    trained = _run_cli(train_dir, *_train_argv(training_yaml))
    assert trained.returncode == 0, trained.stderr

    best = train_dir / BEST_PT
    assert best.is_file(), "training produced no best.pt to decode with"

    decode_dir = tmp_path / "decode"
    decode_dir.mkdir()
    decoded = _run_cli(
        decode_dir,
        f"-d={_write_decode_yaml(tmp_path / 'trained.decoding.yaml', best)}",
        f"-m={SURFACE_MATRIX_YAML}",
        "-e=examples/alist/bsc.error.yaml",
        "-c=examples/alist/lx.check.yaml",
        "-s=examples/alist/perfect.syndrome.yaml",
        f"-bs={batch_size}",
        f"-te={target_error}",
    )
    assert decoded.returncode == 0, decoded.stderr

    written = list(decode_dir.glob("result_phy_err_*.yaml"))
    assert written, "no metric yaml produced"
    metrics = read_yaml(str(written[0]))
    assert metrics["decoder_0"]["algorithm"] == "saq"
    rate = metrics["decoder_0"]["hx"]["logical error rate"]
    assert 0.0 <= float(rate) <= 1.0


def test_resume_continues_optimizer_and_schedule(tmp_path):
    """Reloading training state must continue the run, not warm-start a new one."""
    straight = _fresh_run(4)
    straight_saq, straight_trainer = straight[1], straight[3]
    for epoch in range(4):
        _train_step(straight, seed=epoch)
        straight_trainer.scheduler.step()

    part = _fresh_run(4)
    part_trainer = part[3]
    for epoch in range(2):
        _train_step(part, seed=epoch)
        part_trainer.scheduler.step()
    path = tmp_path / "train_state.pt"
    torch.save(part_trainer.train_state(), path)

    resumed = _fresh_run(4)
    resumed_saq, resumed_trainer = resumed[1], resumed[3]
    resumed_trainer.load_train_state(
        torch.load(path, map_location="cpu", weights_only=True)
    )
    for epoch in range(2, 4):
        _train_step(resumed, seed=epoch)
        resumed_trainer.scheduler.step()

    assert (
        resumed_trainer.scheduler.get_last_lr()[0]
        == straight_trainer.scheduler.get_last_lr()[0]
    )
    reference = straight_saq.state_dict()
    for key, value in resumed_saq.state_dict().items():
        assert torch.equal(value, reference[key]), key
