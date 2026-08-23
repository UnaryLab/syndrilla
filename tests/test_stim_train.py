"""Training a learned decoder on the stim circuit-level path.

What makes the path trainable is that one draw produces all three things a training step
needs: an error over the columns of `H`, the detectors it sets off, and the observables
it flips. These tests pin that consistency, then drive the CLI end to end to check a run
actually learns from it rather than merely finishing.
"""

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import stim
import torch
import yaml

sys.path.append(os.getcwd())

import torch.nn.functional as F

from syndrilla.error_model import create_error_model
from syndrilla.interface import create_interface
from syndrilla.loss.logical_centric.logical_centric import (
    _diff_GF2_mul,
    _parity_llr,
    bin_to_sign,
)
from syndrilla.matrix.stim.stim import _build_dem_matrices


INTERFACE_YAML = "examples/stim/stim_generated.interface.yaml"
TRAIN_ERROR_YAML = "examples/stim/stim_train.error.yaml"
TRAIN_SYNDROME_YAML = "examples/stim/stim_train.syndrome.yaml"
TRAIN_DECODER_YAML = "examples/stim/stim_saq_train.decoder.yaml"
DECODE_ERROR_YAML = "examples/stim/stim_generated.error.yaml"
LOSS_YAML = "examples/stim/logical_centric.loss.yaml"

# The shipped interface is a distance-3 rotated surface code over 3 rounds. Its detector
# error model is what the decoder is built from, so checkpoints are named for it.
CKPT_STEM = "saq_hx_dem24x221"


def _interface(
    error_yaml=TRAIN_ERROR_YAML, syndrome_yaml=TRAIN_SYNDROME_YAML, training=True, **kw
):
    # a swept <rate> is the training-only form, so the mode has to be passed the way
    # main.py passes it or the error model refuses to build
    return create_interface(
        INTERFACE_YAML,
        error_yaml=error_yaml,
        syndrome_yaml=syndrome_yaml,
        training=training,
        **kw,
    )


def _draw(iface, shots=256):
    """One batch: the sampled error, its syndrome, and its observable flips."""
    error_model = iface.error_model
    zeros = torch.zeros([shots, error_model.num_errors], dtype=torch.float64)
    error, _ = error_model.inject_error(zeros, shots)
    syndrome = iface.syndrome_generator.measure_syndrome(error, None)
    return error, syndrome, iface.syndrome_generator.observable_flips


class TestDrawIsSelfConsistent:
    """The error, the syndrome and the observables all describe the same shot."""

    def test_syndrome_is_H_times_error(self):
        iface = _interface()
        error, syndrome, _ = _draw(iface)
        H = torch.as_tensor(
            iface.matrix_bundle.Hx_matrix.get_dense(), dtype=torch.float64
        )
        expected = (error.to(torch.float64) @ H.t()) % 2
        torch.testing.assert_close(syndrome.to(torch.float64), expected)

    def test_observables_are_L_times_error(self):
        iface = _interface()
        error, _, obs = _draw(iface)
        L = torch.as_tensor(
            np.asarray(iface.matrix_bundle.lx_matrix), dtype=torch.float64
        )
        expected = (error.to(torch.float64) @ L.t()) % 2
        torch.testing.assert_close(obs.to(torch.float64), expected)

    def test_error_is_not_the_old_dummy_zero(self):
        """The error used to be a placeholder of zeros, which supervises nothing."""
        _, _, obs = _draw(_interface())
        error, _, _ = _draw(_interface())
        assert error.sum() > 0
        # and the observable flips are not all one class, or the loss has no signal
        assert 0 < float(obs.to(torch.float64).mean()) < 1

    def test_remeasuring_a_stored_error_reproduces_its_syndrome(self):
        """A decode run's deferred queue re-measures errors it kept, so this has to hold."""
        iface = _interface()
        error, syndrome, obs = _draw(iface)
        again = iface.syndrome_generator.measure_syndrome(error, None)
        torch.testing.assert_close(syndrome, again)
        torch.testing.assert_close(obs, iface.syndrome_generator.observable_flips)


class TestSamplingMatchesStim:
    """Sampling with torch rather than stim's DEM sampler has to change nothing."""

    def test_detector_rates_match_stims_own_sampler(self):
        iface = _interface(error_yaml=DECODE_ERROR_YAML)
        torch.manual_seed(0)
        _, syndrome, _ = _draw(iface, shots=40000)
        ours = syndrome.to(torch.float64).mean(dim=0).numpy()

        circuit = iface.syndrome_generator.circuit
        det, _ = circuit.compile_detector_sampler(seed=7).sample(
            shots=40000, separate_observables=True
        )
        theirs = det.astype(np.float64).mean(axis=0)

        # per-detector firing rate, within sampling noise at 40k shots
        np.testing.assert_allclose(ours, theirs, atol=0.015)

    def test_torch_seed_drives_the_draw(self):
        """MetricState reseeds through torch, so a run's error stream has to follow it."""
        iface = _interface()
        torch.manual_seed(1234)
        first, _, _ = _draw(iface)
        torch.manual_seed(1234)
        second, _, _ = _draw(iface)
        torch.testing.assert_close(first, second)

        torch.manual_seed(4321)
        third, _, _ = _draw(iface)
        assert not torch.equal(first, third)


class TestRateSweep:
    """A swept rate is the training-only form, the same one `bsc` takes."""

    def test_sweep_shares_one_H_across_its_points(self):
        iface = _interface()
        error_model = iface.error_model
        assert error_model.rate_is_range
        table = error_model._prior_table
        assert table.shape == (9, error_model.num_errors)
        # each point is a different noise level over the same mechanism set
        assert table[0].mean() < table[-1].mean()

    def test_higher_rate_points_produce_heavier_errors(self):
        iface = _interface()
        error_model = iface.error_model
        torch.manual_seed(0)
        zeros = torch.zeros([20000, error_model.num_errors], dtype=torch.float64)
        error, _ = error_model.inject_error(zeros, 20000)
        weight = error.sum(dim=1)
        # shots drawing the low end of the sweep are lighter than the high end
        prior_mean = error_model._shot_prior.mean(dim=1)
        low = weight[prior_mean <= prior_mean.median()].mean()
        high = weight[prior_mean > prior_mean.median()].mean()
        assert low < high

    def test_range_without_generation_parameters_is_rejected(self):
        """An inline circuit cannot be regenerated at another noise level."""
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_x",
            distance=3,
            rounds=1,
            after_clifford_depolarization=0.003,
        )
        with pytest.raises(ValueError, match="circuit generation parameters"):
            create_error_model(
                cfg={
                    "model": "stim_circuit",
                    "circuit": str(circuit),
                    "rate": [0.001, 0.01],
                    "rate_points": 5,
                },
                training=True,
            )

    def test_decoding_a_swept_rate_is_rejected(self, tmp_path):
        """A result file records one physical error rate, so a decode run needs one."""
        # the shipped bp+osd config, so the run reaches the rate guard rather than
        # stopping earlier on a learned decoder's missing weights
        result = _run_cli(
            tmp_path / "run",
            "-d=examples/stim/stim_generated.decoder.yaml",
            f"-i={INTERFACE_YAML}",
            f"-e={TRAIN_ERROR_YAML}",
            f"-s={TRAIN_SYNDROME_YAML}",
            "-bs=8",
            "-te=1",
        )
        assert result.returncode != 0
        assert "training-only form" in result.stderr


class TestTrainCLI:
    """`syndrilla -t -i=...`, the path this change is for."""

    def test_training_writes_a_checkpoint_named_for_the_dem(self, tmp_path):
        run_dir = tmp_path / "run"
        result = _run_cli(
            run_dir,
            "-t",
            f"-d={_fast_decoder(tmp_path)}",
            f"-i={INTERFACE_YAML}",
            f"-e={TRAIN_ERROR_YAML}",
            f"-s={TRAIN_SYNDROME_YAML}",
            f"-ls={LOSS_YAML}",
            "-bs=256",
        )
        assert result.returncode == 0, result.stderr[-3000:]
        # named for the detector error model, since a DEM column is a fault mechanism
        # and carries no code distance to name it after
        assert (run_dir / f"{CKPT_STEM}.pt").exists()
        assert (run_dir / f"{CKPT_STEM}_last.pt").exists()

    def test_the_run_actually_learns(self, tmp_path):
        """A run that finishes but does not learn is the failure this path used to have.

        With no ground-truth error to supervise against, the logical class error sat at
        chance. It has to beat that now, and by more than the noise on a 2-class draw.
        """
        run_dir = tmp_path / "run"
        result = _run_cli(
            run_dir,
            "-t",
            f"-d={_fast_decoder(tmp_path)}",
            f"-i={INTERFACE_YAML}",
            f"-e={TRAIN_ERROR_YAML}",
            f"-s={TRAIN_SYNDROME_YAML}",
            f"-ls={LOSS_YAML}",
            "-bs=256",
        )
        assert result.returncode == 0, result.stderr[-3000:]
        history = json.loads((run_dir / f"{CKPT_STEM}_history.json").read_text())
        val_errors = [epoch["val"]["class_err"] for epoch in history]
        assert min(val_errors) < 0.25, f"no better than chance: {val_errors}"

    def test_training_needs_a_loss(self, tmp_path):
        """The interface supplies the data, but the objective is still the run's choice."""
        result = _run_cli(
            tmp_path / "run",
            "-t",
            f"-d={TRAIN_DECODER_YAML}",
            f"-i={INTERFACE_YAML}",
            f"-e={TRAIN_ERROR_YAML}",
            f"-s={TRAIN_SYNDROME_YAML}",
            "-bs=8",
        )
        assert result.returncode != 0
        assert "-ls" in result.stderr


class TestParityConditioning:
    """`L_Ent`'s parity, on the support sizes a detector error model produces.

    The probability-domain form multiplies one factor below 1 per bit in the logical
    observable's support. That is exact and, past a couple of dozen bits, worth nothing:
    the product and its gradient both underflow, the term reports a constant `ln 2`, and
    nothing supervises the per-bit llr. These pin the log-domain form against both.
    """

    def test_sign_is_exact(self):
        """No approximation touches the sign, so it must be the true GF(2) parity."""
        torch.manual_seed(0)
        H = (torch.rand(14, 3) < 0.5).double()
        bits = (torch.rand(4096, 14) < 0.5).double()
        parity = _parity_llr(H, bin_to_sign(bits) * 6.0)
        torch.testing.assert_close((parity < 0).double(), (bits @ H) % 2)

    def test_gradient_survives_a_dem_sized_support(self):
        """36 mechanisms, the shipped circuit's observable support, at the llr the old
        term drives the head to. The product form is not merely small there: in the
        float32 the learned decoders train in, it is exactly zero in value and gradient.

        Every tensor is built at an explicit dtype rather than the ambient default,
        because underflowing at all is the property under test and where it lands is
        precisely what the dtype decides.
        """
        n, support = 60, 36
        H = torch.zeros(n, 1, dtype=torch.float32)
        H[:support, 0] = 1
        llr = torch.full((8, n), 1e-3, dtype=torch.float32)
        e = torch.zeros(8, n, dtype=torch.float32)

        old_in = llr.clone().requires_grad_(True)
        p_no_err = torch.sigmoid(old_in)
        p_residual = e * p_no_err + (1 - e) * (1 - p_no_err)
        old = F.binary_cross_entropy(
            _diff_GF2_mul(H, p_residual), torch.zeros(8, 1, dtype=torch.float32)
        )
        old_grad = torch.autograd.grad(old, old_in)[0]
        assert float(old.detach()) == pytest.approx(math.log(2), abs=1e-6)
        assert float(old_grad.abs().max()) == 0.0

        new_in = llr.clone().requires_grad_(True)
        new = F.softplus(-_parity_llr(H, bin_to_sign(e) * new_in)).mean()
        new_grad = torch.autograd.grad(new, new_in)[0]
        assert float(new_grad.abs().max()) > 0.0
        # only bits in the support may be moved by a parity over that support
        assert float(new_grad[:, support:].abs().max()) == 0.0


def _fast_decoder(tmp_path):
    """The shipped training config, shrunk to a handful of epochs so tests stay quick."""
    cfg = yaml.safe_load(open(TRAIN_DECODER_YAML))
    cfg["decoder"]["config"]["train"] = {
        "epochs": 4,
        "batches_per_epoch": 25,
        "val_batches": 5,
        "seed": 42,
    }
    cfg["decoder"]["config"]["model"] = dict(
        cfg["decoder"]["config"]["model"], d_model=64, N_dec=2, h=4
    )
    path = tmp_path / "fast.decoder.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _run_cli(run_dir, *extra):
    """Run the installed `syndrilla` command, the way a user drives a run."""
    cmd = ["syndrilla", f"-r={run_dir}", *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
