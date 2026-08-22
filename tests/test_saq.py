import sys, os
import subprocess, json

import pytest
import torch
from loguru import logger
import yaml

sys.path.append(os.getcwd())

from syndrilla.decoder import create_decoder
from syndrilla.matrix import load_matrices
from syndrilla.loss import create_loss
from syndrilla.utils import read_yaml, get_path, parse_device_dtype
from syndrilla.decoder.decoder import SHARED_KEYS


DECODER_YAML = "examples/alist/saq_hx_train.decoder.yaml"
DECODE_YAML = "examples/alist/saq_hx.decoder.yaml"
SURFACE_MATRIX_YAML = "examples/alist/surface_5.matrix.yaml"
TORIC_MATRIX_YAML = "examples/alist/toric_10.matrix.yaml"

# Checkpoints are named after the configuration that produced them. The CLI tests train
# the shipped hx config on surface_5, whose 41 qubits solve the unrotated surface
# relation `d^2 + (d-1)^2` at distance 5.
CKPT_STEM = "saq_hx_d5"
BEST_PT = f"{CKPT_STEM}.pt"
LAST_PT = f"{CKPT_STEM}_last.pt"
TRAIN_ERROR_YAML = "examples/alist/bsc_train.error.yaml"
TRAIN_SYNDROME_YAML = "examples/alist/perfect.syndrome.yaml"
LOSS_YAML = "examples/alist/logical_centric.loss.yaml"


def _make_decoder(matrix_yaml, training=False, **overrides):
    """Build a saq decoder from the example yaml, shrunk so tests stay fast.

    `training` is a build-time *mode*, not a config key: it is passed to
    `create_decoder` the way `main.py` passes it, so what the decoder does with it is
    under test rather than assumed.
    """
    cfg = read_yaml(get_path(DECODER_YAML))["decoder"]
    # saq's own settings are the `config` entry for its position in `algorithm`
    algo_cfg = cfg["config"]
    algo_cfg["model"] = dict(algo_cfg["model"], d_model=32, N_dec=2, h=4)
    for key, value in overrides.items():
        # framework-wide keys stay at the top of the block, saq's own go under `config`
        target = cfg if key in SHARED_KEYS else algo_cfg
        # a block override merges into the shipped block rather than replacing it, so a
        # test naming one key does not silently drop the rest
        if isinstance(value, dict) and key in target:
            target[key] = dict(target[key], **value)
        else:
            target[key] = value
    bundle = load_matrices(
        read_yaml(get_path(matrix_yaml))["matrix"], *parse_device_dtype(cfg)
    )
    # create_decoder returns the RoundFlattenWrapper; .decoder is the saq module itself
    wrapper = create_decoder(cfg=cfg, bundle=bundle, training=training)[0]
    return wrapper, wrapper.decoder


def _saq_module(saq):
    """The module object the decoder was built from.

    `create_decoder` loads `saq/saq.py` by file location under a synthetic module name, so
    importing it normally would give a *second*, unrelated module object. Going through the
    instance guarantees the module-level CPND helpers under test are the ones that actually
    ran.
    """
    return sys.modules[type(saq).__module__]


def _random_shots(saq, batch_size=8, error_rate=0.05, seed=0):
    """Draw random errors and their exact syndromes for the decoder's check matrix.

    Drawn on the decoder's own device, not the CPU: the shipped yaml asks for cuda, so a
    CPU batch would only ever meet the matrices on a machine without a GPU.
    """
    torch.manual_seed(seed)
    H = saq.H_matrix.to(torch.float32)
    e = (torch.rand(batch_size, saq.n, device=saq.device) < error_rate).to(saq.dtype)
    synd = (e.to(torch.float32) @ H.t()) % 2
    return e, synd.to(saq.dtype), H


def _io_dict(saq, synd, H):
    return {
        "synd": synd,
        "llr0": torch.full(
            (synd.size(0), saq.n), 2.9, dtype=saq.dtype, device=saq.device
        ),
        "H_matrix": H,
    }


@pytest.mark.parametrize(
    "matrix_yaml, m, n, k",
    [
        (SURFACE_MATRIX_YAML, 20, 41, 1),
        (TORIC_MATRIX_YAML, 100, 200, 2),
    ],
)
def test_shapes_and_contract(matrix_yaml, m, n, k):
    """forward() must fill the io_dict keys every syndrilla decoder promises."""
    decoder, saq = _make_decoder(matrix_yaml)
    assert (saq.m, saq.n, saq.k) == (m, n, k)
    assert saq.logical_classes == 2**k
    assert decoder.algo == "saq"
    assert decoder.num_max_iter == 1

    decoder.eval()
    e, synd, H = _random_shots(saq)
    out = decoder(_io_dict(saq, synd, H))

    assert out["e_v"].shape == (e.size(0), n)
    assert out["llr"].shape == (e.size(0), n)
    assert out["iter"].shape == (e.size(0),)
    assert out["converge"].shape == (e.size(0),)
    assert out["logical_logits"].shape == (e.size(0), 2**k)
    assert set(out["e_v"].unique().tolist()) <= {0.0, 1.0}
    assert torch.all(out["iter"] == 1)


def test_converge_flag_matches_syndrome():
    """converge must be exactly 'the estimate reproduces the measured syndrome'."""
    decoder, saq = _make_decoder(SURFACE_MATRIX_YAML)
    decoder.eval()
    _, synd, H = _random_shots(saq, batch_size=16)
    out = decoder(_io_dict(saq, synd, H))

    s_est = (out["e_v"].to(torch.float32) @ H.t()) % 2
    expected = torch.all(s_est == synd, dim=1).long()
    assert torch.equal(out["converge"], expected)


def test_hard_decision_and_syndrome_estimation():
    """The two shared stage helpers must agree with their closed forms."""
    decoder, saq = _make_decoder(SURFACE_MATRIX_YAML)
    e, _, H = _random_shots(saq, batch_size=8, error_rate=0.2)

    # syndrome_estimation is H @ e over GF(2), including for ragged check degrees
    assert torch.equal(
        saq.syndrome_estimation(e), ((e.to(torch.float32) @ H.t()) % 2).to(saq.dtype)
    )

    llr = torch.tensor([[-1.0, 0.0, 1.0, -0.5]], dtype=saq.dtype, device=saq.device)
    assert torch.equal(
        saq.hard_decision(llr),
        torch.tensor([[1.0, 1.0, 0.0, 1.0]], dtype=saq.dtype, device=saq.device),
    )


def test_stage_methods_compose_into_forward():
    """The named stages must reproduce forward() exactly when run by hand."""
    decoder, saq = _make_decoder(SURFACE_MATRIX_YAML)
    decoder.eval()
    _, synd, H = _random_shots(saq)
    out = decoder(_io_dict(saq, synd, H))

    with torch.no_grad():
        syndrome_pm = 1 - 2 * synd
        out_LP = saq.logical_prior(syndrome_pm)
        SN, LN = saq.build_streams(syndrome_pm, out_LP)
        assert SN.shape == (synd.size(0), saq.m + 1, 32)  # +1 for the global token
        assert LN.shape == (synd.size(0), saq.logical_classes, 32)
        for idx in range(saq.N_dec):
            SN = saq.sn_update(SN, idx)
            LN = saq.ln_update(LN, SN, idx)
            if saq.N_dec > 1 and idx == saq.N_dec // 2:
                SN, LN = saq.SN_norm2(SN), saq.LN_norm2(LN)
        l_v, out_L = saq.head_update(SN, LN)

    assert torch.allclose(l_v, out["llr"], atol=1e-6)
    assert torch.allclose(out_L, out["logical_logits"], atol=1e-6)


def test_attention_mask_is_the_code_topology():
    """M_S must allow exactly the stabilizer pairs sharing a qubit, plus the global token."""
    _, saq = _make_decoder(SURFACE_MATRIX_YAML)
    allowed = ~saq.src_mask_SN[0, 0]

    H = saq.H_matrix.to(torch.float32)
    neighbours = (H @ H.t()) > 0
    neighbours.fill_diagonal_(True)

    assert torch.equal(allowed[1:, 1:], neighbours)
    assert torch.all(allowed[0, :]) and torch.all(
        allowed[:, 0]
    )  # global token is dense
    assert not torch.all(allowed)  # the mask actually masks


def test_no_mask_ablation():
    _, saq = _make_decoder(SURFACE_MATRIX_YAML, model={"no_mask": 1})
    assert saq.src_mask_SN is None and saq.src_mask_LN is None


def test_loss_terms_and_gradient_flow():
    """All three logical-loss terms must be finite and reach every parameter."""
    decoder, saq = _make_decoder(SURFACE_MATRIX_YAML)
    saq.train()
    e, synd, H = _random_shots(saq)
    out = decoder(_io_dict(saq, synd, H))

    loss_fn = _make_loss(saq)
    loss_lc, loss_lp, loss_ent = loss_fn.terms(out, e)
    for term in (loss_lc, loss_lp, loss_ent):
        assert torch.isfinite(term) and term.item() >= 0.0

    total = loss_fn.combine(loss_lc, loss_lp, loss_ent)
    expected = (
        loss_fn.lambda_lc * loss_lc
        + loss_fn.lambda_lp * loss_lp
        + loss_fn.lambda_ent * loss_ent
    )
    assert torch.allclose(total, expected)

    total.backward()
    assert all(p.grad is not None for p in saq.parameters() if p.requires_grad)


def _logical_operator(saq, bundle_lz):
    """A vector in ker(H) with odd overlap with some row of the logical matrix.

    Adding it to an error keeps the syndrome identical but flips the logical class, which is
    exactly the failure a decoder must avoid, so it is the right probe for loss polarity.
    """
    H = saq.H_matrix.to(torch.float32)
    lx = saq.logic_matrix.to(torch.float32).t()
    # the bundle hands back a numpy array, so the rows land on the CPU whatever the
    # decoder's device is
    for candidate in torch.as_tensor(bundle_lz).to(torch.float32).to(saq.device):
        if torch.count_nonzero((H @ candidate) % 2) == 0 and torch.any(
            (lx @ candidate) % 2 > 0
        ):
            return candidate
    raise AssertionError("no logical operator found")


@pytest.mark.parametrize(
    "matrix_yaml",
    [SURFACE_MATRIX_YAML, TORIC_MATRIX_YAML],
)
def test_entropy_loss_polarity(matrix_yaml):
    """L_Ent must punish a logical error and reward a correct prediction.

    Upstream feeds `P(residual == 0)` into a term expecting `P(bit == 1)`. Because
    `XOR_i not(r_i) == (XOR_i r_i) XOR (w mod 2)`, that is harmless for even-weight logical
    operators but inverts the term for odd-weight ones (e.g. rotated surface d=5, weight 5),
    where minimising it maximises the logical error rate.
    """
    cfg = read_yaml(get_path(DECODER_YAML))["decoder"]
    cfg["config"]["model"] = dict(cfg["config"]["model"], d_model=32, N_dec=2, h=4)
    bundle = load_matrices(
        read_yaml(get_path(matrix_yaml))["matrix"], *parse_device_dtype(cfg)
    )
    saq = create_decoder(cfg=cfg, bundle=bundle)[0].decoder

    v = _logical_operator(saq, bundle.lz_matrix)
    torch.manual_seed(0)
    e = (torch.rand(16, saq.n, device=saq.device) < 0.06).to(saq.dtype)
    e_wrong = (e + v) % 2

    def confident_llr(pred):
        """LLR decoding exactly to `pred`; positive means no error."""
        return torch.where(pred > 0.5, -8.0, 8.0).to(saq.dtype)

    zeros = torch.zeros(
        e.size(0), saq.logical_classes, dtype=saq.dtype, device=saq.device
    )
    loss_fn = _make_loss(saq)

    def ent(pred):
        io = {
            "llr": confident_llr(pred),
            "logical_logits": zeros,
            "logical_prior": zeros,
        }
        return loss_fn.terms(io, e)[2]

    ent_right = ent(e)
    ent_wrong = ent(e_wrong)

    assert ent_right < ent_wrong, (
        f"L_Ent rewards the logical error: {ent_right.item():.5f} for a perfect "
        f"prediction vs {ent_wrong.item():.5f} for one off by a logical operator"
    )


def test_overfits_a_fixed_batch():
    """The decoder's own training stages must drive the loss down on a fixed batch."""
    decoder, saq = _make_decoder(SURFACE_MATRIX_YAML)
    saq.train()
    loss_fn = _make_loss(saq)
    e, synd, H = _random_shots(saq)

    saq.configure_optimizer(epochs=40)
    # configure_optimizer must leave the gradients clean for the first backward
    assert all(p.grad is None for p in saq.parameters())

    losses = []
    for _ in range(40):
        out = decoder(_io_dict(saq, synd, H))
        loss = loss_fn(out, e)
        saq.backward(loss)
        saq.update()
        losses.append(loss.item())

    assert losses[-1] < losses[0]


# ── CPND (stage 3) ───────────────────────────────────────────────────
CPND_CODES = [
    (SURFACE_MATRIX_YAML, 0),
    (TORIC_MATRIX_YAML, 1),
]


@pytest.mark.parametrize("matrix_yaml, dependent_rows", CPND_CODES)
def test_cpnd_algebra(matrix_yaml, dependent_rows):
    """The precomputed right inverse, kernel basis, and dropped rows must be exact."""
    _, saq = _make_decoder(matrix_yaml)
    H_hat, B = saq.cpnd_H_hat, saq.cpnd_B
    rank = H_hat.shape[0]

    assert torch.equal(
        (H_hat @ B) % 2, torch.eye(rank, dtype=saq.dtype, device=saq.device)
    )
    assert len(saq.cpnd_rows) == saq.m - dependent_rows

    # every stabilizer move must lie in ker([H; L]), and the basis must be complete
    basis = torch.stack(
        [
            torch.zeros(saq.n, dtype=saq.dtype, device=saq.device).index_fill_(
                0, c, 1.0
            )
            for c in saq.cpnd_supports
        ],
        dim=1,
    )
    assert torch.count_nonzero((H_hat @ basis) % 2) == 0
    assert basis.shape[1] == saq.n - rank


@pytest.mark.parametrize("matrix_yaml, _dependent", CPND_CODES)
def test_cpnd_enforces_both_constraints(matrix_yaml, _dependent):
    """With CPND on, every output must satisfy the full syndrome and the predicted class."""
    decoder, saq = _make_decoder(matrix_yaml)
    decoder.eval()
    _, synd, H = _random_shots(saq, batch_size=32, error_rate=0.06)
    out = decoder(_io_dict(saq, synd, H))
    e_v = out["e_v"]

    # the FULL H, including the dependent row CPND drops from the projection
    assert torch.equal((e_v.to(torch.float32) @ H.t()) % 2, synd.to(torch.float32))
    assert torch.all(out["converge"] == 1)

    predicted = (
        _saq_module(saq)
        ._logits_to_logical_bits(out["logical_logits"], saq.k)
        .to(torch.float32)
    )
    assert torch.equal(
        (e_v.to(torch.float32) @ saq.logic_matrix.to(torch.float32)) % 2, predicted
    )


def test_cpnd_descent_lowers_the_weighted_cost():
    """The descent must be monotone in sum(e_i * llr_i) and never leave the coset."""
    decoder, saq = _make_decoder(SURFACE_MATRIX_YAML)
    decoder.eval()
    _, synd, H = _random_shots(saq, batch_size=32)
    out = decoder(_io_dict(saq, synd, H))
    l_v = out["llr"]

    e0 = saq.project(saq.hard_decision(l_v), synd, out["logical_logits"])
    cost_projected = (e0 * l_v).sum(dim=1)
    cost_final = (out["e_v"] * l_v).sum(dim=1)

    assert torch.all(cost_final <= cost_projected + 1e-4)
    assert torch.any(cost_final < cost_projected - 1e-6)

    # extra sweeps keep improving and keep the constraints
    more = (
        _saq_module(saq)
        ._cpnd_descent(e0, saq.cpnd_supports, l_v, passes=5)
        .to(torch.float32)
    )
    assert torch.all((more * l_v).sum(dim=1) <= cost_final + 1e-4)
    assert torch.equal((more @ saq.cpnd_H_hat.t()) % 2, (e0 @ saq.cpnd_H_hat.t()) % 2)


def test_cpnd_sign_vector_stays_consistent():
    """Regression for upstream's no-op sign update: signs must track e after every flip.

    Upstream writes `sign[mask][:, v] *= -1`, where `sign[mask]` is a copy under advanced
    indexing, so the update never lands and every delta after the first accepted flip is
    computed against stale signs.
    """
    decoder, saq = _make_decoder(SURFACE_MATRIX_YAML)
    decoder.eval()
    _, synd, H = _random_shots(saq, batch_size=32)
    out = decoder(_io_dict(saq, synd, H))
    l_v = out["llr"]
    e0 = saq.project(saq.hard_decision(l_v), synd, out["logical_logits"])

    # replay the descent, tracking sign alongside, and require the invariant to hold
    e = e0.to(torch.bool)
    one = torch.ones((), device=saq.device)
    sign = torch.where(e, -one, one)
    flips = 0
    for cols in saq.cpnd_supports:
        rows = ((sign[:, cols] * l_v[:, cols]).sum(dim=1) < 0).nonzero(as_tuple=True)[0]
        if rows.numel():
            flips += int(rows.numel())
            idx = rows.unsqueeze(1)
            e[idx, cols] = ~e[idx, cols]
            sign[idx, cols] = -sign[idx, cols]

    assert flips > 0, "no flips accepted, the invariant would hold vacuously"
    assert torch.equal(sign, torch.where(e, -one, one))
    assert torch.equal(e.to(saq.dtype), out["e_v"])

    # guard the claim: the upstream expression really does write to a copy
    probe = torch.zeros(4, 8)
    probe[torch.tensor([True, False, True, False])][:, torch.tensor([True] * 8)] += 1.0
    assert torch.count_nonzero(probe) == 0


def test_cpnd_weights_use_signed_llr_magnitudes():
    """Regression for upstream's weights: magnitudes must survive and the sign must be right.

    Upstream derives the weights from an already-binarised output, which collapses them to
    {0, -1} and inverts the sign relative to a weight-minimising cost.
    """
    decoder, saq = _make_decoder(SURFACE_MATRIX_YAML)
    decoder.eval()
    _, synd, H = _random_shots(saq, batch_size=32)
    out = decoder(_io_dict(saq, synd, H))
    l_v = out["llr"]
    e_raw = saq.hard_decision(l_v)
    e0 = saq.project(e_raw, synd, out["logical_logits"])

    # what upstream actually computes, from the hard decision rather than the logits
    p = torch.sigmoid(e_raw)
    upstream = -torch.log(p / (1 - p))
    # the claim is that the weights collapse onto two values, not that float32 round-trips
    # sigmoid exactly -- CUDA returns -0.9999998 where the CPU returns -1.0
    collapsed = torch.unique(upstream)
    assert torch.allclose(collapsed, collapsed.round(), atol=1e-6)
    assert set(collapsed.round().tolist()) <= {0.0, -1.0}
    assert (
        l_v.abs().max() > 1.0
    ), "the real LLRs carry magnitude the collapse would lose"

    # descending on the inverted sign must do worse under the true cost
    cost = lambda e: (e * l_v).sum(dim=1).mean()
    fixed = _saq_module(saq)._cpnd_descent(e0, saq.cpnd_supports, l_v).to(torch.float32)
    inverted = (
        _saq_module(saq)._cpnd_descent(e0, saq.cpnd_supports, -l_v).to(torch.float32)
    )
    assert cost(fixed) < cost(inverted)


def test_cpnd_can_be_disabled():
    """`cpnd: false` must skip the stage and its precompute entirely."""
    decoder, saq = _make_decoder(SURFACE_MATRIX_YAML, cpnd={"enable": False})
    decoder.eval()
    assert not saq.use_cpnd
    assert not hasattr(saq, "cpnd_supports")

    _, synd, H = _random_shots(saq, batch_size=16)
    out = decoder(_io_dict(saq, synd, H))
    # the raw hard decision, so it is the pre-CPND behaviour: e_v follows llr's sign
    assert torch.equal(out["e_v"], (out["llr"] <= 0).to(saq.dtype))


def test_checkpoint_round_trip(tmp_path):
    """A saved state_dict must reload into a fresh decoder and reproduce its output."""
    decoder, saq = _make_decoder(SURFACE_MATRIX_YAML)
    decoder.eval()
    _, synd, H = _random_shots(saq)
    reference = decoder(_io_dict(saq, synd, H))["llr"].detach().clone()

    path = tmp_path / "saq.pt"
    torch.save(saq.state_dict(), path)

    reloaded, saq2 = _make_decoder(SURFACE_MATRIX_YAML, checkpoint=str(path))
    reloaded.eval()
    assert torch.allclose(
        reloaded(_io_dict(saq2, synd, H))["llr"], reference, atol=1e-6
    )


def test_end_to_end_cli(tmp_path, batch_size=200, target_error=20):
    """The decoder must drive a full syndrilla run and report saq in the metrics.

    Writes its results under tmp_path rather than tests/test_outputs, so running the suite
    does not overwrite the committed result yamls there.
    """
    result = _run_cli(
        tmp_path,
        f"-d={_write_decoder_yaml(tmp_path / 'cpu.decoder.yaml')}",
        f"-m={SURFACE_MATRIX_YAML}",
        "-e=examples/alist/bsc.error.yaml",
        "-c=examples/alist/lx.check.yaml",
        "-s=examples/alist/perfect.syndrome.yaml",
        f"-bs={batch_size}",
        f"-te={target_error}",
    )
    assert result.returncode == 0, result.stderr

    written = list(tmp_path.glob("result_phy_err_*.yaml"))
    assert written, "no metric yaml produced"
    metrics = read_yaml(str(written[0]))
    assert metrics["decoder_0"]["algorithm"] == "saq"
    assert metrics["decoder_0"]["average iteration"] == 1.0


def _run_cli(run_dir, *extra):
    """Run the installed `syndrilla` command, the way a user drives a run.

    The console script, not `python -m syndrilla.main` on a hand-set PYTHONPATH: what
    these tests are checking is the entry point users type, so an installed package
    whose entry point is broken has to fail here rather than be routed around.
    """
    cmd = ["syndrilla", f"-r={run_dir}", *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


def _plain(value):
    """Plain dicts/lists, so `yaml.safe_dump` can write a config `read_yaml` returned."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _write_decoder_yaml(path, source=None, **overrides):
    """A copy of the shipped decoder yaml with a shrunk schedule, so tests stay fast.

    The schedule lives under the decoder yaml's `train` key, so a run has one decoder
    yaml rather than a decoder yaml plus a training yaml to keep in step with it.
    """
    base = read_yaml(get_path(DECODER_YAML))["decoder"]
    cfg = read_yaml(get_path(source))["decoder"] if source else base
    cfg.setdefault("config", {})["train"] = dict(base["config"]["train"], **overrides)
    # read_yaml hands back OrderedDicts, which safe_dump cannot represent
    path.write_text(yaml.safe_dump(_plain({"decoder": cfg})))
    return path


def test_train_cli_produces_a_loadable_checkpoint(tmp_path):
    """`-t` must train and write a checkpoint the decoder's `checkpoint` key can load."""
    decoder_yaml = _write_decoder_yaml(
        tmp_path / "small.decoder.yaml", epochs=2, batches_per_epoch=4, val_batches=2
    )
    result = _run_cli(
        tmp_path,
        "-t",
        f"-d={decoder_yaml}",
        f"-m={SURFACE_MATRIX_YAML}",
        f"-e={TRAIN_ERROR_YAML}",
        f"-s={TRAIN_SYNDROME_YAML}",
        f"-ls={LOSS_YAML}",
        "-bs=32",
    )
    assert result.returncode == 0, result.stderr

    best = tmp_path / BEST_PT
    assert best.is_file() and (tmp_path / LAST_PT).is_file()
    assert (tmp_path / "history.json").is_file()

    # the schedule and the optimizer settings come from their own blocks of the same
    # decoder yaml
    history = json.loads((tmp_path / "history.json").read_text())
    assert [entry["epoch"] for entry in history] == [1, 2]
    shipped = read_yaml(get_path(DECODER_YAML))["decoder"]
    assert history[0]["lr"] == shipped["config"]["optimizer"]["lr"]
    assert history[1]["lr"] < history[0]["lr"]

    # training must not emit any decode-path artefact
    assert not list(tmp_path.glob("result_phy_err_*.yaml"))
    assert not list(tmp_path.glob("main-*.log"))

    # The checkpoint must load and actually change the decoder's output. Build from the
    # unmodified yaml: the CLI trained that architecture, not the shrunk test one.
    def from_yaml(**overrides):
        cfg = read_yaml(get_path(DECODER_YAML))["decoder"]
        # saq's own settings, `checkpoint` included, live under `config`
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


def _write_decode_yaml(path, checkpoint):
    """The shipped decoder yaml pointed at a checkpoint, for a decode run.

    The architecture is left exactly as shipped: `-t` trained that one, so anything
    rewritten here would be a different network reading the same weights.
    """
    cfg = read_yaml(get_path(DECODER_YAML))["decoder"]
    cfg["config"]["checkpoint"] = str(checkpoint)
    # a decode run has no schedule to follow
    cfg["config"].pop("train", None)
    path.write_text(yaml.safe_dump(_plain({"decoder": cfg})))
    return path


def test_train_then_decode_cli(tmp_path, batch_size=200, target_error=20):
    """Train with `syndrilla -t`, then decode with `syndrilla`, both through the CLI.

    The two commands are what a user types, and the checkpoint is the only thing that
    passes between them: the training run writes `best.pt`, the decode run reads it back
    through the decoder yaml's `checkpoint` key. Run in separate directories, so a decode
    artefact cannot be confused with one the training run left behind.
    """
    train_dir = tmp_path / "train"
    decoder_yaml = _write_decoder_yaml(
        tmp_path / "small.decoder.yaml", epochs=2, batches_per_epoch=4, val_batches=2
    )
    trained = _run_cli(train_dir, *_train_argv(decoder_yaml))
    assert trained.returncode == 0, trained.stderr

    best = train_dir / BEST_PT
    assert best.is_file(), "training produced no best.pt to decode with"

    decode_dir = tmp_path / "decode"
    decode_dir.mkdir()
    decoded = _run_cli(
        decode_dir,
        f"-d={_write_decode_yaml(tmp_path / 'trained.decoder.yaml', best)}",
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
    # a rate, not a quality bar: two epochs on a shrunk schedule is not enough training
    # to beat anything, so what is under test is that the trained weights decode at all
    rate = metrics["decoder_0"]["hx"]["logical error rate"]
    assert 0.0 <= float(rate) <= 1.0


def test_train_cli_rejects_untrainable_and_missing_args(tmp_path):
    """`-t` must fail loudly rather than half-run."""
    decoder_yaml = _write_decoder_yaml(
        tmp_path / "small.decoder.yaml", epochs=1, batches_per_epoch=1, val_batches=1
    )

    no_matrix = _run_cli(
        tmp_path,
        "-t",
        f"-d={decoder_yaml}",
        f"-e={TRAIN_ERROR_YAML}",
        f"-s={TRAIN_SYNDROME_YAML}",
        f"-ls={LOSS_YAML}",
    )
    assert no_matrix.returncode != 0
    assert "requires -m" in no_matrix.stderr

    no_error_yaml = _run_cli(
        tmp_path,
        "-t",
        f"-d={decoder_yaml}",
        f"-m={SURFACE_MATRIX_YAML}",
    )
    assert no_error_yaml.returncode != 0
    assert "requires -e" in no_error_yaml.stderr

    no_syndrome_yaml = _run_cli(
        tmp_path,
        "-t",
        f"-d={decoder_yaml}",
        f"-m={SURFACE_MATRIX_YAML}",
        f"-e={TRAIN_ERROR_YAML}",
    )
    assert no_syndrome_yaml.returncode != 0
    assert "requires -s" in no_syndrome_yaml.stderr

    mwpm_yaml = _write_decoder_yaml(
        tmp_path / "mwpm.decoder.yaml",
        source="examples/alist/mwpm_hx.decoder.yaml",
    )
    not_trainable = _run_cli(
        tmp_path,
        "-t",
        f"-d={mwpm_yaml}",
        f"-m={SURFACE_MATRIX_YAML}",
        f"-e={TRAIN_ERROR_YAML}",
        f"-s={TRAIN_SYNDROME_YAML}",
        f"-ls={LOSS_YAML}",
    )
    assert not_trainable.returncode != 0
    assert "cannot train" in not_trainable.stderr


def test_train_cli_rejects_a_decoder_yaml_with_no_schedule(tmp_path):
    """`-t` on a decoder yaml carrying no `train` block must say so."""
    cfg = read_yaml(get_path(DECODER_YAML))["decoder"]
    cfg["config"].pop("train", None)
    bad = tmp_path / "no_schedule.decoder.yaml"
    bad.write_text(yaml.safe_dump(_plain({"decoder": cfg})))

    result = _run_cli(
        tmp_path,
        "-t",
        f"-d={bad}",
        f"-m={SURFACE_MATRIX_YAML}",
        f"-e={TRAIN_ERROR_YAML}",
        f"-s={TRAIN_SYNDROME_YAML}",
        f"-ls={LOSS_YAML}",
    )
    assert result.returncode != 0
    assert "no 'train' block" in result.stderr


def test_train_cli_rejects_a_schedule_missing_a_key(tmp_path):
    """A `train` block without every key must fail before any training."""
    cfg = read_yaml(get_path(DECODER_YAML))["decoder"]
    cfg["config"]["train"] = {
        k: v for k, v in cfg["config"]["train"].items() if k != "val_batches"
    }
    bad = tmp_path / "bad.decoder.yaml"
    bad.write_text(yaml.safe_dump(_plain({"decoder": cfg})))

    result = _run_cli(
        tmp_path,
        "-t",
        f"-d={bad}",
        f"-m={SURFACE_MATRIX_YAML}",
        f"-e={TRAIN_ERROR_YAML}",
        f"-s={TRAIN_SYNDROME_YAML}",
        f"-ls={LOSS_YAML}",
    )
    assert result.returncode != 0
    assert "missing under 'decoder.train': val_batches" in result.stderr


def test_train_cli_requires_the_loss_yaml(tmp_path):
    """`-t` without `-ls` must fail before anything is constructed."""
    decoder_yaml = _write_decoder_yaml(
        tmp_path / "small.decoder.yaml", epochs=1, batches_per_epoch=1, val_batches=1
    )
    result = _run_cli(
        tmp_path,
        "-t",
        f"-d={decoder_yaml}",
        f"-m={SURFACE_MATRIX_YAML}",
        f"-e={TRAIN_ERROR_YAML}",
        f"-s={TRAIN_SYNDROME_YAML}",
    )
    assert result.returncode != 0
    assert "-ls" in result.stderr
    assert not (tmp_path / BEST_PT).exists()


# ── loss module ──────────────────────────────────────────────────────


def _make_loss(saq, **overrides):
    """Build the logical_centric loss bound to `saq`, from the shipped loss yaml."""
    from syndrilla.loss import create_loss

    cfg = dict(read_yaml(get_path(LOSS_YAML))["loss"], **overrides)
    return create_loss(cfg=cfg, decoder=saq)


def test_create_loss_reads_the_shipped_yaml():
    """The factory must dispatch on `function` and pick up the lambdas from the yaml."""
    from syndrilla.loss import create_loss

    decoder, saq = _make_decoder(SURFACE_MATRIX_YAML)
    loss_fn = create_loss(LOSS_YAML, decoder=saq)

    assert (loss_fn.lambda_lc, loss_fn.lambda_lp, loss_fn.lambda_ent) == (1.0, 0.2, 1.0)

    # the lambdas must actually be applied, not just stored: zeroing two of them must
    # leave exactly the third, doubled
    reweighted = _make_loss(saq, lambda_lc=2.0, lambda_lp=0.0, lambda_ent=0.0)
    e, synd, H = _random_shots(saq)
    out = decoder(_io_dict(saq, synd, H))
    lc, lp, ent = reweighted.terms(out, e)
    assert torch.allclose(reweighted.combine(lc, lp, ent), 2.0 * lc)


def test_loss_combine_matches_call_and_class_error_is_a_fraction():
    """`combine(*terms(...))` is what `__call__` returns, computed once instead of twice."""
    decoder, saq = _make_decoder(SURFACE_MATRIX_YAML)
    loss_fn = _make_loss(saq)
    e, synd, H = _random_shots(saq)
    out = decoder(_io_dict(saq, synd, H))

    lc, lp, ent = loss_fn.terms(out, e)
    for term in (lc, lp, ent):
        assert torch.isfinite(term) and term.item() >= 0.0
    assert torch.allclose(loss_fn.combine(lc, lp, ent), loss_fn(out, e))

    err = loss_fn.class_error(out, e)
    assert isinstance(err, float) and 0.0 <= err <= 1.0


def test_decoder_rejects_the_old_lambda_keys():
    """A stale decoder yaml must fail loudly, not silently train with default lambdas."""
    with pytest.raises(ValueError, match="lambda_loss_lc"):
        _make_decoder(SURFACE_MATRIX_YAML, lambda_loss_lc=1.0)


# --------------------------------------------------------------------------- #
# Resumable training
#
# Weights alone do not describe a run in progress: Adam's moments, the cosine
# schedule's position, the epoch counter, the best-so-far score and the error-stream
# RNG all decide what the next step does. These tests pin that `last.pt` carries
# them and that reloading it continues the run rather than warm-starting a new one.
# --------------------------------------------------------------------------- #


def _training_setup(**overrides):
    """A decoder and its loss, both built from the shipped yamls."""
    decoder, saq = _make_decoder(SURFACE_MATRIX_YAML, **overrides)
    return decoder, saq, _make_loss(saq)


def _train_step(decoder, saq, loss_fn, seed):
    """One batch -> forward -> loss -> backward -> update, on a seeded batch."""
    e, synd, H = _random_shots(saq, batch_size=8, seed=seed)
    out = decoder(_io_dict(saq, synd, H))
    saq.backward(loss_fn.combine(*loss_fn.terms(out, e)))
    saq.update()


def _fresh_run(epochs, seed=0):
    """A decoder seeded identically to every other `_fresh_run`, ready to train."""
    torch.manual_seed(seed)
    decoder, saq, loss_fn = _training_setup()
    saq.configure_optimizer(epochs)
    saq.set_training(True)
    return decoder, saq, loss_fn


def test_resume_continues_optimizer_and_schedule(tmp_path):
    """Reloading training state must continue the run, not warm-start a new one.

    Adam's moments and the cosine schedule's position decide the size and direction
    of the next step, so a decoder that reloads only `state_dict()` diverges from the
    uninterrupted run immediately. Bit-identical final weights are the proof it does
    not: nothing weaker distinguishes a resume from a warm start.
    """
    straight, straight_saq, straight_loss = _fresh_run(4)
    for epoch in range(4):
        _train_step(straight, straight_saq, straight_loss, seed=epoch)
        straight_saq.lr_step()

    part, part_saq, part_loss = _fresh_run(4)
    for epoch in range(2):
        _train_step(part, part_saq, part_loss, seed=epoch)
        part_saq.lr_step()
    path = tmp_path / "train_state.pt"
    torch.save(part_saq.train_state(), path)

    resumed, resumed_saq, resumed_loss = _fresh_run(4)
    resumed_saq.load_train_state(
        torch.load(path, map_location="cpu", weights_only=True)
    )
    for epoch in range(2, 4):
        _train_step(resumed, resumed_saq, resumed_loss, seed=epoch)
        resumed_saq.lr_step()

    assert resumed_saq.current_lr() == straight_saq.current_lr()
    reference = straight_saq.state_dict()
    for key, value in resumed_saq.state_dict().items():
        assert torch.equal(value, reference[key]), key


def test_train_metrics_hand_over_epoch_best_and_history(tmp_path):
    """The training half of `MetricState` must export the run position it owns, and take it back.

    This is the half of `last.pt` the decoder does not own: which epoch is next,
    which was best, and the history so far. `main.py`'s resume calls exactly these.
    """
    from syndrilla.metric import MetricState

    cfg = {"epochs": 4, "batches_per_epoch": 2, "val_batches": 1, "seed": 0}
    metrics = MetricState.for_training(str(tmp_path), cfg)
    metrics.epoch = 3
    metrics.best = 0.25
    metrics.history = [{"epoch": 1}, {"epoch": 2}]

    restored = MetricState.for_training(str(tmp_path), cfg)
    restored.load_train_state(metrics.train_state())

    assert restored.epoch == 3
    assert restored.best == 0.25
    assert restored.history == [{"epoch": 1}, {"epoch": 2}]
    # two epochs of (2 train + 1 val) batches are behind us
    assert restored.batches_done == 6


def _sequence(metrics, batch_index, k=6):
    """The first `k` draws of the phase `batch_index` opens."""
    metrics.begin_batch(batch_index)
    return torch.rand(k)


def test_every_epoch_trains_on_the_same_batches(tmp_path):
    """The training phase is a fixed set: epoch N draws what epoch 1 drew.

    With errors generated per batch, an unseeded stream would hand the model new noise
    every epoch. Pinning the training seed makes the training set finite and repeatable.
    """
    from syndrilla.metric import MetricState

    cfg = {"epochs": 4, "batches_per_epoch": 2, "val_batches": 1, "seed": 7}
    metrics = MetricState.for_training(str(tmp_path), cfg)

    first = _sequence(metrics, 0)  # epoch 1, first training batch
    metrics.epoch = 3
    third = _sequence(metrics, 3 * metrics.period)  # epoch 3, first training batch

    assert torch.equal(first, third)


def test_validation_draws_new_errors_each_epoch(tmp_path):
    """Validation is not the training set replayed, and not the same twice."""
    from syndrilla.metric import MetricState

    cfg = {"epochs": 4, "batches_per_epoch": 2, "val_batches": 1, "seed": 7}
    metrics = MetricState.for_training(str(tmp_path), cfg)

    metrics.epoch = 1
    val_1 = _sequence(metrics, cfg["batches_per_epoch"])
    train = _sequence(metrics, 0)
    metrics.epoch = 2
    val_2 = _sequence(metrics, metrics.period + cfg["batches_per_epoch"])

    assert not torch.equal(val_1, val_2), "validation replayed the same errors"
    assert not torch.equal(val_1, train), "validation replayed the training set"


def test_begin_batch_reports_the_phase(tmp_path):
    """Seeding must not disturb which batches count as training."""
    from syndrilla.metric import MetricState

    cfg = {"epochs": 2, "batches_per_epoch": 2, "val_batches": 1, "seed": 7}
    metrics = MetricState.for_training(str(tmp_path), cfg)

    phases = [metrics.begin_batch(i) for i in range(metrics.period * 2)]

    assert phases == ["train", "train", "val", "train", "train", "val"]
    # the phase it returns is the phase it is left in, so a caller can read it back
    # instead of asking the schedule a second time
    assert metrics.phase == "val"


def test_begin_batch_puts_the_decoder_in_the_phase_it_opened(tmp_path):
    """The phase the metrics pick and the mode the decoder runs in are one decision.

    A validation batch that still built a graph, or a training batch that did not,
    would train on the wrong set while reporting the right one, so `begin_batch` moves
    the bound decoder itself rather than leaving each caller to pair the two.
    """
    from syndrilla.metric import MetricState

    class _Decoder:
        training = None

        def set_training(self, training):
            self.training = training

    cfg = {"epochs": 2, "batches_per_epoch": 2, "val_batches": 1, "seed": 7}
    metrics = MetricState.for_training(str(tmp_path), cfg)
    decoder = _Decoder()
    metrics.bind_decoder(decoder, fingerprint={})

    modes = []
    for i in range(metrics.period):
        metrics.begin_batch(i)
        modes.append(decoder.training)

    assert modes == [True, True, False]


def test_neighbouring_run_seeds_do_not_share_streams(tmp_path):
    """`seed` and `seed + 1` must not produce the same training set."""
    from syndrilla.metric import MetricState

    def train_draw(seed):
        cfg = {"epochs": 4, "batches_per_epoch": 2, "val_batches": 1, "seed": seed}
        return _sequence(MetricState.for_training(str(tmp_path), cfg), 0)

    assert not torch.equal(train_draw(7), train_draw(8))


def _chain(algorithms, configs):
    """Build a decoder chain on the surface_5 matrix."""
    cfg = read_yaml(get_path(DECODER_YAML))["decoder"]
    cfg["config"]["model"] = dict(cfg["config"]["model"], d_model=16, N_dec=1, h=2)
    # osd_0 ships a CUDA port that refuses to run without a GPU
    cfg["device"] = {"device_type": "cpu", "device_idx": 0}
    cfg = dict(cfg, algorithm=algorithms, config=configs)
    bundle = load_matrices(
        read_yaml(get_path(SURFACE_MATRIX_YAML))["matrix"], *parse_device_dtype(cfg)
    )
    return create_decoder(cfg=cfg, bundle=bundle)


def _saq_block():
    cfg = read_yaml(get_path(DECODER_YAML))["decoder"]["config"]
    cfg["model"] = dict(cfg["model"], d_model=16, N_dec=1, h=2)
    return cfg


def test_train_cli_runs_a_chain_ending_in_the_trained_decoder(tmp_path):
    """`[bp_norm_min_sum, saq]` must train end to end, not just pass the static check.

    The earlier stage runs untrained ahead of saq every batch, so this covers the loop
    path that a single-decoder config never reaches.
    """
    cfg = read_yaml(get_path(DECODER_YAML))["decoder"]
    saq_block = dict(cfg["config"])
    saq_block["model"] = dict(saq_block["model"], d_model=16, N_dec=1, h=2)
    saq_block["train"] = dict(
        saq_block["train"], epochs=1, batches_per_epoch=2, val_batches=1
    )
    cfg = dict(
        cfg,
        algorithm=["bp_norm_min_sum", "saq"],
        config=[{"max_iter": 4}, saq_block],
        device={"device_type": "cpu", "device_idx": 0},
    )
    chained = tmp_path / "chained.decoder.yaml"
    chained.write_text(yaml.safe_dump(_plain({"decoder": cfg})))

    assert _run_cli(tmp_path, *_train_argv(chained)).returncode == 0
    assert (tmp_path / BEST_PT).is_file(), "the chain trained but wrote no checkpoint"


def test_only_the_last_decoder_decides_whether_a_chain_can_train():
    """`-t` drives the chain's last stage, so that is the one that must be trainable.

    `[saq, osd_0]` is rejected because a stage after the trained decoder would leave the
    run learning from output its gradient never passed through; `[bp_norm_min_sum, saq]`
    is the shape that trains.
    """
    from syndrilla.decoder import is_trainable

    def tail(algorithms, configs):
        chain = _chain(algorithms, configs)
        return is_trainable(getattr(chain[-1], "decoder", chain[-1]))

    assert not tail(["saq", "osd_0"], [_saq_block(), {}])
    assert tail(["bp_norm_min_sum", "saq"], [{"max_iter": 4}, _saq_block()])
    assert not tail(["bp_norm_min_sum", "osd_0"], [{"max_iter": 4}, {}])


def test_train_cli_exits_when_the_last_decoder_cannot_train(tmp_path):
    """The check has to stop a real `-t` run, not just be true in isolation."""
    cfg = read_yaml(get_path(DECODER_YAML))["decoder"]
    saq_block = dict(cfg["config"])
    saq_block["model"] = dict(saq_block["model"], d_model=16, N_dec=1, h=2)
    cfg = dict(
        cfg,
        algorithm=["saq", "osd_0"],
        config=[saq_block, {}],
        device={"device_type": "cpu", "device_idx": 0},
    )
    bad = tmp_path / "misordered.decoder.yaml"
    bad.write_text(yaml.safe_dump(_plain({"decoder": cfg})))

    assert _run_cli(tmp_path, *_train_argv(bad)).returncode != 0


def _train_argv(decoder_yaml, *extra):
    return [
        "-t",
        f"-d={decoder_yaml}",
        f"-m={SURFACE_MATRIX_YAML}",
        f"-e={TRAIN_ERROR_YAML}",
        f"-s={TRAIN_SYNDROME_YAML}",
        f"-ls={LOSS_YAML}",
        "-bs=16",
        *extra,
    ]


def _interrupt_after_epoch(run_dir, decoder_yaml, epoch):
    """Start a training run and SIGINT it once `epoch`'s line has been printed.

    `record_epoch` writes the checkpoint before printing the line, so seeing the line
    means `last.pt` for that epoch is complete on disk.
    """
    import signal

    env = dict(os.environ, PYTHONUNBUFFERED="1")
    cmd = ["syndrilla", f"-r={run_dir}", *_train_argv(decoder_yaml)]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
    )
    seen = False
    for line in proc.stdout:
        if line.startswith(f"epoch {epoch:4d}/"):
            seen = True
            proc.send_signal(signal.SIGINT)
            break
    proc.stdout.read()
    proc.stdout.close()
    proc.wait(timeout=300)
    assert seen, f"training never reached epoch {epoch}"


def test_resume_cli_finishes_an_interrupted_run(tmp_path):
    """`-tckpt` must finish an interrupted run exactly as if it had never stopped."""
    decoder_yaml = _write_decoder_yaml(
        tmp_path / "small.decoder.yaml", epochs=4, batches_per_epoch=3, val_batches=1
    )

    straight_dir = tmp_path / "straight"
    assert _run_cli(straight_dir, *_train_argv(decoder_yaml)).returncode == 0

    resumed_dir = tmp_path / "resumed"
    _interrupt_after_epoch(resumed_dir, decoder_yaml, 3)
    # an interrupted run writes no history.json -- that is `save_history`'s job at the
    # end -- so the three finished epochs have to be in the checkpoint, or they are lost
    partial = torch.load(resumed_dir / LAST_PT, map_location="cpu", weights_only=True)
    assert [entry["epoch"] for entry in partial["history"]] == [1, 2, 3]
    assert partial["epoch"] == 4

    finished = _run_cli(
        resumed_dir, *_train_argv(decoder_yaml, f"-tckpt={resumed_dir / LAST_PT}")
    )
    assert finished.returncode == 0, finished.stderr

    # the resumed run must reach the same place, epoch by epoch and weight by weight
    assert json.loads((resumed_dir / "history.json").read_text()) == json.loads(
        (straight_dir / "history.json").read_text()
    )
    expected = torch.load(straight_dir / LAST_PT, map_location="cpu", weights_only=True)
    actual = torch.load(resumed_dir / LAST_PT, map_location="cpu", weights_only=True)
    for key, value in expected["state_dict"].items():
        assert torch.equal(actual["state_dict"][key], value), key


def test_resume_rejects_a_changed_schedule(tmp_path):
    """A checkpoint from a different schedule must be refused, not silently resumed."""
    decoder_yaml = _write_decoder_yaml(
        tmp_path / "small.decoder.yaml", epochs=2, batches_per_epoch=2, val_batches=1
    )
    assert _run_cli(tmp_path, *_train_argv(decoder_yaml)).returncode == 0

    changed = _write_decoder_yaml(
        tmp_path / "changed.decoder.yaml", epochs=2, batches_per_epoch=5, val_batches=1
    )
    result = _run_cli(tmp_path, *_train_argv(changed, f"-tckpt={tmp_path / LAST_PT}"))
    assert result.returncode != 0
    assert "batches_per_epoch" in result.stderr


def test_resume_needs_training_mode(tmp_path):
    """`-tckpt` without `-t` must fail rather than be ignored.

    `-tc` is deliberately not the flag: argparse decomposes it into `-t -c`, which would
    turn a typo into a silent training run. `-tckpt` mirrors the decode-side `-ckpt`.
    """
    result = _run_cli(
        tmp_path,
        f"-tckpt={tmp_path / LAST_PT}",
        f"-d={DECODER_YAML}",
        f"-m={SURFACE_MATRIX_YAML}",
        "-e=examples/alist/bsc.error.yaml",
        "-c=examples/alist/lx.check.yaml",
        "-s=examples/alist/perfect.syndrome.yaml",
    )
    assert result.returncode != 0
    assert "-tckpt" in result.stderr


@pytest.mark.parametrize(
    "matrix_yaml, expected",
    [
        # toric pins n = 2d^2, so d is recoverable: 200 qubits -> d10
        (TORIC_MATRIX_YAML, "saq_hx_d10"),
        # surface_5 carries 41 qubits: the unrotated relation d^2 + (d-1)^2 at d=5
        (SURFACE_MATRIX_YAML, "saq_hx_d5"),
    ],
)
def test_checkpoint_stem_names_the_configuration(matrix_yaml, expected):
    """Checkpoints are named after what produced them, and never guess a distance."""
    _, saq = _make_decoder(matrix_yaml)

    assert saq.checkpoint_stem() == expected


def test_code_type_is_rejected_rather_than_ignored():
    """`code_type` was removed; a config still setting it must be told, not ignored.

    The family is measured from the matrix, so a declared one could only ever disagree
    with it. Silently dropping the key would leave that disagreement invisible.
    """
    with pytest.raises(ValueError, match="code_type"):
        _make_decoder(SURFACE_MATRIX_YAML, code_type="toric")


def test_checkpoint_stem_separates_two_configurations(tmp_path):
    """Two configs trained into one run dir must not overwrite each other's weights."""
    _, surface = _make_decoder(SURFACE_MATRIX_YAML)
    _, toric = _make_decoder(TORIC_MATRIX_YAML)

    assert surface.checkpoint_stem() != toric.checkpoint_stem()


def test_best_pt_stays_bare_weights_and_last_pt_still_decodes(tmp_path):
    """`best.pt` must stay a portable state_dict; `last.pt` must still load to decode."""
    decoder_yaml = _write_decoder_yaml(
        tmp_path / "small.decoder.yaml", epochs=1, batches_per_epoch=2, val_batches=1
    )
    assert _run_cli(tmp_path, *_train_argv(decoder_yaml)).returncode == 0

    best = torch.load(tmp_path / BEST_PT, map_location="cpu", weights_only=True)
    assert all(torch.is_tensor(value) for value in best.values())
    assert "optimizer" not in best

    last = torch.load(tmp_path / LAST_PT, map_location="cpu", weights_only=True)
    assert set(last) >= {"state_dict", "optimizer", "scheduler", "fingerprint"}

    # the decoder's `checkpoint` key reads both forms
    for name in (BEST_PT, LAST_PT):
        cfg = read_yaml(get_path(DECODER_YAML))["decoder"]
        cfg["config"]["checkpoint"] = str(tmp_path / name)
        bundle = load_matrices(
            read_yaml(get_path(SURFACE_MATRIX_YAML))["matrix"], *parse_device_dtype(cfg)
        )
        create_decoder(cfg=cfg, bundle=bundle)


# --------------------------------------------------------------------------- #
# Batch-shape constraints
#
# These are saq's, not training's: `logical_logits` / `logical_prior` are written
# per forward row and are not unfolded by RoundFlattenWrapper, so a multi-round
# batch reaches the loss with the llr at [B, d, n] and the logical head at
# [B*d, 2^k]. A second channel is read as a second round for the same reason.
# --------------------------------------------------------------------------- #


def test_check_train_batch_accepts_a_single_round_single_channel_batch():
    """The shape saq is built for must pass without complaint."""
    _, saq = _make_decoder(SURFACE_MATRIX_YAML)
    assert saq.check_train_batch(1, 1) is None


def test_check_train_batch_rejects_multiple_rounds():
    """A multi-round batch must be refused by the decoder, naming rounds."""
    _, saq = _make_decoder(SURFACE_MATRIX_YAML)
    with pytest.raises(ValueError, match="rounds"):
        saq.check_train_batch(2, 1)


def test_check_train_batch_rejects_multiple_channels():
    """A multi-channel batch must be refused by the decoder, naming the channels."""
    _, saq = _make_decoder(SURFACE_MATRIX_YAML)
    with pytest.raises(ValueError, match="number_channel"):
        saq.check_train_batch(1, 2)


def test_train_cli_reports_the_decoders_own_batch_constraint(tmp_path):
    """`-t` on a multi-round measurer must fail with saq's message, not main.py's.

    The constraint belongs to the decoder, so the error has to name the decoder. A
    generic "training needs a single-round batch" from `main.py` would be wrong for a
    decoder that consumes the rounds dimension itself.
    """
    decoder_yaml = _write_decoder_yaml(
        tmp_path / "small.decoder.yaml", epochs=1, batches_per_epoch=1, val_batches=1
    )
    result = _run_cli(
        tmp_path,
        "-t",
        f"-d={decoder_yaml}",
        f"-m={SURFACE_MATRIX_YAML}",
        f"-e={TRAIN_ERROR_YAML}",
        "-s=examples/alist/phenomenological.syndrome.yaml",
        f"-ls={LOSS_YAML}",
        "-bs=16",
    )
    assert result.returncode != 0
    # match the raised message itself, not the loguru lines around it: those mention
    # `create_decoder_with_saq` on every run and would pass no matter who raised
    raised = [ln for ln in result.stderr.splitlines() if ln.startswith("ValueError:")]
    assert raised, result.stderr
    assert "saq" in raised[-1] and "rounds" in raised[-1], raised[-1]


def test_decoder_describes_itself_in_the_resume_fingerprint(tmp_path):
    """The model half of the fingerprint must come from the decoder, not the metrics.

    `MetricState` owns the schedule and the batch size; what algorithm this is, what
    code shape it was built for, and what optimizer settings it will use are the
    decoder's to state. The metrics merge the two rather than reaching into the model.
    """
    from syndrilla.metric import MetricState

    _, saq = _make_decoder(SURFACE_MATRIX_YAML)
    model = saq.train_fingerprint()
    assert model == {
        "algo": "saq",
        "n": saq.n,
        "m": saq.m,
        "k": saq.k,
        "lr": saq.lr,
        "weight_decay": saq.weight_decay,
        "min_lr": saq.min_lr,
    }

    cfg = {"epochs": 4, "batches_per_epoch": 2, "val_batches": 1, "seed": 0}
    merged = MetricState.for_training(str(tmp_path), cfg).fingerprint(
        saq, batch_size=16
    )
    # every model key survives the merge, and the schedule half is added to it
    assert merged.items() >= model.items()
    assert merged["batch_size"] == 16
    assert all(merged[k] == v for k, v in cfg.items())


# --------------------------------------------------------------------------- #
# Config layout
#
# The decoder yaml is grouped so each block has exactly one reader: `model` and
# `cpnd` and `optimizer` are the decoder's, `train` is the metrics'. Nothing
# reaches across, which is what these tests pin.
# --------------------------------------------------------------------------- #


def test_decoder_reads_its_settings_from_their_own_blocks():
    """Architecture, CPND and optimizer settings each come from their own block."""
    _, saq = _make_decoder(
        SURFACE_MATRIX_YAML,
        "rotated_surface",
        model={"d_model": 64, "N_dec": 3, "h": 8, "dropout": 0.25, "no_mask": 1},
        cpnd={"enable": False, "passes": 4},
        optimizer={"lr": 1.0e-3, "weight_decay": 2.0e-7, "min_lr": 3.0e-6},
    )
    assert (saq.d_model, saq.N_dec) == (64, 3)
    assert saq.src_mask_SN is None  # no_mask: 1
    assert saq.use_cpnd is False and saq.cpnd_passes == 4
    assert (saq.lr, saq.weight_decay, saq.min_lr) == (1.0e-3, 2.0e-7, 3.0e-6)


def test_flat_architecture_keys_are_rejected():
    """A stale flat yaml must fail loudly rather than silently use the defaults."""
    with pytest.raises(ValueError, match="d_model"):
        _make_decoder(SURFACE_MATRIX_YAML, d_model=64)


def test_training_mode_turns_cpnd_off_in_the_decoder():
    """The decoder decides what training means for its own inference-only stage.

    `main.py` passes the mode, not a rewritten config: it has no business knowing that
    CPND exists. Deciding at construction also skips CPND's GF(2) precompute, which a
    training run would otherwise pay for and never use.
    """
    _, trained = _make_decoder(
        SURFACE_MATRIX_YAML, training=True, cpnd={"enable": True}
    )
    assert trained.use_cpnd is False
    assert not hasattr(trained, "cpnd_supports")  # precompute skipped entirely

    _, decoded = _make_decoder(SURFACE_MATRIX_YAML, cpnd={"enable": True})
    assert decoded.use_cpnd is True and hasattr(decoded, "cpnd_supports")


def test_train_state_carries_the_random_state():
    """The random state resumes with the decoder, on the decoder's own device.

    Which generators exist is a device question, and the decoder is the thing that has
    a device. `main.py` probing `torch.cuda.is_available()` to answer it is the loop
    reasoning about hardware it does not own.
    """
    _, cpu_saq = _make_decoder(
        SURFACE_MATRIX_YAML,
        "rotated_surface",
        device={"device_type": "cpu", "device_idx": 0},
    )
    cpu_saq.configure_optimizer(2)
    rng = cpu_saq.train_state()["rng"]

    assert torch.is_tensor(rng["cpu"])
    # a cpu decoder has no cuda generators to save, whatever the host happens to have.
    # The device is pinned here rather than taken from the shipped yaml, which asks for
    # cuda: otherwise this asserts something about the test host, not about the decoder.
    assert "cuda" not in rng

    # the mirror image: a cuda decoder does save them, because its stream is the one a
    # resumed run has to carry on drawing from
    if torch.cuda.is_available():
        _, gpu_saq = _make_decoder(
            SURFACE_MATRIX_YAML,
            "rotated_surface",
            device={"device_type": "cuda", "device_idx": 0},
        )
        gpu_saq.configure_optimizer(2)
        assert "cuda" in gpu_saq.train_state()["rng"]


def test_load_train_state_restores_the_error_stream():
    """Restoring must put the draw sequence back, or a resumed run trains on new errors."""
    _, saq = _make_decoder(SURFACE_MATRIX_YAML)
    saq.configure_optimizer(2)

    torch.manual_seed(1234)
    state = saq.train_state()
    expected = torch.rand(6)

    torch.manual_seed(9999)  # somewhere else entirely
    saq.load_train_state(state)
    assert torch.equal(torch.rand(6), expected)


def test_device_resolution_matches_the_matrices():
    """The decoder must land on the same device its matrices did.

    `load_matrices` resolves the device with `parse_device_dtype`, which falls back to
    cpu when cuda is configured but unavailable. A decoder that resolves the device its
    own way can disagree with that, and then the shipped yaml only runs on a GPU host.
    """
    cfg = read_yaml(get_path(DECODER_YAML))["decoder"]
    cfg["device"] = {"device_type": "cuda", "device_idx": 0}
    cfg["config"]["model"] = dict(cfg["config"]["model"], d_model=32, N_dec=2, h=4)
    expected, _ = parse_device_dtype(cfg)

    bundle = load_matrices(
        read_yaml(get_path(SURFACE_MATRIX_YAML))["matrix"], *parse_device_dtype(cfg)
    )
    saq = create_decoder(cfg=cfg, bundle=bundle)[0].decoder

    assert torch.device(saq.device) == expected
    if not torch.cuda.is_available():
        # a cuda yaml must still build on a cpu-only host, not claim cuda:0 and crash
        assert torch.device(saq.device).type == "cpu"
