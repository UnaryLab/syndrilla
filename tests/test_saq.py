import os
import subprocess
import sys

import pytest
import torch
import yaml

sys.path.append(os.getcwd())

from syndrilla.decoder import create_decoder
from syndrilla.decoder.decoder import SHARED_KEYS
from syndrilla.matrix import load_matrices
from syndrilla.utils import get_path, parse_device_dtype, read_yaml

DECODER_YAML = "examples/alist/train_saq_hx.decoder.yaml"
DECODE_YAML = "examples/alist/saq_hx.decoder.yaml"
SURFACE_MATRIX_YAML = "examples/alist/surface_5.matrix.yaml"
TORIC_MATRIX_YAML = "examples/alist/toric_10.matrix.yaml"

# Checkpoints are named after the configuration that produced them. The CLI tests train
# the shipped hx config on surface_5, whose matrix has 41 columns.
CKPT_STEM = "saq_hx_n41"
BEST_PT = f"{CKPT_STEM}_best.pt"
LAST_PT = f"{CKPT_STEM}_last.pt"
RESULT_YAML = f"{CKPT_STEM}_result.yaml"
TRAIN_LOG = f"{CKPT_STEM}_train.log"
# the result yaml's two phase blocks and the columns each carries: the objective and the
# class error, which any trained decoder reports, each named for the phase it belongs to.
# The split of the total into lc/lp/ent is the logical-centric loss's own, so it stays in
# the epoch line and so in the train log. Keyed by the yaml's spelling, not the internal
# `train`/`val` the in-memory history keeps
TERMS = {
    "training": ("training loss", "training error"),
    "validation": ("validation loss", "validation error"),
}
# the terms that split is made of, which the result yaml must not carry
MODEL_TERMS = ("lc", "lp", "ent")
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
    """The hard decision and `syndrome_estimation` must agree with their closed forms.

    The hard decision is read off a forward pass rather than called directly: it is a
    line inside `forward`, and CPND is what would otherwise move `e_v` off it, so the
    decoder is built in training mode, where the decoder turns that stage off itself.
    """
    decoder, saq = _make_decoder(SURFACE_MATRIX_YAML)
    e, _, H = _random_shots(saq, batch_size=8, error_rate=0.2)

    # syndrome_estimation is H @ e over GF(2), including for ragged check degrees
    assert torch.equal(
        saq.syndrome_estimation(e), ((e.to(torch.float32) @ H.t()) % 2).to(saq.dtype)
    )

    train_decoder, train_saq = _make_decoder(SURFACE_MATRIX_YAML, training=True)
    train_decoder.eval()
    _, synd, H = _random_shots(train_saq, batch_size=8)
    out = train_decoder(_io_dict(train_saq, synd, H))

    # non-positive (<= 0) posterior LLR means flipped, zero included
    l_v = out["llr"]
    assert torch.equal(out["e_v"], (l_v <= 0.0).to(train_saq.dtype))
    assert out["e_v"].dtype == train_saq.dtype


def test_forward_matches_the_paper_pipeline():
    """forward() must equal the SAQ pipeline transcribed independently from the paper.

    `forward` runs stages 1 and 2 inline, so this is what pins its wiring: the streams
    are rebuilt here from the learned parameters directly, the layers are driven with
    the two masks by hand, and the heads are applied separately. A mis-plumbed mask, a
    dropped global token, a missed mid-depth norm, or LN reading the *previous* layer's
    SN would all show up as a mismatch against the decoder's own output.
    """
    decoder, saq = _make_decoder(SURFACE_MATRIX_YAML)
    decoder.eval()
    _, synd, H = _random_shots(saq)
    out = decoder(_io_dict(saq, synd, H))

    with torch.no_grad():
        syndrome_pm = 1 - 2 * synd
        out_LP = saq.MLP(syndrome_pm)
        SN = saq.learnable_embed_S.unsqueeze(0) * syndrome_pm.unsqueeze(-1)
        LN = saq.learnable_embed_L.unsqueeze(0) * out_LP.unsqueeze(-1)
        SN = torch.cat([saq.global_tok.expand(SN.size(0), -1, -1), SN], dim=1)
        assert SN.shape == (synd.size(0), saq.m + 1, 32)  # +1 for the global token
        assert LN.shape == (synd.size(0), saq.logical_classes, 32)

        for idx in range(saq.N_dec):
            SN = saq.layers[idx](SN, SN, saq.src_mask_SN, "syndrome")
            LN = saq.layers[idx](LN, SN, saq.src_mask_LN, "logical")
            if saq.N_dec > 1 and idx == saq.N_dec // 2:
                SN, LN = saq.SN_norm2(SN), saq.LN_norm2(LN)

        l_v = saq.out_fc_S(saq.proj_e(saq.SN_norm(SN)[:, 1:, :]).squeeze(-1))
        out_L = saq.out_fc_L(saq.proj_l(saq.LN_norm(LN)).squeeze(-1))

    assert torch.allclose(out_LP, out["logical_prior"], atol=1e-6)
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
        loss.backward()
        saq.optimizer.step()
        saq.optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())

    assert losses[-1] < losses[0]


# ── CPND (stage 4) ───────────────────────────────────────────────────
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

    # the hard decision `forward` makes, recomputed here to feed CPND its raw input
    e0 = saq.project(
        torch.where(l_v <= 0.0, 1.0, 0.0).to(saq.dtype), synd, out["logical_logits"]
    )
    cost_projected = (e0 * l_v).sum(dim=1)
    cost_final = (out["e_v"] * l_v).sum(dim=1)

    assert torch.all(cost_final <= cost_projected + 1e-4)
    assert torch.any(cost_final < cost_projected - 1e-6)

    # extra sweeps keep improving and keep the constraints
    saq.cpnd_passes = 5
    more = saq.nullspace_descent(e0, l_v).to(torch.float32)
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
    # the hard decision `forward` makes, recomputed here to feed CPND its raw input
    e0 = saq.project(
        torch.where(l_v <= 0.0, 1.0, 0.0).to(saq.dtype), synd, out["logical_logits"]
    )

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
    # the hard decision `forward` makes, recomputed here to feed CPND its raw input
    e_raw = torch.where(l_v <= 0.0, 1.0, 0.0).to(saq.dtype)
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
    fixed = saq.nullspace_descent(e0, l_v).to(torch.float32)
    inverted = saq.nullspace_descent(e0, -l_v).to(torch.float32)
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
        tmp_path / "small.decoder.yaml", epochs=2, test_batches=4, validation_batches=2
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
    assert (tmp_path / RESULT_YAML).is_file()

    # the schedule and the optimizer settings come from their own blocks of the same
    # decoder yaml
    epochs = yaml.safe_load((tmp_path / RESULT_YAML).read_text())["training result"]
    # the run's best and its last, so epoch 2 is always there and epoch 1 only if it
    # was the better of the two
    numbers = epochs["epoch"]
    assert numbers[-1] == 2
    rates = epochs["learning rate"]
    shipped_lr = read_yaml(get_path(DECODER_YAML))["decoder"]["config"]["optimizer"][
        "lr"
    ]
    if numbers[0] == 1:
        assert rates[0] == shipped_lr and rates[-1] < rates[0]
    else:
        assert rates[-1] < shipped_lr

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


def test_train_cli_writes_a_result_yaml_indexed_by_epoch(tmp_path):
    """`-t` writes its results as a yaml too, the way a decode run does.

    Two epochs are written, the run's best and its last, stored by column: `epoch` lists
    the epoch numbers and every other list is index-aligned with it, so entry `i` of each
    is one epoch, and every number in it is the one the run's epoch line printed. The
    two collapse to one entry only when the last epoch is itself the best.
    """
    decoder_yaml = _write_decoder_yaml(
        tmp_path / "small.decoder.yaml", epochs=3, test_batches=4, validation_batches=2
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

    written = yaml.safe_load((tmp_path / RESULT_YAML).read_text())
    epochs = written["training result"]
    # the last epoch is always written; the best joins it unless it is that epoch
    assert epochs["epoch"][-1] == 3
    kept = len(epochs["epoch"])
    assert kept in (1, 2)
    if kept == 2:
        assert epochs["best"][0] is True and epochs["epoch"][0] < 3
    # every column is as long as the epoch list, or the alignment they are read by
    # would be silently wrong
    columns = [epochs["learning rate"], epochs["best"]]
    columns += [epochs[phase][term] for phase, terms in TERMS.items() for term in terms]
    assert all(len(column) == kept for column in columns)
    # and nothing else: a term one loss decomposes its total into is not something the
    # result file can name for a model that has no such decomposition
    for phase, terms in TERMS.items():
        assert set(epochs[phase]) == set(terms)

    summary = written["train_full"]
    assert summary["epochs"] == 3
    assert (
        summary["training batches count"] == 4
        and summary["validation batches count"] == 2
    )
    assert summary["batch size"] == 32
    assert summary["best checkpoint"] == str(tmp_path / BEST_PT)
    assert summary["last checkpoint"] == str(tmp_path / LAST_PT)
    # the summary's best has to be the epoch the columns say is best, not a second
    # opinion
    val_class_err = epochs["validation"]["validation error"]
    i = val_class_err.index(min(val_class_err))
    assert summary["best epoch index"] == epochs["epoch"][i]
    assert summary["best validation error"] == pytest.approx(val_class_err[i])
    assert epochs["best"][i] is True

    # the breakdown the yaml drops is still recorded, in the epoch lines of the run's
    # own log rather than in the toolchain's result format
    lines = [
        line
        for line in (tmp_path / TRAIN_LOG).read_text().splitlines()
        if "train_loss=" in line
    ]
    assert len(lines) == 3
    assert all(f"{term}=" in line for line in lines for term in MODEL_TERMS)


def test_train_log_records_every_batch_with_its_loss_and_rate(tmp_path):
    """The log carries the batches an epoch averaged, not only the average.

    An epoch line alone says a run got worse, never which batch it happened on, so
    each batch writes its own loss and the rate it ran at. The two granularities have
    to agree: the epoch's reported average is the mean of the batch losses above it.
    """
    epochs, train_batches, val_batches = 2, 4, 2
    decoder_yaml = _write_decoder_yaml(
        tmp_path / "small.decoder.yaml",
        epochs=epochs,
        test_batches=train_batches,
        validation_batches=val_batches,
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

    period = train_batches + val_batches
    lines = (tmp_path / TRAIN_LOG).read_text().splitlines()
    batch_lines = [line for line in lines if "batch " in line and "loss=" in line]
    assert len(batch_lines) == epochs * period

    # every batch names the phase it was metered as and the rate it ran at, and splits
    # its loss the way the epoch line does
    assert sum("  train  " in line for line in batch_lines) == epochs * train_batches
    assert sum("  val    " in line for line in batch_lines) == epochs * val_batches
    assert all("lr=" in line for line in batch_lines)
    assert all(f"{term}=" in line for line in batch_lines for term in MODEL_TERMS)

    # the console keeps the epoch summaries only: one batch line per batch would bury
    # them, so they go to the log file alone
    assert "batch " not in result.stdout

    def loss_of(line):
        return float(line.split("loss=")[1].split()[0])

    # the first epoch's training average is the mean of the training batches above it
    first = batch_lines[:train_batches]
    epoch_line = next(line for line in lines if "train_loss=" in line)
    assert float(epoch_line.split("train_loss=")[1].split()[0]) == pytest.approx(
        sum(loss_of(line) for line in first) / train_batches, abs=1e-4
    )


def test_train_result_yaml_reports_what_the_run_cost(tmp_path):
    """The run's timing, in the terms a decode result file reports it.

    The averages have to come from the epoch times rather than the wall clock: the
    wall clock includes building the decoder and loading the matrices, so on a short
    run the two differ by more than rounding. `epochs_saved` keeps every epoch's time in
    the file, so the summed column can be checked against the summary that averages it.
    """
    decoder_yaml = _write_decoder_yaml(
        tmp_path / "small.decoder.yaml",
        epochs=3,
        test_batches=4,
        validation_batches=2,
        epochs_saved=3,
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

    written = yaml.safe_load((tmp_path / RESULT_YAML).read_text())
    summary = written["train_full"]
    per_epoch_times = written["training result"]["time (s)"]

    assert len(per_epoch_times) == 3 and all(t > 0 for t in per_epoch_times)
    assert summary["total epoch time (s)"] == pytest.approx(sum(per_epoch_times))
    assert summary["average time per epoch (s)"] == pytest.approx(
        sum(per_epoch_times) / 3
    )
    # a batch is a batch of either phase, so an epoch holds test_batches + val
    period = 4 + 2
    assert summary["average time per batch (s)"] == pytest.approx(
        summary["average time per epoch (s)"] / period
    )
    assert summary["average time per sample (s)"] == pytest.approx(
        summary["average time per batch (s)"] / 32
    )
    # the wall clock covers the epochs and the setup ahead of them
    assert summary["total time (s)"] > summary["total epoch time (s)"]


def test_train_epochs_saved_caps_the_result_yaml(tmp_path):
    """`epochs_saved` bounds the result yaml, so a long run stays readable.

    What survives is the run's tail plus its best epoch: the summary block names the
    best epoch and the saved checkpoint holds it, so a file that dropped it would name
    an epoch it did not carry.
    """
    decoder_yaml = _write_decoder_yaml(
        tmp_path / "small.decoder.yaml",
        epochs=6,
        test_batches=4,
        validation_batches=2,
        epochs_saved=2,
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

    written = yaml.safe_load((tmp_path / RESULT_YAML).read_text())
    epochs = written["training result"]
    numbers = epochs["epoch"]
    summary = written["train_full"]
    # all six ran, and the tail is always the last two of them
    assert summary["epochs"] == 6 and summary["epochs saved"] == 2
    assert numbers[-2:] == [5, 6]
    # the best epoch is kept wherever it fell, so it is the tail alone only when the
    # best is already in it
    i = len(numbers) - 1 - epochs["best"][::-1].index(True)
    assert summary["best epoch index"] == numbers[i]
    assert summary["best validation error"] == pytest.approx(
        epochs["validation"]["validation error"][i]
    )
    assert len(numbers) == (2 if numbers[i] in (5, 6) else 3)
    assert numbers == sorted(numbers)
    # the columns are thinned with the epoch list, not left at their full length
    assert all(
        len(epochs[phase][term]) == len(numbers)
        for phase, terms in TERMS.items()
        for term in terms
    )

    # the resume checkpoint keeps the whole curve: the cap is a file concern, and a
    # run resumed from here has to know every epoch it already ran
    state = torch.load(tmp_path / LAST_PT, map_location="cpu", weights_only=True)
    assert [entry["epoch"] for entry in state["history"]] == [1, 2, 3, 4, 5, 6]


def test_train_epochs_saved_must_be_a_positive_integer(tmp_path):
    """A cap of zero saves nothing, so it is rejected the way a zero schedule is."""
    decoder_yaml = _write_decoder_yaml(
        tmp_path / "small.decoder.yaml",
        epochs=2,
        test_batches=4,
        validation_batches=2,
        epochs_saved=0,
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
    assert result.returncode != 0
    assert "epochs_saved" in result.stderr and "integer >= 1" in result.stderr


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
        tmp_path / "small.decoder.yaml", epochs=2, test_batches=4, validation_batches=2
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
        tmp_path / "small.decoder.yaml", epochs=1, test_batches=1, validation_batches=1
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
        k: v for k, v in cfg["config"]["train"].items() if k != "validation_batches"
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
    assert "missing under 'decoder.train': validation_batches" in result.stderr


def test_train_cli_requires_the_loss_yaml(tmp_path):
    """`-t` without `-ls` must fail before anything is constructed."""
    decoder_yaml = _write_decoder_yaml(
        tmp_path / "small.decoder.yaml", epochs=1, test_batches=1, validation_batches=1
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
    loss_fn.combine(*loss_fn.terms(out, e)).backward()
    saq.optimizer.step()
    saq.optimizer.zero_grad(set_to_none=True)


def _fresh_run(epochs, seed=0):
    """A decoder seeded identically to every other `_fresh_run`, ready to train."""
    torch.manual_seed(seed)
    decoder, saq, loss_fn = _training_setup()
    saq.configure_optimizer(epochs)
    saq.train(True)
    torch.set_grad_enabled(True)
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
        straight_saq.scheduler.step()

    part, part_saq, part_loss = _fresh_run(4)
    for epoch in range(2):
        _train_step(part, part_saq, part_loss, seed=epoch)
        part_saq.scheduler.step()
    path = tmp_path / "train_state.pt"
    torch.save(part_saq.train_state(), path)

    resumed, resumed_saq, resumed_loss = _fresh_run(4)
    resumed_saq.load_train_state(
        torch.load(path, map_location="cpu", weights_only=True)
    )
    for epoch in range(2, 4):
        _train_step(resumed, resumed_saq, resumed_loss, seed=epoch)
        resumed_saq.scheduler.step()

    assert (
        resumed_saq.scheduler.get_last_lr()[0]
        == straight_saq.scheduler.get_last_lr()[0]
    )
    reference = straight_saq.state_dict()
    for key, value in resumed_saq.state_dict().items():
        assert torch.equal(value, reference[key]), key


def _run_stem(decoder):
    """The stem a training run would name its files after, for the given decoder."""
    from syndrilla.metric.metric import _train_stem

    return _train_stem(decoder)


def test_train_metrics_take_back_epoch_best_and_history(tmp_path):
    """The training half of `MetricState` must take back the run position it owns.

    This is the half of `last.pt` the decoder does not own: which epoch is next,
    which was best, and the history so far. `main.py`'s resume reads exactly these
    back out of the checkpoint.
    """
    from syndrilla.metric import MetricState

    class _Decoder:
        def load_train_state(self, state):
            self.state = state

    cfg = {
        "epochs": 4,
        "test_batches": 2,
        "validation_batches": 1,
        "error_random_seed": 0,
    }
    metrics = MetricState.train_initial(cfg, str(tmp_path), DECODER_YAML)
    metrics._decoder = (
        _Decoder()
    )  # what `train_resume_checkpoint` binds, without the run
    metrics._fingerprint = {"epochs": 4}

    path = tmp_path / "hand_over_last.pt"
    torch.save(
        {
            "epoch": 3,
            "best": 0.25,
            "history": [{"epoch": 1}, {"epoch": 2}],
            "fingerprint": {"epochs": 4},
        },
        path,
    )
    metrics.train_load_checkpoint(str(path), "cpu")

    assert metrics.epoch == 3
    assert metrics.best == 0.25
    assert metrics.history == [{"epoch": 1}, {"epoch": 2}]
    # two epochs of (2 train + 1 val) batches are behind us
    assert (metrics.epoch - 1) * metrics.period == 6


def _metrics_under(tmp_path, term_names, **overrides):
    """A training state metering a loss that declares `term_names`."""
    from syndrilla.metric import MetricState

    class _Loss:
        pass

    _Loss.term_names = term_names
    cfg = dict(
        {
            "epochs": 1,
            "test_batches": 1,
            "validation_batches": 1,
            "error_random_seed": 0,
        },
        **overrides,
    )
    metrics = MetricState.train_initial(cfg, str(tmp_path), DECODER_YAML)
    metrics.train_bind_loss(_Loss())
    return metrics


def _metered(metrics, phase="train"):
    """What one phase has accumulated, by the name each slot is keyed on."""
    return dict(zip(metrics.keys, metrics.acc[phase]))


def test_metrics_meter_the_terms_the_loss_declares(tmp_path):
    """The run is keyed by the bound loss's own term names, not by a fixed set.

    `lc`/`lp`/`ent` are the logical-centric loss's decomposition of its total. A second
    loss splits its total some other way, and the metric half has to meter that one
    without being edited, so the names come from the loss.
    """
    metrics = _metrics_under(tmp_path, ("a", "b"))

    assert metrics.keys == ("total", "a", "b", "class_err")
    metrics.train_set_hyperparameter(0)
    metrics.train_update_metric(1, (torch.tensor(1.0), 0.25, 0.75), 0.5)

    assert _metered(metrics) == {
        "total": 1.0,
        "a": 0.25,
        "b": 0.75,
        "class_err": 0.5,
    }


def test_a_loss_with_no_breakdown_is_metered_on_its_total(tmp_path):
    """A loss whose total has no parts worth logging declares none, and still runs."""
    metrics = _metrics_under(tmp_path, ())

    assert metrics.keys == ("total", "class_err")
    metrics.train_set_hyperparameter(0)
    metrics.train_update_metric(1, (2.0,), 0.5)

    assert _metered(metrics) == {"total": 2.0, "class_err": 0.5}


def test_a_loss_handing_back_the_wrong_number_of_terms_is_refused(tmp_path):
    """Each value lands in the slot its position picks, so a miscount cannot be silent.

    An undeclared term would be filed under the next term's name and the run would
    report numbers it never computed, which nothing downstream could detect.
    """
    metrics = _metrics_under(tmp_path, ("a", "b"))
    metrics.train_set_hyperparameter(0)

    with pytest.raises(ValueError, match="term"):
        metrics.train_update_metric(1, (1.0, 0.25), 0.5)


def test_a_loss_cannot_name_a_term_the_run_already_meters(tmp_path):
    """`total` and `class_err` are the run's own, so a loss reusing one is rejected."""
    with pytest.raises(ValueError, match="total"):
        _metrics_under(tmp_path, ("total",))


def test_logical_centric_declares_every_term_it_returns():
    """The loss's `term_names` is the contract the metric half meters it by.

    A name per value `terms` hands back, in that order, so a term added to the loss
    without a name for it is caught here rather than surfacing as a mislabelled column.
    """
    from syndrilla.loss.logical_centric import logical_centric

    decoder, saq = _make_decoder(SURFACE_MATRIX_YAML)
    saq.train()
    e, synd, H = _random_shots(saq)
    loss_fn = _make_loss(saq)

    assert logical_centric.create.term_names == ("lc", "lp", "ent")
    assert len(loss_fn.terms(decoder(_io_dict(saq, synd, H)), e)) == len(
        loss_fn.term_names
    )


def _sequence(metrics, batch_index, k=6):
    """The first `k` draws of the phase `batch_index` opens."""
    metrics.train_set_hyperparameter(batch_index)
    return torch.rand(k)


def test_every_epoch_trains_on_the_same_batches(tmp_path):
    """The training phase is a fixed set: epoch N draws what epoch 1 drew.

    With errors generated per batch, an unseeded stream would hand the model new noise
    every epoch. Pinning the training seed makes the training set finite and repeatable.
    """
    from syndrilla.metric import MetricState

    cfg = {
        "epochs": 4,
        "test_batches": 2,
        "validation_batches": 1,
        "error_random_seed": 7,
    }
    metrics = MetricState.train_initial(cfg, str(tmp_path), DECODER_YAML)

    first = _sequence(metrics, 0)  # epoch 1, first training batch
    metrics.epoch = 3
    third = _sequence(metrics, 3 * metrics.period)  # epoch 3, first training batch

    assert torch.equal(first, third)


def test_validation_draws_new_errors_each_epoch(tmp_path):
    """Validation is not the training set replayed, and not the same twice."""
    from syndrilla.metric import MetricState

    cfg = {
        "epochs": 4,
        "test_batches": 2,
        "validation_batches": 1,
        "error_random_seed": 7,
    }
    metrics = MetricState.train_initial(cfg, str(tmp_path), DECODER_YAML)

    metrics.epoch = 1
    val_1 = _sequence(metrics, cfg["test_batches"])
    train = _sequence(metrics, 0)
    metrics.epoch = 2
    val_2 = _sequence(metrics, metrics.period + cfg["test_batches"])

    assert not torch.equal(val_1, val_2), "validation replayed the same errors"
    assert not torch.equal(val_1, train), "validation replayed the training set"


def test_train_set_hyperparameter_reports_the_phase(tmp_path):
    """Seeding must not disturb which batches count as training."""
    from syndrilla.metric import MetricState

    cfg = {
        "epochs": 2,
        "test_batches": 2,
        "validation_batches": 1,
        "error_random_seed": 7,
    }
    metrics = MetricState.train_initial(cfg, str(tmp_path), DECODER_YAML)

    phases = [metrics.train_set_hyperparameter(i) for i in range(metrics.period * 2)]

    assert phases == ["train", "train", "val", "train", "train", "val"]
    # the phase it returns is the phase it is left in, so a caller can read it back
    # instead of asking the schedule a second time
    assert metrics.phase == "val"


def test_train_set_hyperparameter_puts_the_decoder_in_the_phase_it_opened(tmp_path):
    """The phase the metrics pick and the mode the decoder runs in are one decision.

    A validation batch that still built a graph, or a training batch that did not,
    would train on the wrong set while reporting the right one, so `train_set_hyperparameter` moves
    the bound decoder itself rather than leaving each caller to pair the two.
    """
    from syndrilla.metric import MetricState

    class _Decoder:
        training = None

        def train(self, training):
            self.training = training

    cfg = {
        "epochs": 2,
        "test_batches": 2,
        "validation_batches": 1,
        "error_random_seed": 7,
    }
    metrics = MetricState.train_initial(cfg, str(tmp_path), DECODER_YAML)
    decoder = _Decoder()
    metrics._decoder = decoder  # what `train_resume_checkpoint` binds, without the run

    modes = []
    try:
        for i in range(metrics.period):
            metrics.train_set_hyperparameter(i)
            modes.append((decoder.training, torch.is_grad_enabled()))
    finally:
        # the switch is global, so a val batch left mid-test would follow the process out
        torch.set_grad_enabled(True)

    assert modes == [(True, True), (True, True), (False, False)]


def test_neighbouring_run_seeds_do_not_share_streams(tmp_path):
    """`error_random_seed` and `error_random_seed + 1` must not produce the same training set."""
    from syndrilla.metric import MetricState

    def train_draw(seed):
        cfg = {
            "epochs": 4,
            "test_batches": 2,
            "validation_batches": 1,
            "error_random_seed": seed,
        }
        return _sequence(MetricState.train_initial(cfg, str(tmp_path), DECODER_YAML), 0)

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
        saq_block["train"], epochs=1, test_batches=2, validation_batches=1
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

    `train_compute_avg` writes the checkpoint before printing the line, so seeing the line
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
        tmp_path / "small.decoder.yaml", epochs=4, test_batches=3, validation_batches=1
    )

    straight_dir = tmp_path / "straight"
    assert _run_cli(straight_dir, *_train_argv(decoder_yaml)).returncode == 0

    resumed_dir = tmp_path / "resumed"
    _interrupt_after_epoch(resumed_dir, decoder_yaml, 3)
    # an interrupted run never reaches `train_save_checkpoint`, so the three finished epochs have
    # to be in the checkpoint, or they are lost
    partial = torch.load(resumed_dir / LAST_PT, map_location="cpu", weights_only=True)
    assert [entry["epoch"] for entry in partial["history"]] == [1, 2, 3]
    assert partial["epoch"] == 4

    finished = _run_cli(
        resumed_dir,
        *_train_argv(
            decoder_yaml,
            f"-ckpt={resumed_dir / RESULT_YAML}",
            f"-tckpt={resumed_dir / LAST_PT}",
        ),
    )
    assert finished.returncode == 0, finished.stderr

    # the resumed run must reach the same place, epoch by epoch and weight by weight.
    # The checkpoint's history is the whole curve, the epochs before the interrupt
    # included. Every recorded number is reproducible except how long the epoch took,
    # which is wall clock and is asserted on separately
    expected = torch.load(straight_dir / LAST_PT, map_location="cpu", weights_only=True)
    actual = torch.load(resumed_dir / LAST_PT, map_location="cpu", weights_only=True)
    resumed_history = actual["history"]
    straight_history = expected["history"]
    assert all(entry.pop("time") > 0 for entry in resumed_history)
    assert all(entry.pop("time") > 0 for entry in straight_history)
    assert resumed_history == straight_history
    for key, value in expected["state_dict"].items():
        assert torch.equal(actual["state_dict"][key], value), key


def test_resume_rejects_a_changed_schedule(tmp_path):
    """A checkpoint from a different schedule must be refused, not silently resumed."""
    decoder_yaml = _write_decoder_yaml(
        tmp_path / "small.decoder.yaml", epochs=2, test_batches=2, validation_batches=1
    )
    assert _run_cli(tmp_path, *_train_argv(decoder_yaml)).returncode == 0

    changed = _write_decoder_yaml(
        tmp_path / "changed.decoder.yaml",
        epochs=2,
        test_batches=5,
        validation_batches=1,
    )
    result = _run_cli(
        tmp_path,
        *_train_argv(
            changed, f"-ckpt={tmp_path / RESULT_YAML}", f"-tckpt={tmp_path / LAST_PT}"
        ),
    )
    assert result.returncode != 0
    assert "test_batches" in result.stderr


def test_resume_rejects_a_changed_setup(tmp_path):
    """A resume must be held to what the run was measured on, not only its schedule.

    The `-t` counterpart of what `validate_checkpoint` refuses a decode checkpoint on:
    a different parity-check matrix, physical error rate or dtype. A training run adds
    its objective to that list. Any of them changed and the loaded curve and weights
    would go on being accumulated against something else, so each must stop the run and
    name the field that moved. One trained checkpoint, four ways of not matching it.
    """
    decoder_yaml = _write_decoder_yaml(
        tmp_path / "small.decoder.yaml", epochs=2, test_batches=2, validation_batches=1
    )
    assert _run_cli(tmp_path, *_train_argv(decoder_yaml)).returncode == 0

    # a narrower sweep than the shipped [0.01, 0.20, 9]
    error_yaml = tmp_path / "narrow.error.yaml"
    error_cfg = _plain(read_yaml(get_path(TRAIN_ERROR_YAML)))
    error_cfg["error"]["rate"] = [0.05, 0.20, 9]
    error_yaml.write_text(yaml.safe_dump(error_cfg))

    dtype_yaml = tmp_path / "float64.decoder.yaml"
    decoder_cfg = yaml.safe_load(decoder_yaml.read_text())
    decoder_cfg["decoder"]["dtype"] = "float64"
    dtype_yaml.write_text(yaml.safe_dump(decoder_cfg))

    loss_yaml = tmp_path / "reweighted.loss.yaml"
    loss_cfg = _plain(read_yaml(get_path(LOSS_YAML)))
    loss_cfg["loss"]["lambda_lp"] = 0.5
    loss_yaml.write_text(yaml.safe_dump(loss_cfg))

    for field, override in (
        ("physical_error_rate", f"-e={error_yaml}"),
        ("dtype", f"-d={dtype_yaml}"),
        ("loss_lambda_lp", f"-ls={loss_yaml}"),
        ("H_file_name", f"-m={TORIC_MATRIX_YAML}"),
    ):
        # the override goes last, so argparse reads it over the same flag `_train_argv`
        # already set
        result = _run_cli(
            tmp_path,
            *_train_argv(
                decoder_yaml,
                f"-ckpt={tmp_path / RESULT_YAML}",
                f"-tckpt={tmp_path / LAST_PT}",
                override,
            ),
        )
        assert result.returncode != 0, f"{field} was resumed rather than refused"
        assert field in result.stderr, result.stderr


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


@pytest.mark.parametrize("given, missing", [("-ckpt", "-tckpt"), ("-tckpt", "-ckpt")])
def test_resume_needs_both_checkpoints(tmp_path, given, missing):
    """A training run resumes on the pair or on neither, never on half of it.

    `-tckpt` is where the run got to and `-ckpt` is what it was run as. Accepting one
    alone would resume against half of what a resume is checked against, so the flag
    that was left out is named rather than defaulted to.
    """
    decoder_yaml = _write_decoder_yaml(
        tmp_path / "small.decoder.yaml", epochs=2, test_batches=2, validation_batches=1
    )
    assert _run_cli(tmp_path, *_train_argv(decoder_yaml)).returncode == 0

    path = tmp_path / (RESULT_YAML if given == "-ckpt" else LAST_PT)
    result = _run_cli(tmp_path, *_train_argv(decoder_yaml, f"{given}={path}"))
    assert result.returncode != 0
    assert missing in result.stderr, result.stderr


def test_resume_checks_the_result_yaml_it_was_given(tmp_path):
    """`-ckpt` is validated in its own right, not carried along by a matching `-tckpt`.

    The yaml records the setup and the `*_last.pt` records the setup and the state, so
    a pair naming two different runs is caught only if the yaml is held to this run
    too. Here the checkpoint is this run's and the yaml is another run's, which is the
    one arrangement the `*_last.pt` check cannot see.
    """
    decoder_yaml = _write_decoder_yaml(
        tmp_path / "small.decoder.yaml", epochs=2, test_batches=2, validation_batches=1
    )
    assert _run_cli(tmp_path, *_train_argv(decoder_yaml)).returncode == 0

    other_dir = tmp_path / "other"
    other_yaml = _write_decoder_yaml(
        tmp_path / "other.decoder.yaml", epochs=2, test_batches=3, validation_batches=1
    )
    assert _run_cli(other_dir, *_train_argv(other_yaml)).returncode == 0

    result = _run_cli(
        tmp_path,
        *_train_argv(
            decoder_yaml,
            f"-ckpt={other_dir / RESULT_YAML}",
            f"-tckpt={tmp_path / LAST_PT}",
        ),
    )
    assert result.returncode != 0
    assert "test_batches" in result.stderr, result.stderr


def test_resume_refuses_a_yaml_that_is_not_a_training_result(tmp_path):
    """A decode run's result yaml under `-ckpt` must be named as the wrong file.

    Both modes write a `-r` yaml and only the training one carries `train_full`, so
    the one that cannot describe a training run is refused by what it is missing.
    """
    decoder_yaml = _write_decoder_yaml(
        tmp_path / "small.decoder.yaml", epochs=2, test_batches=2, validation_batches=1
    )
    assert _run_cli(tmp_path, *_train_argv(decoder_yaml)).returncode == 0

    decode_yaml = tmp_path / "decode_result.yaml"
    decode_yaml.write_text(yaml.safe_dump({"decoder_full": {"batch size": 16}}))

    result = _run_cli(
        tmp_path,
        *_train_argv(
            decoder_yaml, f"-ckpt={decode_yaml}", f"-tckpt={tmp_path / LAST_PT}"
        ),
    )
    assert result.returncode != 0
    assert "train_full" in result.stderr, result.stderr


@pytest.mark.parametrize(
    "matrix_yaml, expected",
    [
        # the qubit count is read off the matrix, never turned into a distance
        (TORIC_MATRIX_YAML, "saq_hx_n200"),
        # surface_5's matrix carries 41 columns, which is also not d5
        (SURFACE_MATRIX_YAML, "saq_hx_n41"),
    ],
)
def test_run_stem_names_the_configuration(matrix_yaml, expected):
    """A run names its files after what produced them, and never guesses a distance."""
    _, saq = _make_decoder(matrix_yaml)

    assert _run_stem(saq) == expected


def test_code_type_is_rejected_rather_than_ignored():
    """`code_type` was removed; a config still setting it must be told, not ignored.

    The family is measured from the matrix, so a declared one could only ever disagree
    with it. Silently dropping the key would leave that disagreement invisible.
    """
    with pytest.raises(ValueError, match="code_type"):
        _make_decoder(SURFACE_MATRIX_YAML, code_type="toric")


def test_run_stem_separates_two_configurations():
    """Two configs trained into one run dir must not overwrite each other's weights."""
    _, surface = _make_decoder(SURFACE_MATRIX_YAML)
    _, toric = _make_decoder(TORIC_MATRIX_YAML)

    assert _run_stem(surface) != _run_stem(toric)


def test_best_pt_stays_bare_weights_and_last_pt_still_decodes(tmp_path):
    """`best.pt` must stay a portable state_dict; `last.pt` must still load to decode."""
    decoder_yaml = _write_decoder_yaml(
        tmp_path / "small.decoder.yaml", epochs=1, test_batches=2, validation_batches=1
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
# `logical_logits` / `logical_prior` are written per forward row and are not unfolded
# by RoundFlattenWrapper, so a multi-round batch reaches the loss with the llr at
# [B, d, n] and the logical head at [B*d, 2^k]. A second channel is read as a second
# round for the same reason. Nothing checks the shape up front, so what stops such a
# run is the mismatch itself, raised by the loss.
# --------------------------------------------------------------------------- #


def test_train_cli_does_not_train_on_a_batch_shape_saq_cannot_learn_from(tmp_path):
    """`-t` on a multi-round measurer must fail rather than train on paired-up shapes.

    The rows are what disagree: the loss is handed one logical row per forward row,
    `rounds` times as many as the targets it is scored against. This is the run
    failing where the mismatch lands, not being refused for a shape it declared.
    """
    decoder_yaml = _write_decoder_yaml(
        tmp_path / "small.decoder.yaml", epochs=1, test_batches=1, validation_batches=1
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
    assert "batch_size" in raised[-1], raised[-1]


def test_decoder_describes_itself_in_the_resume_fingerprint():
    """Each half of the fingerprint must come from whoever owns it, not the metrics.

    `MetricState` owns the schedule and the batch size; what algorithm this is, what
    code shape it was built for, what dtype and device it runs on and what optimizer
    settings it will use are the decoder's to state, and what weights the objective is
    the loss's. The metrics merge the halves rather than reaching into either.
    """
    from syndrilla.loss import create_loss
    from syndrilla.metric.metric import _train_fingerprint

    _, saq = _make_decoder(SURFACE_MATRIX_YAML)
    model = saq.train_fingerprint()
    assert model == {
        "algo": "saq",
        "n": saq.n,
        "m": saq.m,
        "k": saq.k,
        "dtype": str(saq.dtype),
        "device": str(saq.device),
        "lr": saq.lr,
        "weight_decay": saq.weight_decay,
        "min_lr": saq.min_lr,
    }

    cfg = {
        "epochs": 4,
        "test_batches": 2,
        "validation_batches": 1,
        "error_random_seed": 0,
    }
    loss = create_loss(LOSS_YAML, decoder=saq)
    merged = _train_fingerprint(
        cfg, saq, 16, [0.01, 0.20, 9], "/codes/surface_5.hx.alist", loss
    )
    # every model key survives the merge, and the schedule half is added to it
    assert merged.items() >= model.items()
    assert merged["batch_size"] == 16
    assert all(merged[k] == v for k, v in cfg.items())
    # what the run decodes, and at what noise: the same ground the decode side's
    # `validate_checkpoint` holds a resumed run to
    assert merged["H_file_name"] == "/codes/surface_5.hx.alist"
    assert merged["physical_error_rate"] == [0.01, 0.20, 9]
    # and the objective: which loss it is, and the block that weights its terms
    assert merged["loss_function"] == "logical_centric"
    assert merged["loss_lambda_lp"] == loss.lambda_lp


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
        model={"d_model": 64, "N_dec": 3, "h": 8, "dropout": 0.25, "no_mask": 1},
        cpnd={"enable": False, "passes": 4},
        optimizer={"lr": 1.0e-3, "weight_decay": 2.0e-7, "min_lr": 3.0e-6},
    )
    # d_model is checked where it lands, on the token embeddings, rather than on a copy
    # of the setting: an architecture built at some other width would still pass that
    assert (saq.learnable_embed_S.shape[1], len(saq.layers)) == (64, 3)
    assert saq.N_dec == 3
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
