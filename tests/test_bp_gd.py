"""Guided-decimation rules, budgets and the shared sum-product steps."""
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from syndrilla.decoder.bp_gd.bp_gd import create as Gd
from syndrilla.decoder.bp_sum_prod.bp_sum_prod import create as Spa
from syndrilla.decoder import create_decoder
from syndrilla.matrix import load_matrices
from syndrilla.matrix.matrix import dense_to_index_format
from syndrilla.utils import read_yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tiny(**knobs):
    h = torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0],
                      [0, 1, 1, 1, 0, 0]], dtype=torch.float64)
    index = dense_to_index_format(h, "cpu")
    bundle = SimpleNamespace(Hx_matrix=None, Hz_matrix=None, lx_matrix=None, lz_matrix=None,
                             select=lambda _: index)
    cfg = dict(device={"device_type": "cpu"}, dtype="float64", max_iter=12,
               round_iters=2, compile=False, rebatch_opt=False, **knobs)
    io = dict(llr0=torch.full((9, 6), 2., dtype=torch.float64),
              synd=torch.tensor([[0, 0, 0]] + [[0, 1, 1]] * 8))
    return cfg, bundle, io


def test_factory_defaults_and_example():
    cfg, bundle, io = tiny()
    dec = create_decoder(cfg=dict(algorithm="bp_gd", device=cfg["device"], dtype=cfg["dtype"],
        config={key: value for key, value in cfg.items() if key not in ("device", "dtype")}), bundle=bundle)[0]
    out = dec(dict(io))
    assert dec.algo == "bp_gd" and out["rounds"].shape == (9,)
    default = Gd(dict(device=cfg["device"], rebatch_opt=False), bundle=bundle)
    assert default.round_iters == 10 and default.llr_max == 25
    assert default.saturation == 25
    assert default.max_rounds == 6 and default.order == "reliability"
    assert default.random_machine == "sobol"
    example = read_yaml(str(ROOT / "examples/alist/bp_gd_hx.decoding.yaml"))["decoding"]
    assert create_decoder(cfg=example, bundle=bundle)[0].max_iter == 181


@pytest.mark.parametrize("key,value", [
    ("round_iters", 0), ("round_iters", True), ("round_iters", 1.5),
    ("max_iter", 0), ("max_iter", False), ("max_iter", 1),
    ("max_rounds", 0), ("max_rounds", True), ("max_rounds", 2.),
    ("llr_max", 0), ("llr_max", -1), ("llr_max", float("nan")),
    ("llr_max", float("inf")), ("llr_max", True), ("order", "other"),
    ("saturation", 0), ("saturation", -1), ("saturation", True),
    ("saturation", float("nan")), ("saturation", float("inf")), ("saturation", "25"),
    ("random_machine", "other"), ("dtype", "float16"),
    ("check_type", "other"), ("early_termination", 1),
])
def test_bad_knob_rejected(key, value):
    cfg, bundle, _ = tiny()
    cfg[key] = value
    with pytest.raises(ValueError, match=key):
        Gd(cfg, bundle=bundle)


def test_unknown_key_before_setup():
    with pytest.raises(ValueError, match="unsupported key misspelled for bp_gd"):
        Gd({"misspelled": 1})


@pytest.mark.parametrize("reuse", [True, False])
def test_no_decimation_equals_sum_product_bit_exact(reuse):
    cfg, bundle, io = tiny()
    cfg.update(round_iters=12, reuse_buffers=reuse)
    gd, plain = Gd(dict(cfg, saturation=1e6), bundle=bundle), Spa({key: value for key, value in cfg.items() if key != "round_iters"}, bundle=bundle)
    actual, expected = gd(dict(io)), plain(dict(io))
    assert not actual["converge"][1:].any()
    assert not actual["rounds"].any()
    for key in ("e_v", "llr", "iter", "converge", "defer"):
        assert torch.equal(actual[key], expected[key]), key


def test_saturation_clips_outgoing_messages_only():
    h = torch.tensor([[1, 1]], dtype=torch.float64)
    index = dense_to_index_format(h, "cpu")
    bundle = SimpleNamespace(Hx_matrix=None, Hz_matrix=None, lx_matrix=None, lz_matrix=None,
                             select=lambda _: index)
    gd = Gd(dict(device={"device_type": "cpu"}, dtype="float64", max_iter=1,
                round_iters=1, saturation=2., compile=False, rebatch_opt=False), bundle=bundle)
    gd.batch_size, gd.i, gd._gd_buffers = 1, 1, None
    prior = torch.tensor([[12., -9., float("inf")]], dtype=torch.float64)
    messages = torch.tensor([[[7., -4.]]], dtype=torch.float64)
    posterior = torch.tensor([[19., -13., float("inf")]], dtype=torch.float64)
    incoming, out, _, _ = gd._bp_step(messages, posterior, prior, torch.zeros(1, 1, 1, dtype=torch.bool))
    torch.testing.assert_close(incoming, torch.tensor([[[-2., 2.]]], dtype=torch.float64), rtol=0, atol=1e-14)
    torch.testing.assert_close(out[:, :2], torch.tensor([[10., -7.]], dtype=torch.float64), rtol=0, atol=1e-14)
    assert prior.tolist() == [[12., -9., float("inf")]]


def test_large_saturation_matches_pre_clip_surface_outputs():
    bundle = load_matrices(read_yaml(str(ROOT / "examples/alist/surface_10.matrix.yaml"))["matrix"],
                           torch.device("cpu"), torch.float64)
    h = bundle.select("hx")[3].to_dense().double()
    error = (torch.rand((512, h.shape[1]), generator=torch.Generator().manual_seed(269)) < .05).double()
    io = dict(synd=(error @ h.T) % 2, llr0=torch.full_like(error, math.log(19)))
    gd = Gd(dict(device={"device_type": "cpu"}, dtype="float64", max_iter=181,
                round_iters=1, saturation=1e6, rebatch_opt=False, compile=False), bundle=bundle)
    out = gd(io)
    expected = {
        "e_v": "217bd290a8cc8125df7c90b33338bf816a692755a0bd45d72e41cfafe328c9b7",
        "llr": "169058fce063d2969707fe5719f8b7f6cc41b489cb4e392c1ab093d429d3e35a",
        "converge": "50c8ee0d29539257e571ef1c9a1bb54715120abdd20b180eae3fd052e3bb79e7",
        "iter": "b7508e0a6a302a4698d5658d0f1886f350ea237751e93f255a023282bf54325d",
        "rounds": "315ab19393b233e5eab20664938df3391533d77349a67bebbb8d22412f59e93d",
        "defer": "076a27c79e5ace2a3d47f9dd2e83e4ff6ea8872b3c2218f66c92b89b55f36560",
    }
    for key, digest in expected.items():
        assert hashlib.sha256(out[key].numpy().tobytes()).hexdigest() == digest, key


def test_hand_decimation_ties_zero_sign_and_retained_messages():
    cfg, bundle, _ = tiny()
    gd = Gd(cfg, bundle=bundle)
    prior = torch.tensor([[2., 3., 4., 5., 6., 7., float("inf")],
                          [0., 0., 0., 0., 0., 0., float("inf")],
                          [2., 3., 4., 5., 6., 7., float("inf")]], dtype=torch.float64)
    posterior = torch.tensor([[4., -9., 9., 1., 2., 0., float("inf")],
                              [0., 0., 0., 0., 0., 0., float("inf")],
                              [-9., 1., 2., 3., 4., 5., float("inf")]], dtype=torch.float64)
    gd._gd_current = prior[:, :-1].clone()
    gd._gd_undecimated = torch.ones(3, 6, dtype=torch.bool)
    gd._gd_rounds = torch.zeros(3, dtype=torch.long)
    messages = gd._new_messages(3)
    before = messages.clone()
    gd._decimate(torch.arange(3), posterior, prior, messages, torch.tensor([True, True, False]))
    assert gd.last_gd_nodes.tolist() == [1, 0, -1]
    assert prior[0].tolist() == [2, -25, 4, 5, 6, 7, float("inf")]
    assert prior[1].tolist() == [-25, 0, 0, 0, 0, 0, float("inf")]
    assert gd._gd_rounds.tolist() == [1, 1, 0]
    assert (~gd._gd_undecimated).sum(1).tolist() == [1, 1, 0]
    assert not gd._gd_undecimated[0, 1] and not gd._gd_undecimated[1, 0]
    assert torch.equal(messages, before)
    assert torch.equal(posterior, prior)
    posterior[0, 2] = 10
    gd._decimate(torch.arange(3), posterior, prior, messages, torch.tensor([True, False, False]))
    assert gd.last_gd_nodes.tolist() == [2, -1, -1]
    assert prior[0, 1] == -25 and prior[0, 2] == 25


@pytest.mark.parametrize("compact", [0, .5, .75])
def test_prior_sign_persists_and_nodes_are_not_reused(compact):
    cfg, bundle, io = tiny(compact_frac=compact)
    io["synd"][:6] = 0
    gd = Gd(cfg, bundle=bundle)
    original = gd._decimate
    history = []

    def record(rows, posterior, prior, messages, active):
        previous = gd._gd_current.clone()
        fixed = ~gd._gd_undecimated.clone()
        original(rows, posterior, prior, messages, active)
        assert torch.equal(gd._gd_current[fixed], previous[fixed])
        assert torch.equal(prior[:, :-1][fixed[rows]], previous[rows][fixed[rows]])
        history.append(gd.last_gd_nodes.clone())

    gd._decimate = record
    out = gd(dict(io))
    assert len(history) == 5
    assert out["rounds"].tolist() == [0] * 6 + [5] * 3
    assert out["iter"].tolist() == [1] * 6 + [12] * 3
    assert torch.equal(out["llr"] <= 0, out["e_v"].bool())
    assert torch.equal(out["e_v"][0], torch.zeros(6, dtype=torch.float64))


@pytest.mark.parametrize("maximum,rounds,expected_iter,edits", [(20, 2, 6, 1), (14, 20, 12, 3), (21, 99, 18, 5)])
def test_stops_at_round_and_iteration_limits(maximum, rounds, expected_iter, edits):
    cfg, bundle, io = tiny()
    cfg.update(max_iter=maximum, max_rounds=rounds, round_iters=3)
    out = Gd(cfg, bundle=bundle)(dict(io))
    assert not out["converge"][1:].any()
    assert out["iter"][1:].tolist() == [expected_iter] * 8
    assert out["rounds"][1:].tolist() == [edits] * 8


@pytest.mark.parametrize("early,expected", [(True, 1), (False, 3)])
def test_convergence_check_frequency(early, expected):
    cfg, bundle, io = tiny()
    cfg.update(round_iters=3, early_termination=early)
    out = Gd(cfg, bundle=bundle)(dict(io))
    assert out["converge"][0] == 1 and out["iter"][0] == expected
    assert out["rounds"][0] == 0
    wrapped = create_decoder(cfg=dict(algorithm="bp_gd", device=cfg.pop("device"),
        dtype=cfg.pop("dtype"), config=cfg), bundle=bundle)[0]
    round_io = {key: value[:2, None].expand(-1, 2, -1).clone() for key, value in io.items()}
    result = wrapped(round_io)
    assert result["rounds"].shape == result["iter"].shape == (2, 2)


def test_random_sobol_reproduces_prior_and_outputs():
    cfg, bundle, io = tiny(order="random", random_machine="sobol")
    gd = Gd(cfg, bundle=bundle)
    a = gd(dict(io))
    fixed = gd._gd_current.clone()
    torch.manual_seed(9381)
    b = gd(dict(io))
    for key in ("e_v", "llr", "converge", "iter", "rounds"):
        assert torch.equal(a[key], b[key]), key
    assert torch.equal(fixed, gd._gd_current)


def test_all_groups_off_preserve_hard_results():
    cfg, bundle, io = tiny()
    reference = Gd(cfg, bundle=bundle)(dict(io))
    groups = {key: False for key in ("pruning_opt", "fusion_opt", "mapping_opt", "gather_opt", "memory_opt", "rebatch_opt")}
    actual = Gd(dict(cfg, **groups), bundle=bundle)(dict(io))
    for key in ("e_v", "converge", "iter", "rounds"):
        assert torch.equal(actual[key], reference[key]), key


def test_rebatch_cap_defers_open_shots_without_decimation():
    from syndrilla.decoder.decoder import RebatchSpeedup

    cfg, bundle, io = tiny()
    io["synd"][:6] = 0
    gd = Gd(cfg, bundle=bundle)
    gd.cap = RebatchSpeedup()
    gd.cap.frac = .5
    out = gd(dict(io))
    assert gd.cap_active_last
    assert out["defer"].tolist() == [False] * 6 + [True] * 3
    assert out["iter"].tolist() == [1] * 9
    assert not out["rounds"].any()


def test_surface_decimation_beats_sum_product():
    bundle = load_matrices(read_yaml(str(ROOT / "examples/alist/surface_10.matrix.yaml"))["matrix"],
                           torch.device("cpu"), torch.float64)
    h = bundle.select("hx")[3].to_dense().double()
    error = (torch.rand((512, h.shape[1]), generator=torch.Generator().manual_seed(269)) < .05).double()
    io = dict(synd=(error @ h.T) % 2, llr0=torch.full_like(error, math.log(19)))
    cfg = dict(device={"device_type": "cpu"}, dtype="float64", max_iter=181, rebatch_opt=False, compile=False)
    plain, guided = Spa(cfg, bundle=bundle)(dict(io)), Gd(dict(cfg, round_iters=1), bundle=bundle)(dict(io))
    plain_ok, guided_ok = plain["converge"].bool(), guided["converge"].bool()
    rescues = int((~plain_ok & guided_ok).sum())
    harms = int((plain_ok & ~guided_ok).sum())
    assert int(guided_ok.sum()) > int(plain_ok.sum()), (
        f"guided={int(guided_ok.sum())}, plain={int(plain_ok.sum())}, rescues={rescues}, harms={harms}")
    assert int(guided["converge"].sum()) > 0
    assert torch.equal((guided["e_v"][guided["converge"].bool()] @ h.T) % 2, io["synd"][guided["converge"].bool()])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_compiled_step_matches_eager():
    cfg, bundle, io = tiny()
    cfg["device"] = {"device_type": "cuda"}
    eager = Gd(cfg, bundle=bundle)(dict(io))
    compiled = Gd(dict(cfg, compile=True), bundle=bundle)
    assert compiled.compile
    actual = compiled(dict(io))
    for key in ("e_v", "converge", "iter", "rounds"):
        assert torch.equal(actual[key], eager[key]), key
