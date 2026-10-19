"""CPU and CUDA checks for the lottery study knobs."""

import math
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

from syndrilla.decoder.bp_lottery import bp_lottery as lottery
from syndrilla.decoder.bp_lottery import bp_lottery_cuda as lottery_cuda
from syndrilla.decoder.bp_lottery_layered import bp_lottery_layered as layered
from syndrilla.decoder.bp_lottery_layered import bp_lottery_layered_cuda as layered_cuda
from syndrilla.decoder.bp_lottery_quant import bp_lottery_quant as quant
from syndrilla.decoder.bp_lottery_quant import bp_lottery_quant_cuda as quant_cuda
from syndrilla.decoder.bp_lottery_policy import bp_lottery_policy as policy
from syndrilla.decoder.knobs import BASE_CONFIG_KEYS, GROUP_OFF
from syndrilla.matrix.matrix import dense_to_index_format


LOTTERY_ALGORITHMS = ("bp_lottery", "bp_lottery_pre", "bp_lottery_quant",
                      "bp_lottery_layered", "bp_lottery_policy")


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def bundle(H):
    index = dense_to_index_format(H, torch.device("cpu"))
    return SimpleNamespace(Hx_matrix=None, Hz_matrix=None, lx_matrix=None,
                           lz_matrix=None, select=lambda _: index)


def decoder(**extra):
    H = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1], [0, 0, 1, 0]])
    cfg = dict(device={"device_type": "cpu"}, dtype="float64", max_iter=8,
               compile=False, rebatch_opt=False)
    cfg.update(extra)
    dec = lottery.create(cfg, bundle=bundle(H))
    dec.i, dec._B, dec.batch_size = 1, 3, 3
    dec._hook_rows = torch.arange(3)
    dec._active = torch.tensor([True, False, True])
    dec._draw_r = lambda n: torch.full((n,), 0.8, dtype=dec.dtype)
    dec.r_v = torch.full((8, 1), 0.2, dtype=dec.dtype)
    llr = torch.tensor([[0.5, -2, 1, 4, float("inf")]], dtype=dec.dtype).repeat(3, 1)
    syndrome = torch.tensor([[1, 1, 1], [1, 1, 1], [0, 0, 0]], dtype=dec.dtype)
    return dec, llr, syndrome


def assert_negation(dec, before, after, variable):
    expected = before.clone()
    expected[0, variable] *= -1
    assert torch.equal(after, expected)
    assert dec.last_flip_valid.tolist() == [True, False, False]


@pytest.mark.parametrize("value,cn,vn", [("rand_unsat", 2, 2), ("max_err_unsat", 1, 2)])
def test_cn_pick(value, cn, vn):
    dec, llr, syndrome = decoder(cn_pick=value, cn_pick_tie_break=False)
    dec._flip_e_v = torch.tensor([[0, 0, 1, 1, 1]], dtype=dec.dtype).repeat(3, 1)
    before = llr.clone()
    dec.sign_flip_cn_rand_new(syndrome, torch.zeros_like(syndrome), llr)
    assert dec.last_flip_cn[0] == cn
    assert dec.last_flip_vn[0] == vn
    assert_negation(dec, before, llr, vn)


@pytest.mark.parametrize("flag,cn,vn", [(False, 0, 1), (True, 1, 2),
                                      pytest.param(None, 1, 2, id="default")])
def test_max_err_unsat_random_tie_break(flag, cn, vn):
    dec, llr, syndrome = decoder(cn_pick="max_err_unsat",
                                **({"cn_pick_tie_break": flag} if flag is not None else {}))
    e_v = torch.tensor([[1, 1, 0, 1, 0]], dtype=dec.dtype).repeat(3, 1)
    before = llr.clone()
    dec._iter_hook(1, llr, e_v, dec._active, syndrome)
    assert dec.last_flip_cn[:2].tolist() == [cn, cn]
    assert dec.last_flip_vn[:2].tolist() == [vn, vn]
    assert_negation(dec, before, llr, vn)


@pytest.mark.parametrize("flag", [False, True])
def test_rand_unsat_ignores_cn_tie_break(flag):
    dec, llr, syndrome = decoder(cn_pick="rand_unsat", cn_pick_tie_break=flag)
    before = llr.clone()
    dec._iter_hook(1, llr, torch.zeros_like(llr), dec._active, syndrome)
    assert dec.last_flip_cn.tolist() == [2, 2, 0]
    assert dec.last_flip_vn.tolist() == [2, 2, 0]
    assert_negation(dec, before, llr, 2)


@pytest.mark.parametrize("value", [0, 1, None, "true"])
@pytest.mark.parametrize("module", [lottery, lottery_cuda, layered, layered_cuda, quant, quant_cuda])
def test_invalid_cn_pick_tie_break(module, value):
    with pytest.raises(ValueError, match="cn_pick_tie_break"):
        module.create({"cn_pick_tie_break": value})


def test_factory_rejects_top_level_cn_tie_break():
    from syndrilla.decoder.decoder import create_decoder

    with pytest.raises(ValueError, match="at the top level"):
        create_decoder(cfg={"algorithm": "bp_lottery", "cn_pick_tie_break": True})


@pytest.mark.parametrize("algorithm", LOTTERY_ALGORITHMS)
@pytest.mark.parametrize("suffix", ["", "_cuda"])
@pytest.mark.parametrize("key", ["vn_pik", "pre", "pre_typo", "pre_parallel_typo"])
def test_unknown_lottery_key_rejected_before_device_setup(algorithm, suffix, key):
    module = import_module(f"syndrilla.decoder.{algorithm}.{algorithm}{suffix}")
    with pytest.raises(ValueError, match=f"unsupported key {key} for {algorithm}"):
        module.create({key: "random"})


@pytest.mark.parametrize("first", ["vn_pik", "vn_pick_k"])
def test_first_illegal_key_is_reported(first):
    with pytest.raises(ValueError, match=first) as error:
        lottery.reject_unknown_keys({first: 2, "another_typo": True})
    assert "another_typo" not in str(error.value)


@pytest.mark.parametrize("suffix", ["", "_cuda"])
@pytest.mark.parametrize("key,value", [
    ("int_width", 3), ("frac_width", 4), ("sign_flip_policy", "Proposed"),
    ("pre_parallel_max_lambda", 2), ("pre_parallel_max_iter", 2),
    ("pre_parallel_start_iter", 2),
])
def test_vanilla_lottery_rejects_subtype_keys(suffix, key, value):
    module = import_module(f"syndrilla.decoder.bp_lottery.bp_lottery{suffix}")
    with pytest.raises(ValueError, match=key):
        module.create({key: value})


@pytest.mark.parametrize("algorithm", LOTTERY_ALGORITHMS)
@pytest.mark.parametrize("backend", ["cpu", "pytorch", "cuda"])
def test_factory_accepts_full_lottery_config(algorithm, backend, seeded_inputs):
    from syndrilla.decoder.decoder import create_decoder

    if backend != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    settings = {key: value for members in GROUP_OFF.values() for key, value in members.items()
                if key in BASE_CONFIG_KEYS}
    settings.update({**seeded_inputs["cfg"],
        "early_termination": True,
        "pruning_opt": False, "fusion_opt": False, "mapping_opt": False,
        "gather_opt": False, "memory_opt": False,
        "compact_frac": 0.5, "reuse_buffers": True, "c2v_gather": True,
        "cn_sign_parity": True, "persistent_kernel": False,
        "host_check_every": 2, "skip_converged": False, "warp_per_check": False,
        "fuse_vn": False, "f64_int_compare": False, "vn_gather": False,
        "edge_layout": "csr", "random_machine": "system",
        "cn_pick": "all_unsat", "vn_pick": "random",
        "cn_pick_tie_break": False, "vn_pick_tie_break": False,
        "flip_phase_iters": [], "flip_mode": "lottery"})
    shared = {key: settings.pop(key) for key in ("device", "dtype")}
    shared.update(algorithm=algorithm, check_type="hx", force_pytorch=backend != "cuda",
                  rebatch_opt_params={"kl_min": 2})
    shared["device"] = {"device_type": "cpu" if backend == "cpu" else "cuda"}
    if algorithm == "bp_lottery_policy":
        settings["sign_flip_policy"] = "global_optimal"
    else:
        settings.update(flip_phase_iters=[2], flip_mode=["lottery", "pre"],
                        pre_max_lambda=2, pre_max_iter=2)
    if algorithm == "bp_lottery_pre":
        settings.update(pre_parallel_max_lambda=3, pre_parallel_max_iter=2,
                        pre_parallel_start_iter=8)
    if algorithm == "bp_lottery_quant":
        settings.update(int_width=4, frac_width=5)
    dec = create_decoder(cfg={**shared, "config": settings},
                         bundle=bundle(seeded_inputs["H"]))[0].decoder
    assert dec.algo == algorithm
    assert dec.max_iter == 8 and dec.check_type == "hx"
    assert dec.cap is None
    if backend != "cuda" or algorithm == "bp_lottery_layered":
        assert dec.compact_frac == 0.5 and dec.reuse_buffers
        assert dec.c2v_gather and dec.cn_sign_parity
        assert not dec.compile
    else:
        assert dec._host_check_every == 2 and not dec._skip_converged
        assert not dec._warp_per_check and not dec._fuse_vn and not dec._vn_gather
    if algorithm == "bp_lottery_policy":
        assert dec.sign_flip_policy == "global_optimal"
    else:
        assert dec.flip_phase_iters == [2]
        assert [phase["flip_mode"] for phase in dec.phase_knobs] == ["lottery", "pre"]
        assert (dec.pre_max_lambda, dec.pre_max_iter) == (2, 2)
    if algorithm == "bp_lottery_pre":
        assert dec.parallel_enabled
        assert (dec.max_lambda, dec.pre_it, dec.parallel_start) == (3, 2, 8)
    if algorithm == "bp_lottery_quant":
        assert (dec.intwidth, dec.fracwidth) == (4, 5)


@pytest.mark.parametrize("hard,unsat,cn,vn", [
    ([0, 0, 1, 1, 1], [1, 1, 1], 1, 2),
    ([0, 1, 0, 0, 1], [1, 1, 1], 0, 1),
    ([0, 0, 1, 1, 1], [1, 0, 1], 2, 2),
])
def test_max_err_unsat_cn_pick_hook(hard, unsat, cn, vn):
    dec, llr, _ = decoder(cn_pick="max_err_unsat", cn_pick_tie_break=False)
    e_v = torch.tensor([hard], dtype=dec.dtype).repeat(3, 1)
    syndrome = dec.syndrome_estimation(e_v)
    syndrome[0] = (syndrome[0] + torch.tensor(unsat)) % 2
    before = llr.clone()
    dec._iter_hook(1, llr, e_v, dec._active, syndrome)
    assert dec.last_flip_cn[0] == cn
    assert dec.last_flip_vn[0] == vn
    assert_negation(dec, before, llr, vn)


@pytest.mark.parametrize("value,vn", [("max_unsat", 1), ("random", 0)])
def test_vn_pick(value, vn):
    dec, llr, syndrome = decoder(cn_pick="rand_unsat", vn_pick=value)
    dec._draw_r = lambda n: torch.full((n,), 0.2, dtype=dec.dtype)
    before = llr.clone()
    dec.sign_flip_cn_rand_new(syndrome, torch.zeros_like(syndrome), llr)
    assert dec.last_flip_vn[0] == vn
    assert_negation(dec, before, llr, vn)


@pytest.mark.parametrize("module", [lottery, policy])
@pytest.mark.parametrize("count,small,large", [(1, 0.30, 0.33), (9, 0.5, 1.2)])
@pytest.mark.parametrize("smaller_index", [0, 1])
def test_max_unsat_exact_llr_tie_break(module, count, small, large, smaller_index):
    H = torch.ones(count, 2, dtype=torch.long)
    dec = module.create(dict(device={"device_type": "cpu"}, dtype="float32",
                             max_iter=2, compile=False, rebatch_opt=False), bundle=bundle(H))
    dec.i, dec._B, dec._hook_rows = 1, 1, None
    dec._active = torch.tensor([True])
    dec.r = torch.zeros(2, 1)
    llr = torch.tensor([[large, large, float("inf")]])
    llr[0, smaller_index] = small
    unsat = torch.ones(1, count, dtype=torch.bool)
    if module is lottery:
        _, selected = lottery.select_flip_vn(dec, unsat, llr, torch.zeros_like(llr))
        assert selected.item() == smaller_index
    else:
        expected = llr.clone()
        expected[0, smaller_index] *= -1
        dec.sign_flip_lottery(unsat.float(), torch.zeros_like(unsat).float(), llr)
        assert torch.equal(llr, expected)


@pytest.mark.parametrize("value,llrs,without,with_tie", [
    ("max_unsat", [4, 2, 0.5, 4], 1, 2),
    ("random", [4, 2, 0.5, 4], 0, 0),
])
def test_vn_pick_tie_break_flag(value, llrs, without, with_tie):
    for flag, vn in ((False, without), (True, with_tie)):
        dec, llr, syndrome = decoder(cn_pick="all_unsat", vn_pick=value,
                                    vn_pick_tie_break=flag)
        llr[:, :-1] = torch.tensor(llrs, dtype=dec.dtype)
        before = llr.clone()
        dec.sign_flip_cn_rand_new(syndrome, torch.zeros_like(syndrome), llr)
        assert dec.last_flip_vn[0] == vn
        assert_negation(dec, before, llr, vn)


@pytest.mark.parametrize("value", [0, 1, None, "false"])
@pytest.mark.parametrize("module", [lottery, lottery_cuda, layered, layered_cuda, quant, quant_cuda])
def test_invalid_vn_pick_tie_break(module, value):
    with pytest.raises(ValueError, match="vn_pick_tie_break"):
        module.create({"vn_pick_tie_break": value})


@pytest.mark.parametrize("pick", ["max_unsat", "random"])
def test_zero_filter_backend_parity_and_empty_pool(pick):
    H = torch.tensor([[1, 1, 0], [0, 1, 1]])
    outputs = []
    for backend in ("cpu", "pytorch", "cuda"):
        if backend != "cpu" and not torch.cuda.is_available():
            continue
        dec = phase_decoder(H, backend, vn_pick=pick)
        dec._B, dec._hook_rows = 3, None
        dec._draw_r = lambda n: torch.zeros(n, device=dec.device, dtype=dec.dtype)
        dec.r_v = torch.full((8, 1), 0.8, device=dec.device, dtype=dec.dtype)
        llr = torch.tensor([[0., 0., 2., float("inf")], [0., 0., 0., float("inf")],
                            [2., 0., 0., float("inf")]], device=dec.device, dtype=dec.dtype)
        hard = (llr <= 0).to(dec.dtype)
        syndrome = ((hard[:, :-1] @ H.to(dec.device, dec.dtype).T) + 1) % 2
        dec._iter_hook(1, llr, hard, torch.ones(3, device=dec.device, dtype=torch.bool), syndrome)
        assert dec.last_flip_cn.tolist() == [0, 0, 0]
        assert dec.last_flip_vn.tolist() == [1, 1, 0]
        assert dec.last_flip_valid.tolist() == [True, True, True]
        expected = llr.new_tensor([[0., 0., 2., float("inf")], [0., 0., 0., float("inf")],
                                   [-2., 0., 0., float("inf")]])
        assert torch.equal(llr, expected)
        outputs.append(llr.cpu())
    assert all(torch.equal(value, outputs[0]) for value in outputs)


@pytest.mark.parametrize("value,vn", [("max_unsat", 2), ("random", 3)])
def test_global_vn_pick(value, vn):
    dec, llr, syndrome = decoder(cn_pick="all_unsat", vn_pick=value)
    dec._draw_r = lambda n: torch.full((n,), 0.2, dtype=dec.dtype)
    dec.r_v.fill_(0.8)
    llr[:, :-1] = torch.tensor([4, 2, 0.5, 1], dtype=dec.dtype)
    before = llr.clone()
    dec.sign_flip_cn_rand_new(syndrome, torch.zeros_like(syndrome), llr)
    assert dec.last_flip_cn.tolist() == [-1, -1, -1]
    assert dec.last_flip_vn[0] == vn
    assert_negation(dec, before, llr, vn)


@pytest.mark.parametrize("value,vn", [("max_unsat", 1)])
def test_all_vn_pick(value, vn):
    for mode, expected_vn in (("all_unsat", 1), ("all", vn)):
        dec, llr, syndrome = decoder(cn_pick=mode, vn_pick=value)
        syndrome[:2, 1:] = 0
        llr[:, :-1] = torch.tensor([4, 2, -0.25, 1], dtype=dec.dtype)
        before = llr.clone()
        dec.sign_flip_cn_rand_new(syndrome, torch.zeros_like(syndrome), llr)
        assert dec.last_flip_vn[0] == expected_vn
        assert_negation(dec, before, llr, expected_vn)


@pytest.mark.parametrize("mode,checks", [("all_unsat", [-1, -1, -1]),
                                       ("all", [-1, -1, -1]),
                                       ("rand_unsat", [2, 2, 0])])
def test_flip_hook_check_marker_preserves_vn_pick(mode, checks):
    dec, llr, syndrome = decoder(cn_pick=mode)
    before = llr.clone()
    dec._iter_hook(1, llr, torch.zeros_like(llr), dec._active, syndrome)
    assert dec.last_flip_cn.tolist() == checks
    assert dec.last_flip_cn.dtype == torch.int64
    assert dec.last_flip_vn.tolist() == [2, 2, 0]
    assert_negation(dec, before, llr, 2)


@pytest.fixture(scope="module")
def seeded_inputs():
    H = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1], [1, 0, 1, 0]], dtype=torch.float64)
    error = (torch.rand((16, H.shape[1]), generator=torch.Generator().manual_seed(37)) < .15).double()
    return dict(H=H, synd=(error @ H.T).remainder(2),
                llr0=torch.full_like(error, 2.0),
                cfg=dict(device={"device_type": "cpu"}, dtype="float64", max_iter=8,
                         compile=False, rebatch_opt=False))


@pytest.mark.parametrize("compact_frac", [0, 0.75])
def test_global_max_unsat_matches_global_optimal(compact_frac):
    from syndrilla.decoder.bp_lottery_policy import bp_lottery_policy as policy
    from syndrilla.matrix import load_matrices
    from syndrilla.utils import parse_device_dtype, read_yaml

    cfg = {"device": {"device_type": "cpu"}, "dtype": "float64", "check_type": "hx",
           "compile": False, "rebatch_opt": False, "max_iter": 20, "early_termination": True,
           "compact_frac": compact_frac, "force_pytorch": True,
           "cn_pick": "all_unsat",
           "vn_pick": "max_unsat"}
    matrices = load_matrices(read_yaml("examples/alist/surface_10.matrix.yaml")["matrix"],
                             *parse_device_dtype(cfg))
    H = matrices.select("hx")[3].to_dense().double()
    error = (torch.rand((24, H.shape[1]), generator=torch.Generator().manual_seed(17)) < .1).double()
    inputs = dict(synd=(error @ H.T).remainder(2), llr0=torch.full_like(error, 2.0))
    outputs = []
    for module in (lottery, policy):
        extra = {"sign_flip_policy": "global_optimal"} if module is policy else {}
        dec = module.create({**cfg, **extra}, bundle=matrices)
        with torch.no_grad():
            outputs.append(dec({key: value.clone() for key, value in inputs.items()}))
    for key in ("e_v", "converge"):
        assert torch.equal(outputs[0][key], outputs[1][key]), key
    with torch.no_grad():
        off = lottery.create({**cfg, "flip_mode": "off"}, bundle=matrices)(dict(inputs))
        mutant = lottery.create({**cfg, "cn_pick": "rand_unsat"}, bundle=matrices)(dict(inputs))
    assert ((off["converge"] == 0) & (off["iter"] == cfg["max_iter"])).any()
    assert any(not torch.equal(outputs[0][key], mutant[key]) for key in ("e_v", "converge"))


@pytest.mark.parametrize("knob", list(lottery.STUDY_KNOBS))
@pytest.mark.parametrize("value", ["bad", None, [], 3])
def test_invalid_choice(knob, value):
    with pytest.raises(ValueError, match=knob):
        lottery.create({knob: value})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("modules", [(lottery, lottery_cuda), (quant, quant_cuda)],
                         ids=["lottery", "quant"])
@pytest.mark.parametrize("random_machine", ["sobol", "system"])
@pytest.mark.parametrize("knob,value", [(key, value) for key, values in lottery.STUDY_KNOBS.items()
                                     for value in values[1:]] + [("vn_pick_tie_break", False),
                                                               ("cn_pick_tie_break", False)])
def test_cuda_study_knobs_match_pytorch(modules, knob, value, random_machine):
    from test_lottery_bp import _surface10_system_outputs

    cfg = {knob: value, "random_machine": random_machine, "max_iter": 60,
           "rebatch_opt": False, "force_pytorch": True}
    if knob == "cn_pick_tie_break":
        cfg["cn_pick"] = "max_err_unsat"
    ref = _surface10_system_outputs(modules[0], cfg)
    got = _surface10_system_outputs(modules[1], cfg)
    for expected, actual in zip(ref, got):
        for key in ("e_v", "converge", "cpu_rng", "cuda_rng"):
            assert torch.equal(expected[key], actual[key]), key


def test_quant_sobol_pick_draws_are_rounded(seeded_inputs):
    dec = quant.create({**seeded_inputs["cfg"], "vn_pick": "random", "frac_width": 1},
                       bundle=bundle(seeded_inputs["H"]))
    dec.i = 1
    dec.r = torch.full((1, 1), 0.8, device=dec.device, dtype=dec.dtype)
    dec.r_v = dec.r.clone()
    assert torch.equal(dec._draw_r(3), dec.r.new_full((3,), 0.5))
    assert torch.equal(dec.r_v, dec.r.new_full((1, 1), 0.5))


def test_sobol_check_draws_pair_across_vn_arms(seeded_inputs):
    class Recorded(lottery.create):
        def _iter_hook(self, i, *args):
            super()._iter_hook(i, *args)
            if lottery.is_flip_iter(self, i) and not hasattr(self, "first_cn"):
                self.first_cn = self.last_flip_cn.clone()

    draws, picks = [], []
    for value in lottery.STUDY_KNOBS["vn_pick"]:
        cfg = {**seeded_inputs["cfg"], "vn_pick": value}
        dec = Recorded(cfg, bundle=bundle(seeded_inputs["H"]))
        with torch.no_grad():
            dec({key: seeded_inputs[key].clone() for key in ("synd", "llr0")})
        draws.append(dec.r)
        picks.append(dec.first_cn)
    for stream, checks in zip(draws[1:], picks[1:]):
        assert torch.equal(stream, draws[0])
        assert torch.equal(checks, picks[0])


@pytest.mark.parametrize("knob,value", [("cn_pick", "max_err_unsat"),
                                      ("vn_pick", "random")])
def test_layered_supported_knobs_run(seeded_inputs, knob, value):
    dec = layered.create({**seeded_inputs["cfg"], knob: value}, bundle=bundle(seeded_inputs["H"]))
    with torch.no_grad():
        out = dec({key: seeded_inputs[key].clone() for key in ("synd", "llr0")})
    parity = (out["e_v"] @ seeded_inputs["H"].T) % 2
    assert torch.equal(out["converge"].bool(), (parity == seeded_inputs["synd"]).all(1))


@pytest.mark.parametrize("knob,value", [(key, value) for key, values in lottery.STUDY_KNOBS.items()
                                     for value in values])
@pytest.mark.parametrize("compact_frac", [0, 0.75])
def test_cpu_decode_contract(seeded_inputs, knob, value, compact_frac):
    cfg = {**seeded_inputs["cfg"], knob: value, "compact_frac": compact_frac,
           "early_termination": True, "max_iter": 20}
    dec = lottery.create(cfg, bundle=bundle(seeded_inputs["H"]))
    io = {key: seeded_inputs[key].clone() for key in ("synd", "llr0")}
    torch.manual_seed(37)
    with torch.no_grad():
        out = dec(io)
    assert torch.equal(io["synd"], seeded_inputs["synd"])
    assert torch.equal(io["llr0"], seeded_inputs["llr0"])
    assert torch.isfinite(out["llr"]).all()
    parity = (out["e_v"] @ seeded_inputs["H"].T) % 2
    assert torch.equal(out["converge"].bool(), (parity == io["synd"]).all(1))


@pytest.mark.parametrize("module", [lottery, lottery_cuda])
@pytest.mark.parametrize("value", [None, (), [-1], [True], [1.5], [3, 2], [1, 2, 3]])
def test_invalid_phase_boundaries(module, value):
    with pytest.raises(ValueError, match="flip_phase_iters"):
        module.create({"flip_phase_iters": value})


@pytest.mark.parametrize("module", [lottery, lottery_cuda])
@pytest.mark.parametrize("key,value", [
    ("cn_pick", "rand_unsat"), ("vn_pick", "max_unsat"),
    ("flip_mode", "lottery"),
])
@pytest.mark.parametrize("length", [0, 1, 4])
def test_phase_list_length(module, key, value, length):
    with pytest.raises(ValueError, match=key + " list must have 3 entries"):
        module.create({"flip_phase_iters": [2, 4], key: [value] * length})


@pytest.mark.parametrize("key,value", [
    ("flip_mode", "bad"), ("flip_mode", None),
])
def test_invalid_phase_value(key, value):
    for module in (lottery, lottery_cuda):
        with pytest.raises(ValueError, match=key):
            module.create({key: value})


@pytest.mark.parametrize("module", [lottery, lottery_cuda])
def test_invalid_vn_choice_in_phase(module):
    with pytest.raises(ValueError, match="vn_pick"):
        module.create({"flip_phase_iters": [2], "vn_pick": ["max_unsat", "bad"]})


def test_phase_knobs_at_boundaries():
    choices = ["max_unsat", "max_unsat", "random"]
    dec, _, _ = decoder(flip_phase_iters=[2, 4], cn_pick=["rand_unsat", "all_unsat", "all"],
                        vn_pick=choices,
                        flip_mode=["off", "lottery", "lottery"])
    expected = [False, False, True, True, True, True, True, False]
    assert [lottery.is_flip_iter(dec, i) for i in range(1, 9)] == expected
    lottery.set_phase(dec, 4)
    assert (dec.cn_pick, dec.vn_pick, dec.flip_mode) == ("all_unsat", "max_unsat", "lottery")
    lottery.set_phase(dec, 5)
    assert (dec.cn_pick, dec.vn_pick, dec.flip_mode) == ("all", "random", "lottery")
    assert choices == ["max_unsat", "max_unsat", "random"]


@pytest.mark.parametrize("t", [1, 2, 42, 82])
def test_single_flip_phase_boundaries(t):
    dec, _, _ = decoder(max_iter=121, flip_phase_iters=[t - 1, t],
                        flip_mode=["off", "lottery", "off"])
    assert [i for i in range(122) if lottery.is_flip_iter(dec, i)] == [t]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("pick", ["max_unsat", "random"])
@pytest.mark.parametrize("extra", [
    {}, {"flip_phase_iters": [2, 4], "vn_pick": ["random", "max_unsat", "random"]},
    {"cn_pick": "all_unsat"},
])
def test_retained_surface_cpu_pytorch_cuda_parity(pick, extra):
    from syndrilla.matrix import load_matrices
    from syndrilla.utils import parse_device_dtype, read_yaml

    outputs = []
    for backend in ("cpu", "pytorch", "cuda"):
        cfg = dict(device={"device_type": "cpu" if backend == "cpu" else "cuda"},
                   dtype="float64", check_type="hx", max_iter=12, compile=False,
                   rebatch_opt=False, random_machine="sobol", flip_mode="lottery",
                   vn_pick=pick)
        cfg.update(extra)
        matrices = load_matrices(read_yaml("examples/alist/surface_10.matrix.yaml")["matrix"],
                                 *parse_device_dtype(cfg))
        H = matrices.select("hx")[3].to_dense().cpu().double()
        error = (torch.rand((24, H.shape[1]), generator=torch.Generator().manual_seed(17)) < .1).double()
        dec = (lottery_cuda if backend == "cuda" else lottery).create(cfg, bundle=matrices)
        torch.manual_seed(123)
        with torch.no_grad():
            out = dec(dict(synd=(error @ H.T).remainder(2),
                           llr0=torch.full_like(error, math.log(9))))
        outputs.append({key: out[key].cpu() for key in ("e_v", "llr", "converge", "iter", "defer")})
    for out in outputs[1:]:
        for key in out:
            assert torch.equal(out[key], outputs[0][key]), key


def phase_decoder(H, backend="cpu", **extra):
    device = torch.device("cpu" if backend == "cpu" else "cuda")
    index = dense_to_index_format(H, device)
    matrix = SimpleNamespace(select=lambda _: index, Hx_matrix=None, Hz_matrix=None,
                             lx_matrix=None, lz_matrix=None)
    module = lottery_cuda if backend == "cuda" else lottery
    return module.create(dict(device={"device_type": device.type}, dtype="float64", max_iter=8,
                              compile=False, early_termination=False, rebatch_opt=False, **extra), bundle=matrix)


@pytest.mark.parametrize("backend", ["cpu", "pytorch", "cuda"])
def test_equal_phase_boundaries_skip_middle_phase(backend):
    if backend != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    H = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1], [0, 0, 1, 0]])
    synd = torch.tensor([[1., 0., 1.], [0., 1., 0.], [0., 0., 0.]], dtype=torch.float64)
    llr = torch.tensor([[1., 2., 3., 4.]], dtype=torch.float64).repeat(3, 1)
    outputs = []
    for boundaries, modes in (([2, 2], ["lottery", "lottery", "off"]),
                              ([2], ["lottery", "off"])):
        dec = phase_decoder(H, backend, flip_phase_iters=boundaries, flip_mode=modes)
        assert len(dec.phase_knobs) == len(boundaries) + 1
        lottery.set_phase(dec, 2)
        assert dec._phase_index == 0 and dec.flip_mode == "lottery"
        lottery.set_phase(dec, 3)
        assert dec._phase_index == len(boundaries) and dec.flip_mode == "off"
        with torch.no_grad():
            out = dec({"synd": synd.clone(), "llr0": llr.clone()})
        outputs.append({k: out[k].clone() for k in ("e_v", "llr", "converge", "iter")})
    for key in outputs[0]:
        assert torch.equal(outputs[0][key], outputs[1][key]), key


@pytest.mark.parametrize("extra", [
    {"flip_phase_iters": [2, 4]},
    {"cn_pick": ["all", "rand_unsat", "all_unsat"]},
    {"vn_pick": ["max_unsat", "random", "max_unsat"]},
    {"flip_mode": ["off", "lottery", "lottery"]},
])
@pytest.mark.parametrize("early", [False, True])
def test_phase_cpu_pytorch_cuda_parity_and_batch_reset(extra, early):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    H = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1], [0, 0, 1, 0]])
    cfg = {"flip_phase_iters": [2, 4], **extra}
    synd = torch.tensor([[1., 0., 1.], [0., 1., 0.], [0., 0., 0.]], dtype=torch.float64)
    llr = torch.tensor([[1., 2., 3., 4.]], dtype=torch.float64).repeat(3, 1)
    outputs = []
    for backend in ("cpu", "pytorch", "cuda"):
        dec = phase_decoder(H, backend, **cfg)
        dec.early_termination = early
        dec.compact_frac = 0.75
        batches = []
        for _ in range(2):
            with torch.no_grad():
                out = dec({"synd": synd.clone(), "llr0": llr.clone()})
            batches.append({k: out[k].cpu() for k in ("e_v", "llr", "converge", "iter")})
            parity = (out["e_v"].cpu() @ H.T.double()) % 2
            assert torch.equal(out["converge"].cpu().bool().flatten(), (parity == synd).all(1))
            assert torch.equal(out["llr"] <= 0, out["e_v"].bool())
        for key in batches[0]:
            assert torch.equal(batches[0][key], batches[1][key]), key
        outputs.append(batches[0])
    for out in outputs[1:]:
        for key in out:
            assert torch.equal(out[key], outputs[0][key]), key


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("extra", [
    {"vn_pick": ["max_unsat", "random", "max_unsat"]},
    {"flip_mode": ["off", "lottery", "lottery"]},
])
def test_phase_system_rng_pytorch_cuda_parity(extra):
    from test_lottery_bp import _surface10_system_outputs

    cfg = dict(flip_phase_iters=[3, 7], max_iter=20, rebatch_opt=False,
               force_pytorch=True, random_machine="system", **extra)
    expected = _surface10_system_outputs(lottery, cfg)
    actual = _surface10_system_outputs(lottery_cuda, cfg)
    for a, b in zip(expected, actual):
        for key in ("e_v", "converge", "iter", "cpu_rng", "cuda_rng"):
            assert torch.equal(a[key], b[key]), key
        torch.testing.assert_close(a["llr"], b["llr"], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("pick_policy", sorted(policy.create._sign_flip_policies))
def test_policy_skips_zero_candidates_with_all_zero_fallback(pick_policy, monkeypatch):
    H = torch.ones(2, 3, dtype=torch.float64)
    dec = policy.create(dict(device={"device_type": "cpu"}, dtype="float64", max_iter=2,
        compile=False, rebatch_opt=False, sign_flip_policy=pick_policy), bundle=bundle(H))
    dec.i, dec._B, dec.batch_size, dec._hook_rows = 1, 2, 2, None
    dec._active = torch.ones(2, dtype=torch.bool)
    dec.r = torch.full((2, 1), .8, dtype=dec.dtype)
    posterior = torch.tensor([[0., 2., 0., float("inf")], [0., 0., 0., float("inf")]])
    picked = []
    native = policy.flip_rows
    def record(values, nodes, valid):
        picked.append(nodes.clone())
        native(values, nodes, valid)
    monkeypatch.setattr(policy, "flip_rows", record)
    dec._apply_policy(torch.ones(2, 2), torch.zeros(2, 2), posterior)
    assert len(picked) == 1 and picked[0][0] == 1
    assert 0 <= picked[0][1] < 3
    assert posterior[0, 1] == -2
