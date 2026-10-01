"""Each CUDA BP variant (the *_cuda.py modules other than bp_norm_min_sum_cuda)
gives the same outputs, bit for bit, as frozen references saved from an earlier
tree. A reference lives in REF/<variant>/<case>_<dtype>/ as inputs.pt (synd,
llr0), outputs.pt (e_v, llr, iter, converge) and config.json (seeds, bundle
source and decoder config). Cases: stim rotated_memory_x d=5 rounds=5 at p=3e-3
on all four noise knobs, and surface_10 hx from alist, at float64 and float32,
B=64. The quant variants (bp_norm_min_sum_quant, bp_lottery_quant) take their
outputs from the PyTorch references instead (PT_REF). Tests skip without CUDA
and fail when the reference files are missing; the failure message gives the
command that rebuilds them."""
import importlib
import json
import math
import os
import random

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.environ.get("SYNDRILLA_BP_VARIANT_REF", "/tmp/claude-1110527820/t37/ref")
T37 = "/tmp/claude-1110527820/t37"
# The CUDA quant variants follow the PyTorch rounding, so their outputs are checked
# against the PyTorch (CPU) references of tests/test_bp_variants_pytorch_equiv.py.
PT_REF = os.environ.get(
    "SYNDRILLA_BP_VARIANT_PT_REF", "/tmp/claude-1110527820/t38/refs"
)
PT_REF_VARIANTS = {"bp_norm_min_sum_quant", "bp_lottery_quant"}
DEV = {"device_type": "cuda", "device_idx": 0}
KEYS = ("e_v", "llr", "iter", "converge")
B, SEED, P_STIM, P_ALIST = 64, 0, 3e-3, 0.03
# variant -> example decoding yaml that gives its default config
VARIANTS = {
    "bp_sf": "bp_sf_hx",
    "bp_lottery": "lottery_bp_hx",
    "bp_lottery_policy": "lottery_policy_hx",
    "bp_norm_min_sum_quant": "bp_quant_hx",
    "bp_lottery_quant": "lottery_bp_quant_hx",
    "bp_branch_assisted": "bbp_hz",
    "relay_bp": "relay_bp_hx",
}
CASES = ("stim_d5", "surface10")
DTYPES = ("float64", "float32")
NOISE = (
    "after_clifford_depolarization",
    "after_reset_flip_probability",
    "before_measure_flip_probability",
    "before_round_data_depolarization",
)


def decoder_cfg(variant, dtype):
    """Flat decoder config from the variant's example yaml, on cuda:0 with the
    given dtype and check_type hx (the stim bundle has hx only)."""
    from syndrilla.decoder.decoder import resolve_configs
    from syndrilla.utils import read_yaml

    path = os.path.join(ROOT, "examples", "alist", f"{VARIANTS[variant]}.decoding.yaml")
    cfg = resolve_configs(read_yaml(path)["decoding"])[0]
    return {**cfg, "device": DEV, "dtype": dtype, "check_type": "hx"}


def bundle_and_inputs(case, dtype):
    """(bundle, synd, llr0) for B shots of the case, drawn with seed SEED."""
    if case == "stim_d5":
        from syndrilla.interface.stim.stim import create as stim_iface

        it = stim_iface(
            {"code": "surface_code:rotated_memory_x", "distance": 5},
            error_cfg={k: P_STIM for k in NOISE},
            syndrome_cfg={"rounds": 5},
            decoding_cfg={
                "algorithm": "bp_norm_min_sum",
                "check_type": "hx",
                "config": {"max_iter": 1},
                "dtype": dtype,
                "device": DEV,
                "force_pytorch": True,
            },
        )
        torch.manual_seed(SEED)
        z = torch.zeros(
            B, it.error_model.num_errors, dtype=getattr(torch, dtype), device="cuda"
        )
        _, dl = it.error_model.inject_error(z, B)
        e, llr0, _ = next(iter(dl))
        return it.matrix_bundle, it.syndrome_generator.measure_syndrome(e, None), llr0
    from syndrilla.matrix import load_matrices
    from syndrilla.utils import parse_device_dtype, read_yaml

    mcfg = read_yaml(os.path.join(ROOT, "examples", "alist", "surface_10.matrix.yaml"))[
        "matrix"
    ]
    for v in mcfg.values():
        if isinstance(v, dict) and "path" in v:
            v["path"] = os.path.join(ROOT, v["path"])
    bundle = load_matrices(mcfg, *parse_device_dtype({"device": DEV, "dtype": dtype}))
    H = bundle.select("hx")[3].to_dense().cpu().double()
    g = torch.Generator().manual_seed(SEED)
    err = (torch.rand(B, H.shape[1], generator=g) < P_ALIST).double()
    synd = ((err @ H.T) % 2).to(torch.uint8)
    llr0 = torch.full(
        (B, H.shape[1]), math.log((1 - P_ALIST) / P_ALIST), dtype=torch.float64
    )
    return bundle, synd, llr0


def decode(variant, cfg, bundle, synd, llr0):
    """Build the variant's CUDA decoder, seed torch (CPU and CUDA), numpy and
    python random with SEED, run forward once and return the outputs on CPU."""
    mod = importlib.import_module(f"syndrilla.decoder.{variant}.{variant}_cuda")
    dec = mod.create(cfg, bundle=bundle)
    assert type(dec).__module__.endswith("_cuda")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    with torch.no_grad():
        out = dec({"synd": synd.clone(), "llr0": llr0.clone()})
    return {k: out[k].detach().cpu().clone() for k in KEYS}


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_matches_reference(variant, case, dtype):
    d = os.path.join(REF, variant, f"{case}_{dtype}")
    if not os.path.isfile(os.path.join(d, "outputs.pt")):
        pytest.fail(
            f"no reference in {d}. Rebuild all of them with"
            " tests/freeze_bp_variant_refs.py run against the pre-change source snapshot"
            f" {T37}/src-before (not the working tree):\n"
            f"for v in {' '.join(VARIANTS)}; do PYTHONPATH={T37}/src-before"
            f" TORCH_EXTENSIONS_DIR={T37}/ext_before conda run -n syndrilla python"
            f" tests/freeze_bp_variant_refs.py $v {REF}; done\nSee {T37}/README.md."
        )
    with open(os.path.join(d, "config.json")) as f:
        meta = json.load(f)
    if not meta["deterministic"]:
        pytest.skip(f"{variant} is not run-to-run deterministic: {meta['note']}")
    x = torch.load(os.path.join(d, "inputs.pt"))
    if variant in PT_REF_VARIANTS:
        p = os.path.join(PT_REF, variant, f"{case}_{dtype}", "outputs.pt")
        if not os.path.isfile(p):
            pytest.fail(
                f"no reference {p}. The CUDA quant variants are checked against the"
                " PyTorch references; set SYNDRILLA_BP_VARIANT_PT_REF, or rebuild them"
                " (needs a GPU): conda run -n syndrilla python"
                f" tests/freeze_bp_variant_pytorch_refs.py {PT_REF}"
            )
        ref = torch.load(p)
    else:
        ref = torch.load(os.path.join(d, "outputs.pt"))
    bundle, _, _ = bundle_and_inputs(case, dtype)
    got = decode(
        variant, meta["decoder_cfg"], bundle, x["synd"].cuda(), x["llr0"].cuda()
    )
    for k in KEYS:
        assert got[k].dtype == ref[k].dtype, (k, got[k].dtype, ref[k].dtype)
        diff = (got[k] != ref[k]).reshape(len(ref[k]), -1).any(1)
        assert torch.equal(
            got[k], ref[k]
        ), f"{k} differs in samples {diff.nonzero().flatten().tolist()}"


def test_iter_hook_forces_per_step_and_keeps_outputs():
    """A subclass that overrides _iter_hook runs the per-step path even where the
    persistent one would be picked, sees every non-breaking iteration with the
    live active mask, and with a read-only hook matches the plain per-step run."""
    from syndrilla.decoder.bp_norm_min_sum import bp_norm_min_sum_cuda as bp_gpu

    class Hooked(bp_gpu.create):
        def _iter_hook(self, i, l_v, e_v, active, syndrome):
            self.calls.append((i, int(active.sum())))

    bundle, synd, llr0 = bundle_and_inputs("surface10", "float64")
    cfg = dict(device=DEV, dtype="float64", check_type="hx", max_iter=20)
    hooked = Hooked(cfg, bundle=bundle)
    hooked.calls = []
    assert bp_gpu.create(cfg, bundle=bundle)._use_persistent(B, False)
    assert not hooked._use_persistent(B, False)
    plain = bp_gpu.create({**cfg, "force_per_step": True}, bundle=bundle)
    ref, got = (decode_with(d, synd, llr0) for d in (plain, hooked))
    for k in KEYS:
        assert torch.equal(got[k], ref[k]), k
    assert [c[0] for c in hooked.calls] == list(range(1, len(hooked.calls) + 1))
    assert hooked.calls[-1][1] == int((ref["converge"] == 0).sum())


def decode_with(dec, synd, llr0):
    with torch.no_grad():
        out = dec({"synd": synd.clone(), "llr0": llr0.clone()})
    return {k: out[k].cpu() for k in KEYS}


@pytest.mark.parametrize("per_step", [False, True])
def test_exit_hook_called_once(per_step):
    """_exit_hook runs exactly once per forward on both paths, with num_iters
    final, and does not change path selection."""
    from syndrilla.decoder.bp_norm_min_sum import bp_norm_min_sum_cuda as bp_gpu

    class Hooked(bp_gpu.create):
        def _exit_hook(self, l_v, e_v, num_iters, converges):
            self.exits.append(int(num_iters.min()))

    bundle, synd, llr0 = bundle_and_inputs("surface10", "float64")
    cfg = dict(device=DEV, dtype="float64", check_type="hx", max_iter=20)
    dec = Hooked({**cfg, "force_per_step": per_step}, bundle=bundle)
    dec.exits = []
    assert dec._use_persistent(B, False) is not per_step
    decode_with(dec, synd, llr0)
    assert len(dec.exits) == 1 and dec.exits[0] >= 1


def test_iter_hook_l_v_edit_carries_to_next_iteration():
    """An l_v edit at iteration 1 feeds iteration 2: the l_v seen at iteration 2
    differs from a read-only run on every row active there in both runs."""
    from syndrilla.decoder.bp_norm_min_sum import bp_norm_min_sum_cuda as bp_gpu

    class Hooked(bp_gpu.create):
        def _iter_hook(self, i, l_v, e_v, active, syndrome):
            if i == 1 and self.flip:
                l_v[active, :-1] = -l_v[active, :-1]
            if i == 2:
                self.seen = (l_v.clone(), active.clone())

    bundle, synd, llr0 = bundle_and_inputs("surface10", "float64")
    cfg = dict(device=DEV, dtype="float64", check_type="hx", max_iter=20)
    runs = []
    for flip in (False, True):
        dec = Hooked(cfg, bundle=bundle)
        dec.flip = flip
        decode_with(dec, synd, llr0)
        runs.append(dec.seen)
    (l_ro, act_ro), (l_ed, act_ed) = runs
    both = act_ro & act_ed
    assert both.any()
    assert (l_ro[both] != l_ed[both]).any(1).all()


def test_iter_hook_active_under_cap():
    """With the rebatch cap active, the active mask at iteration i is exactly the
    rows still running after i (final iter > i), and the loop stops early."""
    from syndrilla.decoder.bp_norm_min_sum import bp_norm_min_sum_cuda as bp_gpu
    from syndrilla.decoder.decoder import RebatchSpeedup

    class Hooked(bp_gpu.create):
        def _iter_hook(self, i, l_v, e_v, active, syndrome):
            self.calls.append((i, active.clone()))

    bundle, synd, llr0 = bundle_and_inputs("surface10", "float64")
    cfg = dict(device=DEV, dtype="float64", check_type="hx", max_iter=20)
    dec = Hooked(cfg, bundle=bundle)
    dec.calls, dec.cap = [], RebatchSpeedup()
    dec.cap.frac = 0.6
    out = decode_with(dec, synd, llr0)
    assert dec.cap_active_last and dec.calls
    stop = int(out["iter"].max())
    assert stop < 20 and dec.calls[-1][0] == stop - 1
    for i, active in dec.calls:
        assert torch.equal(active.cpu(), out["iter"] > i), i
