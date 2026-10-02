"""The pure-PyTorch osd_0 gives the same e_v, bit for bit, as references saved
from the earlier osd_0 (LU decomposition over all rank(H) pivot columns), on CPU
at float64 and float32, with the full order and with short order prefixes that
send samples through the 4k retry and the full-order fallback. On CUDA its e_v,
iter and converge equal osd_0_cuda's on a stim rotated surface code circuit
(d=5, 5 rounds). On every path iter is the input iter (zeros when absent) with
N on the samples OSD decodes. A small workspace_bytes budget splits the scan
into chunks without changing e_v. REF holds, per case, the inputs (synd, float32 BP
output llr) and the reference e_v: surface_10 hx (B=32), stim d=5 (B=16), and
a hand-made H with a duplicated row and syndromes outside its column space."""
import os

import numpy as np
import pytest
import torch

from syndrilla.decoder.osd_0 import osd_0 as osd_cpu
from syndrilla.matrix.matrix import dense_to_index_format

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "tests", "data", "osd0_pytorch_ref.pt")
B_CASE = {"surface10": 32, "stim_d5": 16}
P = 3e-3
NOISE = (
    "after_clifford_depolarization",
    "after_reset_flip_probability",
    "before_measure_flip_probability",
    "before_round_data_depolarization",
)
CUDA = {"device_type": "cuda", "device_idx": 0}
needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


class _Bundle:
    """Minimal MatrixBundle stand-in for a hand-made H."""

    def __init__(self, H):
        self.index = dense_to_index_format(H, "cpu")

    def select(self, check_type):
        return self.index


def _stim(device, dtype="float64", max_iter=181):
    from syndrilla.interface.stim.stim import create as stim_iface

    return stim_iface(
        {"code": "surface_code:rotated_memory_x", "distance": 5},
        error_cfg={k: P for k in NOISE},
        syndrome_cfg={"rounds": 5},
        decoding_cfg={
            "algorithm": "bp_norm_min_sum",
            "check_type": "hx",
            "config": {"max_iter": max_iter},
            "dtype": dtype,
            "device": device,
        },
    )


def bundle_for(case):
    """The hx matrix bundle of a REF case, on CPU."""
    if case == "stim_d5":
        return _stim({"device_type": "cpu"}).matrix_bundle
    from syndrilla.matrix import load_matrices
    from syndrilla.utils import parse_device_dtype, read_yaml

    ex = os.path.join(ROOT, "examples", "alist")
    cfg = dict(read_yaml(os.path.join(ex, "bposd_hx.decoding.yaml"))["decoding"])
    cfg["device"] = {"device_type": "cpu"}
    mcfg = read_yaml(os.path.join(ex, "surface_10.matrix.yaml"))["matrix"]
    for v in mcfg.values():
        if isinstance(v, dict) and "path" in v:
            v["path"] = os.path.join(ROOT, v["path"])
    return load_matrices(mcfg, *parse_device_dtype(cfg))


def inconsistent_case():
    """(bundle, synd, llr): M=30, N=240, B=12, rows 0 and 1 equal and every
    syndrome differs on them, so no syndrome is in the column space of H."""
    rng = np.random.default_rng(4)
    M, N, B = 30, 240, 12
    H = (rng.random((M, N)) < 0.1).astype(np.uint8)
    H[1] = H[0]
    s = rng.integers(0, 2, (B, M))
    s[:, 1] = 1 - s[:, 0]
    return (
        _Bundle(H),
        torch.from_numpy(s).to(torch.uint8),
        torch.from_numpy(rng.random((B, N))),
    )


def _run(dec, synd, llr, H, converge=None, e_v=None, iter_in=None):
    B, N = llr.shape
    dev = llr.device
    io = {
        "synd": synd,
        "llr": llr,
        "converge": torch.zeros(B, dtype=torch.long, device=dev)
        if converge is None
        else converge,
        "e_v": torch.zeros(B, N, dtype=torch.uint8, device=dev) if e_v is None else e_v,
        "H_matrix": H.to(dev),
    }
    if iter_in is not None:
        io["iter"] = iter_in
    with torch.no_grad():
        return dec(io)


def _iter_rule(out, converge, N, iter_in=None):
    """iter is the input iter (zeros [B] int64 when absent) with N on the
    samples OSD decodes, as in osd_0_cuda."""
    want = torch.zeros(len(converge), dtype=torch.long) if iter_in is None else iter_in.clone()
    want[converge == 0] = N
    assert out["iter"].dtype == want.dtype and torch.equal(out["iter"], want)


@pytest.fixture(scope="module")
def ref():
    if not os.path.isfile(REF):
        pytest.skip(f"reference file {REF} missing")
    return torch.load(REF)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("case", ["surface10", "stim_d5"])
def test_cpu_matches_reference(ref, case, dtype):
    """e_v equals the reference with the full order and with prefixes of 12
    and N // 16 columns; converge is all ones; iter follows the rule in
    _iter_rule, with and without an input iter."""
    r = ref[case]
    bundle = bundle_for(case)
    H = bundle.select("hx")[3]
    llr = r["llr"].to(getattr(torch, dtype))
    B, N = llr.shape
    bp_iter = torch.arange(B, dtype=torch.long) % 7 + 1
    for prefix in (None, 12, N // 16):
        for iter_in in (None, bp_iter):
            dec = osd_cpu.create({"device": {"device_type": "cpu"}, "dtype": dtype}, bundle=bundle)
            assert dec.prefix == N  # N < 16384: full sort by default
            if prefix:
                dec.prefix = prefix
            out = _run(dec, r["synd"], llr, H, iter_in=iter_in)
            assert torch.equal(out["e_v"], r["e_v"]), f"prefix={prefix}"
            assert out["converge"].dtype == torch.long and bool((out["converge"] == 1).all())
            _iter_rule(out, torch.zeros(B, dtype=torch.long), N, iter_in)


def test_cpu_inconsistent_matches_reference(ref):
    """No syndrome is in the column space, so no scan stops early and every
    sample goes through all stages of a 15-column prefix: e_v equals the
    reference."""
    bundle, synd, llr = inconsistent_case()
    H = bundle.select("hx")[3]
    for prefix in (None, 15):
        dec = osd_cpu.create({"device": {"device_type": "cpu"}}, bundle=bundle)
        if prefix:
            dec.prefix = prefix
        out = _run(dec, synd, llr, H)
        assert torch.equal(out["e_v"], ref["inconsistent"]["e_v"]), f"prefix={prefix}"
        _iter_rule(out, torch.zeros(len(synd), dtype=torch.long), llr.shape[1])


def test_cpu_partly_converged(ref):
    """Converged samples keep their input e_v row and their input iter."""
    r = ref["surface10"]
    bundle = bundle_for("surface10")
    B, N = r["llr"].shape
    converge = (torch.arange(B) % 2).long()
    e_in = torch.randint(0, 2, (B, N), dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
    iter_in = torch.arange(B, dtype=torch.long) % 7 + 1
    dec = osd_cpu.create({"device": {"device_type": "cpu"}}, bundle=bundle)
    out = _run(dec, r["synd"], r["llr"].double(), bundle.select("hx")[3], converge, e_in.clone(), iter_in.clone())
    keep = converge == 1
    assert torch.equal(out["e_v"][keep], e_in[keep])
    assert torch.equal(out["e_v"][~keep], r["e_v"][~keep])
    _iter_rule(out, converge, N, iter_in)


@pytest.fixture(scope="module", params=["float64", "float32"])
def stim_d5_cuda(request):
    """(it, dtype, BP output io dict) for 64 circuit-noise shots with seed 0."""
    dtype = request.param
    it = _stim(CUDA, dtype)
    torch.manual_seed(0)
    z = torch.zeros(64, it.error_model.num_errors, dtype=getattr(torch, dtype), device="cuda")
    _, dl = it.error_model.inject_error(z, 64)
    e, llr0, _ = next(iter(dl))
    synd = it.syndrome_generator.measure_syndrome(e, None)
    bp = it.decoders[0]
    bp.eval()
    H = it.matrix_bundle.select("hx")[3]
    with torch.no_grad():
        io = bp({"synd": synd, "llr0": llr0, "H_matrix": H})
    io["H_matrix"] = H
    return it, dtype, io


@needs_cuda
def test_cuda_matches_osd_0_cuda(stim_d5_cuda):
    """With BP's own converge flags and with every sample sent to OSD, with
    and without BP's iter in the input, and with the full order and a
    12-column prefix on both decoders, e_v, iter and converge equal
    osd_0_cuda's (values, dtype, shape and device)."""
    from syndrilla.decoder.osd_0 import osd_0_cuda as osd_gpu

    it, dtype, io = stim_d5_cuda
    cfg = {"device": CUDA, "dtype": dtype, "check_type": "hx"}
    B = io["llr"].shape[0]
    all_in = torch.zeros(B, dtype=torch.long, device="cuda")
    assert 0 < int(io["converge"].sum()) < B
    for converge in (io["converge"], all_in):
        for keys in (("synd", "llr", "e_v", "iter"), ("synd", "llr", "e_v")):
            for prefix in (None, 12):
                outs = []
                for mod in (osd_cpu, osd_gpu):
                    dec = mod.create(cfg, bundle=it.matrix_bundle)
                    if prefix:
                        dec.prefix = prefix
                    sub = {k: io[k].clone() for k in keys}
                    sub.update(converge=converge.clone(), H_matrix=io["H_matrix"])
                    with torch.no_grad():
                        outs.append(dec(sub))
                label = f"prefix={prefix} all_in={converge is all_in} iter_in={'iter' in keys}"
                for k in ("e_v", "iter", "converge"):
                    a, b = outs[0][k], outs[1][k]
                    assert (a.dtype, a.shape, a.device) == (b.dtype, b.shape, b.device), (label, k)
                    assert torch.equal(a, b), (label, k)


@pytest.mark.parametrize("case", ["surface10", "stim_d5"])
def test_cpu_chunked(ref, case):
    """A workspace_bytes budget of 5 samples (the per-sample bytes of
    _scan_chunks) splits the sort and scan into chunks of at most 5 samples
    (counted via _scan calls), with the full order and with a 12-column prefix
    (so the retry stages are chunked too); e_v equals the one-chunk run and
    the reference."""
    r = ref[case]
    bundle = bundle_for(case)
    H = bundle.select("hx")[3]
    B, N = r["llr"].shape
    M = r["synd"].shape[1]
    cfg = {"device": {"device_type": "cpu"}, "dtype": "float32"}
    for prefix in (None, 12):
        whole = osd_cpu.create(cfg, bundle=bundle)
        D = whole.cols.shape[1]
        U = M // 64 + 1
        per_sample = 12 * (M + 1) * U + 16 * D * U + 64 * (M + 1) + 48 * U + 60 * N
        dec = osd_cpu.create(dict(cfg, workspace_bytes=5 * per_sample), bundle=bundle)
        calls, scan = [], dec._scan
        dec._scan = lambda cols, llr, synd, width: calls.append(len(llr)) or scan(cols, llr, synd, width)
        if prefix:
            whole.prefix = dec.prefix = prefix
        want = _run(whole, r["synd"], r["llr"], H)["e_v"]
        got = _run(dec, r["synd"], r["llr"], H)["e_v"]
        assert max(calls) <= 5 and calls[: -(-B // 5)] == [5] * (B // 5) + [B % 5] * (B % 5 > 0)
        assert torch.equal(got, want) and torch.equal(got, r["e_v"]), f"prefix={prefix}"


@pytest.mark.parametrize("group", ["pruning_opt", "memory_opt", "osd_packed_transform"])
@pytest.mark.parametrize("case", ["surface10", "inconsistent"])
def test_cpu_group_off(ref, case, group):
    """With `group` false (osd_packed_transform alone too, since memory_opt
    also turns osd_column_scan off), e_v and iter equal the default run with a small
    prefix (12 columns on surface10, 15 on the inconsistent H), so early stop
    and the prefix retry take a path there; every other sample is marked
    converged with a random input e_v."""
    if case == "inconsistent":
        bundle, synd, llr = inconsistent_case()
        k = 15
    else:
        r = ref[case]
        bundle, synd, llr, k = bundle_for(case), r["synd"], r["llr"].double(), 12
    H = bundle.select("hx")[3]
    B, N = llr.shape
    converge = (torch.arange(B) % 2).long()
    e_in = torch.randint(0, 2, (B, N), dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
    iter_in = torch.arange(B, dtype=torch.long) % 7 + 1
    cpu = {"device": {"device_type": "cpu"}}
    on = osd_cpu.create(cpu, bundle=bundle)
    on.prefix = k
    calls, scan = [], on._scan
    on._scan = lambda cols, llr, synd, width: calls.append(width) or scan(cols, llr, synd, width)
    want = _run(on, synd, llr, H, converge.clone(), e_in.clone(), iter_in.clone())
    assert len(calls) > 1  # the prefix retry ran
    off = osd_cpu.create(dict(cpu, **{group: False}), bundle=bundle)
    if group == "osd_packed_transform":
        off.prefix = k
    got = _run(off, synd, llr, H, converge.clone(), e_in.clone(), iter_in.clone())
    for key in ("e_v", "iter"):
        assert torch.equal(got[key], want[key]), key


def test_cpu_knob_resolution():
    """osd_early_stop false turns osd_prefix_scan off; a key written explicitly
    wins over its group key; groups without OSD members are ignored."""
    bundle = inconsistent_case()[0]
    cpu = {"device": {"device_type": "cpu"}}
    dec = osd_cpu.create(dict(cpu, osd_early_stop=False), bundle=bundle)
    assert not dec.osd_prefix_scan and dec.osd_solve_by_pivots
    dec = osd_cpu.create(dict(cpu, pruning_opt=False, osd_skip_converged=True, mapping_opt=False), bundle=bundle)
    assert not (dec.osd_early_stop or dec.osd_prefix_scan or dec.osd_solve_by_pivots)
    assert dec.osd_skip_converged and dec.osd_column_scan
    dec = osd_cpu.create(dict(cpu, memory_opt=False), bundle=bundle)
    assert not (dec.osd_column_scan or dec.osd_packed_transform) and dec.workspace_bytes == 1 << 60
