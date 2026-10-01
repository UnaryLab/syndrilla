"""osd_0_cuda gives the same e_v, bit for bit, as the CPU osd_0 on the fused
and the per-step path, with one chunk and with several chunks. The tests also
check last_pivot_pos, the early-stop flags and pivot counts in scan_stats, and
the GPU rank fallback, and that scan pool overflow, the order prefix (with
ties at its boundary) and split elimination batches leave e_v unchanged."""
import os

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

from syndrilla.decoder import create_decoder
from syndrilla.decoder.osd_0 import osd_0 as osd_cpu
from syndrilla.decoder.osd_0 import osd_0_cuda as osd_gpu
from syndrilla.error_model import create_error_model
from syndrilla.matrix import load_matrices
from syndrilla.matrix.matrix import dense_to_index_format
from syndrilla.syndrome import create_syndrome
from syndrilla.utils import parse_device_dtype, read_yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CUDA_CFG = {"device": {"device_type": "cuda", "device_idx": 0}, "dtype": "float64"}


class _Bundle:
    """Minimal MatrixBundle stand-in for a hand-made H."""

    def __init__(self, H):
        self.index = dense_to_index_format(H, "cpu")

    def select(self, check_type):
        return self.index


def _run(dec, synd, llr, H):
    B, N = llr.shape
    dev = (
        dec.device if isinstance(dec.device, torch.device) else torch.device(dec.device)
    )
    io = {
        "synd": synd.to(dev),
        "llr": llr.to(dev),
        "converge": torch.zeros(B, dtype=torch.long, device=dev),
        "e_v": torch.zeros(B, N, dtype=torch.uint8, device=dev),
        "H_matrix": H.to(dev),
    }
    with torch.no_grad():
        return dec(io)["e_v"].to(torch.uint8).cpu()


def _cpu_osd(bundle, synd, llr):
    H = bundle.select("hx")[3]
    dec = osd_cpu.create(
        {"device": {"device_type": "cpu"}, "dtype": "float64"}, bundle=bundle
    )
    return _run(dec, synd, llr, H)


@pytest.fixture(scope="module")
def surface10():
    """64 BP outputs on surface_10 with the bposd_hx config (all sent to OSD)."""
    torch.manual_seed(0)
    np.random.seed(0)
    ex = os.path.join(ROOT, "examples", "alist")
    cfg = read_yaml(os.path.join(ex, "bposd_hx.decoding.yaml"))["decoding"]
    mcfg = read_yaml(os.path.join(ex, "surface_10.matrix.yaml"))["matrix"]
    for v in mcfg.values():
        if isinstance(v, dict) and "path" in v:
            v["path"] = os.path.join(ROOT, v["path"])
    bundle = load_matrices(mcfg, *parse_device_dtype(cfg))
    bp = create_decoder(cfg=dict(cfg, algorithm=["bp_norm_min_sum"]), bundle=bundle)[0]
    bp.eval()
    em = create_error_model(os.path.join(ex, "bsc.error.yaml"))
    sg = create_syndrome(os.path.join(ex, "perfect.syndrome.yaml"))
    B, N = 64, bundle.select("hx")[0][1]
    _, loader = em.inject_error(torch.zeros([B, N], dtype=bp.dtype), B)
    err, llr0, _ = next(iter(loader))
    synd = sg.measure_syndrome(err, bp)
    H = bundle.select("hx")[3]
    with torch.no_grad():
        io = bp({"synd": synd, "llr0": llr0, "H_matrix": H})
    return cfg, bundle, io["synd"].cpu(), io["llr"].double().cpu()


def test_surface10_matches_cpu(surface10):
    cfg, bundle, synd, llr = surface10
    H = bundle.select("hx")[3]
    ref = _cpu_osd(bundle, synd, llr)
    for per_step in (False, True):
        dec = osd_gpu.create(dict(cfg, force_per_step=per_step), bundle=bundle)
        assert torch.equal(_run(dec, synd, llr, H), ref), f"force_per_step={per_step}"


def test_surface10_chunked(surface10):
    """A workspace_bytes budget of 5 scan samples (_scan_bytes) splits the 64
    samples into 13 scan chunks (counted via _scan calls); e_v and
    last_pivot_pos equal the one-chunk run and the CPU osd_0, on the fused and
    the per-step path."""
    cfg, bundle, synd, llr = surface10
    H = bundle.select("hx")[3]
    ref = _cpu_osd(bundle, synd, llr)
    whole = osd_gpu.create(dict(cfg), bundle=bundle)
    assert torch.equal(_run(whole, synd, llr, H), ref)
    per_sample = whole._scan_bytes(whole.pool_rows, whole.pool_cols, whole.A_rank)
    for per_step in (False, True):
        dec = osd_gpu.create(
            dict(cfg, force_per_step=per_step, workspace_bytes=5 * per_sample),
            bundle=bundle,
        )
        calls, scan = [], dec._scan
        dec._scan = lambda *a: calls.append(a[0].shape[0]) or scan(*a)
        assert torch.equal(_run(dec, synd, llr, H), ref), f"force_per_step={per_step}"
        assert calls == [5] * 12 + [4]
        assert dec.last_pivot_pos == whole.last_pivot_pos


@pytest.mark.parametrize("M", [20, 80])
def test_rank_deficient_late_pivots(M):
    """H with a duplicated row (rank < M) whose 2M lowest-LLR columns span rank 1,
    so pivots lie past order position 2M (asserted via last_pivot_pos); M=80
    leaves more than 32 rows without a pivot. e_v equals the CPU osd_0 on the
    fused and the per-step path."""
    rng = np.random.default_rng(M)
    N, B = 20 * M, 8
    H = (rng.random((M, N)) < 0.08).astype(np.uint8)
    H[1] = H[0]  # rank(H) < M
    H[:, : 2 * M] = 0
    H[2, : 2 * M] = 1  # first 2M columns span rank 1
    bundle = _Bundle(H)
    e = (rng.random((B, N)) < 0.05).astype(np.int64)
    synd = torch.from_numpy((e @ H.T.astype(np.int64)) % 2).to(torch.uint8)
    llr = torch.from_numpy(rng.random((B, N)) + 1.0)
    llr[:, : 2 * M] -= 1.0  # these columns sort first
    Ht = bundle.select("hx")[3]

    ref = _cpu_osd(bundle, synd, llr)
    for per_step in (False, True):
        dec = osd_gpu.create(dict(CUDA_CFG, force_per_step=per_step), bundle=bundle)
        assert dec.A_rank < M
        out = _run(dec, synd, llr, Ht)
        assert dec.last_pivot_pos >= 2 * M  # a pivot beyond the first 2M columns
        assert torch.equal(out, ref), f"force_per_step={per_step}"


def _gpu_variants(cfg, bundle, B):
    """(label, decoder) for fused and per-step, each with one chunk and with a
    workspace_bytes budget of 3 scan samples (several chunks)."""
    probe = osd_gpu.create(dict(cfg), bundle=bundle)
    per_sample = probe._scan_bytes(probe.pool_rows, probe.pool_cols, probe.A_rank)
    out = []
    for per_step in (False, True):
        for budget in (None, 3 * per_sample):
            c = dict(cfg, force_per_step=per_step)
            if budget:
                c["workspace_bytes"] = budget
                assert B > 3
            out.append(
                (
                    f"per_step={per_step} chunked={bool(budget)}",
                    osd_gpu.create(c, bundle=bundle),
                )
            )
    return out


def test_early_stop_low_weight(surface10):
    """Errors of weight 1 to 3 whose columns come first in the order: the
    syndrome lies in the span of the first pivots, so every scan stops before
    the last pivot position of the full scan. e_v equals the CPU osd_0."""
    cfg, bundle, _, _ = surface10
    H = bundle.select("hx")[3]
    Hd = H.to_dense().cpu().numpy().astype(np.int64)
    M, N = Hd.shape
    B = 16
    rng = np.random.default_rng(3)
    e = np.zeros((B, N), np.int64)
    for b in range(B):
        e[b, rng.choice(N, size=1 + b % 3, replace=False)] = 1
    synd = torch.from_numpy((e @ Hd.T) % 2).to(torch.uint8)
    llr = torch.from_numpy(rng.random((B, N)) + 1.0 - e)  # error columns sort first
    ref = _cpu_osd(bundle, synd, llr)
    for label, dec in _gpu_variants(cfg, bundle, B):
        assert torch.equal(_run(dec, synd, llr, H), ref), label
        st = dec.scan_stats
        assert bool(st["stopped"].all()), label
        order = torch.sort(llr.to(dec.device), dim=1, stable=True)[1].int().contiguous()
        piv_pos, found, _, _ = dec._scan(order, dec.A_rank)
        assert bool((found == dec.A_rank).all())
        full_last = piv_pos[:, -1].cpu()
        assert bool((st["end"].cpu() - 1 < full_last).all()), label
        assert bool((st["pivots"].cpu() < dec.A_rank).all()), label


def test_no_early_stop_inconsistent():
    """H with a duplicated row and syndromes that differ on those two rows, so
    no syndrome is in the column space: no scan stops early, every sample uses
    all rank(H) pivots, and e_v equals the CPU osd_0."""
    rng = np.random.default_rng(4)
    M, N, B = 30, 240, 12
    Hm = (rng.random((M, N)) < 0.1).astype(np.uint8)
    Hm[1] = Hm[0]
    bundle = _Bundle(Hm)
    s = rng.integers(0, 2, (B, M))
    s[:, 1] = 1 - s[:, 0]
    synd = torch.from_numpy(s).to(torch.uint8)
    llr = torch.from_numpy(rng.random((B, N)))
    H = bundle.select("hx")[3]
    ref = _cpu_osd(bundle, synd, llr)
    for label, dec in _gpu_variants(CUDA_CFG, bundle, B):
        assert torch.equal(_run(dec, synd, llr, H), ref), label
        st = dec.scan_stats
        assert not bool(st["stopped"].any()), label
        assert bool((st["pivots"] == dec.A_rank).all()), label


def test_gpu_rank_fallback(monkeypatch):
    """With ldpc hidden, rank(H) comes from the GPU scan and equals the host
    elimination in _gf2_rank."""
    rng = np.random.default_rng(2)
    H = (rng.random((30, 90)) < 0.1).astype(np.uint8)
    H[5] = H[3] ^ H[4]
    rows, cols = (x.astype(np.int64) for x in np.nonzero(H))
    ref = osd_gpu._gf2_rank(osd_gpu._pack_H(rows, cols, 30, 2), 30, 90)
    dec = osd_gpu.create(dict(CUDA_CFG), bundle=_Bundle(H))
    import builtins

    real_import = builtins.__import__

    def no_ldpc(name, *a, **k):
        if name.startswith("ldpc"):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_ldpc)
    assert dec._rank(rows, cols) == (ref, "GPU scan")
    assert dec.A_rank == ref


def _same_stats(dec, ref):
    """scan_stats end / pivots / stopped and last_pivot_pos equal ref's."""
    for key in ("end", "pivots", "stopped"):
        assert torch.equal(dec.scan_stats[key], ref.scan_stats[key]), key
    assert dec.last_pivot_pos == ref.last_pivot_pos


def _same_scan(dec, ref, llr):
    """_scan on the full order returns the same 4-tuple for dec and ref."""
    order = torch.sort(llr.to(dec.device), dim=1, stable=True)[1].int().contiguous()
    for a, b in zip(dec._scan(order, dec.A_rank), ref._scan(order, ref.A_rank)):
        assert torch.equal(a, b)


@pytest.mark.parametrize("pool", [1, 12])
def test_pool_overflow(surface10, pool):
    """Scan pools of `pool` Tr and TuT rows (M = 90): with 1 row every sample
    overflows and is rescanned with the full T; with 12 rows some samples
    overflow and some do not. e_v, scan_stats, last_pivot_pos and the _scan
    4-tuple equal the default pools (which never overflow here) and the CPU
    osd_0."""
    cfg, bundle, synd, llr = surface10
    H = bundle.select("hx")[3]
    ref = osd_gpu.create(dict(cfg), bundle=bundle)
    ref_ev = _run(ref, synd, llr, H)
    assert not bool(ref.scan_stats["overflow"].any())
    dec = osd_gpu.create(dict(cfg), bundle=bundle)
    dec.pool_rows = dec.pool_cols = pool
    out = _run(dec, synd, llr, H)
    ovf = dec.scan_stats["overflow"]
    assert torch.equal(ovf, dec.last_overflow)
    if pool == 1:
        assert bool(ovf.all())
    else:
        assert bool(ovf.any()) and not bool(ovf.all())
    assert torch.equal(out, ref_ev)
    assert torch.equal(out, _cpu_osd(bundle, synd, llr))
    _same_stats(dec, ref)
    _same_scan(dec, ref, llr)


def test_prefix_redo(surface10):
    """A 12-column prefix of the order (N = 181): samples whose full scan ends
    at or past column 12 are solved again on the first 48 columns, and those
    whose full scan ends at or past column 48 are then solved with the full
    order. The sample count of each stage is read from the _solve calls; e_v,
    scan_stats and last_pivot_pos equal the full-sort run."""
    cfg, bundle, synd, llr = surface10
    H = bundle.select("hx")[3]
    ref = osd_gpu.create(dict(cfg), bundle=bundle)
    assert ref.prefix == ref.N  # N < 16384: full sort by default
    ref_ev = _run(ref, synd, llr, H)
    k = 12
    retry = int((ref.scan_stats["end"] > k).sum())
    full = int((ref.scan_stats["end"] > 4 * k).sum())
    assert 0 < full < retry < llr.shape[0]

    dec = osd_gpu.create(dict(cfg), bundle=bundle)
    dec.prefix = k
    widths, solve = [], dec._solve

    def spy(s, order):
        widths.append(tuple(order.shape))
        return solve(s, order)

    dec._solve = spy
    out = _run(dec, synd, llr, H)
    assert widths == [(llr.shape[0], k), (retry, 4 * k), (full, dec.N)]
    assert torch.equal(out, ref_ev)
    _same_stats(dec, ref)


def test_prefix_ties(surface10):
    """LLRs rounded to 4 levels, so the value at the 30-column prefix boundary
    is shared by columns inside and outside the prefix (on some samples). e_v equals the CPU
    osd_0 and the full-sort run; scan_stats equal the full-sort run."""
    cfg, bundle, synd, llr = surface10
    H = bundle.select("hx")[3]
    k = 30
    lo, hi = llr.min(), llr.max()
    q = torch.round((llr - lo) / (hi - lo) * 3)  # levels 0..3
    srt = torch.sort(q, dim=1, stable=True)[0]
    assert bool((srt[:, k - 1] == srt[:, k]).any())  # ties cross the boundary
    ref = osd_gpu.create(dict(cfg), bundle=bundle)
    ref_ev = _run(ref, synd, q, H)
    dec = osd_gpu.create(dict(cfg), bundle=bundle)
    dec.prefix = k
    out = _run(dec, synd, q, H)
    assert torch.equal(out, ref_ev)
    assert torch.equal(out, _cpu_osd(bundle, synd, q))
    _same_stats(dec, ref)


def test_elimination_sub_batches():
    """M = 400 with a duplicated row and inconsistent syndromes, so every
    sample uses all rank(H) pivots (K >= 320). A workspace_bytes budget of 3
    elimination samples splits each scan chunk into several elimination
    batches (counted via _eliminate calls). e_v and scan_stats equal the
    default budget."""
    rng = np.random.default_rng(5)
    M, N, B = 400, 1200, 12
    Hm = (rng.random((M, N)) < 0.01).astype(np.uint8)
    Hm[1] = Hm[0]
    bundle = _Bundle(Hm)
    s = rng.integers(0, 2, (B, M))
    s[:, 1] = 1 - s[:, 0]
    synd = torch.from_numpy(s).to(torch.uint8)
    llr = torch.from_numpy(rng.random((B, N)))
    H = bundle.select("hx")[3]

    ref = osd_gpu.create(dict(CUDA_CFG), bundle=bundle)
    ref_ev = _run(ref, synd, llr, H)
    K = int(ref.scan_stats["pivots"].max())
    elim = M * (((K + 64) >> 6) * 8 + 16)
    budget = 3 * elim
    chunk = budget // ref._scan_bytes(ref.pool_rows, ref.pool_cols, ref.A_rank)
    assert 3 < chunk  # several elimination batches per scan chunk
    n_chunks = -(-B // chunk)

    dec = osd_gpu.create(dict(CUDA_CFG, workspace_bytes=budget), bundle=bundle)
    calls, elim_fn = [], dec._eliminate

    def spy(s_, cols, rank):
        calls.append(cols.shape[0])
        return elim_fn(s_, cols, rank)

    dec._eliminate = spy
    out = _run(dec, synd, llr, H)
    assert max(calls) <= 3 and sum(calls) == B
    assert len(calls) > n_chunks
    assert torch.equal(out, ref_ev)
    _same_stats(dec, ref)


def _run_partly_converged(dec, synd, llr):
    """e_v and iter with every other sample marked converged (input e_v
    random, input iter 1 to 7)."""
    B, N = llr.shape
    dev = dec.device
    io = {
        "synd": synd.to(dev),
        "llr": llr.to(dev),
        "converge": (torch.arange(B, device=dev) % 2).long(),
        "e_v": torch.randint(
            0, 2, (B, N), dtype=torch.uint8, generator=torch.Generator().manual_seed(0)
        ).to(dev),
        "iter": (torch.arange(B, device=dev) % 7 + 1).long(),
    }
    with torch.no_grad():
        out = dec(io)
    return out["e_v"].cpu(), out["iter"].cpu()


@pytest.mark.parametrize("group", ["pruning_opt", "memory_opt"])
@pytest.mark.parametrize("case", ["surface10", "inconsistent"])
def test_group_off(surface10, case, group):
    """With `group` false, e_v and iter equal the default run with a small
    prefix (12 columns on surface10, 15 on the inconsistent H), so early stop
    and the prefix retry take a path there; every other sample is marked
    converged. Fused and per-step path."""
    if case == "surface10":
        cfg, bundle, synd, llr = surface10
        k = 12
    else:
        rng = np.random.default_rng(4)
        M, N, B = 30, 240, 12
        Hm = (rng.random((M, N)) < 0.1).astype(np.uint8)
        Hm[1] = Hm[0]
        bundle = _Bundle(Hm)
        s = rng.integers(0, 2, (B, M))
        s[:, 1] = 1 - s[:, 0]
        synd = torch.from_numpy(s).to(torch.uint8)
        llr = torch.from_numpy(rng.random((B, N)))
        cfg, k = CUDA_CFG, 15
    for per_step in (False, True):
        on = osd_gpu.create(dict(cfg, force_per_step=per_step), bundle=bundle)
        on.prefix = k
        widths, solve = [], on._solve
        on._solve = lambda s, o: widths.append(o.shape[1]) or solve(s, o)
        want = _run_partly_converged(on, synd, llr)
        assert len(widths) > 1  # the prefix retry ran
        off = osd_gpu.create(
            dict(cfg, force_per_step=per_step, **{group: False}), bundle=bundle
        )
        assert off.prefix == off.N
        got = _run_partly_converged(off, synd, llr)
        for a, b in zip(got, want):
            assert torch.equal(a, b), f"per_step={per_step}"


def test_knob_resolution():
    """osd_early_stop false turns osd_prefix_scan off; a key written explicitly
    wins over its group key; groups without OSD members are ignored."""
    H = (np.random.default_rng(2).random((30, 90)) < 0.1).astype(np.uint8)
    bundle = _Bundle(H)
    dec = osd_gpu.create(dict(CUDA_CFG, osd_early_stop=False), bundle=bundle)
    assert not dec.osd_prefix_scan and dec.osd_solve_by_pivots
    dec = osd_gpu.create(
        dict(CUDA_CFG, pruning_opt=False, osd_skip_converged=True, fusion_opt=False),
        bundle=bundle,
    )
    assert not (dec.osd_early_stop or dec.osd_prefix_scan or dec.osd_solve_by_pivots)
    assert dec.osd_skip_converged and dec.osd_column_scan
    dec = osd_gpu.create(dict(CUDA_CFG, memory_opt=False), bundle=bundle)
    assert not dec.osd_column_scan and dec.workspace_bytes == 1 << 60
