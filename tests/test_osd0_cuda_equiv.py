"""osd_0_cuda gives the same e_v, bit for bit, as the CPU osd_0 and as a
reference osd_0_cuda.py (full-width Gauss-Jordan over all N columns) given by the
SYNDRILLA_REF_OSD environment variable. Each test runs its CPU comparisons first
and skips, with a message, when that variable is unset or its file is missing."""
import importlib.util
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

REF_OSD = os.environ.get("SYNDRILLA_REF_OSD", "")
CUDA_CFG = {"device": {"device_type": "cuda", "device_idx": 0}, "dtype": "float64"}


class _Bundle:
    """Minimal MatrixBundle stand-in for a hand-made H."""

    def __init__(self, H):
        self.index = dense_to_index_format(H, "cpu")

    def select(self, check_type, dense=False):
        return self.index


def _reference_decoder(cfg, bundle):
    """The reference osd_0_cuda, loaded under another module name and built in
    its own extension directory; skips the calling test when unavailable."""
    if not REF_OSD:
        pytest.skip("SYNDRILLA_REF_OSD is unset: set it to a reference osd_0_cuda.py")
    if not os.path.isfile(REF_OSD):
        pytest.skip(f"SYNDRILLA_REF_OSD file not found: {REF_OSD}")
    spec = importlib.util.spec_from_file_location("osd_0_cuda_reference", REF_OSD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    root = os.environ.get("TORCH_EXTENSIONS_DIR")
    if root is None:
        from torch.utils.cpp_extension import _get_build_directory

        root = os.path.dirname(_get_build_directory("x", False))
    prev = os.environ.get("TORCH_EXTENSIONS_DIR")
    os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join(root, "osd_reference")
    try:
        return mod.create(dict(cfg), bundle=bundle)
    finally:
        if prev is None:
            del os.environ["TORCH_EXTENSIONS_DIR"]
        else:
            os.environ["TORCH_EXTENSIONS_DIR"] = prev


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


def test_surface10_matches_cpu_and_reference(surface10):
    cfg, bundle, synd, llr = surface10
    H = bundle.select("hx")[3]
    ref = _cpu_osd(bundle, synd, llr)
    for per_step in (False, True):
        dec = osd_gpu.create(dict(cfg, force_per_step=per_step), bundle=bundle)
        assert torch.equal(_run(dec, synd, llr, H), ref), f"force_per_step={per_step}"
    refk = _reference_decoder(cfg, bundle)
    assert torch.equal(_run(refk, synd, llr, H), ref)


def test_surface10_chunked(surface10):
    """A workspace_bytes budget of 5 samples splits the 64 samples into 13
    chunks; e_v and last_pivot_pos equal the one-chunk run, the CPU osd_0 and
    the reference kernel, on the fused and the per-step path."""
    cfg, bundle, synd, llr = surface10
    H = bundle.select("hx")[3]
    ref = _cpu_osd(bundle, synd, llr)
    whole = osd_gpu.create(dict(cfg), bundle=bundle)
    assert torch.equal(_run(whole, synd, llr, H), ref)
    M, rank = whole.M, whole.A_rank
    per_sample = M * max(2 * ((M + 63) >> 6), (rank + 64) >> 6) * 8
    for per_step in (False, True):
        dec = osd_gpu.create(
            dict(cfg, force_per_step=per_step, workspace_bytes=5 * per_sample),
            bundle=bundle,
        )
        assert torch.equal(_run(dec, synd, llr, H), ref), f"force_per_step={per_step}"
        assert dec.last_pivot_pos == whole.last_pivot_pos
    refk = _reference_decoder(cfg, bundle)
    assert torch.equal(_run(refk, synd, llr, H), ref)


@pytest.mark.parametrize("M", [20, 80])
def test_rank_deficient_late_pivots(M):
    """H with a duplicated row (rank < M) whose 2M lowest-LLR columns span rank 1,
    so pivots lie past order position 2M (asserted via last_pivot_pos); M=80
    leaves more than 32 rows without a pivot. e_v equals the CPU osd_0 and the
    reference kernel on the fused and the per-step path, and equals the
    reference kernel for random syndromes too."""
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
    outs = []
    for per_step in (False, True):
        dec = osd_gpu.create(dict(CUDA_CFG, force_per_step=per_step), bundle=bundle)
        assert dec.A_rank < M
        outs.append(_run(dec, synd, llr, Ht))
        assert dec.last_pivot_pos >= 2 * M  # a pivot beyond the first 2M columns
        assert torch.equal(outs[-1], ref), f"force_per_step={per_step}"
    refk = _reference_decoder(CUDA_CFG, bundle)
    assert refk.A_rank == dec.A_rank
    assert torch.equal(_run(refk, synd, llr, Ht), ref)
    # Random syndromes, mostly outside the column space of H.
    rs = torch.from_numpy(rng.integers(0, 2, (B, M))).to(torch.uint8)
    assert torch.equal(_run(dec, rs, llr, Ht), _run(refk, rs, llr, Ht))


def _gpu_variants(cfg, bundle, B):
    """(label, decoder) for fused and per-step, each with one chunk and with a
    workspace_bytes budget of 3 samples (several chunks)."""
    probe = osd_gpu.create(dict(cfg), bundle=bundle)
    M, rank = probe.M, probe.A_rank
    per_sample = M * max(2 * ((M + 63) >> 6), (rank + 64) >> 6) * 8
    out = []
    for per_step in (False, True):
        for budget in (None, 3 * per_sample):
            c = dict(cfg, force_per_step=per_step)
            if budget:
                c["workspace_bytes"] = budget
                assert B > 3
            out.append((f"per_step={per_step} chunked={bool(budget)}",
                        osd_gpu.create(c, bundle=bundle)))
    return out


def test_early_stop_low_weight(surface10):
    """Errors of weight 1 to 3 whose columns come first in the order: the
    syndrome lies in the span of the first pivots, so every scan stops before
    the last pivot position of the full scan. e_v equals the CPU osd_0 and the
    reference kernel."""
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
    refk = _reference_decoder(cfg, bundle)
    assert torch.equal(_run(refk, synd, llr, H), ref)


def test_no_early_stop_inconsistent():
    """H with a duplicated row and syndromes that differ on those two rows, so
    no syndrome is in the column space: no scan stops early, every sample uses
    all rank(H) pivots, and e_v equals the CPU osd_0 and the reference kernel."""
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
    refk = _reference_decoder(CUDA_CFG, bundle)
    assert torch.equal(_run(refk, synd, llr, H), ref)


def test_gpu_rank_fallback(monkeypatch):
    """With ldpc hidden, rank(H) comes from the GPU scan and equals the host
    reference elimination."""
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
