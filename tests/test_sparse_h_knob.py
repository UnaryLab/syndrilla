"""The top-level sparse_h knob picks the H storage (sparse COO when true, dense
int64 when false) and the stim syndrome path. Bundle index tables, syndromes
and bp_norm_min_sum + osd_0 outputs are identical for both values, on the
surface_5 alist code and a stim d=5 circuit, on CPU and on CUDA."""
import copy
import os

import pytest
import yaml
import torch

from syndrilla.decoder import create_decoder
from syndrilla.interface.stim.stim import create as stim_iface
from syndrilla.matrix import load_matrices
from syndrilla.syndrome import create_syndrome
from syndrilla.utils import read_yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = ("e_v", "iter", "converge", "llr")
P, B, SEED = 0.01, 32, 0
NOISE = (
    "after_clifford_depolarization",
    "after_reset_flip_probability",
    "before_measure_flip_probability",
    "before_round_data_depolarization",
)
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _alist_cfg():
    mcfg = read_yaml(os.path.join(ROOT, "examples/alist/surface_5.matrix.yaml"))[
        "matrix"
    ]
    for v in mcfg.values():
        if isinstance(v, dict) and "path" in v:
            v["path"] = os.path.join(ROOT, v["path"])
    return mcfg


def _stim(dev):
    return stim_iface(
        {"code": "surface_code:rotated_memory_x", "distance": 5},
        error_cfg={k: P for k in NOISE},
        syndrome_cfg={"rounds": 5},
        decoding_cfg={
            "check_type": "hx",
            "dtype": "float64",
            "device": {"device_type": dev},
        },
    )


def _bundle(mcfg, dev, **top):
    return load_matrices(
        {**copy.deepcopy(mcfg), **top}, torch.device(dev), torch.float64
    )


def _decoders(bundle, dev):
    cfg = {
        "algorithm": ["bp_norm_min_sum", "osd_0"],
        "check_type": "hx",
        "dtype": "float64",
        "device": {"device_type": dev},
        "force_pytorch": dev == "cpu",
        "config": [{"max_iter": 50}, {}],
    }
    decs = create_decoder(cfg=cfg, bundle=bundle)
    for d in decs:
        d.eval()
    return decs


def _decode(decs, synd, llr0, H):
    io = {"synd": synd.clone(), "llr0": llr0.clone(), "H_matrix": H}
    with torch.no_grad():
        for d in decs:
            io = d(io)
    return {k: io[k].cpu() for k in KEYS}


def _check_bundles(sp, de):
    assert sp.sparse_h is True and de.sparse_h is False
    for ct in ("hx", "hz"):
        a, b = sp.select(ct), de.select(ct)
        assert a[0] == b[0]
        assert torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])
        assert a[3].is_sparse and not b[3].is_sparse and b[3].dtype == torch.int64
        assert torch.equal(a[3].to_dense().long(), b[3])
    for k in ("lx_matrix", "lz_matrix"):
        assert torch.equal(
            torch.as_tensor(getattr(sp, k)), torch.as_tensor(getattr(de, k))
        )


def _check_decoders(sp, de, synd, llr0):
    dev = synd.device.type
    got = [_decode(_decoders(b, dev), synd, llr0, b.select("hx")[3]) for b in (sp, de)]
    for k in KEYS:
        assert torch.equal(got[0][k], got[1][k]), k


def test_memory_opt_false_gives_dense():
    mcfg = _alist_cfg()
    assert _bundle(mcfg, "cpu", memory_opt=False).sparse_h is False
    assert _bundle(mcfg, "cpu", memory_opt=False, sparse_h=True).sparse_h is True
    assert _bundle(mcfg, "cpu").sparse_h is True


@pytest.mark.parametrize("sparse_h", [True, False])
def test_select_returns_stored_h(sparse_h):
    b = _bundle(_alist_cfg(), "cpu", sparse_h=sparse_h)
    for ct in ("hx", "hz"):
        assert b.select(ct)[3] is b.select(ct)[3]
    assert b.select("hx")[3] is not b.select("hz")[3]


@pytest.mark.parametrize("dev", DEVICES)
def test_alist(dev):
    mcfg = _alist_cfg()
    sp, de = _bundle(mcfg, dev), _bundle(mcfg, dev, sparse_h=False)
    _check_bundles(sp, de)
    H = de.select("hx")[3].double()
    g = torch.Generator().manual_seed(SEED)
    e = (torch.rand(B, H.shape[1], generator=g) < 0.05).double().to(dev)
    synd = ((e @ H.t()) % 2).long()
    llr0 = torch.full_like(e, float(torch.log(torch.tensor((1 - P) / P))))
    _check_decoders(sp, de, synd, llr0)


@pytest.mark.parametrize("dev", DEVICES)
def test_stim(dev):
    it = _stim(dev)
    sp, de = _bundle(it.matrix_cfg, dev), _bundle(it.matrix_cfg, dev, sparse_h=False)
    _check_bundles(sp, de)

    torch.manual_seed(SEED)
    z = torch.zeros(B, it.error_model.num_errors, dtype=torch.float64, device=dev)
    _, dl = it.error_model.inject_error(z, B)
    e, llr0, _ = next(iter(dl))
    circ = str(it.circuit)
    syn = {
        v: create_syndrome(
            cfg={"measure": "stim", "rounds": 5, "circuit": circ, "sparse_h": v}
        )
        for v in (True, False)
    }
    s_sp, s_de = (syn[v].measure_syndrome(e, None) for v in (True, False))
    assert torch.equal(s_sp, s_de)
    assert torch.equal(syn[True].observable_flips, syn[False].observable_flips)
    assert s_sp.any()
    _check_decoders(sp, de, s_sp, llr0)


# (algorithm, decoding yaml, devices): decoders that read H's COO indices at init
H_INDEX_DECODERS = [
    ("union_find", "examples/alist/union_find_hx.decoding.yaml", DEVICES),
    ("mwpm", "examples/alist/mwpm_hx.decoding.yaml", ["cpu"]),
    ("saq", "examples/alist/saq_hx.decoding.yaml", ["cpu"]),
]


@pytest.mark.parametrize(
    "algo,yaml_path,dev",
    [(a, y, d) for a, y, devs in H_INDEX_DECODERS for d in devs],
)
def test_h_index_decoders(algo, yaml_path, dev):
    """union_find (union_find_cuda on CUDA), mwpm and saq give identical outputs
    for both sparse_h values on the surface_5 alist code."""
    cfg = read_yaml(os.path.join(ROOT, yaml_path))["decoding"]
    cfg["device"] = {"device_type": dev}
    if "checkpoint" in cfg.get("config", {}):
        cfg["config"]["checkpoint"] = os.path.join(ROOT, cfg["config"]["checkpoint"])
    mcfg = _alist_cfg()
    sp, de = _bundle(mcfg, dev), _bundle(mcfg, dev, sparse_h=False)
    H = de.select("hx")[3].double()
    g = torch.Generator().manual_seed(SEED)
    e = (torch.rand(B, H.shape[1], generator=g) < 0.05).double().to(dev)
    synd = ((e @ H.t()) % 2).long()
    llr0 = torch.full_like(e, float(torch.log(torch.tensor((1 - P) / P))))
    got = []
    for b in (sp, de):
        dec = create_decoder(cfg=copy.deepcopy(cfg), bundle=b)[0]
        dec.eval()
        io = {"synd": synd.clone(), "llr0": llr0.clone(), "H_matrix": b.select("hx")[3]}
        with torch.no_grad():
            io = dec(io)
        got.append(
            {
                k: v.cpu()
                for k, v in io.items()
                if k != "H_matrix" and torch.is_tensor(v)
            }
        )
    assert got[0].keys() == got[1].keys() and "e_v" in got[0]
    for k in got[0]:
        assert torch.equal(got[0][k], got[1][k]), k


def test_stim_interface_memory_opt_false(tmp_path):
    """memory_opt false in an interface yaml reaches the matrix bundle (dense H)
    and the stim syndrome measurer, and decodes as the default interface does."""
    from syndrilla.interface import create_interface

    dec_cfg = {
        "algorithm": ["bp_norm_min_sum", "osd_0"],
        "check_type": "hx",
        "dtype": "float64",
        "device": {"device_type": "cpu"},
        "force_pytorch": True,
        "config": [{"max_iter": 50}, {}],
    }
    its = []
    for top in ({}, {"memory_opt": False}):
        path = tmp_path / f"stim_{len(its)}.interface.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "interface": {
                        "backend": "stim",
                        "code": "surface_code:rotated_memory_x",
                        "distance": 5,
                        **top,
                    }
                }
            )
        )
        its.append(
            create_interface(
                str(path),
                error_cfg={k: P for k in NOISE},
                syndrome_cfg={"rounds": 5},
                decoding_cfg=copy.deepcopy(dec_cfg),
            )
        )
    sp, de = its
    assert sp.matrix_bundle.sparse_h is True and de.matrix_bundle.sparse_h is False
    assert sp.syndrome_generator.sparse_h is True
    assert de.syndrome_generator.sparse_h is False
    _check_bundles(sp.matrix_bundle, de.matrix_bundle)

    torch.manual_seed(SEED)
    z = torch.zeros(B, sp.error_model.num_errors, dtype=torch.float64)
    _, dl = sp.error_model.inject_error(z, B)
    e, llr0, _ = next(iter(dl))
    synd = [it.syndrome_generator.measure_syndrome(e, None) for it in its]
    assert torch.equal(synd[0], synd[1]) and synd[0].any()
    got = [
        _decode(it.decoders, synd[0], llr0, it.matrix_bundle.select("hx")[3])
        for it in its
    ]
    for k in KEYS:
        assert torch.equal(got[0][k], got[1][k]), k


def test_syndrome_memory_opt_false():
    """memory_opt false in the syndrome config alone gives the dense stim
    syndrome path, with the same syndromes as the sparse path."""
    it = _stim("cpu")
    torch.manual_seed(SEED)
    z = torch.zeros(B, it.error_model.num_errors, dtype=torch.float64)
    _, dl = it.error_model.inject_error(z, B)
    e, _, _ = next(iter(dl))
    base = {"measure": "stim", "rounds": 5, "circuit": str(it.circuit)}
    sp = create_syndrome(cfg=base)
    de = create_syndrome(cfg={**base, "memory_opt": False})
    assert sp.sparse_h is True and de.sparse_h is False
    s_sp, s_de = sp.measure_syndrome(e, None), de.measure_syndrome(e, None)
    assert torch.equal(s_sp, s_de) and s_sp.any()
