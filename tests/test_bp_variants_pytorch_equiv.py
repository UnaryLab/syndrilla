"""Each PyTorch BP variant other than bp_norm_min_sum gives the same outputs on CPU,
bit for bit, as frozen references saved from an earlier tree. A reference lives in
REF/<variant>/<case>_<dtype>/ as inputs.pt (synd, llr0), outputs.pt (e_v, llr,
iter, converge) and config.json (seeds and the decoding block, from the variant's
example yaml with check_type hx). Cases: stim rotated_memory_x d=5 rounds=5 at
p=3e-3 on all four noise knobs, and surface_10 hx from alist with BSC p=0.03 (bp4:
depolarizing p=0.03, its 2-channel syndrome), at float64 and float32, B=64. Tests
fail when the reference files are missing, with the command that rebuilds them."""
import json
import os
import random

import numpy as np
import pytest
import torch

from syndrilla.decoder import create_decoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.environ.get("SYNDRILLA_BP_VARIANT_PT_REF", "/tmp/claude-1110527820/t38/refs")
CPU = {"device_type": "cpu", "device_idx": 0}
KEYS = ("e_v", "llr", "iter", "converge")
SEED = 0
NOISE = (
    "after_clifford_depolarization",
    "after_reset_flip_probability",
    "before_measure_flip_probability",
    "before_round_data_depolarization",
)
VARIANTS = (
    "bp_sf",
    "bp_lottery",
    "bp_lottery_policy",
    "bp_norm_min_sum_quant",
    "bp_lottery_quant",
    "bp_branch_assisted",
    "relay_bp",
    "bp_sum_prod",
)
CASES = [(v, c) for v in VARIANTS for c in ("stim_d5", "surface10")]
CASES.append(("bp4", "surface10_depol"))


def bundle(case, dtype):
    """The matrix bundle of the case on CPU."""
    if case == "stim_d5":
        from syndrilla.interface.stim.stim import create as stim_iface

        return stim_iface(
            {"code": "surface_code:rotated_memory_x", "distance": 5},
            error_cfg={k: 3e-3 for k in NOISE},
            syndrome_cfg={"rounds": 5},
            decoding_cfg={"dtype": dtype, "device": CPU},
        ).matrix_bundle
    from syndrilla.matrix import load_matrices
    from syndrilla.utils import read_yaml

    mcfg = read_yaml(os.path.join(ROOT, "examples", "alist", "surface_10.matrix.yaml"))[
        "matrix"
    ]
    for v in mcfg.values():
        if isinstance(v, dict) and "path" in v:
            v["path"] = os.path.join(ROOT, v["path"])
    return load_matrices(mcfg, torch.device("cpu"), getattr(torch, dtype))


def decode(cfg, case, dtype, synd, llr0):
    """Build the PyTorch decoder on CPU, seed torch, numpy and python random with
    SEED, run forward once and return the outputs."""
    dec = create_decoder(cfg={**cfg, "device": CPU}, bundle=bundle(case, dtype))[0]
    dec = getattr(dec, "decoder", dec)
    assert not type(dec).__module__.endswith("_cuda"), type(dec).__module__
    default = torch.get_default_dtype()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    try:
        with torch.no_grad():
            out = dec({"synd": synd.clone(), "llr0": llr0.to(getattr(torch, dtype))})
    finally:
        torch.set_default_dtype(default)  # bp_sum_prod sets it globally
    return {k: out[k] for k in KEYS}


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("variant,case", CASES, ids=[f"{v}-{c}" for v, c in CASES])
def test_matches_reference(variant, case, dtype):
    d = os.path.join(REF, variant, f"{case}_{dtype}")
    if not all(
        os.path.isfile(os.path.join(d, f))
        for f in ("inputs.pt", "outputs.pt", "config.json")
    ):
        pytest.fail(
            f"no reference in {d}. Set SYNDRILLA_BP_VARIANT_PT_REF, or rebuild all of them"
            " with tests/freeze_bp_variant_pytorch_refs.py run against a tree whose variant"
            " files match the references (needs a GPU for the stim_d5 inputs):\n"
            f"conda run -n syndrilla python tests/freeze_bp_variant_pytorch_refs.py {REF}"
        )
    with open(os.path.join(d, "config.json")) as f:
        meta = json.load(f)
    x = torch.load(os.path.join(d, "inputs.pt"))
    ref = torch.load(os.path.join(d, "outputs.pt"))
    got = decode(meta["decoder_cfg"], case, dtype, x["synd"], x["llr0"])
    for k in KEYS:
        assert got[k].dtype == ref[k].dtype, (k, got[k].dtype, ref[k].dtype)
        diff = (got[k] != ref[k]).reshape(len(ref[k]), -1).any(1)
        assert torch.equal(
            got[k], ref[k]
        ), f"{k} differs in samples {diff.nonzero().flatten().tolist()}"
