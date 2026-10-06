import math

import pytest
import torch

from syndrilla.decoder.decoder import _create_one_decoder, create_decoder
from syndrilla.matrix import load_matrices
from syndrilla.utils import parse_device_dtype, read_yaml

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
ALGORITHMS = ("bp_memory", "bp_ots", "bp_self_corrected_min_sum")


@pytest.fixture(scope="module")
def bundle():
    cfg = {"device": {"device_type": "cuda", "device_idx": 0}, "dtype": "float64"}
    return load_matrices(
        read_yaml("examples/alist/surface_5.matrix.yaml")["matrix"],
        *parse_device_dtype(cfg),
    )


@pytest.mark.parametrize("algo", ALGORITHMS)
@pytest.mark.parametrize("check_type", ["hx", "hz"])
def test_factory_and_repeated_batches(bundle, algo, check_type):
    cfg = read_yaml(f"examples/alist/{algo}_hx.decoding.yaml")["decoding"]
    cfg = {**cfg, **cfg.pop("config"), "check_type": check_type, "max_iter": 20}
    dec = _create_one_decoder(cfg, "decoding", "algorithm", bundle=bundle)
    H = bundle.select(check_type)[3].to_dense().double()
    torch.manual_seed(9)
    errors = (torch.rand(16, H.shape[1], device="cuda") < 0.08).double()
    synd = ((errors @ H.T) % 2).to(torch.uint8)
    prior = torch.full_like(errors, math.log(0.92 / 0.08))
    inputs = {"synd": synd, "llr0": prior}
    a = dec(dict(inputs))
    b = dec(dict(inputs))
    for key in ("e_v", "llr", "iter", "converge", "defer"):
        assert torch.equal(a[key], b[key])
    assert a["llr"].shape == prior.shape
    assert not a["defer"].any()
    assert torch.equal(((a["e_v"] @ H.T) % 2).to(torch.uint8)[a["converge"].bool()],
                       synd[a["converge"].bool()])
    assert torch.equal(inputs["synd"], synd)
    assert torch.equal(inputs["llr0"], prior)


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_empty_batch(bundle, algo):
    cfg = dict(algorithm=algo, device={"device_type": "cuda"}, max_iter=3)
    dec = _create_one_decoder(cfg, "decoding", "algorithm", bundle=bundle)
    result = dec({"synd": torch.empty(0, dec.H_shape[0]),
                  "llr0": torch.empty(0, dec.N)})
    assert result["llr"].shape == (0, dec.N)
    assert result["iter"].shape == (0,)


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_public_factory_rounds_and_syndrome_generator(bundle, algo):
    from syndrilla.syndrome.perfect.perfect import create

    dec = create_decoder(
        yaml_path=f"examples/alist/{algo}_hx.decoding.yaml", bundle=bundle,
    )[0]
    error = torch.zeros(2, dec.N, device="cuda", dtype=torch.float64)
    syndrome = create({}).measure_syndrome(error, dec)
    syndrome = syndrome[:, None, :].expand(-1, 3, -1)
    result = dec({"synd": syndrome, "llr0": torch.full_like(error, 2)})
    assert result["e_v"].shape == (2, 3, dec.N)
    assert result["converge"].shape == (2, 3)
    assert result["converge"].all()


@pytest.mark.parametrize("algo,extra,match", [
    ("bp_memory", {"alpha": 0}, "alpha"),
    ("bp_ots", {"interval": 0}, "interval"),
    ("bp_ots", {"strength": float("nan")}, "strength"),
    ("bp_memory", {"dtype": "float16"}, "dtype"),
    ("bp_self_corrected_min_sum", {"max_iter": 0}, "max_iter"),
])
def test_invalid_configuration(bundle, algo, extra, match):
    cfg = dict(algorithm=algo, device={"device_type": "cuda"}, **extra)
    with pytest.raises(ValueError, match=match):
        _create_one_decoder(cfg, "decoding", "algorithm", bundle=bundle)
