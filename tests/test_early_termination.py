import math
from importlib import import_module

import pytest
import torch

from syndrilla.decoder import create_decoder
from syndrilla.matrix import load_matrices
from syndrilla.utils import parse_device_dtype, read_yaml


def _case(backend, **extra):
    if backend != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cfg = dict(device={"device_type": "cpu" if backend == "cpu" else "cuda"},
               dtype="float64", max_iter=12, compile=False, early_termination=False)
    cfg.update(extra)
    matrix_cfg = read_yaml("examples/alist/surface_10.matrix.yaml")["matrix"]
    bundle = load_matrices(matrix_cfg, *parse_device_dtype(cfg))
    H = bundle.select("hx")[3].to_dense().cpu().double()
    errors = (torch.rand(32, H.shape[1], generator=torch.Generator().manual_seed(0)) < 0.1).double()
    io = {"synd": ((errors @ H.T) % 2).to(torch.uint8),
          "llr0": torch.full_like(errors, math.log(9))}
    return cfg, bundle, H, io


@pytest.mark.parametrize("backend", ["cpu", "pytorch"])
@pytest.mark.parametrize("algorithm", ["bp_norm_min_sum", "bp_lottery", "bp_lottery_policy"])
def test_full_iterations_and_hook_contract(backend, algorithm):
    module = import_module(f"syndrilla.decoder.{algorithm}.{algorithm}")
    cfg, bundle, H, io = _case(backend)
    calls, last = [], {}

    class Recorded(module.create):
        def _iter_hook(self, i, llr, hard, active, syndrome):
            expected = ~torch.all(self.syndrome_estimation(hard) == syndrome, dim=1)
            assert torch.equal(active, expected)
            assert torch.equal(self._hook_rows, torch.arange(32, device=self.device))
            before = llr.clone()
            super()._iter_hook(i, llr, hard, active, syndrome)
            assert torch.equal(llr[~active], before[~active])
            calls.append((i, bool(active.any()), bool((~active).any())))
            last.update(e_v=hard[:, :-1].clone(), llr=llr[:, :-1].clone())

    decoder = Recorded(cfg, bundle=bundle)
    with torch.no_grad():
        out = decoder(io)
    assert [i for i, _, _ in calls] == list(range(1, decoder.max_iter + 1))
    assert any(active and inactive for _, active, inactive in calls)
    assert decoder.cap is None and not decoder.cap_active_last
    assert not bool(out["defer"].any())
    assert bool((out["iter"] == decoder.max_iter).all())
    assert torch.equal(out["e_v"], last["e_v"])
    assert torch.equal(out["llr"], last["llr"])
    parity = ((out["e_v"].cpu().double() @ H.T) % 2) == io["synd"]
    assert torch.equal(out["converge"].cpu().bool(), parity.all(dim=1))


@pytest.mark.parametrize("suffix", ["", "_cuda"])
@pytest.mark.parametrize("value", ["false", 0, None])
def test_non_bool_rejected(suffix, value):
    module = import_module(f"syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum{suffix}")
    cfg, bundle, _, _ = _case("cpu", early_termination=value)
    with pytest.raises(ValueError, match="early_termination must be a bool"):
        module.create(cfg, bundle=bundle)


@pytest.mark.parametrize("algorithm", ["bp_norm_min_sum", "bp_lottery"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("skip_converged", [True, False])
def test_cuda_run_to_end_matches_pytorch(algorithm, dtype, skip_converged, monkeypatch):
    from syndrilla.decoder.bp_lottery.bp_lottery import is_flip_iter

    cfg, bundle, H, io = _case("cuda", dtype=dtype, skip_converged=skip_converged,
                             random_machine="sobol", rebatch_opt=True, host_check_every=1)
    outputs, histories, flips = [], [], []
    for suffix in ("", "_cuda"):
        module = import_module(f"syndrilla.decoder.{algorithm}.{algorithm}{suffix}")
        history, selected, launches = [], [], []

        class Recorded(module.create):
            def _iter_hook(self, i, llr, hard, active, syndrome):
                parity = hard[:, self.V_c_col].long().sum(2) % 2
                assert torch.equal(active, (parity != syndrome).any(1))
                before = llr.clone()
                history.append((hard[:, :-1].bool().clone(), active.clone()))
                super()._iter_hook(i, llr, hard, active, syndrome)
                assert torch.equal(llr[~active], before[~active])
                if algorithm == "bp_lottery" and is_flip_iter(self, i):
                    selected.append((self.last_flip_cn.clone(), self.last_flip_vn.clone(), self.last_flip_valid.clone()))

        decoder = Recorded(cfg, bundle=bundle)
        if suffix:
            native = module.create(cfg, bundle=bundle)
            assert not native._use_persistent(32, False)
            with torch.no_grad():
                native_out = native({key: value.clone() for key, value in io.items()})
            assert not decoder._use_persistent(32, False)
            step = decoder._ext.vn_cn_update_csr

            def counted(*args, **kwargs):
                launches.append(1)
                return step(*args, **kwargs)

            monkeypatch.setattr(decoder._ext, "vn_cn_update_csr", counted)
        with torch.no_grad():
            out = decoder({key: value.clone() for key, value in io.items()})
        assert len(history) == decoder.max_iter
        assert not suffix or len(launches) == decoder.max_iter
        assert decoder.cap is None and not decoder.cap_active_last
        assert bool((out["iter"] == decoder.max_iter).all()) and not bool(out["defer"].any())
        parity = ((out["e_v"].cpu().double() @ H.T) % 2) == io["synd"]
        assert torch.equal(out["converge"].cpu().bool(), parity.all(1))
        if suffix:
            assert all(torch.equal(out[key], native_out[key]) for key in ("e_v", "llr", "iter", "converge", "defer"))
        outputs.append(out)
        histories.append(history)
        flips.append(selected)
    for key in ("e_v", "llr", "iter", "converge", "defer"):
        assert torch.equal(outputs[0][key], outputs[1][key]), key
    assert len(histories[0]) == len(histories[1])
    assert len(flips[0]) == len(flips[1])
    for left, right in zip(histories[0] + flips[0], histories[1] + flips[1]):
        assert all(torch.equal(a, b) for a, b in zip(left, right))


@pytest.mark.parametrize("algorithm", ["bp4", "bp_branch_assisted", "relay_bp", "osd_0", "union_find"])
def test_factory_rejects_unsupported_run_to_end(algorithm):
    cfg, bundle, _, _ = _case("cpu")
    config = {key: cfg.pop(key) for key in ("max_iter", "compile", "early_termination")}
    cfg.update(algorithm=algorithm, check_type="hx", config=config)
    with pytest.raises(ValueError, match=f"{algorithm} does not support early_termination=false"):
        create_decoder(cfg=cfg, bundle=bundle)


def test_factory_accepts_supported_run_to_end():
    cfg, bundle, _, _ = _case("cpu")
    config = {key: cfg.pop(key) for key in ("max_iter", "compile", "early_termination")}
    cfg.update(algorithm="bp_norm_min_sum", check_type="hx", config=config)
    decoder = create_decoder(cfg=cfg, bundle=bundle)[0]
    assert decoder.early_termination is False


@pytest.mark.parametrize("algorithm,extra", [
    ("bp_lottery", {}),
    ("bp_lottery_quant", {}),
    ("bp_lottery_policy", {}),
    ("bp_lottery_policy", {"random_machine": "sobol", "sign_flip_policy": "local_random"}),
])
@pytest.mark.parametrize("zero_error", [True, False])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_run_to_end_rng_and_flips_match(algorithm, extra, zero_error, dtype):
    cfg, bundle, _, io = _case("cuda", dtype=dtype, random_machine="system", rebatch_opt=False)
    cfg.update(extra)
    if zero_error:
        io["synd"].zero_()
    outputs, histories, flips, rng_states = [], [], [], []
    for suffix in ("", "_cuda"):
        module = import_module(f"syndrilla.decoder.{algorithm}.{algorithm}{suffix}")
        history, changes = [], []

        class Recorded(module.create):
            def _iter_hook(self, i, llr, hard, active, syndrome):
                history.append((hard[:, :-1].bool().clone(), active.clone()))
                before = llr.clone()
                super()._iter_hook(i, llr, hard, active, syndrome)
                changes.append(llr != before)
                assert not bool(changes[-1][~active].any())

        decoder = Recorded(cfg, bundle=bundle)
        torch.manual_seed(71)
        with torch.no_grad():
            outputs.append(decoder({key: value.clone() for key, value in io.items()}))
        rng_states.append(torch.cuda.get_rng_state().clone())
        histories.append(history)
        flips.append(changes)
        assert len(history) == len(changes) == decoder.max_iter
        if zero_error:
            assert not any(bool(active.any()) for _, active in history)
    assert torch.equal(rng_states[0], rng_states[1])
    assert len(histories[0]) == len(histories[1]) and len(flips[0]) == len(flips[1])
    for key in ("e_v", "llr", "iter", "converge"):
        assert torch.equal(outputs[0][key], outputs[1][key]), key
    for left, right in zip(histories[0], histories[1]):
        assert all(torch.equal(a, b) for a, b in zip(left, right))
    assert all(torch.equal(a, b) for a, b in zip(flips[0], flips[1]))
