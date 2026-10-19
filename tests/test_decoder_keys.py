"""Decoder configs reject unknown keys before matrix or device setup."""

import importlib
import sys
from pathlib import Path

import pytest
import torch
import yaml

from syndrilla.decoder import create_decoder
from syndrilla.decoder.decoder import resolve_configs
from syndrilla.decoder import knobs
from syndrilla.matrix import load_matrices

ROOT = Path(__file__).resolve().parents[1]
DECODER_DIR = ROOT / "src/syndrilla/decoder"
MODULE_PATHS = sorted(
    path for path in DECODER_DIR.glob("*/*.py")
    if path.stem in {path.parent.name, f"{path.parent.name}_cuda"}
)
MODULES = [
    importlib.import_module(".".join(path.relative_to(ROOT / "src").with_suffix("").parts))
    for path in MODULE_PATHS
]
MODULE_IDS = [f"{path.parent.name}/{path.stem}" for path in MODULE_PATHS]
EXAMPLE_CASES = []
MISSING_EXAMPLE_ALGORITHMS = []
for path in sorted((ROOT / "examples").rglob("*.decoding.yaml")):
    cfg = yaml.safe_load(path.read_text())["decoding"]
    algorithms = cfg["algorithm"]
    algorithms = [algorithms] if isinstance(algorithms, str) else algorithms
    for stage, (algorithm, flat) in enumerate(zip(algorithms, resolve_configs(cfg))):
        found = False
        for module_path, module in zip(MODULE_PATHS, MODULES):
            if module_path.parent.name == algorithm:
                flat = {**flat, "algorithm": algorithm}
                EXAMPLE_CASES.append(pytest.param(
                    module, flat,
                    id=f"{path.relative_to(ROOT)}:{stage}:{module_path.stem}",
                ))
                found = True
        if not found:
            MISSING_EXAMPLE_ALGORITHMS.append((str(path.relative_to(ROOT)), algorithm))


class _NoSetup:
    def __getattr__(self, name):
        pytest.fail(f"matrix setup ran before config validation: {name}")


def _no_device_setup(*args, **kwargs):
    pytest.fail("device setup ran before config validation")


@pytest.mark.parametrize("module", MODULES, ids=MODULE_IDS)
def test_unknown_key_rejected_before_setup(module, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", _no_device_setup)
    for decoder_module in MODULES:
        if hasattr(decoder_module, "parse_device_dtype"):
            monkeypatch.setattr(decoder_module, "parse_device_dtype", _no_device_setup)
    algorithm = module.__name__.split(".")[-2]
    cfg = {"algorithm": algorithm, "t258_unknown_key": 1, "second_unknown_key": 2}
    with pytest.raises(ValueError, match=f"unsupported key t258_unknown_key for {algorithm}$"):
        module.create(cfg, bundle=_NoSetup())


def test_saq_reports_unknown_key_before_later_legacy_key():
    from syndrilla.decoder.saq.saq import create

    with pytest.raises(ValueError, match="unsupported key first_typo for saq$"):
        create({"first_typo": 1, "code_type": "surface"}, bundle=_NoSetup())


@pytest.mark.parametrize("algorithm", ["bp_norm_min_sum", "osd_0"])
@pytest.mark.parametrize("suffix", ["", "_cuda"])
def test_retired_execution_alias_rejected_before_setup(algorithm, suffix, monkeypatch):
    module = importlib.import_module(f"syndrilla.decoder.{algorithm}.{algorithm}{suffix}")
    monkeypatch.setattr(torch.cuda, "is_available", _no_device_setup)
    with pytest.raises(ValueError, match=f"unsupported key force_per_step for {algorithm}$"):
        module.create({"algorithm": algorithm, "force_per_step": True}, bundle=_NoSetup())


@pytest.mark.parametrize("suffix", ["", "_cuda"])
def test_osd_prefixed_key_rejected_before_setup(suffix, monkeypatch):
    module = importlib.import_module(f"syndrilla.decoder.osd_0.osd_0{suffix}")
    key = "osd_early_stop"
    monkeypatch.setattr(torch.cuda, "is_available", _no_device_setup)
    with pytest.raises(ValueError, match=f"unsupported key {key} for osd_0$"):
        module.create({"algorithm": "osd_0", key: False}, bundle=_NoSetup())


class _Validated(Exception):
    pass


@pytest.fixture
def stop_after_key_validation(monkeypatch):
    validate = knobs.reject_unknown_keys

    def stop(cfg, legal, algorithm):
        validate(cfg, legal, algorithm)
        raise _Validated

    # Constructors import this function locally, including inherited constructors.
    for name, module in list(sys.modules.items()):
        if name.startswith("syndrilla.decoder") and getattr(module, "reject_unknown_keys", None) is validate:
            monkeypatch.setattr(module, "reject_unknown_keys", stop)
    return stop


@pytest.mark.parametrize("module", MODULES, ids=MODULE_IDS)
def test_minimal_config_reaches_key_validation(module, stop_after_key_validation):
    algorithm = module.__name__.split(".")[-2]
    with pytest.raises(_Validated):
        module.create({"algorithm": algorithm, "rebatch_opt": False}, bundle=_NoSetup())


@pytest.mark.parametrize("module,cfg", EXAMPLE_CASES)
def test_example_config_keys_accepted(module, cfg, stop_after_key_validation):
    with pytest.raises(_Validated):
        module.create(cfg, bundle=_NoSetup())


def test_examples_reference_available_decoders():
    assert not MISSING_EXAMPLE_ALGORITHMS


@pytest.fixture
def small_bundle():
    cfg = yaml.safe_load((ROOT / "examples/alist/surface_3.matrix.yaml").read_text())["matrix"]
    return load_matrices(cfg, torch.device("cpu"), torch.float64)


@pytest.mark.parametrize("stage", [0, 1])
def test_factory_rejects_unknown_key_in_each_chain_stage(stage, small_bundle):
    blocks = [{"max_iter": 2, "compile": False}, {}]
    blocks[stage]["t258_unknown_key"] = 1
    cfg = {
        "algorithm": ["bp_norm_min_sum", "osd_0"],
        "dtype": "float64", "device": {"device_type": "cpu"},
        "config": blocks,
    }
    algorithm = cfg["algorithm"][stage]
    with pytest.raises(ValueError, match=f"unsupported key t258_unknown_key for {algorithm}$"):
        create_decoder(cfg=cfg, bundle=small_bundle)


def test_factory_does_not_share_stage_specific_keys(small_bundle):
    cfg = {
        "algorithm": ["bp_norm_min_sum", "osd_0"],
        "dtype": "float64", "device": {"device_type": "cpu"},
        "config": [{"max_iter": 2}, {"max_iter": 2}],
    }
    with pytest.raises(ValueError, match="unsupported key max_iter for osd_0$"):
        create_decoder(cfg=cfg, bundle=small_bundle)


@pytest.mark.parametrize("shared", [{"fusion_opt": False}, {"skip_converged": False}])
def test_chain_routes_top_level_knobs_to_legal_stages(shared, small_bundle):
    cfg = {
        "algorithm": ["bp_norm_min_sum", "osd_0"],
        "dtype": "float64", "device": {"device_type": "cpu"},
        **shared, "config": [{"max_iter": 2, "compile": False}, {}],
    }
    stages = resolve_configs(cfg)
    assert all(key in stages[0] for key in shared)
    if "skip_converged" in shared:
        assert stages[1]["skip_converged"] is False
    assert len(create_decoder(cfg=cfg, bundle=small_bundle)) == 2


def test_chain_stage_entry_overrides_top_level_knob(small_bundle):
    cfg = {
        "algorithm": ["bp_norm_min_sum", "osd_0"],
        "dtype": "float64", "device": {"device_type": "cpu"},
        "fusion_opt": False, "skip_converged": False,
        "config": [{"max_iter": 2, "compile": True, "skip_converged": True}, {}],
    }
    stages = resolve_configs(cfg)
    assert stages[0]["fusion_opt"] is False
    assert stages[0]["compile"] is True
    assert stages[0]["skip_converged"] is True
    assert stages[1]["skip_converged"] is False
    decoders = create_decoder(cfg=cfg, bundle=small_bundle)
    assert len(decoders) == 2
    assert decoders[1].skip_converged is False


def test_chain_rejects_top_level_key_no_stage_accepts(small_bundle):
    cfg = {
        "algorithm": ["bp_norm_min_sum", "osd_0"],
        "device": {"device_type": "cpu"}, "t258_unknown_key": False,
        "config": [{"max_iter": 2}, {}],
    }
    with pytest.raises(ValueError, match="unsupported key t258_unknown_key for"):
        create_decoder(cfg=cfg, bundle=small_bundle)


@pytest.mark.parametrize("tail", ["mwpm", "union_find"])
def test_chain_routes_group_to_non_bp_stage(tail):
    stages = resolve_configs({
        "algorithm": ["bp_norm_min_sum", tail],
        "fusion_opt": False, "compile": False,
    })
    assert stages[0]["compile"] is False
    assert "compile" not in stages[1]
    assert stages[1]["fusion_opt"] is False


def test_chain_does_not_filter_an_explicit_unsupported_member(small_bundle):
    cfg = {
        "algorithm": ["bp_norm_min_sum", "osd_0"],
        "device": {"device_type": "cpu"}, "early_stop": False,
        "config": [{"max_iter": 2, "early_stop": False}, {}],
    }
    with pytest.raises(ValueError, match="unsupported key early_stop for bp_norm_min_sum$"):
        create_decoder(cfg=cfg, bundle=small_bundle)


@pytest.mark.parametrize("key", sorted(knobs.OSD_ONLY))
def test_chain_routes_osd_only_keys_to_osd_stage(key):
    stages = resolve_configs({
        "algorithm": ["bp_norm_min_sum", "osd_0"],
        "device": {"device_type": "cuda"}, key: False,
    })
    assert key not in knobs.BASE_CONFIG_KEYS
    assert key not in stages[0]
    assert stages[1][key] is False


@pytest.mark.parametrize("name", ["all_off", "+memory_opt", "+pruning_opt", "+fusion_opt",
                                  "+mapping_opt", "+gather_opt", "+rebatch_opt"])
def test_ablation_driver_build_accepts_all_group_configs(name, monkeypatch):
    import importlib.util
    from types import SimpleNamespace

    path = ROOT / "zoo/opt_ablation_bposd/ablation.py"
    spec = importlib.util.spec_from_file_location("t258_ablation_driver", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    monkeypatch.setattr(driver, "DEV", {"device_type": "cpu"})
    monkeypatch.setattr(driver, "MAX_ITER", 2)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    matrix_cfg = yaml.safe_load((ROOT / "examples/alist/surface_3.matrix.yaml").read_text())["matrix"]
    _, decoders, _, _ = driver.build(SimpleNamespace(matrix_cfg=matrix_cfg), "pytorch", "float64", name)
    assert len(decoders) == 2
