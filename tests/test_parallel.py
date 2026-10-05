from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import yaml

from syndrilla import parallel as par


def _args(monkeypatch, run_dir, *flags):
    monkeypatch.setattr(sys, "argv", ["syndrilla-parallel", "run", f"-r={run_dir}", *flags])
    return par.parse_args()


def _result(target=4, seed=10):
    return {
        "decoder_0": {"algorithm": "test", "sample count": 10,
                      "iteration count": [5, 5], "iteration distribution": [1, 2],
                      "average iteration": 1.5, "total time (s)": 2.0,
                      "hx": {"logical error rate": 0.2}},
        "decoder_full": {"target batch": target, "target error": None,
                         "target error reached": 2, "batch count": 2,
                         "total time (s)": 2.0, "physical error rate": 0.01,
                         "seed": seed, "hx": {"logical error rate": 0.2}},
    }


def test_run_flat_workers_gpu_major_shares_and_seeds(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,7")
    args, flags = _args(monkeypatch, tmp_path, "--gpus", "0", "1",
                        "--workers-per-gpu=2", "-tb=7", "--seed=100", "-bs=16")
    plan = par.plan_workers(args, flags)
    assert [w["name"] for w in plan] == ["w0", "w1", "w2", "w3"]
    assert [str(w["device"]) for w in plan] == ["2", "2", "7", "7"]
    assert [w["target"] for w in plan] == [2, 2, 2, 1]
    assert [w["seed"] for w in plan] == [100, 101, 102, 103]
    for worker in plan:
        assert Path(worker["dir"]).name == worker["name"]
        assert "gpu" not in worker and "worker" not in worker
        assert worker["env"]["CUDA_VISIBLE_DEVICES"] == str(worker["device"])
        assert f"--seed={worker['seed']}" in worker["cmd"]
        assert "-bs=16" in worker["cmd"]
        assert not worker["resumed"]
    assert not (tmp_path / "w0").exists()


def test_cpu_uses_one_flat_worker_without_cuda_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,6")
    path = tmp_path / "cpu.decoding.yaml"
    path.write_text(yaml.safe_dump({"decoding": {"device": {"device_type": "cpu"}}}))
    args, flags = _args(monkeypatch, tmp_path, "--gpus", "9", "8",
                        "--workers-per-gpu=3", "-tb=7", "--seed=20", f"-d={path}")
    plan = par.plan_workers(args, flags)
    assert [(w["name"], w["device"], w["target"], w["seed"]) for w in plan] == [("w0", "cpu", 7, 20)]
    assert "CUDA_VISIBLE_DEVICES" not in plan[0]["env"]


def test_pooled_targets_and_too_small_split(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    args, flags = _args(monkeypatch, tmp_path, "--gpus", "0", "1", "-te=1", "-tb=3")
    plan = par.plan_workers(args, flags)
    assert [w["target"] for w in plan] == [3, 3]
    assert all("-tb=3" in w["cmd"] and not any(x.startswith("-te=") for x in w["cmd"]) for w in plan)
    args, flags = _args(monkeypatch, tmp_path, "--gpus", "0", "1", "-tb=1")
    with pytest.raises(SystemExit, match="smaller|workers"):
        par.plan_workers(args, flags)


def test_launch_roundtrip_resume_and_merge(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(par, "spawn_workers", lambda *args: None)
    args, flags = _args(monkeypatch, tmp_path, "--gpus", "2", "7", "-tb=8", "--seed=10")
    plan = par.plan_workers(args, flags)
    par.start_launch(args, plan)
    manifest = yaml.safe_load((tmp_path / "launch.yaml").read_text())
    assert manifest["workers"] == [{"name": "w0", "device": "2"}, {"name": "w1", "device": "7"}]
    assert manifest["target batch"] == 8 and manifest["target error"] is None
    for worker in plan:
        Path(worker["dir"]).mkdir()
        Path(worker["dir"], "result_phy_err_0.01.yaml").write_text(yaml.safe_dump(_result(seed=worker["seed"])))
    resumed = par.plan_workers(args, flags)
    assert all(w["resumed"] and any(x.startswith("-ckpt=") for x in w["cmd"]) for w in resumed)
    assert [w["before"]["shots"] for w in resumed] == [10, 10]
    assert par.pooled_now(resumed)[0] == {"shots": 20, "fails": 4, "batches": 4}
    for worker in resumed:
        worker.update(proc=SimpleNamespace(returncode=0), wall=1.0)
    merged = par.merge(args, resumed, {"rule": "worker targets"}, 2.0)
    assert set(merged["per worker"]) == {"w0", "w1"}
    assert [merged["per worker"][f"w{i}"]["device"] for i in range(2)] == ["2", "7"]
    assert merged["shots"] == 20 and merged["fails"] == 4
    pooled = yaml.safe_load((tmp_path / "result_phy_err_0.01.yaml").read_text())
    assert set(pooled) == {"decoder_0", "decoder_full"}
    assert pooled["decoder_0"]["iteration count"] == [10, 10]
    assert pooled["decoder_full"]["target batch"] == 8
    assert pooled["decoder_full"]["seed"] is None


@pytest.mark.parametrize("change", ["device", "count", "target"])
def test_resume_refuses_changed_manifest(tmp_path, monkeypatch, change):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(par, "spawn_workers", lambda *args: None)
    args, flags = _args(monkeypatch, tmp_path, "--gpus", "2", "7", "-tb=8")
    par.start_launch(args, par.plan_workers(args, flags))
    if change == "device":
        args.gpus = [7, 2]
    elif change == "count":
        args.workers_per_gpu = 2
    else:
        args.target_batch = 10
        args.worker_target = ("-tb", "target batch", 10)
    with pytest.raises(SystemExit, match="resume|layout|target|worker|device"):
        par.plan_workers(args, flags)


@pytest.mark.parametrize("old_dir", ["gpu0_w0", "cpu_w0"])
def test_old_layout_refused_without_resume_command(tmp_path, monkeypatch, old_dir):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    (tmp_path / old_dir).mkdir()
    args, flags = _args(monkeypatch, tmp_path, "--gpus", "0", "-tb=2")
    with pytest.raises(SystemExit) as exc:
        par.plan_workers(args, flags)
    message = str(exc.value).lower()
    assert "layout" in message and ("old" in message or "legacy" in message)
    assert "-ckpt=" not in message


def test_resume_uses_saved_seed_and_checks_worker_share(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(par, "spawn_workers", lambda *args: None)
    args, flags = _args(monkeypatch, tmp_path, "--gpus", "0", "-tb=4")
    par.start_launch(args, par.plan_workers(args, flags))
    (tmp_path / "w0").mkdir()
    result = tmp_path / "w0" / "result_phy_err_0.01.yaml"
    result.write_text(yaml.safe_dump(_result(seed=77)))
    assert par.plan_workers(args, flags)[0]["seed"] == 77
    result.write_text(yaml.safe_dump(_result(target=5)))
    with pytest.raises(SystemExit, match="target|share|resume"):
        par.plan_workers(args, flags)


def test_targets_only_old_manifest_is_not_upgraded(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    (tmp_path / "launch.yaml").write_text(yaml.safe_dump({"target batch": 4, "target error": None}))
    (tmp_path / "w0").mkdir()
    args, flags = _args(monkeypatch, tmp_path, "--gpus", "0", "-tb=4")
    with pytest.raises(SystemExit) as exc:
        par.plan_workers(args, flags)
    assert "-ckpt=" not in str(exc.value)
    assert "workers" not in yaml.safe_load((tmp_path / "launch.yaml").read_text())


def test_merge_missing_first_worker_keeps_later_identity(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    args, flags = _args(monkeypatch, tmp_path, "--gpus", "2", "7", "-tb=8")
    plan = par.plan_workers(args, flags)
    for worker in plan:
        Path(worker["dir"]).mkdir()
        worker.update(proc=SimpleNamespace(returncode=0), wall=1.0)
    Path(plan[1]["dir"], "result_phy_err_0.01.yaml").write_text(yaml.safe_dump(_result()))
    merged = par.merge(args, plan, {"rule": "worker targets"}, 2.0)
    assert merged["workers with no result yaml"] == ["w0"]
    assert list(merged["per worker"]) == ["w1"]
    assert merged["per worker"]["w1"]["device"] == "7"
