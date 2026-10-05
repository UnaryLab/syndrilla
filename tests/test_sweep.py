from pathlib import Path
import sys

import pytest
import yaml

from syndrilla import parallel as par
from syndrilla import sweep


def _args(monkeypatch, root, *flags):
    monkeypatch.setattr(sys, "argv", ["syndrilla-parallel", "sweep", f"-r={root}",
                                     "--no-probe", "-tb=20", *flags])
    return par.parse_args()


def _point(root, name, device="cuda"):
    folder = root / name
    folder.mkdir()
    for filename in ("test.decoding.yaml", "bsc.error.yaml", "lx.check.yaml",
                     "perfect.syndrome.yaml", "matrix.yaml"):
        config = {"decoding": {"device": {"device_type": device}}} if "decoding" in filename else {}
        (folder / filename).write_text(yaml.safe_dump(config))
    return folder


def _fake_launches(monkeypatch):
    started = []

    def start(args, plan, label=""):
        started.append((Path(args.run_dir).name, plan))
        return {"plan": plan}

    monkeypatch.setattr(par, "start_launch", start)
    monkeypatch.setattr(par, "step", lambda state: True)
    monkeypatch.setattr(par, "finish", lambda state: ([], {"rule": "worker targets"}, 1.0))
    monkeypatch.setattr(par, "merge", lambda *args: {
        "shots": 10, "fails": 1, "logical error rate": 0.1,
        "logical error rate 95% Wilson interval": [0.01, 0.3],
    })
    monkeypatch.setattr(sweep.time, "sleep", lambda seconds: None)
    return started


@pytest.mark.parametrize("used,capacity,expected", [
    ({2: 0, 7: 0}, 1, [2, 7]),
    ({2: 0, 7: 0}, 2, [2, 7, 2, 7]),
    ({2: 1, 7: 0}, 3, [7, 2, 7, 2, 7]),
    ({9: 2, 3: 0}, 2, [3, 3]),
    ({2: 2, 7: 2}, 2, []),
])
def test_free_slots_include_repeated_devices_in_load_order(used, capacity, expected):
    assert sweep.free_slots(used, capacity) == expected


def test_worker_count_default_and_only_supported_flag(tmp_path, monkeypatch, capsys):
    args, _ = _args(monkeypatch, tmp_path)
    assert args.workers_per_point == 1
    args, _ = _args(monkeypatch, tmp_path, "--workers-per-point=3")
    assert args.workers_per_point == 3
    output = capsys.readouterr()
    assert output.out == output.err == ""
    monkeypatch.setattr(sys, "argv", ["syndrilla-parallel", "sweep", "--help"])
    with pytest.raises(SystemExit) as exc:
        par.parse_args()
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--workers-per-point" in help_text


@pytest.mark.parametrize("mixed", [False, True])
def test_cpu_points_share_one_independent_slot(tmp_path, monkeypatch, mixed):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    for name in ("a_cpu", "c_cpu", "d_cpu", "e_cpu"):
        _point(tmp_path, name, device="cpu")
    if mixed:
        _point(tmp_path, "b_gpu")
    started = _fake_launches(monkeypatch)
    start, finish = par.start_launch, par.finish
    active = {"cpu": 0, "gpu": 0}
    occupancy = []

    def record_start(args, plan, label=""):
        kind = "cpu" if plan[0]["device"] == "cpu" else "gpu"
        active[kind] += 1
        occupancy.append(dict(active))
        return {**start(args, plan, label), "kind": kind}

    def record_finish(state):
        active[state["kind"]] -= 1
        return finish(state)

    monkeypatch.setattr(par, "start_launch", record_start)
    monkeypatch.setattr(par, "finish", record_finish)
    args, flags = _args(monkeypatch, tmp_path, "--gpus", "0", "--workers-per-gpu=1")
    sweep.run_sweep(args, flags)
    assert len(started) == 4 + mixed
    assert max(item["cpu"] for item in occupancy) == 1
    assert active == {"cpu": 0, "gpu": 0}
    if mixed:
        assert {"cpu": 1, "gpu": 1} in occupancy
    for _, plan in started:
        if plan[0]["device"] == "cpu":
            assert len(plan) == 1 and plan[0]["name"] == "w0"
            assert "CUDA_VISIBLE_DEVICES" not in plan[0]["env"]


@pytest.mark.parametrize("capacity,workers,expected", [
    (1, 2, ["2", "7"]),
    (2, 4, ["2", "7", "2", "7"]),
])
def test_sweep_assigns_real_devices_flat_names_and_point_seeds(
        tmp_path, monkeypatch, capacity, workers, expected):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,7")
    _point(tmp_path, "b")
    _point(tmp_path, "a")
    started = _fake_launches(monkeypatch)
    plan_workers = par.plan_workers
    planned_devices = []

    def record_plan(args, flags, devices=None):
        assert devices is not None
        planned_devices.append(list(devices))
        return plan_workers(args, flags, devices=devices)

    monkeypatch.setattr(par, "plan_workers", record_plan)
    args, flags = _args(monkeypatch, tmp_path, "--gpus", "0", "1",
                        f"--workers-per-gpu={capacity}", f"--workers-per-point={workers}", "--seed=100")
    sweep.run_sweep(args, flags)
    assert [name for name, _ in started] == ["a", "b"]
    assert planned_devices == [expected, expected]
    for point_index, (_, plan) in enumerate(started):
        assert [w["name"] for w in plan] == [f"w{i}" for i in range(workers)]
        assert [str(w["device"]) for w in plan] == expected
        assert [w["seed"] for w in plan] == [100 + point_index * workers + i for i in range(workers)]
        assert [w["env"]["CUDA_VISIBLE_DEVICES"] for w in plan] == expected
    assert (tmp_path / "sweep_results.csv").exists()


def test_sweep_refuses_more_workers_than_total_capacity(tmp_path, monkeypatch):
    _point(tmp_path, "a")
    started = _fake_launches(monkeypatch)
    args, flags = _args(monkeypatch, tmp_path, "--gpus", "0", "1",
                        "--workers-per-gpu=2", "--workers-per-point=5")
    with pytest.raises(SystemExit, match="capacity|slots|workers"):
        sweep.run_sweep(args, flags)
    assert started == []


@pytest.mark.parametrize("done", [False, True])
def test_unavailable_resume_devices_refused_upfront_unless_done(tmp_path, monkeypatch, done):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _point(tmp_path, "a_new")
    folder = _point(tmp_path, "z_resume")
    (folder / "w0").mkdir()
    (folder / "launch.yaml").write_text(yaml.safe_dump({
        "target batch": 20, "target error": None,
        "workers": [{"name": "w0", "device": "9"}],
    }))
    if done:
        (folder / "merged_result.yaml").write_text(yaml.safe_dump({
            "shots": 10, "fails": 1, "logical error rate": 0.1,
            "logical error rate 95% Wilson interval": [0.01, 0.3], "launch wall (s)": 1.0,
        }))
    started = _fake_launches(monkeypatch)
    args, flags = _args(monkeypatch, tmp_path, "--gpus", "2", "7")
    if done:
        sweep.run_sweep(args, flags)
        assert [name for name, _ in started] == ["a_new"]
    else:
        with pytest.raises(SystemExit, match="device|GPU|available"):
            sweep.run_sweep(args, flags)
        assert started == []


@pytest.mark.parametrize("capacity", [1, 2])
def test_resume_pins_physical_devices_instead_of_substituting(tmp_path, monkeypatch, capacity):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    folder = _point(tmp_path, "a")
    for i in range(2):
        (folder / f"w{i}").mkdir()
    (folder / "launch.yaml").write_text(yaml.safe_dump({
        "target batch": 20, "target error": None,
        "workers": [{"name": "w0", "device": "7"}, {"name": "w1", "device": "7"}],
    }))
    started = _fake_launches(monkeypatch)
    args, flags = _args(monkeypatch, tmp_path, "--gpus", "2", "7",
                        f"--workers-per-gpu={capacity}", "--workers-per-point=2")
    if capacity == 1:
        with pytest.raises(SystemExit, match="capacity|slots|workers"):
            sweep.run_sweep(args, flags)
        assert started == []
    else:
        sweep.run_sweep(args, flags)
        assert [str(w["device"]) for w in started[0][1]] == ["7", "7"]


def test_bad_later_point_preflight_precedes_any_launch(tmp_path, monkeypatch):
    _point(tmp_path, "a_valid")
    folder = _point(tmp_path, "z_legacy")
    (folder / "cpu_w0").mkdir()
    started = _fake_launches(monkeypatch)
    args, flags = _args(monkeypatch, tmp_path, "--gpus", "0")
    with pytest.raises(SystemExit, match="layout|legacy|old"):
        sweep.run_sweep(args, flags)
    assert started == []


def test_completed_legacy_point_skips_configs_layout_and_targets(tmp_path, monkeypatch):
    folder = tmp_path / "done"
    folder.mkdir()
    (folder / "test.decoding.yaml").write_text("{}")
    (folder / "gpu9_w0").mkdir()
    (folder / "launch.yaml").write_text(yaml.safe_dump({"target batch": 999, "target error": 123}))
    merged = folder / "merged_result.yaml"
    merged.write_text(yaml.safe_dump({
        "shots": 10, "fails": 1, "logical error rate": 0.1,
        "logical error rate 95% Wilson interval": [0.01, 0.3], "launch wall (s)": 1.0,
    }))
    before = merged.read_text()
    started = _fake_launches(monkeypatch)
    monkeypatch.setattr(par, "gpu_list", lambda args: pytest.fail("completed points need no GPU discovery"))
    args, flags = _args(monkeypatch, tmp_path, "--gpus", "0")
    sweep.run_sweep(args, flags)
    assert started == []
    assert merged.read_text() == before
    assert ",done" in (tmp_path / "sweep_results.csv").read_text()
