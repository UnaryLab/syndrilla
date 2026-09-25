import os

import pytest
import torch
import yaml

from syndrilla.metric.metric import MetricState


def _batch(ler, number_channel):
    zeros = [0.0] * number_channel
    return {
        "total_time": 0.0,
        "sample_size": 100,
        "average_time_sample": 0.0,
        "average_iter": 1.0,
        "distribution": torch.zeros(3),
        "average_time_sample_iter": 0.0,
        "data_qubit_acc": zeros,
        "data_frame_error_rate": zeros,
        "synd_frame_error_rate": zeros,
        "correction_acc": zeros,
        "logical_error_rate": ler,
        "invoke_rate": 1.0,
        "converge_fail": zeros,
        "converge_succ": zeros,
    }


@pytest.mark.parametrize(
    "lers, check_num, expected",
    [
        # 1-channel BP -> OSD chain on hx
        ([[0.25], [0.01]], 0, {"hx": 0.01}),
        # 2-channel chain (bp4-style), both checks reported
        ([[0.25, 0.5], [0.01, 0.02]], 0, {"hx": 0.01, "hz": 0.02}),
    ],
)
def test_decoder_full_reports_last_decoder(tmp_path, lers, check_num, expected):
    number_channel = len(lers[0])
    m = MetricState(len(lers), number_channel, "cpu")
    for i, ler in enumerate(lers):
        m.update_metric(i, _batch(ler, number_channel))
    out = m.get_all_metrics(1, ["bp_norm_min_sum", "osd_0"])
    m.save_metric(out, str(tmp_path), 100, 10, "torch.float64", 0.01, 1, 1, "H", check_num)

    with open(os.path.join(tmp_path, "result_phy_err_0.01.yaml")) as f:
        full = yaml.safe_load(f)["decoder_full"]
    for check_name, ler in expected.items():
        assert full[check_name]["logical error rate"] == pytest.approx(ler)


def test_time_per_iteration_zero_when_decoder_not_invoked():
    # decoder ran on no samples (every row converged earlier) but took nonzero time
    m = MetricState(2, 1, "cpu")
    e = torch.zeros(4, 1, 5)
    out = m.report_metric(3, e[:, 0, :], e, torch.empty(0), [0.01], torch.zeros(4, 1), torch.ones(4), torch.ones(4), 1)
    assert out["average_time_sample_iter"] == 0.0
