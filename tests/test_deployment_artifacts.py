"""Tests for Flower detection deployment artifacts.

These exercise the artifact writer with minimal/fake strategy results. They do
not require a real Flower network, GPU, or dataset; ``ArrayRecord`` /
``MetricRecord`` / ``Result`` are constructed directly in-process.
"""

import json

import numpy as np
import pytest
from flwr.app import ArrayRecord, MetricRecord
from flwr.serverapp.strategy.result import Result

from src.fl.deployment_artifacts import (
    AGGREGATION_STRATEGY,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PARTIAL,
    WEIGHTED_BY_KEY,
    build_deployment_summary,
    derive_status,
    finalize_deployment_artifacts,
    latest_round_metrics,
    result_head_arrays,
    save_head_npz,
    write_deployment_artifacts,
)
from src.utils.detection_config import DetectionConfig


def _fake_arrays():
    return [
        np.arange(6, dtype=np.float32).reshape(2, 3),
        np.ones(4, dtype=np.float32),
    ]


def _fake_result(arrays=None, rounds=2):
    train = {
        rnd: MetricRecord({"train_loss": 0.6 - 0.1 * rnd, "num-examples": 10})
        for rnd in range(1, rounds + 1)
    }
    evaluate = {
        rnd: MetricRecord(
            {"map": 0.1 * rnd, "map_50": 0.2 * rnd, "map_75": 0.05 * rnd, "num-examples": 10}
        )
        for rnd in range(1, rounds + 1)
    }
    return Result(
        arrays=ArrayRecord(arrays if arrays is not None else _fake_arrays()),
        train_metrics_clientapp=train,
        evaluate_metrics_clientapp=evaluate,
    )


def _empty_result():
    """A result with no aggregated head and no completed rounds."""
    return Result(
        arrays=ArrayRecord([]),
        train_metrics_clientapp={},
        evaluate_metrics_clientapp={},
    )


def _config(tmp_path, **overrides):
    base = dict(
        exp_id="EXP-TEST",
        output_dir=str(tmp_path),
        num_clients=2,
        num_rounds=2,
        pretrained=False,
    )
    base.update(overrides)
    return DetectionConfig(**base)


def test_result_head_arrays_returns_none_when_empty():
    assert result_head_arrays(Result()) is None
    assert result_head_arrays(None) is None


def test_result_head_arrays_extracts_arrays():
    arrays = result_head_arrays(_fake_result())
    assert arrays is not None
    assert [a.shape for a in arrays] == [(2, 3), (4,)]


def test_latest_round_metrics_picks_highest_round():
    metrics = {
        1: MetricRecord({"map": 0.1}),
        3: MetricRecord({"map": 0.3}),
        2: MetricRecord({"map": 0.2}),
    }
    assert latest_round_metrics(metrics)["map"] == pytest.approx(0.3)
    assert latest_round_metrics({}) == {}


def test_save_head_npz_with_names(tmp_path):
    arrays = _fake_arrays()
    path = save_head_npz(tmp_path / "final_head.npz", arrays, names=["fc.weight", "fc.bias"])
    loaded = np.load(path)
    assert set(loaded.files) == {"fc.weight", "fc.bias"}
    np.testing.assert_array_equal(loaded["fc.weight"], arrays[0])


def test_save_head_npz_without_names_uses_positional_keys(tmp_path):
    path = save_head_npz(tmp_path / "head.npz", _fake_arrays())
    loaded = np.load(path)
    assert set(loaded.files) == {"arr_0", "arr_1"}


def test_build_deployment_summary_contains_expected_keys(tmp_path):
    config = _config(tmp_path)
    summary = build_deployment_summary(
        config,
        status="completed",
        update_size_bytes=1000,
        started_at="2026-06-30T00:00:00+00:00",
        ended_at="2026-06-30T00:01:00+00:00",
        runtime_seconds=60.0,
        result=_fake_result(),
        output_paths={"deployment_summary": "x.json"},
    )

    assert summary["exp_id"] == "EXP-TEST"
    assert summary["status"] == "completed"
    assert summary["server"]["aggregation_strategy"] == AGGREGATION_STRATEGY
    assert summary["server"]["weighted_by_key"] == WEIGHTED_BY_KEY
    assert summary["update_size_bytes"] == 1000
    # planned cost uses config.num_rounds; completed uses rounds_completed.
    # Here both rounds finished so the two match: 1000 * 2 clients * 2 rounds * 2.
    assert summary["planned_communication_cost_bytes"] == 1000 * 2 * 2 * 2
    assert summary["estimated_completed_communication_cost_bytes"] == 1000 * 2 * 2 * 2
    assert "communication_cost_bytes" not in summary
    assert summary["rounds_completed"] == 2
    for key in ("num_clients", "num_rounds", "image_size", "batch_size",
                "local_epochs", "lr", "device", "pretrained", "seed"):
        assert key in summary["config"]
    # latest-round metrics flow through
    assert summary["final_metrics"]["map"] == pytest.approx(0.2)
    assert summary["final_metrics"]["map_50"] == pytest.approx(0.4)
    assert summary["final_metrics"]["train_loss"] == pytest.approx(0.4)
    assert "no raw data" in summary["note"]


def test_write_deployment_artifacts_writes_summary_and_head(tmp_path):
    config = _config(tmp_path)
    paths = write_deployment_artifacts(
        config,
        result=_fake_result(),
        update_size_bytes=2048,
        started_at="2026-06-30T00:00:00+00:00",
        ended_at="2026-06-30T00:05:00+00:00",
        runtime_seconds=300.0,
        status="completed",
        head_param_names=["fc.weight", "fc.bias"],
    )

    summary_path = tmp_path / "deployment_summary.json"
    head_path = tmp_path / "final_head.npz"
    assert summary_path.is_file()
    assert head_path.is_file()

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert data["status"] == "completed"
    assert data["update_size_bytes"] == 2048
    assert data["output_paths"]["final_head"] == str(head_path)
    assert data["output_paths"]["deployment_summary"] == str(summary_path)

    loaded = np.load(head_path)
    assert set(loaded.files) == {"fc.weight", "fc.bias"}
    assert paths["final_head"] == str(head_path)


def test_write_deployment_artifacts_failed_run_has_no_head(tmp_path):
    config = _config(tmp_path)
    paths = write_deployment_artifacts(
        config,
        result=None,
        update_size_bytes=2048,
        started_at="2026-06-30T00:00:00+00:00",
        ended_at="2026-06-30T00:00:30+00:00",
        runtime_seconds=30.0,
        status="failed",
    )

    summary_path = tmp_path / "deployment_summary.json"
    assert summary_path.is_file()
    assert not (tmp_path / "final_head.npz").exists()
    assert "final_head" not in paths

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert data["status"] == "failed"
    assert data["rounds_completed"] == 0
    assert data["final_metrics"] == {}


# --- status derivation -----------------------------------------------------


def test_derive_status_completed_when_all_rounds_and_head():
    result = _fake_result(rounds=2)
    assert derive_status(result, expected_rounds=2) == STATUS_COMPLETED


def test_derive_status_failed_on_exception():
    assert derive_status(_fake_result(), expected_rounds=2, exception_raised=True) == (
        STATUS_FAILED
    )


def test_derive_status_failed_when_result_none():
    assert derive_status(None, expected_rounds=2) == STATUS_FAILED


def test_derive_status_failed_when_no_rounds():
    assert derive_status(_empty_result(), expected_rounds=2) == STATUS_FAILED


def test_derive_status_partial_when_fewer_rounds_than_expected():
    result = _fake_result(rounds=1)
    assert derive_status(result, expected_rounds=2) == STATUS_PARTIAL


def test_derive_status_partial_when_rounds_done_but_no_head():
    result = _fake_result(arrays=[], rounds=2)
    assert derive_status(result, expected_rounds=2) == STATUS_PARTIAL


# --- finalize seam (ServerApp-side, network-free) --------------------------


def test_finalize_completed_writes_head_and_status(tmp_path):
    config = _config(tmp_path)
    status, paths = finalize_deployment_artifacts(
        config,
        result=_fake_result(rounds=2),
        update_size_bytes=1000,
        started_at="2026-06-30T00:00:00+00:00",
        ended_at="2026-06-30T00:05:00+00:00",
        runtime_seconds=300.0,
        exception_raised=False,
        head_param_names=["fc.weight", "fc.bias"],
    )

    assert status == STATUS_COMPLETED
    data = json.loads((tmp_path / "deployment_summary.json").read_text(encoding="utf-8"))
    assert data["status"] == STATUS_COMPLETED
    assert (tmp_path / "final_head.npz").is_file()
    assert paths["final_head"] == str(tmp_path / "final_head.npz")


def test_finalize_failed_empty_result_writes_summary_without_head(tmp_path):
    config = _config(tmp_path)
    status, paths = finalize_deployment_artifacts(
        config,
        result=_empty_result(),
        update_size_bytes=1000,
        started_at="2026-06-30T00:00:00+00:00",
        ended_at="2026-06-30T00:00:30+00:00",
        runtime_seconds=30.0,
        exception_raised=False,
    )

    assert status == STATUS_FAILED
    assert (tmp_path / "deployment_summary.json").is_file()
    assert not (tmp_path / "final_head.npz").exists()
    assert "final_head" not in paths
    data = json.loads((tmp_path / "deployment_summary.json").read_text(encoding="utf-8"))
    assert data["status"] == STATUS_FAILED
    assert data["estimated_completed_communication_cost_bytes"] == 0


def test_finalize_partial_uses_completed_rounds_for_cost(tmp_path):
    # config asks for 2 rounds; only 1 completed -> partial, and the completed
    # communication cost is half the planned cost.
    config = _config(tmp_path, num_rounds=2)
    status, _ = finalize_deployment_artifacts(
        config,
        result=_fake_result(rounds=1),
        update_size_bytes=1000,
        started_at="2026-06-30T00:00:00+00:00",
        ended_at="2026-06-30T00:02:00+00:00",
        runtime_seconds=120.0,
        exception_raised=False,
        head_param_names=["fc.weight", "fc.bias"],
    )

    assert status == STATUS_PARTIAL
    data = json.loads((tmp_path / "deployment_summary.json").read_text(encoding="utf-8"))
    assert data["status"] == STATUS_PARTIAL
    assert data["rounds_completed"] == 1
    # planned: 1000 * 2 clients * 2 rounds * 2 ; completed: ... * 1 round * 2
    assert data["planned_communication_cost_bytes"] == 1000 * 2 * 2 * 2
    assert data["estimated_completed_communication_cost_bytes"] == 1000 * 2 * 1 * 2
    # final head still saved because round 1 produced an aggregated head
    assert (tmp_path / "final_head.npz").is_file()
