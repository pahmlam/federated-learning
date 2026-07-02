"""Tests for Flower detection deployment artifacts.

These exercise the artifact writer with minimal/fake strategy results. They do
not require a real Flower network, GPU, or dataset; ``ArrayRecord`` /
``MetricRecord`` / ``Result`` are constructed directly in-process.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from flwr.app import ArrayRecord, MetricRecord
from flwr.serverapp.strategy.result import Result

from src.fl.deployment_artifacts import (
    AGGREGATION_STRATEGY,
    DEPLOYMENT_SITE_LABELS,
    METRICS_SEMANTICS,
    PARTICIPATION_NOTE,
    STATUS_COMPLETED,
    STATUS_EXPECTED,
    STATUS_FAILED,
    STATUS_PARTIAL,
    WEIGHTED_BY_KEY,
    build_deployment_summary,
    build_log_paths,
    build_round_metrics,
    deployment_log_dir,
    derive_status,
    finalize_deployment_artifacts,
    latest_round_metrics,
    log_file_targets,
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
                "local_epochs", "lr", "device", "pretrained", "seed",
                "min_train_nodes", "min_evaluate_nodes", "min_available_nodes"):
        assert key in summary["config"]
    # Strict default: min-node thresholds equal num_clients when unset.
    assert summary["config"]["min_train_nodes"] == summary["config"]["num_clients"]
    assert summary["config"]["min_evaluate_nodes"] == summary["config"]["num_clients"]
    assert summary["config"]["min_available_nodes"] == summary["config"]["num_clients"]
    # latest-round metrics flow through
    assert summary["final_metrics"]["map"] == pytest.approx(0.2)
    assert summary["final_metrics"]["map_50"] == pytest.approx(0.4)
    assert summary["final_metrics"]["train_loss"] == pytest.approx(0.4)
    assert "no raw data" in summary["note"]


def test_summary_records_explicit_min_node_overrides(tmp_path):
    config = _config(
        tmp_path,
        min_train_nodes=1,
        min_evaluate_nodes=1,
        min_available_nodes=1,
    )
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

    assert summary["config"]["num_clients"] == 2  # baseline node count unchanged
    assert summary["config"]["min_train_nodes"] == 1
    assert summary["config"]["min_evaluate_nodes"] == 1
    assert summary["config"]["min_available_nodes"] == 1


def test_deployment_summary_min_nodes_survive_json_roundtrip(tmp_path):
    config = _config(tmp_path, min_train_nodes=1)
    write_deployment_artifacts(
        config,
        result=_fake_result(),
        update_size_bytes=1000,
        started_at="2026-06-30T00:00:00+00:00",
        ended_at="2026-06-30T00:01:00+00:00",
        runtime_seconds=60.0,
        status="completed",
        head_param_names=["fc.weight", "fc.bias"],
    )
    data = json.loads((tmp_path / "deployment_summary.json").read_text(encoding="utf-8"))
    assert data["config"]["min_train_nodes"] == 1
    # unset fields fall back to strict num_clients
    assert data["config"]["min_evaluate_nodes"] == 2
    assert data["config"]["min_available_nodes"] == 2


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


# --- round_metrics.json ----------------------------------------------------


def _round_metrics_kwargs():
    return dict(status="completed", update_size_bytes=1000, result=_fake_result(rounds=2))


def test_build_round_metrics_structure(tmp_path):
    config = _config(tmp_path)
    metrics = build_round_metrics(config, **_round_metrics_kwargs())

    assert metrics["exp_id"] == "EXP-TEST"
    assert metrics["num_rounds_configured"] == 2
    assert metrics["rounds_completed"] == 2
    assert metrics["num_clients"] == 2
    assert metrics["min_train_nodes"] == 2  # strict default == num_clients
    assert metrics["min_evaluate_nodes"] == 2
    assert metrics["min_available_nodes"] == 2
    assert metrics["metrics_semantics"] == METRICS_SEMANTICS
    assert metrics["weighted_by_key"] == WEIGHTED_BY_KEY
    assert metrics["planned_communication_cost_bytes"] == 1000 * 2 * 2 * 2
    assert metrics["estimated_completed_communication_cost_bytes"] == 1000 * 2 * 2 * 2

    rounds = metrics["rounds"]
    assert [r["round"] for r in rounds] == [1, 2]
    first = rounds[0]
    # weighted aggregate holds scalar metrics with num-examples stripped out.
    assert "train_loss" in first["train"]["weighted_aggregate"]
    assert WEIGHTED_BY_KEY not in first["train"]["weighted_aggregate"]
    assert first["train"]["weighted_num_examples"] == 10
    assert "map" in first["evaluate"]["weighted_aggregate"]
    assert first["evaluate"]["weighted_num_examples"] == 10


def test_build_round_metrics_client_participation_is_null_not_invented(tmp_path):
    metrics = build_round_metrics(_config(tmp_path), **_round_metrics_kwargs())

    participation = metrics["client_participation"]
    assert participation["actual_train_clients"] is None
    assert participation["actual_evaluate_clients"] is None
    assert participation["actual_client_ids"] is None
    assert "does not expose" in participation["participation_note"]
    assert participation["participation_note"] == PARTICIPATION_NOTE
    # per-round entries are equally honest about unknown participation
    for entry in metrics["rounds"]:
        assert entry["actual_train_clients"] is None
        assert entry["actual_evaluate_clients"] is None
        assert entry["actual_client_ids"] is None


def test_build_round_metrics_is_json_serializable(tmp_path):
    metrics = build_round_metrics(_config(tmp_path), **_round_metrics_kwargs())
    # Must not raise; also round-trips to the same structure.
    assert json.loads(json.dumps(metrics))["rounds_completed"] == 2


def test_build_round_metrics_empty_result_has_no_rounds(tmp_path):
    metrics = build_round_metrics(
        _config(tmp_path),
        status="failed",
        update_size_bytes=1000,
        result=_empty_result(),
    )
    assert metrics["rounds"] == []
    assert metrics["rounds_completed"] == 0
    assert metrics["estimated_completed_communication_cost_bytes"] == 0
    json.dumps(metrics)  # still serializable


def test_build_round_metrics_none_result_has_no_rounds(tmp_path):
    metrics = build_round_metrics(
        _config(tmp_path), status="failed", update_size_bytes=1000, result=None
    )
    assert metrics["rounds"] == []
    assert metrics["rounds_completed"] == 0


# --- raw-log paths ---------------------------------------------------------


def test_log_file_targets_single_source(tmp_path):
    log_dir = Path("outputs/logs/EXP-015")
    targets = log_file_targets(log_dir)

    assert targets["log_dir"] == log_dir
    assert targets["flower_run_log"] == log_dir / "flower_run_log.txt"
    assert targets["server_log"] == log_dir / "server_log.txt"
    assert set(targets["client_logs"]) == set(DEPLOYMENT_SITE_LABELS)
    assert targets["client_logs"]["site-b"] == log_dir / "client_site_b_log.txt"
    assert targets["client_logs"]["site-c"] == log_dir / "client_site_c_log.txt"


def test_deployment_log_dir_maps_output_dir_to_logs_subtree(tmp_path):
    config = _config(tmp_path, output_dir="outputs/EXP-012", exp_id="EXP-012")
    assert deployment_log_dir(config) == Path("outputs/logs/EXP-012")


def test_build_log_paths_expected_when_files_absent(tmp_path):
    config = _config(tmp_path, output_dir=str(tmp_path / "EXP-TEST"), exp_id="EXP-TEST")
    logs = build_log_paths(config, run_id=42)

    log_dir = tmp_path / "logs" / "EXP-TEST"
    assert logs["log_dir"] == str(log_dir)
    assert logs["server_log"]["path"] == str(log_dir / "server_log.txt")
    assert logs["server_log"]["status"] == STATUS_EXPECTED
    assert set(logs["client_logs"]) == set(DEPLOYMENT_SITE_LABELS)
    assert logs["client_logs"]["site-b"]["path"] == str(
        log_dir / "client_site_b_log.txt"
    )
    assert logs["client_logs"]["site-c"]["path"] == str(
        log_dir / "client_site_c_log.txt"
    )
    assert all(entry["status"] == STATUS_EXPECTED for entry in logs["client_logs"].values())
    assert logs["flower_log_command"] == "flwr log 42 deploy --show"


def test_build_log_paths_present_when_file_exists(tmp_path):
    config = _config(tmp_path, output_dir=str(tmp_path / "EXP-TEST"), exp_id="EXP-TEST")
    log_dir = tmp_path / "logs" / "EXP-TEST"
    log_dir.mkdir(parents=True)
    (log_dir / "server_log.txt").write_text("boot\n", encoding="utf-8")

    logs = build_log_paths(config)
    assert logs["server_log"]["status"] == "present"
    assert logs["flower_log_command"] == "flwr log <run-id> deploy --show"


# --- write path wires round_metrics + logs together ------------------------


def test_write_deployment_artifacts_writes_round_metrics_and_log_dir(tmp_path):
    config = _config(tmp_path, output_dir=str(tmp_path / "EXP-TEST"), exp_id="EXP-TEST")
    paths = write_deployment_artifacts(
        config,
        result=_fake_result(rounds=2),
        update_size_bytes=1000,
        started_at="2026-06-30T00:00:00+00:00",
        ended_at="2026-06-30T00:05:00+00:00",
        runtime_seconds=300.0,
        status="completed",
        head_param_names=["fc.weight", "fc.bias"],
        run_id=7,
    )

    round_metrics_path = tmp_path / "EXP-TEST" / "round_metrics.json"
    assert round_metrics_path.is_file()
    assert paths["round_metrics_path"] == str(round_metrics_path)
    # raw-log dir is created for manual logs
    assert (tmp_path / "logs" / "EXP-TEST").is_dir()

    round_data = json.loads(round_metrics_path.read_text(encoding="utf-8"))
    assert round_data["rounds_completed"] == 2

    summary = json.loads(
        (tmp_path / "EXP-TEST" / "deployment_summary.json").read_text(encoding="utf-8")
    )
    # summary references round_metrics + log locations + log command
    assert summary["output_paths"]["round_metrics_path"] == str(round_metrics_path)
    assert summary["logs"]["server_log"]["path"].endswith("logs/EXP-TEST/server_log.txt")
    assert summary["flower_log_command"] == "flwr log 7 deploy --show"
    assert "estimated_completed_communication_cost_note" in summary


def test_build_deployment_summary_backward_compatible_keys(tmp_path):
    # Existing callers pass no run_id; summary must still carry the logs section.
    summary = build_deployment_summary(
        _config(tmp_path),
        status="completed",
        update_size_bytes=1000,
        started_at="2026-06-30T00:00:00+00:00",
        ended_at="2026-06-30T00:01:00+00:00",
        runtime_seconds=60.0,
        result=_fake_result(),
        output_paths={"deployment_summary": "x.json"},
    )
    assert summary["flower_log_command"] == "flwr log <run-id> deploy --show"
    assert summary["logs"]["client_logs"]["site-b"]["status"] == STATUS_EXPECTED


# --- strict 2-client deployment: required-field contract -------------------


def _strict_two_client_config(tmp_path):
    """A strict 2-client config: every min-node threshold equals num_clients."""
    return _config(
        tmp_path,
        num_clients=2,
        min_train_nodes=2,
        min_evaluate_nodes=2,
        min_available_nodes=2,
    )


def test_strict_two_client_summary_exposes_required_fields(tmp_path):
    config = _strict_two_client_config(tmp_path)
    summary = build_deployment_summary(
        config,
        status="completed",
        update_size_bytes=1000,
        started_at="2026-06-30T00:00:00+00:00",
        ended_at="2026-06-30T00:01:00+00:00",
        runtime_seconds=60.0,
        result=_fake_result(rounds=2),
        output_paths={"deployment_summary": "x.json"},
    )

    cfg = summary["config"]
    # Strict participation is derivable: every min-node threshold == num_clients.
    assert cfg["num_clients"] == 2
    assert cfg["min_train_nodes"] == 2
    assert cfg["min_evaluate_nodes"] == 2
    assert cfg["min_available_nodes"] == 2
    assert (
        cfg["min_train_nodes"]
        == cfg["min_evaluate_nodes"]
        == cfg["min_available_nodes"]
        == cfg["num_clients"]
    )

    # Interpreting a run needs completion, cost, timing, and eval semantics.
    assert isinstance(summary["rounds_completed"], int)
    assert isinstance(summary["update_size_bytes"], int)
    assert isinstance(summary["planned_communication_cost_bytes"], int)
    assert isinstance(summary["estimated_completed_communication_cost_bytes"], int)
    timing = summary["timing"]
    assert timing["started_at"] == "2026-06-30T00:00:00+00:00"
    assert timing["ended_at"] == "2026-06-30T00:01:00+00:00"
    assert isinstance(timing["runtime_seconds"], float)
    assert summary["server"]["evaluation"] == "distributed"


def test_strict_two_client_round_metrics_expose_required_fields(tmp_path):
    config = _strict_two_client_config(tmp_path)
    metrics = build_round_metrics(
        config, status="completed", update_size_bytes=1000, result=_fake_result(rounds=2)
    )

    assert metrics["num_clients"] == 2
    assert metrics["min_train_nodes"] == 2
    assert metrics["min_evaluate_nodes"] == 2
    assert metrics["min_available_nodes"] == 2

    assert metrics["rounds"], "strict run must record per-round entries"
    for entry in metrics["rounds"]:
        assert isinstance(entry["round"], int)
        assert "weighted_aggregate" in entry["train"]
        assert "weighted_aggregate" in entry["evaluate"]
        # Participation stays honestly null -- Flower Result cannot expose it.
        assert entry["actual_train_clients"] is None
        assert entry["actual_evaluate_clients"] is None
        assert entry["actual_client_ids"] is None
