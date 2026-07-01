"""Tests for the raw Flower log capture helper.

These exercise path creation, dry-run (no subprocess), the exact ``flwr log``
command issued, and honest failure recording -- all without a real Flower network,
a real ``flwr`` binary, or a GPU (``subprocess.run`` is faked/monkeypatched).
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "capture_flower_logs.py"
    spec = importlib.util.spec_from_file_location("capture_flower_logs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = _load_module()


def _base_args(tmp_path, *extra):
    return [
        "--exp-id",
        "EXP-012",
        "--run-id",
        "123",
        "--logs-root",
        str(tmp_path),
        *extra,
    ]


# --- pure helpers ----------------------------------------------------------


def test_build_log_targets_paths(tmp_path):
    targets = module.build_log_targets("EXP-012", tmp_path)
    log_dir = tmp_path / "EXP-012"
    assert targets["log_dir"] == log_dir
    assert targets["flower_run_log"] == log_dir / "flower_run_log.txt"
    assert targets["server_log"] == log_dir / "server_log.txt"
    assert targets["client_logs"]["site-b"] == log_dir / "client_site_b_log.txt"
    assert targets["client_logs"]["site-c"] == log_dir / "client_site_c_log.txt"


def test_flwr_log_command_shape():
    assert module.flwr_log_command("123", "deploy") == [
        "flwr",
        "log",
        "123",
        "deploy",
        "--show",
    ]


def test_tee_commands_reference_each_site_and_server(tmp_path):
    targets = module.build_log_targets("EXP-012", tmp_path)
    commands = "\n".join(module.tee_commands(targets))
    assert str(targets["server_log"]) in commands
    for label in ("site-b", "site-c"):
        assert str(targets["client_logs"][label]) in commands


def test_build_capture_summary_json_serializable(tmp_path):
    targets = module.build_log_targets("EXP-012", tmp_path)
    summary = module.build_capture_summary(
        exp_id="EXP-012",
        run_id="123",
        federation="deploy",
        targets=targets,
        command=module.flwr_log_command("123", "deploy"),
        return_code=0,
    )
    assert summary["status"] == "success"
    assert summary["command"] == "flwr log 123 deploy --show"
    assert summary["log_paths"]["client_logs"]["site-b"].endswith("client_site_b_log.txt")
    json.dumps(summary)  # must not raise


# --- main orchestration ----------------------------------------------------


def test_main_creates_log_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="ok", stderr=""),
    )
    module.main(_base_args(tmp_path))
    assert (tmp_path / "EXP-012").is_dir()


def test_dry_run_does_not_execute_subprocess(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: calls.append(a))

    module.main(_base_args(tmp_path, "--dry-run", "--print-commands"))

    assert calls == []  # subprocess never invoked
    log_dir = tmp_path / "EXP-012"
    assert log_dir.is_dir()  # dir still created
    assert not (log_dir / "flower_run_log.txt").exists()
    assert not (log_dir / "log_capture_summary.json").exists()


def test_non_dry_run_issues_expected_command_and_writes_artifacts(tmp_path, monkeypatch):
    captured = {}

    def _fake_run(command, *args, **kwargs):
        captured["command"] = command
        return SimpleNamespace(returncode=0, stdout="round 1 ok\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    module.main(_base_args(tmp_path))

    assert captured["command"] == ["flwr", "log", "123", "deploy", "--show"]
    log_dir = tmp_path / "EXP-012"
    assert (log_dir / "flower_run_log.txt").read_text(encoding="utf-8") == "round 1 ok\n"
    summary = json.loads(
        (log_dir / "log_capture_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "success"
    assert summary["return_code"] == 0
    assert summary["run_id"] == "123"


def test_failure_return_code_still_writes_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="boom"),
    )

    module.main(_base_args(tmp_path))

    log_dir = tmp_path / "EXP-012"
    summary = json.loads(
        (log_dir / "log_capture_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "failed"
    assert summary["return_code"] == 1
    assert (log_dir / "flower_run_log.txt").read_text(encoding="utf-8") == "boom"


def test_missing_flwr_binary_records_failure(tmp_path, monkeypatch):
    def _raise(*a, **k):
        raise FileNotFoundError("flwr not found")

    monkeypatch.setattr(module.subprocess, "run", _raise)

    module.main(_base_args(tmp_path))  # must not raise

    log_dir = tmp_path / "EXP-012"
    summary = json.loads(
        (log_dir / "log_capture_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "failed"
    assert summary["return_code"] is None
    assert "flwr" in (log_dir / "flower_run_log.txt").read_text(encoding="utf-8")


def test_custom_federation_flows_into_command(tmp_path, monkeypatch):
    captured = {}

    def _fake_run(command, *args, **kwargs):
        captured["command"] = command
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    module.main(_base_args(tmp_path, "--federation", "local-sim"))

    assert captured["command"] == ["flwr", "log", "123", "local-sim", "--show"]
