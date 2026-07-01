#!/usr/bin/env python3
"""Capture raw runtime logs for a Flower PPE-detection deployment run.

Structured artifacts (``deployment_summary.json`` / ``round_metrics.json``) cannot
show actual client IDs/counts, sampled nodes, or disconnects -- Flower's ``Result``
does not expose them. Those details only live in runtime stdout/stderr. This helper:

- captures ``flwr log <RUN_ID> <federation> --show`` output (available *after* a run,
  by run id) into ``outputs/logs/<EXP-ID>/flower_run_log.txt``;
- creates the log directory and prints the expected manual log paths;
- with ``--print-commands``, prints ``tee`` redirects to wire into *future* runs so
  SuperLink/SuperNode terminals are captured live.

It does not try to scrape already-running terminals after the fact -- that is not
reliable. This is observability tooling only; it changes no deployment behavior.

Examples::

    # Capture flwr log after a run:
    venv/bin/python scripts/capture_flower_logs.py --exp-id EXP-012 --run-id 123

    # Just show the paths and tee commands for a future run (no flwr log call):
    venv/bin/python scripts/capture_flower_logs.py \
        --exp-id EXP-012 --run-id 123 --dry-run --print-commands
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fl.deployment_artifacts import DEPLOYMENT_SITE_LABELS
from src.utils.io import write_json

DEFAULT_LOGS_ROOT = "outputs/logs"
DEFAULT_FEDERATION = "deploy"
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"


def build_log_targets(exp_id: str, logs_root: str | Path) -> dict[str, Any]:
    """Resolve every log path for a run under ``<logs_root>/<exp_id>/``."""

    log_dir = Path(logs_root) / exp_id
    client_logs = {
        label: log_dir / f"client_{label.replace('-', '_')}_log.txt"
        for label in DEPLOYMENT_SITE_LABELS
    }
    return {
        "log_dir": log_dir,
        "flower_run_log": log_dir / "flower_run_log.txt",
        "server_log": log_dir / "server_log.txt",
        "client_logs": client_logs,
    }


def flwr_log_command(run_id: str, federation: str) -> list[str]:
    """The ``flwr log`` argv used to fetch a completed run's logstream."""

    return ["flwr", "log", str(run_id), federation, "--show"]


def tee_commands(targets: dict[str, Any]) -> list[str]:
    """Manual ``tee`` redirects to capture live SuperLink/SuperNode terminals.

    These are hints for *future* runs -- printed, never executed here.
    """

    commands = [
        f"flower-superlink --insecure ... 2>&1 | tee {targets['server_log']}",
    ]
    for label, path in targets["client_logs"].items():
        commands.append(
            f"proxychains4 flower-supernode --insecure ...  # {label}\n"
            f"    ... 2>&1 | tee {path}"
        )
    return commands


def _run_flwr_log(command: list[str]) -> tuple[int | None, str]:
    """Run ``flwr log`` and return (return_code, combined stdout+stderr).

    A missing ``flwr`` binary yields ``(None, <error>)`` rather than crashing, so the
    capture summary still records the failure honestly.
    """

    try:
        completed = subprocess.run(command, capture_output=True, text=True)
    except FileNotFoundError as exc:
        return None, f"Failed to execute {command[0]!r}: {exc}"
    output = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode, output


def build_capture_summary(
    *,
    exp_id: str,
    run_id: str,
    federation: str,
    targets: dict[str, Any],
    command: list[str],
    return_code: int | None,
) -> dict[str, Any]:
    """Assemble the JSON-ready ``log_capture_summary.json`` content (pure)."""

    status = STATUS_SUCCESS if return_code == 0 else STATUS_FAILED
    return {
        "exp_id": exp_id,
        "run_id": run_id,
        "federation": federation,
        "command": " ".join(command),
        "return_code": return_code,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "log_paths": {
            "flower_run_log": str(targets["flower_run_log"]),
            "server_log": str(targets["server_log"]),
            "client_logs": {
                label: str(path) for label, path in targets["client_logs"].items()
            },
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-id", required=True, help="Experiment id, e.g. EXP-012")
    parser.add_argument("--run-id", required=True, help="Flower run id from `flwr run`")
    parser.add_argument("--federation", default=DEFAULT_FEDERATION)
    parser.add_argument("--logs-root", default=DEFAULT_LOGS_ROOT)
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help="Print tee commands to capture live SuperLink/SuperNode logs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print paths/commands without running `flwr log`",
    )
    return parser.parse_args(argv)


def _print_expected_paths(targets: dict[str, Any]) -> None:
    print(f"[capture-logs] log_dir: {targets['log_dir']}", flush=True)
    print(f"[capture-logs] expected server log (manual): {targets['server_log']}")
    for label, path in targets["client_logs"].items():
        print(f"[capture-logs] expected client log (manual, {label}): {path}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    targets = build_log_targets(args.exp_id, args.logs_root)
    targets["log_dir"].mkdir(parents=True, exist_ok=True)

    _print_expected_paths(targets)
    command = flwr_log_command(args.run_id, args.federation)
    if args.print_commands:
        print("[capture-logs] Redirect live logs during a future run with:", flush=True)
        for cmd in tee_commands(targets):
            print(f"    {cmd}")

    if args.dry_run:
        print(f"[capture-logs] DRY RUN -- would run: {' '.join(command)}", flush=True)
        print(f"[capture-logs] would capture into: {targets['flower_run_log']}")
        return

    print(f"[capture-logs] running: {' '.join(command)}", flush=True)
    return_code, output = _run_flwr_log(command)
    targets["flower_run_log"].write_text(output, encoding="utf-8")

    summary = build_capture_summary(
        exp_id=args.exp_id,
        run_id=args.run_id,
        federation=args.federation,
        targets=targets,
        command=command,
        return_code=return_code,
    )
    summary_path = targets["log_dir"] / "log_capture_summary.json"
    write_json(summary_path, summary)

    print(
        f"[capture-logs] status={summary['status']} return_code={return_code} "
        f"run_log={targets['flower_run_log']} summary={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
