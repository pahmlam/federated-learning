"""Artifacts for Flower detection deployment runs.

Deployment relies on Flower's logstream, while simulation writes structured
JSON. These helpers turn a strategy ``Result`` (Flower 1.30) into the same kind
of durable artifacts so EXP-012 deployment runs are reproducible and easy to
journal without scraping logs.

The server holds no raw data, so every metric here is a weighted
distributed-evaluation aggregate returned by clients, not a server-side pooled
evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.utils.detection_config import DetectionConfig
from src.utils.io import write_json

AGGREGATION_STRATEGY = "FedAvg"
WEIGHTED_BY_KEY = "num-examples"
SERVER_DATA_NOTE = (
    "Server holds no raw data; metrics are weighted distributed-evaluation "
    "aggregates returned by clients."
)
# Flower's Result carries only round-keyed, already-aggregated MetricRecords -- it
# never exposes which nodes replied. We record client counts/ids as null with this
# note rather than inventing participation data.
PARTICIPATION_NOTE = (
    "Flower Result does not expose per-client identities in this artifact path"
)
METRICS_SEMANTICS = "flower_weighted_aggregate"
COMPLETED_COST_NOTE = (
    "estimated_completed_communication_cost_bytes is derived from configured "
    "num_clients x rounds_completed, not from actual client replies (Flower Result "
    "does not expose actual per-round participation)."
)
# Active deployment SuperNodes. Used only to name *expected* raw-log files; their
# presence is existence-checked, never fabricated.
DEPLOYMENT_SITE_LABELS: tuple[str, ...] = ("site-b", "site-c")

STATUS_COMPLETED = "completed"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"
STATUS_PRESENT = "present"
STATUS_EXPECTED = "expected"

_TRAIN_METRIC_KEYS = ("train_loss",)
_EVAL_METRIC_KEYS = ("map", "map_50", "map_75")


def result_head_arrays(result: Any) -> list[np.ndarray] | None:
    """Return the final aggregated head arrays from a strategy result, or None.

    Flower 1.30 returns a ``Result`` whose ``arrays`` is an ``ArrayRecord``. An
    empty record means aggregation produced no final head (for example every
    round failed), so there is nothing to save.
    """

    arrays = getattr(result, "arrays", None)
    if arrays is None or len(arrays) == 0:
        return None
    return list(arrays.to_numpy_ndarrays())


def save_head_npz(
    path: str | Path,
    arrays: list[np.ndarray],
    names: list[str] | None = None,
) -> str:
    """Save head arrays to a ``.npz`` file; return the path written.

    When ``names`` line up with ``arrays`` the archive is keyed by parameter
    name (stable across clients); otherwise positional ``arr_<i>`` keys are used.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if names is not None and len(names) == len(arrays):
        keyed = {name: array for name, array in zip(names, arrays, strict=True)}
    else:
        keyed = {f"arr_{index}": array for index, array in enumerate(arrays)}
    np.savez(output_path, **keyed)
    return str(output_path)


def latest_round_metrics(metrics_by_round: dict[int, Any]) -> dict[str, Any]:
    """Return the highest-round metric record as a plain dict (empty if none)."""

    if not metrics_by_round:
        return {}
    latest = max(metrics_by_round)
    return dict(metrics_by_round[latest])


def _rounds_completed(result: Any) -> int:
    keys: list[int] = []
    for attr in ("train_metrics_clientapp", "evaluate_metrics_clientapp"):
        keys.extend(getattr(result, attr, {}) or {})
    return max(keys) if keys else 0


def _select_metrics(source: dict[str, Any], keys: tuple[str, ...]) -> dict[str, float]:
    return {key: float(source[key]) for key in keys if key in source}


def _communication_costs(
    update_size_bytes: int,
    clients: int,
    num_rounds: int,
    rounds_completed: int,
) -> tuple[int, int]:
    """Return (planned, estimated_completed) head-transfer byte costs.

    Each round sends the head to clients and collects it back -> factor of 2. The
    estimated-completed cost uses ``rounds_completed`` but still assumes all
    configured ``clients`` replied, because Flower's Result does not expose the
    actual per-round responder count.
    """

    planned = update_size_bytes * clients * num_rounds * 2
    estimated_completed = update_size_bytes * clients * rounds_completed * 2
    return int(planned), int(estimated_completed)


def _jsonify_scalar(value: Any) -> Any:
    """Coerce a MetricRecord scalar to a JSON-safe value (guards numpy scalars)."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    item = getattr(value, "item", None)
    if callable(item):  # numpy scalar / 0-d array
        return item()
    if isinstance(value, (list, tuple)):
        return [_jsonify_scalar(element) for element in value]
    return value


def _split_round_record(record: Any) -> tuple[dict[str, Any], int | None]:
    """Split a round MetricRecord into (weighted metrics, weighted num-examples).

    Pops the ``num-examples`` weighting count and returns every remaining scalar
    metric (not a fixed allowlist) JSON-coerced. These are Flower weighted
    aggregates, not per-client raw values.
    """

    data = dict(record)
    raw_count = data.pop(WEIGHTED_BY_KEY, None)
    num_examples = None if raw_count is None else int(raw_count)
    weighted = {key: _jsonify_scalar(value) for key, value in data.items()}
    return weighted, num_examples


def deployment_log_dir(config: DetectionConfig) -> Path:
    """Raw-text-log directory for a run: ``outputs/logs/<EXP-ID>/``.

    Derived from ``output_dir``'s parent so ``outputs/EXP-012`` maps to
    ``outputs/logs/EXP-012``. Raw logs are captured manually; this only defines
    (and lets the writer create) the location.
    """

    return Path(config.output_dir).parent / "logs" / config.exp_id


def log_file_targets(log_dir: Path) -> dict[str, Any]:
    """Canonical raw-log filenames under a run's ``log_dir`` (single source of truth).

    Both the deployment artifact writer (via :func:`build_log_paths`) and the
    ``scripts/capture_flower_logs.py`` capture helper resolve their log paths through
    this one function, so a run's logs always land where its summary expects them.
    Client log filenames come from ``DEPLOYMENT_SITE_LABELS``.
    """

    return {
        "log_dir": log_dir,
        "flower_run_log": log_dir / "flower_run_log.txt",
        "server_log": log_dir / "server_log.txt",
        "client_logs": {
            label: log_dir / f"client_{label.replace('-', '_')}_log.txt"
            for label in DEPLOYMENT_SITE_LABELS
        },
    }


def _log_entry(path: Path) -> dict[str, str]:
    status = STATUS_PRESENT if path.is_file() else STATUS_EXPECTED
    return {"path": str(path), "status": status}


def build_log_paths(
    config: DetectionConfig,
    *,
    run_id: str | int | None = None,
) -> dict[str, Any]:
    """Describe the expected raw-log locations for this run (JSON-ready).

    Paths come from :func:`log_file_targets`; each entry is marked
    ``present``/``expected`` by existence check -- we never fabricate log files.
    """

    targets = log_file_targets(deployment_log_dir(config))
    client_logs = {
        label: _log_entry(path) for label, path in targets["client_logs"].items()
    }
    return {
        "log_dir": str(targets["log_dir"]),
        "server_log": _log_entry(targets["server_log"]),
        "client_logs": client_logs,
        "flower_log_command": _flower_log_command(run_id),
        "note": (
            "Raw Flower/SuperLink/SuperNode logs are captured manually under log_dir; "
            "structured metrics live in round_metrics.json / deployment_summary.json."
        ),
    }


def _flower_log_command(run_id: str | int | None) -> str:
    return f"flwr log {run_id if run_id is not None else '<run-id>'} deploy --show"


def build_round_metrics(
    config: DetectionConfig,
    *,
    status: str,
    update_size_bytes: int,
    result: Any | None,
    num_clients: int | None = None,
) -> dict[str, Any]:
    """Assemble the per-round deployment metrics artifact (JSON-ready).

    Metrics are Flower weighted aggregates keyed by round. Client participation is
    recorded as null with ``PARTICIPATION_NOTE`` because Flower's Result exposes no
    per-client identities on this path -- this must not be invented.
    """

    clients = config.num_clients if num_clients is None else num_clients
    rounds_completed = _rounds_completed(result) if result is not None else 0
    planned_cost, estimated_cost = _communication_costs(
        update_size_bytes, clients, config.num_rounds, rounds_completed
    )

    return {
        "exp_id": config.exp_id,
        "status": status,
        "num_rounds_configured": config.num_rounds,
        "rounds_completed": int(rounds_completed),
        "num_clients": clients,
        "min_train_nodes": config.effective_min_train_nodes,
        "min_evaluate_nodes": config.effective_min_evaluate_nodes,
        "min_available_nodes": config.effective_min_available_nodes,
        "update_size_bytes": int(update_size_bytes),
        "planned_communication_cost_bytes": planned_cost,
        "estimated_completed_communication_cost_bytes": estimated_cost,
        "metrics_semantics": METRICS_SEMANTICS,
        "weighted_by_key": WEIGHTED_BY_KEY,
        "client_participation": {
            "actual_train_clients": None,
            "actual_evaluate_clients": None,
            "actual_client_ids": None,
            "participation_note": PARTICIPATION_NOTE,
        },
        "rounds": _build_round_entries(result),
        "note": COMPLETED_COST_NOTE,
    }


def _build_round_entries(result: Any | None) -> list[dict[str, Any]]:
    if result is None:
        return []
    train_by_round = getattr(result, "train_metrics_clientapp", {}) or {}
    eval_by_round = getattr(result, "evaluate_metrics_clientapp", {}) or {}
    round_numbers = sorted(set(train_by_round) | set(eval_by_round))
    return [
        _round_entry(
            round_number,
            train_by_round.get(round_number),
            eval_by_round.get(round_number),
        )
        for round_number in round_numbers
    ]


def _round_entry(
    round_number: int,
    train_record: Any | None,
    eval_record: Any | None,
) -> dict[str, Any]:
    return {
        "round": int(round_number),
        "train": _phase_metrics(train_record),
        "evaluate": _phase_metrics(eval_record),
        # Per-round participation is unknown for the same reason as the top-level
        # block; recorded as null rather than invented.
        "actual_train_clients": None,
        "actual_evaluate_clients": None,
        "actual_client_ids": None,
    }


def _phase_metrics(record: Any | None) -> dict[str, Any]:
    if record is None:
        return {"weighted_aggregate": {}, "weighted_num_examples": None}
    weighted, num_examples = _split_round_record(record)
    return {"weighted_aggregate": weighted, "weighted_num_examples": num_examples}


def derive_status(
    result: Any | None,
    *,
    expected_rounds: int,
    exception_raised: bool = False,
) -> str:
    """Derive a deployment status from the strategy result.

    - ``failed``: an exception was raised, no result was returned, or no round
      completed (Flower produced nothing usable).
    - ``completed``: at least the expected number of rounds completed *and* a
      final aggregated head is available.
    - ``partial``: rounds completed but the final head is missing, or fewer than
      the expected rounds finished (e.g. early stop / stragglers dropping out).
    """

    if exception_raised or result is None:
        return STATUS_FAILED
    rounds_completed = _rounds_completed(result)
    if rounds_completed == 0:
        return STATUS_FAILED
    has_head = result_head_arrays(result) is not None
    if has_head and rounds_completed >= expected_rounds:
        return STATUS_COMPLETED
    return STATUS_PARTIAL


def build_deployment_summary(
    config: DetectionConfig,
    *,
    status: str,
    update_size_bytes: int,
    started_at: str,
    ended_at: str,
    runtime_seconds: float,
    result: Any | None,
    output_paths: dict[str, str],
    num_clients: int | None = None,
    run_id: str | int | None = None,
) -> dict[str, Any]:
    """Assemble the deployment summary dict (JSON-ready)."""

    clients = config.num_clients if num_clients is None else num_clients
    final_metrics: dict[str, float] = {}
    rounds_completed = 0
    if result is not None:
        train_latest = latest_round_metrics(getattr(result, "train_metrics_clientapp", {}))
        eval_latest = latest_round_metrics(getattr(result, "evaluate_metrics_clientapp", {}))
        final_metrics = {
            **_select_metrics(train_latest, _TRAIN_METRIC_KEYS),
            **_select_metrics(eval_latest, _EVAL_METRIC_KEYS),
        }
        rounds_completed = _rounds_completed(result)

    planned_communication_cost_bytes, estimated_completed_communication_cost_bytes = (
        _communication_costs(update_size_bytes, clients, config.num_rounds, rounds_completed)
    )

    return {
        "exp_id": config.exp_id,
        "status": status,
        "mode": "federated-deployment",
        "config": {
            "num_clients": clients,
            "num_rounds": config.num_rounds,
            "image_size": config.image_size,
            "batch_size": config.batch_size,
            "local_epochs": config.local_epochs,
            "lr": config.lr,
            "device": config.device,
            "pretrained": config.pretrained,
            "seed": config.seed,
            # Effective min-node thresholds actually passed to FedAvg (strict by
            # default: equal to num_clients unless explicitly overridden).
            "min_train_nodes": config.effective_min_train_nodes,
            "min_evaluate_nodes": config.effective_min_evaluate_nodes,
            "min_available_nodes": config.effective_min_available_nodes,
        },
        "server": {
            "aggregation_strategy": AGGREGATION_STRATEGY,
            "weighted_by_key": WEIGHTED_BY_KEY,
            "evaluation": "distributed",
        },
        "rounds_completed": int(rounds_completed),
        "update_size_bytes": int(update_size_bytes),
        "planned_communication_cost_bytes": int(planned_communication_cost_bytes),
        "estimated_completed_communication_cost_bytes": int(
            estimated_completed_communication_cost_bytes
        ),
        "estimated_completed_communication_cost_note": COMPLETED_COST_NOTE,
        "timing": {
            "started_at": started_at,
            "ended_at": ended_at,
            "runtime_seconds": round(float(runtime_seconds), 3),
        },
        "final_metrics": final_metrics,
        "output_paths": output_paths,
        "logs": build_log_paths(config, run_id=run_id),
        "flower_log_command": _flower_log_command(run_id),
        "note": SERVER_DATA_NOTE,
    }


def write_deployment_artifacts(
    config: DetectionConfig,
    *,
    result: Any | None,
    update_size_bytes: int,
    started_at: str,
    ended_at: str,
    runtime_seconds: float,
    status: str,
    head_param_names: list[str] | None = None,
    num_clients: int | None = None,
    run_id: str | int | None = None,
) -> dict[str, str]:
    """Write ``deployment_summary.json``, ``round_metrics.json`` and ``final_head.npz``.

    Returns the map of artifact name -> path written. ``final_head.npz`` is only
    written when the result exposes final aggregated arrays. The raw-log directory
    (``outputs/logs/<EXP-ID>/``) is created so manual logs have a home, but log
    text files themselves are not fabricated here.
    """

    output_dir = Path(config.output_dir)
    output_paths: dict[str, str] = {}

    # Create the raw-log directory up front so its referenced paths are real.
    deployment_log_dir(config).mkdir(parents=True, exist_ok=True)

    arrays = result_head_arrays(result)
    if arrays:
        output_paths["final_head"] = save_head_npz(
            output_dir / "final_head.npz", arrays, names=head_param_names
        )

    # Register both structured artifact paths before building the summary so the
    # summary's embedded output_paths lists round_metrics.json too.
    round_metrics_path = output_dir / "round_metrics.json"
    output_paths["round_metrics_path"] = str(round_metrics_path)
    summary_path = output_dir / "deployment_summary.json"
    output_paths["deployment_summary"] = str(summary_path)

    round_metrics = build_round_metrics(
        config,
        status=status,
        update_size_bytes=update_size_bytes,
        result=result,
        num_clients=num_clients,
    )
    write_json(round_metrics_path, round_metrics)

    summary = build_deployment_summary(
        config,
        status=status,
        update_size_bytes=update_size_bytes,
        started_at=started_at,
        ended_at=ended_at,
        runtime_seconds=runtime_seconds,
        result=result,
        output_paths=output_paths,
        num_clients=num_clients,
        run_id=run_id,
    )
    write_json(summary_path, summary)
    return output_paths


def finalize_deployment_artifacts(
    config: DetectionConfig,
    *,
    result: Any | None,
    update_size_bytes: int,
    started_at: str,
    ended_at: str,
    runtime_seconds: float,
    exception_raised: bool,
    head_param_names: list[str] | None = None,
    num_clients: int | None = None,
    run_id: str | int | None = None,
) -> tuple[str, dict[str, str]]:
    """Derive the run status from the result, then write the artifacts.

    This is the network-free seam used by the ServerApp: status is derived from
    the strategy ``result`` (not assumed "completed"), so a run where every
    round failed or only some rounds finished is recorded honestly. Returns
    ``(status, output_paths)``.
    """

    status = derive_status(
        result,
        expected_rounds=config.num_rounds,
        exception_raised=exception_raised,
    )
    output_paths = write_deployment_artifacts(
        config,
        result=result,
        update_size_bytes=update_size_bytes,
        started_at=started_at,
        ended_at=ended_at,
        runtime_seconds=runtime_seconds,
        status=status,
        head_param_names=head_param_names,
        num_clients=num_clients,
        run_id=run_id,
    )
    return status, output_paths
