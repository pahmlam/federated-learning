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

STATUS_COMPLETED = "completed"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"

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

    # Each round sends the head to clients and collects it back -> factor of 2.
    planned_communication_cost_bytes = update_size_bytes * clients * config.num_rounds * 2
    estimated_completed_communication_cost_bytes = (
        update_size_bytes * clients * rounds_completed * 2
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
        "timing": {
            "started_at": started_at,
            "ended_at": ended_at,
            "runtime_seconds": round(float(runtime_seconds), 3),
        },
        "final_metrics": final_metrics,
        "output_paths": output_paths,
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
) -> dict[str, str]:
    """Write ``deployment_summary.json`` and ``final_head.npz`` (when available).

    Returns the map of artifact name -> path written. ``final_head.npz`` is only
    written when the result exposes final aggregated arrays.
    """

    output_dir = Path(config.output_dir)
    output_paths: dict[str, str] = {}

    arrays = result_head_arrays(result)
    if arrays:
        output_paths["final_head"] = save_head_npz(
            output_dir / "final_head.npz", arrays, names=head_param_names
        )

    summary_path = output_dir / "deployment_summary.json"
    output_paths["deployment_summary"] = str(summary_path)

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
    )
    return status, output_paths
