"""Centralized and local-only detection baselines (mAP per-client).

Mirrors ``embedding_baselines`` but for detection: the centralized baseline trains
one head on pooled data (a reference outside the FL privacy model); local-only
trains a separate head per client on its own shard. Both report per-client mAP and
a weighted global aggregate.
"""

from __future__ import annotations

import time
from typing import Any

from src.data.detection_data import DetectionClientData, DetectionDatasetBundle
from src.evaluation.metrics import parameter_bytes
from src.models.detection_model import (
    build_detection_model,
    get_detection_head_parameters,
    resolve_device,
)
from src.training.detection_trainer import evaluate_detection, train_detection_head
from src.utils.detection_config import DetectionConfig

# mAP metric keys carried into per-client / global reports.
_REPORT_KEYS = ("map", "map_50", "map_75")


def run_detection_centralized(
    config: DetectionConfig,
    bundle: DetectionDatasetBundle,
) -> dict[str, Any]:
    device = resolve_device(config.device)
    start = time.perf_counter()
    print(f"[centralized] start on {device}", flush=True)
    model = _build(config, bundle)
    train_detection_head(
        model,
        bundle.pooled_train,
        epochs=config.centralized_epochs,
        **_train_kwargs(config, device),
        log_prefix="[centralized]",
    )
    global_metrics = _evaluate(config, model, bundle.pooled_val, device, "[centralized/global]")
    per_client = [
        _client_record(
            client,
            _evaluate(config, model, client.val, device, f"[centralized/{client.client_id}]"),
        )
        for client in bundle.clients
    ]
    update_size = parameter_bytes(get_detection_head_parameters(model))
    print(f"[centralized] done", flush=True)
    return _result("centralized", config, global_metrics, per_client, start, update_size)


def run_detection_local_only(
    config: DetectionConfig,
    bundle: DetectionDatasetBundle,
) -> dict[str, Any]:
    device = resolve_device(config.device)
    start = time.perf_counter()
    print(f"[local-only] start on {device}", flush=True)
    per_client: list[dict[str, Any]] = []
    update_size = 0
    for client in bundle.clients:
        print(f"[local-only/{client.client_id}] start", flush=True)
        model = _build(config, bundle)
        train_detection_head(
            model,
            client.train,
            epochs=config.local_epochs,
            **_train_kwargs(config, device),
            log_prefix=f"[local-only/{client.client_id}]",
        )
        per_client.append(
            _client_record(
                client,
                _evaluate(config, model, client.val, device, f"[local-only/{client.client_id}]"),
            )
        )
        update_size = parameter_bytes(get_detection_head_parameters(model))
    global_metrics = _weighted_global(per_client)
    print(f"[local-only] done", flush=True)
    return _result("local-only", config, global_metrics, per_client, start, update_size)


def _build(config: DetectionConfig, bundle: DetectionDatasetBundle):
    return build_detection_model(
        num_classes=bundle.num_classes,
        pretrained=config.pretrained,
        seed=config.seed,
    )


def _train_kwargs(config: DetectionConfig, device: str) -> dict[str, Any]:
    return {
        "batch_size": config.batch_size,
        "lr": config.lr,
        "momentum": config.momentum,
        "weight_decay": config.weight_decay,
        "device": device,
        "num_workers": config.num_workers,
        "seed": config.seed,
    }


def _evaluate(
    config: DetectionConfig,
    model,
    dataset,
    device: str,
    log_prefix: str | None = None,
) -> dict[str, Any]:
    return evaluate_detection(
        model,
        dataset,
        batch_size=config.batch_size,
        device=device,
        num_workers=config.num_workers,
        score_threshold=config.score_threshold,
        log_prefix=log_prefix,
    )


def _client_record(client: DetectionClientData, metrics: dict[str, Any]) -> dict[str, Any]:
    record = {
        "client_id": client.client_id,
        "client_label": client.client_label,
        "num_examples": len(client.val),
        "label_histogram": client.label_histogram,
    }
    record.update({key: float(metrics.get(key, -1.0)) for key in _REPORT_KEYS})
    record["map_per_class"] = metrics.get("map_per_class")
    return record


def _weighted_global(per_client: list[dict[str, Any]]) -> dict[str, Any]:
    """Weighted average of per-client mAP by validation image count (ignore -1)."""

    global_metrics: dict[str, Any] = {}
    for key in _REPORT_KEYS:
        total = sum(
            record["num_examples"] for record in per_client if record.get(key, -1.0) >= 0
        )
        if total == 0:
            global_metrics[key] = -1.0
            continue
        global_metrics[key] = sum(
            record["num_examples"] * record[key]
            for record in per_client
            if record.get(key, -1.0) >= 0
        ) / total
    return global_metrics


def _result(
    mode: str,
    config: DetectionConfig,
    global_metrics: dict[str, Any],
    per_client: list[dict[str, Any]],
    start: float,
    update_size: int,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "exp_id": config.exp_id,
        "num_clients": len(per_client),
        "num_classes": config.num_classes,
        "global": {key: float(global_metrics.get(key, -1.0)) for key in _REPORT_KEYS},
        "per_client": per_client,
        "training_time_sec": time.perf_counter() - start,
        "update_size_bytes": update_size,
        "communication_cost_bytes": 0,  # baselines do not communicate
    }
