"""FedAvg over detection heads, in-process simulation (Phase A validation).

A lightweight, fully testable FedAvg loop: each round every client trains its head
from the current global head, the server averages the heads weighted by train-set
size, and evaluation is **distributed** (each client evaluates the global head on
its own val shard -- the server holds no data). The deployment ServerApp/ClientApp
reuse the exact same train/eval functions and averaging math.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.data.detection_data import DetectionClientData, DetectionDatasetBundle
from src.evaluation.metrics import parameter_bytes
from src.models.detection_model import (
    build_detection_model,
    get_detection_head_parameters,
    resolve_device,
    set_detection_head_parameters,
)
from src.training.detection_trainer import evaluate_detection, train_detection_head
from src.utils.detection_config import DetectionConfig

_REPORT_KEYS = ("map", "map_50", "map_75")


def federated_average(
    param_lists: list[list[np.ndarray]], weights: list[float]
) -> list[np.ndarray]:
    """Weighted element-wise average of head parameter lists (FedAvg)."""

    total = float(sum(weights))
    if total == 0:
        raise ValueError("federated_average requires a positive total weight")
    num_arrays = len(param_lists[0])
    return [
        sum(weight * params[i] for params, weight in zip(param_lists, weights)) / total
        for i in range(num_arrays)
    ]


def run_detection_federated(
    config: DetectionConfig,
    bundle: DetectionDatasetBundle,
) -> dict[str, Any]:
    device = resolve_device(config.device)
    start = time.perf_counter()

    global_params = get_detection_head_parameters(_build(config, bundle))
    update_size = parameter_bytes(global_params)
    history: list[dict[str, Any]] = []

    for server_round in range(1, config.num_rounds + 1):
        client_params: list[list[np.ndarray]] = []
        weights: list[float] = []
        for client in bundle.clients:
            model = _build(config, bundle)
            set_detection_head_parameters(model, global_params)
            train_detection_head(
                model,
                client.train,
                epochs=config.local_epochs,
                **_train_kwargs(config, device),
            )
            client_params.append(get_detection_head_parameters(model))
            weights.append(float(len(client.train)))
        global_params = federated_average(client_params, weights)
        round_records = _distributed_eval(config, bundle, global_params, device)
        history.append(
            {"round": server_round, **_weighted_global(round_records)}
        )

    per_client = _distributed_eval(config, bundle, global_params, device)
    return {
        "mode": "federated",
        "exp_id": config.exp_id,
        "num_clients": len(bundle.clients),
        "num_classes": config.num_classes,
        "rounds": config.num_rounds,
        "global": _weighted_global(per_client),
        "per_client": per_client,
        "history": history,
        "training_time_sec": time.perf_counter() - start,
        "update_size_bytes": update_size,
        "communication_cost_bytes": update_size * len(bundle.clients) * config.num_rounds * 2,
    }


def _distributed_eval(
    config: DetectionConfig,
    bundle: DetectionDatasetBundle,
    global_params: list[np.ndarray],
    device: str,
) -> list[dict[str, Any]]:
    records = []
    for client in bundle.clients:
        model = _build(config, bundle)
        set_detection_head_parameters(model, global_params)
        metrics = evaluate_detection(
            model,
            client.val,
            batch_size=config.batch_size,
            device=device,
            num_workers=config.num_workers,
            score_threshold=config.score_threshold,
        )
        records.append(_client_record(client, metrics))
    return records


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
    global_metrics: dict[str, Any] = {}
    for key in _REPORT_KEYS:
        total = sum(r["num_examples"] for r in per_client if r.get(key, -1.0) >= 0)
        if total == 0:
            global_metrics[key] = -1.0
            continue
        global_metrics[key] = sum(
            r["num_examples"] * r[key] for r in per_client if r.get(key, -1.0) >= 0
        ) / total
    return global_metrics
