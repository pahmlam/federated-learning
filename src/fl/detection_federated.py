"""FedAvg over detection heads, in-process simulation (Phase A validation).

A lightweight, fully testable FedAvg loop: each round every client trains its head
from the current global head, the server averages the heads weighted by train-set
size, and evaluation is **distributed** (each client evaluates the global head on
its own val shard -- the server holds no data). The deployment ServerApp/ClientApp
reuse the exact same train/eval functions and averaging math.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import numpy as np
from torch.utils.data import Subset

from src.data.detection_data import DetectionClientData, DetectionDatasetBundle
from src.data.detection_dataset import PPEDetectionDataset
from src.evaluation.metrics import parameter_bytes
from src.fl.edge_profile import (
    EdgeProfile,
    apply_profile_overrides,
    edge_decision,
    profile_metrics,
    resolve_edge_profile,
)
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
    print(f"[federated] start on {device}, rounds={config.num_rounds}", flush=True)

    global_params = get_detection_head_parameters(_build(config, bundle))
    update_size = parameter_bytes(global_params)
    history: list[dict[str, Any]] = []

    for server_round in range(1, config.num_rounds + 1):
        print(f"[federated] round {server_round}/{config.num_rounds} start", flush=True)
        client_params: list[list[np.ndarray]] = []
        weights: list[float] = []
        train_records: list[dict[str, Any]] = []
        train_failures: list[dict[str, Any]] = []
        for client in bundle.clients:
            print(
                f"[federated] round {server_round}/{config.num_rounds} "
                f"client {client.client_id} train",
                flush=True,
            )
            profile = _profile_for(config, client)
            client_config = apply_profile_overrides(config, profile)
            effective_client = _client_with_profile(client, profile, bundle)
            decision = edge_decision(
                profile,
                seed=client_config.seed,
                client_id=client.client_id,
                round_number=server_round,
                stage="train",
            )
            if not decision.should_run:
                train_failures.append(
                    _edge_failure_record(client, profile, decision, stage="train")
                )
                print(
                    f"[federated] round {server_round}/{config.num_rounds} "
                    f"client {client.client_id} skipped: {decision.reason}",
                    flush=True,
                )
                continue

            model = _build(client_config, bundle)
            set_detection_head_parameters(model, global_params)
            train_data = _train_dataset(effective_client, profile)
            client_start = time.perf_counter()
            if profile and profile.artificial_train_delay_sec > 0:
                time.sleep(profile.artificial_train_delay_sec)
            train_detection_head(
                model,
                train_data,
                epochs=client_config.local_epochs,
                **_train_kwargs(client_config, device),
                log_prefix=f"[federated/r{server_round}/{client.client_id}]",
            )
            elapsed = time.perf_counter() - client_start
            client_params.append(get_detection_head_parameters(model))
            weights.append(float(len(train_data)))
            train_record = {
                "client_id": client.client_id,
                "client_label": client.client_label,
                "num_examples": len(train_data),
                "effective_train_time": elapsed,
            }
            train_record.update(
                profile_metrics(profile, decision, update_size_bytes=update_size)
            )
            train_records.append(train_record)
        if not client_params:
            raise RuntimeError("No detection clients completed training in this round")
        global_params = federated_average(client_params, weights)
        print(f"[federated] round {server_round}/{config.num_rounds} aggregate done", flush=True)
        round_records = _distributed_eval(
            config,
            bundle,
            global_params,
            device,
            f"[federated/r{server_round}/eval]",
            round_number=server_round,
            update_size_bytes=update_size,
        )
        history_record = {"round": server_round, **_weighted_global(round_records)}
        if _has_edge_profiles(config):
            history_record["train_clients"] = train_records
            history_record["train_failures"] = train_failures
        history.append(history_record)
        round_metrics = history[-1]
        print(
            f"[federated] round {server_round}/{config.num_rounds} done: "
            f"map={round_metrics.get('map', -1.0):.4f}, "
            f"map_50={round_metrics.get('map_50', -1.0):.4f}",
            flush=True,
        )

    per_client = _distributed_eval(
        config,
        bundle,
        global_params,
        device,
        "[federated/final]",
        round_number=config.num_rounds + 1,
        update_size_bytes=update_size,
    )
    print(f"[federated] done", flush=True)
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
    log_prefix: str | None = None,
    *,
    round_number: int = 1,
    update_size_bytes: int = 0,
) -> list[dict[str, Any]]:
    records = []
    for client in bundle.clients:
        profile = _profile_for(config, client)
        client_config = apply_profile_overrides(config, profile)
        effective_client = _client_with_profile(client, profile, bundle)
        decision = edge_decision(
            profile,
            seed=client_config.seed,
            client_id=client.client_id,
            round_number=round_number,
            stage="evaluate",
        )
        if not decision.should_run:
            records.append(
                _edge_failure_record(client, profile, decision, stage="evaluate")
            )
            continue

        model = _build(client_config, bundle)
        set_detection_head_parameters(model, global_params)
        eval_start = time.perf_counter()
        metrics = evaluate_detection(
            model,
            effective_client.val,
            batch_size=client_config.batch_size,
            device=device,
            num_workers=client_config.num_workers,
            score_threshold=client_config.score_threshold,
            log_prefix=f"{log_prefix}/{client.client_id}" if log_prefix else None,
        )
        elapsed = time.perf_counter() - eval_start
        record = _client_record(effective_client, metrics)
        edge_metrics = profile_metrics(profile, decision, update_size_bytes=update_size_bytes)
        if edge_metrics:
            edge_metrics["effective_eval_time"] = elapsed
            record.update(edge_metrics)
        records.append(record)
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


def _profile_for(config: DetectionConfig, client: DetectionClientData) -> EdgeProfile | None:
    return resolve_edge_profile(
        edge_profile=config.edge_profile,
        edge_profiles=config.edge_profiles,
        client_label=client.client_label,
        client_id=client.client_id,
    )


def _client_with_profile(
    client: DetectionClientData,
    profile: EdgeProfile | None,
    bundle: DetectionDatasetBundle,
) -> DetectionClientData:
    if profile is None or profile.image_size is None or profile.image_size == bundle.image_size:
        return client
    return replace(
        client,
        train=PPEDetectionDataset(client.train.records, image_size=profile.image_size),
        val=PPEDetectionDataset(client.val.records, image_size=profile.image_size),
    )


def _train_dataset(client: DetectionClientData, profile: EdgeProfile | None):
    if profile is None or profile.max_train_samples is None:
        return client.train
    limit = min(profile.max_train_samples, len(client.train))
    if limit == len(client.train):
        return client.train
    return Subset(client.train, range(limit))


def _edge_failure_record(
    client: DetectionClientData,
    profile: EdgeProfile | None,
    decision,
    *,
    stage: str,
) -> dict[str, Any]:
    record = {
        "client_id": client.client_id,
        "client_label": client.client_label,
        "num_examples": 0,
        "stage": stage,
        "edge_profile_enabled": 1.0 if profile else 0.0,
        "edge_availability_decision": 1.0 if decision.available else 0.0,
        "edge_dropout_decision": 1.0 if decision.dropped else 0.0,
        "edge_dropout_reason_code": float(decision.reason_code),
    }
    for key in _REPORT_KEYS:
        record[key] = -1.0
    return record


def _has_edge_profiles(config: DetectionConfig) -> bool:
    return bool(config.edge_profile or config.edge_profiles)


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
