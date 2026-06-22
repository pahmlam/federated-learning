"""Centralized and local-only baselines for embedding artifacts."""

from __future__ import annotations

import time
from typing import Any

from src.data.embedding import EmbeddingDatasetBundle
from src.evaluation.metrics import parameter_bytes, weighted_average_metric
from src.models.embedding_head import (
    create_embedding_head_model,
    get_embedding_head_parameters,
)
from src.training.trainer import evaluate_model, train_head
from src.utils.config import DemoConfig
from src.utils.resources import get_resource_snapshot


def run_embedding_centralized(
    config: DemoConfig,
    bundle: EmbeddingDatasetBundle,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    model = create_embedding_head_model(
        embedding_dim=bundle.embedding_dim,
        num_classes=bundle.num_classes,
        seed=config.seed,
    )
    update_size = parameter_bytes(get_embedding_head_parameters(model))
    train_head(
        model=model,
        train_x=bundle.pooled.train_x,
        train_y=bundle.pooled.train_y,
        epochs=config.centralized_epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    positive_class_id = _unsafe_class_id(bundle)
    global_metrics = evaluate_model(
        model,
        bundle.pooled.val_x,
        bundle.pooled.val_y,
        positive_class_id=positive_class_id,
    )
    per_client = []
    for client in bundle.clients:
        metrics = evaluate_model(
            model,
            client.val_x,
            client.val_y,
            positive_class_id=positive_class_id,
        )
        per_client.append(
            {
                "client_id": client.client_id,
                "client_label": client.client_label,
                "num_examples": int(client.val_y.numel()),
                "label_histogram": client.label_histogram,
                **_float_metrics(metrics),
            }
        )

    return _with_embedding_metadata(
        {
            "mode": "centralized",
            "profile": config.profile,
            "global": global_metrics,
            "per_client": per_client,
            "training_time_sec": time.perf_counter() - start_time,
            "update_size_bytes": update_size,
            "communication_cost_bytes": 0,
            "resource_snapshot": get_resource_snapshot(),
        },
        bundle,
    )


def run_embedding_local_only(
    config: DemoConfig,
    bundle: EmbeddingDatasetBundle,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    initial_model = create_embedding_head_model(
        embedding_dim=bundle.embedding_dim,
        num_classes=bundle.num_classes,
        seed=config.seed,
    )
    update_size = parameter_bytes(get_embedding_head_parameters(initial_model))
    positive_class_id = _unsafe_class_id(bundle)
    per_client = []
    for client in bundle.clients:
        model = create_embedding_head_model(
            embedding_dim=bundle.embedding_dim,
            num_classes=bundle.num_classes,
            seed=config.seed,
        )
        train_head(
            model=model,
            train_x=client.train_x,
            train_y=client.train_y,
            epochs=config.local_epochs,
            batch_size=config.batch_size,
            lr=config.lr,
            seed=config.seed + client.client_id,
            num_workers=config.num_workers,
        )
        metrics = evaluate_model(
            model,
            client.val_x,
            client.val_y,
            positive_class_id=positive_class_id,
        )
        per_client.append(
            {
                "client_id": client.client_id,
                "client_label": client.client_label,
                "num_examples": int(client.val_y.numel()),
                "label_histogram": client.label_histogram,
                **_float_metrics(metrics),
            }
        )

    return _with_embedding_metadata(
        {
            "mode": "local-only",
            "profile": config.profile,
            "global": weighted_average_metric(per_client),
            "per_client": per_client,
            "training_time_sec": time.perf_counter() - start_time,
            "update_size_bytes": update_size,
            "communication_cost_bytes": 0,
            "resource_snapshot": get_resource_snapshot(),
        },
        bundle,
    )


def _with_embedding_metadata(
    result: dict[str, Any],
    bundle: EmbeddingDatasetBundle,
) -> dict[str, Any]:
    result.update(
        {
            "data_source": "embedding",
            "artifact_path": bundle.artifact_path,
            "num_clients": len(bundle.clients),
            "embedding_dim": bundle.embedding_dim,
            "num_classes": bundle.num_classes,
            "label_mapping": bundle.label_mapping,
        }
    )
    return result


def _unsafe_class_id(bundle: EmbeddingDatasetBundle) -> int | None:
    return bundle.label_mapping.get("unsafe")


def _float_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {name: float(value) for name, value in metrics.items()}
