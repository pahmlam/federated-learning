"""Centralized and local-only baselines for the synthetic demo."""

from __future__ import annotations

import time
from typing import Any

from src.data.synthetic import make_client_splits, make_pooled_dataset
from src.evaluation.metrics import parameter_bytes, weighted_average_metric
from src.models.head_model import create_model, get_head_parameters
from src.training.trainer import evaluate_model, train_head
from src.utils.config import DemoConfig


def run_centralized(config: DemoConfig) -> dict[str, Any]:
    start_time = time.perf_counter()
    clients = make_client_splits(config)
    pooled = make_pooled_dataset(clients)

    model = create_model(config)
    update_size = parameter_bytes(get_head_parameters(model))
    train_head(
        model=model,
        train_x=pooled.train_x,
        train_y=pooled.train_y,
        epochs=config.centralized_epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        seed=config.seed,
    )
    global_metrics = evaluate_model(model, pooled.val_x, pooled.val_y)

    per_client = []
    for client in clients:
        metrics = evaluate_model(model, client.val_x, client.val_y)
        per_client.append(
            {
                "client_id": client.client_id,
                "num_examples": int(client.val_y.numel()),
                "loss": float(metrics["loss"]),
                "accuracy": float(metrics["accuracy"]),
                "label_histogram": client.label_histogram,
            }
        )

    return {
        "mode": "centralized",
        "partition": config.partition,
        "num_clients": config.num_clients,
        "global": global_metrics,
        "per_client": per_client,
        "training_time_sec": time.perf_counter() - start_time,
        "update_size_bytes": update_size,
        "communication_cost_bytes": 0,
    }


def run_local_only(config: DemoConfig) -> dict[str, Any]:
    start_time = time.perf_counter()
    clients = make_client_splits(config)
    per_client = []

    initial_model = create_model(config)
    update_size = parameter_bytes(get_head_parameters(initial_model))

    for client in clients:
        model = create_model(config)
        train_head(
            model=model,
            train_x=client.train_x,
            train_y=client.train_y,
            epochs=config.local_epochs,
            batch_size=config.batch_size,
            lr=config.lr,
            seed=config.seed + client.client_id,
        )
        metrics = evaluate_model(model, client.val_x, client.val_y)
        per_client.append(
            {
                "client_id": client.client_id,
                "num_examples": int(client.val_y.numel()),
                "loss": float(metrics["loss"]),
                "accuracy": float(metrics["accuracy"]),
                "label_histogram": client.label_histogram,
            }
        )

    global_metrics = weighted_average_metric(per_client)
    return {
        "mode": "local-only",
        "partition": config.partition,
        "num_clients": config.num_clients,
        "global": global_metrics,
        "per_client": per_client,
        "training_time_sec": time.perf_counter() - start_time,
        "update_size_bytes": update_size,
        "communication_cost_bytes": 0,
    }
