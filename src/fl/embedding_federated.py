"""Federated baseline for precomputed embedding artifacts."""

from __future__ import annotations

import time
from typing import Any

from flwr.client import NumPyClient
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import NDArrays, Parameters
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation

from src.data.embedding import EmbeddingClientDataset, EmbeddingDatasetBundle
from src.evaluation.metrics import parameter_bytes
from src.models.embedding_head import (
    create_embedding_head_model,
    get_embedding_head_parameters,
    set_embedding_head_parameters,
)
from src.training.trainer import evaluate_model, train_head
from src.utils.config import DemoConfig
from src.utils.resources import get_resource_snapshot


class CapturingEmbeddingFedAvg(FedAvg):
    """FedAvg strategy that stores the latest aggregated embedding-head params."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.latest_parameters: Parameters | None = None

    def aggregate_fit(self, server_round, results, failures):
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if parameters is not None:
            self.latest_parameters = parameters
        return parameters, metrics


class EmbeddingHeadClient(NumPyClient):
    """Flower client for head-only training on precomputed embeddings."""

    def __init__(
        self,
        client_data: EmbeddingClientDataset,
        bundle: EmbeddingDatasetBundle,
        config: DemoConfig,
    ) -> None:
        self.client_data = client_data
        self.bundle = bundle
        self.config = config

    def fit(self, parameters: NDArrays, config: dict[str, Any]):
        model = create_embedding_head_model(
            embedding_dim=self.bundle.embedding_dim,
            num_classes=self.bundle.num_classes,
            seed=self.config.seed,
        )
        set_embedding_head_parameters(model, parameters)
        metrics = train_head(
            model=model,
            train_x=self.client_data.train_x,
            train_y=self.client_data.train_y,
            epochs=int(config.get("local_epochs", self.config.local_epochs)),
            batch_size=int(config.get("batch_size", self.config.batch_size)),
            lr=float(config.get("lr", self.config.lr)),
            seed=self.config.seed + self.client_data.client_id,
            num_workers=self.config.num_workers,
            weight_decay=float(config.get("weight_decay", self.config.weight_decay)),
        )
        updated = get_embedding_head_parameters(model)
        return (
            updated,
            int(self.client_data.train_y.numel()),
            {
                "train_loss": float(metrics["loss"]),
                "train_accuracy": float(metrics["accuracy"]),
                "update_bytes": parameter_bytes(updated),
            },
        )

    def evaluate(self, parameters: NDArrays, config: dict[str, Any]):
        model = create_embedding_head_model(
            embedding_dim=self.bundle.embedding_dim,
            num_classes=self.bundle.num_classes,
            seed=self.config.seed,
        )
        set_embedding_head_parameters(model, parameters)
        metrics = evaluate_model(
            model,
            self.client_data.val_x,
            self.client_data.val_y,
            positive_class_id=_unsafe_class_id(self.bundle),
        )
        return (
            float(metrics["loss"]),
            int(self.client_data.val_y.numel()),
            _metric_payload(metrics, exclude={"loss"}),
        )


def run_embedding_federated(
    config: DemoConfig,
    bundle: EmbeddingDatasetBundle,
) -> dict[str, Any]:
    """Run FedAvg over embedding clients and return JSON-ready metrics."""

    start_time = time.perf_counter()
    initial_model = create_embedding_head_model(
        embedding_dim=bundle.embedding_dim,
        num_classes=bundle.num_classes,
        seed=config.seed,
    )
    initial_parameters = get_embedding_head_parameters(initial_model)
    update_bytes = parameter_bytes(initial_parameters)
    strategy = CapturingEmbeddingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=len(bundle.clients),
        min_evaluate_clients=len(bundle.clients),
        min_available_clients=len(bundle.clients),
        initial_parameters=ndarrays_to_parameters(initial_parameters),
        evaluate_fn=_server_evaluate_fn(config, bundle),
        on_fit_config_fn=lambda server_round: {
            "server_round": server_round,
            "local_epochs": config.local_epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
        },
        fit_metrics_aggregation_fn=_weighted_fit_metrics,
        evaluate_metrics_aggregation_fn=_weighted_eval_metrics,
    )

    def client_fn(context: Context):
        partition_id = int(context.node_config.get("partition-id", context.node_id))
        client = bundle.clients[partition_id % len(bundle.clients)]
        return EmbeddingHeadClient(client, bundle, config).to_client()

    ray_init_args: dict[str, Any] = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
    }
    if config.ray_num_cpus is not None:
        ray_init_args["num_cpus"] = config.ray_num_cpus

    history = start_simulation(
        client_fn=client_fn,
        num_clients=len(bundle.clients),
        config=ServerConfig(num_rounds=config.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": config.client_num_cpus},
        ray_init_args=ray_init_args,
    )
    final_parameters = (
        parameters_to_ndarrays(strategy.latest_parameters)
        if strategy.latest_parameters is not None
        else initial_parameters
    )
    final_model = create_embedding_head_model(
        embedding_dim=bundle.embedding_dim,
        num_classes=bundle.num_classes,
        seed=config.seed,
    )
    set_embedding_head_parameters(final_model, final_parameters)
    global_metrics = evaluate_model(
        final_model,
        bundle.pooled.val_x,
        bundle.pooled.val_y,
        positive_class_id=_unsafe_class_id(bundle),
    )
    per_client = _evaluate_per_client(config, bundle, final_parameters)
    return {
        "mode": "federated",
        "profile": config.profile,
        "data_source": "embedding",
        "artifact_path": bundle.artifact_path,
        "num_clients": len(bundle.clients),
        "rounds": config.num_rounds,
        "embedding_dim": bundle.embedding_dim,
        "num_classes": bundle.num_classes,
        "label_mapping": bundle.label_mapping,
        "global": global_metrics,
        "per_client": per_client,
        "training_time_sec": time.perf_counter() - start_time,
        "update_size_bytes": update_bytes,
        "communication_cost_bytes": update_bytes * len(bundle.clients) * config.num_rounds * 2,
        "resource_snapshot": get_resource_snapshot(),
        "flower_history": _history_to_dict(history),
    }


def _server_evaluate_fn(config: DemoConfig, bundle: EmbeddingDatasetBundle):
    def evaluate(server_round: int, parameters: NDArrays, run_config: dict[str, Any]):
        model = create_embedding_head_model(
            embedding_dim=bundle.embedding_dim,
            num_classes=bundle.num_classes,
            seed=config.seed,
        )
        set_embedding_head_parameters(model, parameters)
        metrics = evaluate_model(
            model,
            bundle.pooled.val_x,
            bundle.pooled.val_y,
            positive_class_id=_unsafe_class_id(bundle),
        )
        return float(metrics["loss"]), _metric_payload(metrics, exclude={"loss"})

    return evaluate


def _evaluate_per_client(
    config: DemoConfig,
    bundle: EmbeddingDatasetBundle,
    parameters: NDArrays,
) -> list[dict[str, Any]]:
    records = []
    for client in bundle.clients:
        model = create_embedding_head_model(
            embedding_dim=bundle.embedding_dim,
            num_classes=bundle.num_classes,
            seed=config.seed,
        )
        set_embedding_head_parameters(model, parameters)
        metrics = evaluate_model(
            model,
            client.val_x,
            client.val_y,
            positive_class_id=_unsafe_class_id(bundle),
        )
        records.append(
            {
                "client_id": client.client_id,
                "client_label": client.client_label,
                "num_examples": int(client.val_y.numel()),
                "label_histogram": client.label_histogram,
                **_json_metrics(metrics),
            }
        )
    return records


def _weighted_fit_metrics(records):
    total = sum(num_examples for num_examples, _ in records)
    if total == 0:
        return {}
    return {
        "train_loss": sum(
            num_examples * float(metrics["train_loss"])
            for num_examples, metrics in records
        )
        / total,
        "train_accuracy": sum(
            num_examples * float(metrics["train_accuracy"])
            for num_examples, metrics in records
        )
        / total,
    }


def _weighted_eval_metrics(records):
    total = sum(num_examples for num_examples, _ in records)
    if total == 0:
        return {}
    metric_names = sorted({key for _, metrics in records for key in metrics})
    return {
        metric_name: sum(
            num_examples * float(metrics[metric_name])
            for num_examples, metrics in records
            if metric_name in metrics
        )
        / total
        for metric_name in metric_names
    }


def _history_to_dict(history) -> dict[str, Any]:
    return {
        "losses_centralized": history.losses_centralized,
        "metrics_centralized": history.metrics_centralized,
        "losses_distributed": history.losses_distributed,
        "metrics_distributed": history.metrics_distributed,
    }


def _unsafe_class_id(bundle: EmbeddingDatasetBundle) -> int | None:
    return bundle.label_mapping.get("unsafe")


def _metric_payload(
    metrics: dict[str, Any],
    exclude: set[str] | None = None,
) -> dict[str, float]:
    excluded = exclude or set()
    return {
        name: float(value)
        for name, value in metrics.items()
        if name not in excluded and isinstance(value, (int, float))
    }


def _json_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        name: float(value) if isinstance(value, (int, float)) else value
        for name, value in metrics.items()
    }
