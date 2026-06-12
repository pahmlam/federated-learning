"""Federated baseline using Flower simulation and head-only updates."""

from __future__ import annotations

import time
from typing import Any

from flwr.client import NumPyClient
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import NDArrays, Parameters
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation

from src.data.synthetic import ClientDataset, make_client_splits, make_pooled_dataset
from src.evaluation.metrics import parameter_bytes
from src.models.head_model import create_model, get_head_parameters, set_head_parameters
from src.training.trainer import evaluate_model, train_head
from src.utils.config import DemoConfig


class CapturingFedAvg(FedAvg):
    """FedAvg strategy that keeps the latest aggregated parameters."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.latest_parameters: Parameters | None = None

    def aggregate_fit(self, server_round, results, failures):
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if parameters is not None:
            self.latest_parameters = parameters
        return parameters, metrics


class HeadOnlyClient(NumPyClient):
    """Flower NumPyClient that trains only classifier head parameters."""

    def __init__(self, client_data: ClientDataset, config: DemoConfig) -> None:
        self.client_data = client_data
        self.config = config

    def fit(self, parameters: NDArrays, config: dict[str, Any]):
        model = create_model(self.config)
        set_head_parameters(model, parameters)
        metrics = train_head(
            model=model,
            train_x=self.client_data.train_x,
            train_y=self.client_data.train_y,
            epochs=int(config.get("local_epochs", self.config.local_epochs)),
            batch_size=int(config.get("batch_size", self.config.batch_size)),
            lr=float(config.get("lr", self.config.lr)),
            seed=self.config.seed + self.client_data.client_id,
        )
        updated = get_head_parameters(model)
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
        model = create_model(self.config)
        set_head_parameters(model, parameters)
        metrics = evaluate_model(model, self.client_data.val_x, self.client_data.val_y)
        return (
            float(metrics["loss"]),
            int(self.client_data.val_y.numel()),
            {"accuracy": float(metrics["accuracy"])},
        )


def run_federated(config: DemoConfig) -> dict[str, Any]:
    """Run FedAvg over simulated clients and return JSON-ready metrics."""

    start_time = time.perf_counter()
    clients = make_client_splits(config)
    pooled = make_pooled_dataset(clients)
    initial_model = create_model(config)
    initial_parameters = get_head_parameters(initial_model)
    update_bytes = parameter_bytes(initial_parameters)

    strategy = CapturingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=config.num_clients,
        min_evaluate_clients=config.num_clients,
        min_available_clients=config.num_clients,
        initial_parameters=ndarrays_to_parameters(initial_parameters),
        evaluate_fn=_server_evaluate_fn(config, pooled.val_x, pooled.val_y),
        on_fit_config_fn=lambda server_round: {
            "server_round": server_round,
            "local_epochs": config.local_epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
        },
        fit_metrics_aggregation_fn=_weighted_fit_metrics,
        evaluate_metrics_aggregation_fn=_weighted_eval_metrics,
    )

    def client_fn(context: Context):
        partition_id = int(context.node_config.get("partition-id", context.node_id))
        client = clients[partition_id % config.num_clients]
        return HeadOnlyClient(client, config).to_client()

    history = start_simulation(
        client_fn=client_fn,
        num_clients=config.num_clients,
        config=ServerConfig(num_rounds=config.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1},
        ray_init_args={"ignore_reinit_error": True, "include_dashboard": False},
    )

    final_parameters = (
        parameters_to_ndarrays(strategy.latest_parameters)
        if strategy.latest_parameters is not None
        else initial_parameters
    )
    final_model = create_model(config)
    set_head_parameters(final_model, final_parameters)
    global_metrics = evaluate_model(final_model, pooled.val_x, pooled.val_y)
    per_client = _evaluate_per_client(config, clients, final_parameters)
    duration = time.perf_counter() - start_time

    return {
        "mode": "federated",
        "partition": config.partition,
        "num_clients": config.num_clients,
        "rounds": config.num_rounds,
        "global": global_metrics,
        "per_client": per_client,
        "training_time_sec": duration,
        "update_size_bytes": update_bytes,
        "communication_cost_bytes": update_bytes * config.num_clients * config.num_rounds * 2,
        "flower_history": _history_to_dict(history),
    }


def _server_evaluate_fn(config: DemoConfig, val_x, val_y):
    def evaluate(server_round: int, parameters: NDArrays, run_config: dict[str, Any]):
        model = create_model(config)
        set_head_parameters(model, parameters)
        metrics = evaluate_model(model, val_x, val_y)
        return float(metrics["loss"]), {"accuracy": float(metrics["accuracy"])}

    return evaluate


def _evaluate_per_client(
    config: DemoConfig, clients: list[ClientDataset], parameters: NDArrays
) -> list[dict[str, Any]]:
    records = []
    for client in clients:
        model = create_model(config)
        set_head_parameters(model, parameters)
        metrics = evaluate_model(model, client.val_x, client.val_y)
        records.append(
            {
                "client_id": client.client_id,
                "num_examples": int(client.val_y.numel()),
                "loss": float(metrics["loss"]),
                "accuracy": float(metrics["accuracy"]),
                "label_histogram": client.label_histogram,
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
    return {
        "accuracy": sum(
            num_examples * float(metrics["accuracy"]) for num_examples, metrics in records
        )
        / total
    }


def _history_to_dict(history) -> dict[str, Any]:
    return {
        "losses_centralized": history.losses_centralized,
        "metrics_centralized": history.metrics_centralized,
        "losses_distributed": history.losses_distributed,
        "metrics_distributed": history.metrics_distributed,
    }
