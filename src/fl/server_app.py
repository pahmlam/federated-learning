"""Flower ServerApp for the synthetic head-only demo."""

from __future__ import annotations

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from src.data.synthetic import make_client_splits, make_pooled_dataset
from src.models.head_model import create_model, get_head_parameters, set_head_parameters
from src.training.trainer import evaluate_model
from src.utils.config import DemoConfig

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    config = DemoConfig.from_run_config(context.run_config)
    clients = make_client_splits(config)
    pooled = make_pooled_dataset(clients)

    model = create_model(config)
    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=1.0,
        min_train_nodes=config.num_clients,
        min_evaluate_nodes=config.num_clients,
        min_available_nodes=config.num_clients,
        weighted_by_key="num-examples",
    )

    strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(get_head_parameters(model)),
        num_rounds=config.num_rounds,
        train_config=ConfigRecord(
            {
                "local_epochs": config.local_epochs,
                "batch_size": config.batch_size,
                "lr": config.lr,
            }
        ),
        evaluate_fn=lambda server_round, arrays: _evaluate_global(
            config, arrays, pooled.val_x, pooled.val_y
        ),
    )


def _evaluate_global(config: DemoConfig, arrays: ArrayRecord, val_x, val_y):
    model = create_model(config)
    set_head_parameters(model, arrays.to_numpy_ndarrays())
    metrics = evaluate_model(model, val_x, val_y)
    return MetricRecord(
        {
            "num-examples": int(val_y.numel()),
            "loss": float(metrics["loss"]),
            "accuracy": float(metrics["accuracy"]),
        }
    )
