"""Flower ServerApp for PPE detection head-only FedAvg deployment."""

from __future__ import annotations

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from src.models.detection_model import build_detection_model, get_detection_head_parameters
from src.utils.detection_config import DetectionConfig

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    config = DetectionConfig.from_env_and_overrides(
        dict(context.run_config),
        env_overrides=True,
    )
    model = build_detection_model(
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        seed=config.seed,
    )
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
        initial_arrays=ArrayRecord(get_detection_head_parameters(model)),
        num_rounds=config.num_rounds,
        train_config=_round_config(config),
        evaluate_config=_round_config(config),
    )


def _round_config(config: DetectionConfig) -> ConfigRecord:
    return ConfigRecord(
        {
            "local_epochs": config.local_epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "momentum": config.momentum,
            "weight_decay": config.weight_decay,
            "score_threshold": config.score_threshold,
        }
    )
