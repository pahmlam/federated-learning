"""Flower ClientApp for the synthetic head-only demo."""

from __future__ import annotations

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from src.data.synthetic import make_client_splits
from src.models.head_model import create_model, set_head_parameters, get_head_parameters
from src.training.trainer import evaluate_model, train_head
from src.utils.config import DemoConfig

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    config = DemoConfig.from_run_config(context.run_config)
    client = make_client_splits(config)[_partition_id(context, config.num_clients)]

    model = create_model(config)
    set_head_parameters(model, msg.content["arrays"].to_numpy_ndarrays())
    train_metrics = train_head(
        model=model,
        train_x=client.train_x,
        train_y=client.train_y,
        epochs=config.local_epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        seed=config.seed + client.client_id,
    )

    content = RecordDict(
        {
            "arrays": ArrayRecord(get_head_parameters(model)),
            "metrics": MetricRecord(
                {
                    "num-examples": int(client.train_y.numel()),
                    "train_loss": float(train_metrics["loss"]),
                    "train_accuracy": float(train_metrics["accuracy"]),
                }
            ),
        }
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    config = DemoConfig.from_run_config(context.run_config)
    client = make_client_splits(config)[_partition_id(context, config.num_clients)]

    model = create_model(config)
    set_head_parameters(model, msg.content["arrays"].to_numpy_ndarrays())
    metrics = evaluate_model(model, client.val_x, client.val_y)

    content = RecordDict(
        {
            "metrics": MetricRecord(
                {
                    "num-examples": int(client.val_y.numel()),
                    "loss": float(metrics["loss"]),
                    "accuracy": float(metrics["accuracy"]),
                }
            )
        }
    )
    return Message(content=content, reply_to=msg)


def _partition_id(context: Context, num_clients: int) -> int:
    partition = context.node_config.get("partition-id", context.node_id)
    return int(partition) % num_clients
