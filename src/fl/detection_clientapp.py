"""Flower ClientApp for PPE detection head-only federated training.

The Flower glue (context -> config/data, message record assembly) lives here; the
detection compute (build model, get/set head arrays, train, evaluate) is delegated
to ``DetectionTask`` so the same ClientApp shape can serve other workloads. The
context-resolution helpers are re-exported from ``detection_task`` for backward
compatibility with existing imports.
"""

from __future__ import annotations

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from src.fl.detection_task import (
    DetectionTask,
    detection_config_from_context,
    load_detection_client_context,
    select_detection_client,
)

__all__ = [
    "app",
    "detection_config_from_context",
    "load_detection_client_context",
    "select_detection_client",
]

app = ClientApp()

_TASK = DetectionTask()


@app.train()
def train(msg: Message, context: Context) -> Message:
    ctx = _TASK.load_client_context(context)
    model = _TASK.build_model(ctx.config, num_classes=ctx.bundle.num_classes)
    _TASK.set_global_arrays(model, msg.content["arrays"].to_numpy_ndarrays())
    out = _TASK.train_round(model, ctx)

    content = RecordDict(
        {
            "arrays": ArrayRecord(_TASK.get_global_arrays(model)),
            "metrics": MetricRecord({"num-examples": out.num_examples, **out.metrics}),
        }
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    ctx = _TASK.load_client_context(context)
    model = _TASK.build_model(ctx.config, num_classes=ctx.bundle.num_classes)
    _TASK.set_global_arrays(model, msg.content["arrays"].to_numpy_ndarrays())
    out = _TASK.evaluate_round(model, ctx)

    return Message(
        content=RecordDict(
            {"metrics": MetricRecord({"num-examples": out.num_examples, **out.metrics})}
        ),
        reply_to=msg,
    )
