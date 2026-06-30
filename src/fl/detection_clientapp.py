"""Flower ClientApp for PPE detection head-only federated training."""

from __future__ import annotations

from dataclasses import replace
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from src.data.detection_data import (
    DetectionClientData,
    DetectionDatasetBundle,
    load_detection_bundle,
)
from src.models.detection_model import (
    build_detection_model,
    get_detection_head_parameters,
    resolve_device,
    set_detection_head_parameters,
)
from src.training.detection_trainer import evaluate_detection, train_detection_head
from src.utils.detection_config import DetectionConfig

app = ClientApp()

_REPORT_KEYS = ("map", "map_50", "map_75")


@app.train()
def train(msg: Message, context: Context) -> Message:
    config, bundle, client = load_detection_client_context(context)
    device = resolve_device(config.device)

    model = build_detection_model(
        num_classes=bundle.num_classes,
        pretrained=config.pretrained,
        seed=config.seed,
    )
    set_detection_head_parameters(model, msg.content["arrays"].to_numpy_ndarrays())
    metrics = train_detection_head(
        model,
        client.train,
        epochs=config.local_epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        device=device,
        num_workers=config.num_workers,
        seed=config.seed + client.client_id,
    )

    content = RecordDict(
        {
            "arrays": ArrayRecord(get_detection_head_parameters(model)),
            "metrics": MetricRecord(
                {
                    "num-examples": len(client.train),
                    "train_loss": float(metrics["train_loss"]),
                }
            ),
        }
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    config, bundle, client = load_detection_client_context(context)
    device = resolve_device(config.device)

    model = build_detection_model(
        num_classes=bundle.num_classes,
        pretrained=config.pretrained,
        seed=config.seed,
    )
    set_detection_head_parameters(model, msg.content["arrays"].to_numpy_ndarrays())
    metrics = evaluate_detection(
        model,
        client.val,
        batch_size=config.batch_size,
        device=device,
        num_workers=config.num_workers,
        score_threshold=config.score_threshold,
    )

    record = {
        "num-examples": len(client.val),
        **{key: float(metrics.get(key, -1.0)) for key in _REPORT_KEYS},
    }
    return Message(
        content=RecordDict({"metrics": MetricRecord(record)}),
        reply_to=msg,
    )


def load_detection_client_context(
    context: Context,
) -> tuple[DetectionConfig, DetectionDatasetBundle, DetectionClientData]:
    """Load config, bundle, and selected client for a Flower node."""

    config = detection_config_from_context(context)
    bundle = load_detection_bundle(
        config.manifest_path,
        config.root_dir,
        image_size=config.image_size,
    )
    return config, bundle, select_detection_client(context, bundle, client_id=config.client_id)


def detection_config_from_context(context: Context) -> DetectionConfig:
    """Build detection config, letting node_config override data location."""

    config = DetectionConfig.from_env_and_overrides(
        dict(context.run_config),
        env_overrides=True,
    )
    node_values = {}
    for source, target in (
        ("manifest-path", "manifest-path"),
        ("manifest_path", "manifest-path"),
        ("root-dir", "root-dir"),
        ("root_dir", "root-dir"),
        ("data-root", "root-dir"),
        ("data_root", "root-dir"),
        ("client-id", "client-id"),
        ("client_id", "client-id"),
    ):
        if source in context.node_config:
            node_values[target.replace("-", "_")] = context.node_config[source]
    if node_values:
        config = replace(config, **node_values).normalized()
    return replace(config, num_clients=max(1, int(config.num_clients))).normalized()


def select_detection_client(
    context: Context,
    bundle: DetectionDatasetBundle,
    *,
    client_id: str | None = None,
) -> DetectionClientData:
    """Choose a site shard for this node.

    A deployed client usually receives a one-site manifest and therefore selects
    the sole client. Local simulation uses ``partition-id`` modulo client count.
    ``client-id`` is supported for explicit site selection in notebooks or shell
    commands.
    """

    if len(bundle.clients) == 1:
        return bundle.clients[0]

    explicit = context.node_config.get(
        "client-id",
        context.node_config.get("client_id", client_id),
    )
    if explicit is not None:
        explicit_text = str(explicit)
        for client in bundle.clients:
            if explicit_text in {client.client_label, str(client.client_id)}:
                return client
        raise ValueError(f"Unknown detection client-id: {explicit_text}")

    partition = context.node_config.get(
        "partition-id",
        context.node_config.get("partition_id", context.node_id),
    )
    return bundle.clients[int(partition) % len(bundle.clients)]
