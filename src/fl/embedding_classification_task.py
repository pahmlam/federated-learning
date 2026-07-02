"""``FederatedTask`` implementation for head-only embedding classification.

A second, lightweight workload proving ``FederatedTask`` is not detection-shaped.
It wraps the existing stage-1 embedding-head stack (``src/models/embedding_head.py``,
``src/training/trainer.py``, ``src/data/embedding.py``) which operates entirely on
in-memory embedding tensors -- no images, GPU, or Flower network -- so it also serves
as the seam a future face/embedding workload would fill.

This module holds no new training logic; it composes existing helpers. It is a
library/seam and is intentionally not wired into the active ClientApp/ServerApp
(PPE detection stays the deployed workload).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from flwr.app import Context

from src.data.embedding import (
    EmbeddingClientDataset,
    EmbeddingDatasetBundle,
    load_embedding_dataset_bundle,
)
from src.fl.task import RoundOutput
from src.models.embedding_head import (
    create_embedding_head_model,
    get_embedding_head_parameters,
    set_embedding_head_parameters,
)
from src.training.trainer import evaluate_model, train_head
from src.utils.config import DemoConfig

_ARTIFACT_PATH_KEYS = ("embedding-artifact-path", "artifact-path")


@dataclass(frozen=True)
class EmbeddingClientContext:
    """Per-node embedding context: resolved config + loaded bundle + selected site."""

    config: DemoConfig
    bundle: EmbeddingDatasetBundle
    client: EmbeddingClientDataset


class EmbeddingClassificationTask:
    """Head-only embedding classification as a ``FederatedTask``."""

    def load_client_context(self, context: Context) -> EmbeddingClientContext:
        config = DemoConfig.from_run_config(dict(context.run_config))
        path = _artifact_path_from_context(context)
        bundle = load_embedding_dataset_bundle(path)
        client = _select_embedding_client(context, bundle, config.num_clients)
        return EmbeddingClientContext(config=config, bundle=bundle, client=client)

    def build_model(self, config: DemoConfig, *, num_classes: int):
        return create_embedding_head_model(
            embedding_dim=config.embedding_dim,
            num_classes=num_classes,
            seed=config.seed,
            normalize_input=config.normalize_embedding,
            hidden_dim=config.head_hidden_dim,
        )

    def get_global_arrays(self, model) -> list[np.ndarray]:
        return get_embedding_head_parameters(model)

    def set_global_arrays(self, model, arrays: list[np.ndarray]) -> None:
        set_embedding_head_parameters(model, arrays)

    def train_round(self, model, ctx: EmbeddingClientContext) -> RoundOutput:
        config, client = ctx.config, ctx.client
        metrics = train_head(
            model,
            client.train_x,
            client.train_y,
            epochs=config.local_epochs,
            batch_size=config.batch_size,
            lr=config.lr,
            seed=config.seed + client.client_id,
            num_workers=config.num_workers,
            weight_decay=config.weight_decay,
        )
        return RoundOutput(
            num_examples=int(client.train_y.numel()),
            metrics={
                "train_loss": float(metrics["loss"]),
                "train_accuracy": float(metrics["accuracy"]),
                "train_macro_f1": float(metrics["macro_f1"]),
            },
        )

    def evaluate_round(self, model, ctx: EmbeddingClientContext) -> RoundOutput:
        client = ctx.client
        metrics = evaluate_model(model, client.val_x, client.val_y)
        return RoundOutput(
            num_examples=int(client.val_y.numel()),
            metrics={
                "loss": float(metrics["loss"]),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
            },
        )


def _artifact_path_from_context(context: Context) -> str:
    """Resolve the embedding artifact path from node_config, then run_config."""

    for source in (context.node_config, context.run_config):
        for key in _ARTIFACT_PATH_KEYS:
            if key in source:
                return str(source[key])
    raise ValueError(
        "Embedding artifact path not found; set one of "
        f"{list(_ARTIFACT_PATH_KEYS)} in node_config or run_config"
    )


def _select_embedding_client(
    context: Context,
    bundle: EmbeddingDatasetBundle,
    num_clients: int,
) -> EmbeddingClientDataset:
    """Choose a site shard: single-client selects itself; else client-id/partition-id."""

    if len(bundle.clients) == 1:
        return bundle.clients[0]

    explicit = context.node_config.get(
        "client-id", context.node_config.get("client_id")
    )
    if explicit is not None:
        explicit_text = str(explicit)
        for client in bundle.clients:
            if explicit_text in {client.client_label, str(client.client_id)}:
                return client
        raise ValueError(f"Unknown embedding client-id: {explicit_text}")

    partition = context.node_config.get(
        "partition-id", context.node_config.get("partition_id", context.node_id)
    )
    return bundle.clients[int(partition) % len(bundle.clients)]
