"""``FederatedTask`` implementation for the PPE detection workload.

Wraps the existing detection data/model/trainer functions behind the
``FederatedTask`` interface so the Flower ClientApp/ServerApp drive detection
through the same seam any future workload would use. This module holds no new
training logic -- it composes ``src/models/detection_model.py``,
``src/training/detection_trainer.py`` and ``src/data/detection_data.py``.

The Flower-context resolution helpers (``detection_config_from_context``,
``select_detection_client``, ``load_detection_client_context``) live here and are
re-exported from ``detection_clientapp`` for backward compatibility.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np
from flwr.app import Context
from torch.utils.data import Subset

from src.data.detection_data import (
    DetectionClientData,
    DetectionDatasetBundle,
    load_detection_bundle,
)
from src.data.detection_dataset import PPEDetectionDataset
from src.evaluation.metrics import parameter_bytes
from src.fl.edge_profile import (
    EdgeDecision,
    EdgeProfile,
    apply_profile_overrides,
    edge_decision,
    profile_metrics,
    resolve_edge_profile,
)
from src.fl.task import RoundOutput
from src.models.detection_model import (
    build_detection_model,
    get_detection_head_parameters,
    resolve_device,
    set_detection_head_parameters,
)
from src.training.detection_trainer import evaluate_detection, train_detection_head
from src.utils.detection_config import DetectionConfig

# Evaluate metrics reported back to the server (weighted-aggregated by Flower).
_REPORT_KEYS = ("map", "map_50", "map_75")


@dataclass(frozen=True)
class DetectionTaskContext:
    """Per-node detection context: resolved config + loaded bundle + selected site."""

    config: DetectionConfig
    bundle: DetectionDatasetBundle
    client: DetectionClientData
    edge_profile: EdgeProfile | None = None
    server_round: int = 1
    update_size_bytes: int = 0


class DetectionTask:
    """PPE detection as a ``FederatedTask`` (thin wrappers over existing funcs)."""

    def load_client_context(self, context: Context) -> DetectionTaskContext:
        config = detection_config_from_context(context)
        global_profile = (
            None
            if config.edge_profiles
            else resolve_edge_profile(
                edge_profile=config.edge_profile,
                edge_profiles=None,
                client_label="default",
                client_id=0,
            )
        )
        load_config = apply_profile_overrides(config, global_profile)
        bundle = load_detection_bundle(
            load_config.manifest_path,
            load_config.root_dir,
            image_size=load_config.image_size,
        )
        client = select_detection_client(context, bundle, client_id=config.client_id)
        profile = resolve_edge_profile(
            edge_profile=config.edge_profile,
            edge_profiles=config.edge_profiles,
            client_label=client.client_label,
            client_id=client.client_id,
        )
        effective_config = apply_profile_overrides(config, profile)
        if effective_config.image_size != bundle.image_size:
            client = _client_with_image_size(client, effective_config.image_size)
        return DetectionTaskContext(
            config=effective_config,
            bundle=bundle,
            client=client,
            edge_profile=profile,
        )

    def build_model(self, config: DetectionConfig, *, num_classes: int):
        return build_detection_model(
            num_classes=num_classes,
            pretrained=config.pretrained,
            seed=config.seed,
        )

    def get_global_arrays(self, model) -> list[np.ndarray]:
        return get_detection_head_parameters(model)

    def set_global_arrays(self, model, arrays: list[np.ndarray]) -> None:
        set_detection_head_parameters(model, arrays)

    def train_round(self, model, ctx: DetectionTaskContext) -> RoundOutput:
        config, client = ctx.config, ctx.client
        decision = _ensure_edge_client_can_run(ctx, stage="train")
        train_data = _train_dataset(client, ctx.edge_profile)
        started = time.perf_counter()
        if ctx.edge_profile and ctx.edge_profile.artificial_train_delay_sec > 0:
            time.sleep(ctx.edge_profile.artificial_train_delay_sec)
        metrics = train_detection_head(
            model,
            train_data,
            epochs=config.local_epochs,
            batch_size=config.batch_size,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            device=resolve_device(config.device),
            num_workers=config.num_workers,
            seed=config.seed + client.client_id,
        )
        elapsed = time.perf_counter() - started
        output_metrics = {"train_loss": float(metrics["train_loss"])}
        output_metrics.update(
            _edge_metrics(ctx, decision, effective_time_key="effective_train_time", elapsed=elapsed)
        )
        return RoundOutput(
            num_examples=len(train_data),
            metrics=output_metrics,
        )

    def evaluate_round(self, model, ctx: DetectionTaskContext) -> RoundOutput:
        config, client = ctx.config, ctx.client
        decision = _ensure_edge_client_can_run(ctx, stage="evaluate")
        started = time.perf_counter()
        metrics = evaluate_detection(
            model,
            client.val,
            batch_size=config.batch_size,
            device=resolve_device(config.device),
            num_workers=config.num_workers,
            score_threshold=config.score_threshold,
        )
        elapsed = time.perf_counter() - started
        output_metrics = {key: float(metrics.get(key, -1.0)) for key in _REPORT_KEYS}
        output_metrics.update(
            _edge_metrics(
                ctx,
                decision,
                effective_time_key="effective_eval_time",
                elapsed=elapsed,
            )
        )
        return RoundOutput(
            num_examples=len(client.val),
            metrics=output_metrics,
        )


def load_detection_client_context(
    context: Context,
) -> tuple[DetectionConfig, DetectionDatasetBundle, DetectionClientData]:
    """Load config, bundle, and selected client for a Flower node.

    Retained (and re-exported from ``detection_clientapp``) for backward
    compatibility; ``DetectionTask.load_client_context`` is the preferred entry.
    """

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


def context_with_transfer_size(
    ctx: DetectionTaskContext,
    arrays: list[np.ndarray],
) -> DetectionTaskContext:
    """Attach head-transfer size for EdgeProfile communication metrics."""

    return replace(ctx, update_size_bytes=parameter_bytes(arrays))


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


def _client_with_image_size(
    client: DetectionClientData,
    image_size: int,
) -> DetectionClientData:
    return replace(
        client,
        train=PPEDetectionDataset(client.train.records, image_size=image_size),
        val=PPEDetectionDataset(client.val.records, image_size=image_size),
    )


def _train_dataset(client: DetectionClientData, profile: EdgeProfile | None):
    if profile is None or profile.max_train_samples is None:
        return client.train
    limit = min(profile.max_train_samples, len(client.train))
    if limit == len(client.train):
        return client.train
    return Subset(client.train, range(limit))


def _ensure_edge_client_can_run(
    ctx: DetectionTaskContext,
    *,
    stage: str,
) -> EdgeDecision:
    decision = edge_decision(
        ctx.edge_profile,
        seed=ctx.config.seed,
        client_id=ctx.client.client_id,
        round_number=ctx.server_round,
        stage=stage,
    )
    if not decision.should_run:
        profile_name = "none" if ctx.edge_profile is None else ctx.edge_profile.tier
        raise RuntimeError(
            f"EdgeProfile blocked client {ctx.client.client_label} "
            f"at {stage}: {decision.reason} (profile={profile_name})"
        )
    return decision


def _edge_metrics(
    ctx: DetectionTaskContext,
    decision: EdgeDecision,
    *,
    effective_time_key: str,
    elapsed: float,
) -> dict[str, float]:
    metrics = profile_metrics(
        ctx.edge_profile,
        decision,
        update_size_bytes=ctx.update_size_bytes,
    )
    if metrics:
        metrics[effective_time_key] = float(elapsed)
    return metrics
