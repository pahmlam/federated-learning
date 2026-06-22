"""Adapters from embedding artifacts to train/eval client tensors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.data.real_data import EmbeddingArtifact, load_embedding_artifact


@dataclass(frozen=True)
class EmbeddingClientDataset:
    """Train/validation tensors for one client from an embedding artifact."""

    client_id: int
    client_label: str
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    label_histogram: dict[int, int]


@dataclass(frozen=True)
class EmbeddingPooledDataset:
    """Pooled train/validation tensors for centralized embedding baseline."""

    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor


@dataclass(frozen=True)
class EmbeddingDatasetBundle:
    """Embedding artifact converted to client datasets and label ids."""

    clients: list[EmbeddingClientDataset]
    pooled: EmbeddingPooledDataset
    label_mapping: dict[str, int]
    artifact_path: str
    embedding_dim: int
    num_classes: int


def load_embedding_dataset_bundle(path: str) -> EmbeddingDatasetBundle:
    artifact = load_embedding_artifact(path)
    return embedding_artifact_to_bundle(artifact=artifact, artifact_path=path)


def embedding_artifact_to_bundle(
    artifact: EmbeddingArtifact,
    artifact_path: str = "",
) -> EmbeddingDatasetBundle:
    validate_embedding_baseline_splits(artifact)
    label_mapping = artifact.label_mapping
    label_ids = np.asarray(
        [label_mapping[str(label)] for label in artifact.labels.tolist()],
        dtype=np.int64,
    )
    client_labels = sorted(set(artifact.client_ids.astype(str).tolist()))
    clients = [
        _build_client(artifact, label_ids, client_label, client_id, len(label_mapping))
        for client_id, client_label in enumerate(client_labels)
    ]
    pooled = EmbeddingPooledDataset(
        train_x=torch.cat([client.train_x for client in clients], dim=0),
        train_y=torch.cat([client.train_y for client in clients], dim=0),
        val_x=torch.cat([client.val_x for client in clients], dim=0),
        val_y=torch.cat([client.val_y for client in clients], dim=0),
    )
    return EmbeddingDatasetBundle(
        clients=clients,
        pooled=pooled,
        label_mapping=label_mapping,
        artifact_path=artifact_path,
        embedding_dim=artifact.embedding_dim,
        num_classes=len(label_mapping),
    )


def validate_embedding_baseline_splits(artifact: EmbeddingArtifact) -> None:
    """Ensure each client has train and val rows for baseline runs."""

    client_labels = sorted(set(artifact.client_ids.astype(str).tolist()))
    splits = artifact.splits.astype(str)
    client_ids = artifact.client_ids.astype(str)
    for client_label in client_labels:
        client_mask = client_ids == client_label
        missing = [
            split_name
            for split_name in ("train", "val")
            if not (client_mask & (splits == split_name)).any()
        ]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(
                f"Client {client_label} is missing required baseline split(s): "
                f"{missing_text}"
            )


def _build_client(
    artifact: EmbeddingArtifact,
    label_ids: np.ndarray,
    client_label: str,
    client_id: int,
    num_classes: int,
) -> EmbeddingClientDataset:
    client_mask = artifact.client_ids.astype(str) == client_label
    train_mask = client_mask & (artifact.splits.astype(str) == "train")
    val_mask = client_mask & (artifact.splits.astype(str) == "val")
    if not train_mask.any():
        raise ValueError(f"Client {client_label} has no train samples")
    if not val_mask.any():
        raise ValueError(f"Client {client_label} has no val samples")

    client_label_ids = label_ids[client_mask]
    histogram = {
        class_id: int((client_label_ids == class_id).sum())
        for class_id in range(num_classes)
    }
    return EmbeddingClientDataset(
        client_id=client_id,
        client_label=client_label,
        train_x=torch.from_numpy(artifact.features[train_mask].astype(np.float32)),
        train_y=torch.from_numpy(label_ids[train_mask]).long(),
        val_x=torch.from_numpy(artifact.features[val_mask].astype(np.float32)),
        val_y=torch.from_numpy(label_ids[val_mask]).long(),
        label_histogram=histogram,
    )
