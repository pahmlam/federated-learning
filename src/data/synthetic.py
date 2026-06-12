"""Synthetic camera-like classification data for FL smoke tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.utils.config import DemoConfig


@dataclass(frozen=True)
class ClientDataset:
    """Train/validation tensors for one simulated client/site."""

    client_id: int
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    label_histogram: dict[int, int]


@dataclass(frozen=True)
class PooledDataset:
    """Pooled train/validation tensors used for centralized baseline."""

    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor


def make_client_splits(config: DemoConfig) -> list[ClientDataset]:
    """Create deterministic IID or non-IID synthetic client splits."""

    rng = np.random.default_rng(config.seed)
    total_samples = config.num_clients * config.samples_per_client
    labels = _balanced_labels(total_samples, config.num_classes, rng)
    features = _features_from_labels(labels, config, rng)

    if config.partition == "iid":
        client_indices = _iid_indices(total_samples, config.num_clients, rng)
    elif config.partition == "non_iid":
        client_indices = _dirichlet_label_skew_indices(
            labels=labels,
            num_clients=config.num_clients,
            alpha=config.dirichlet_alpha,
            rng=rng,
        )
    else:
        raise ValueError(f"Unsupported partition: {config.partition}")

    return [
        _build_client_dataset(client_id, indices, features, labels, config, rng)
        for client_id, indices in enumerate(client_indices)
    ]


def make_pooled_dataset(clients: list[ClientDataset]) -> PooledDataset:
    """Concatenate all clients into one centralized dataset."""

    return PooledDataset(
        train_x=torch.cat([client.train_x for client in clients], dim=0),
        train_y=torch.cat([client.train_y for client in clients], dim=0),
        val_x=torch.cat([client.val_x for client in clients], dim=0),
        val_y=torch.cat([client.val_y for client in clients], dim=0),
    )


def _balanced_labels(
    total_samples: int, num_classes: int, rng: np.random.Generator
) -> np.ndarray:
    labels = np.arange(total_samples, dtype=np.int64) % num_classes
    rng.shuffle(labels)
    return labels


def _features_from_labels(
    labels: np.ndarray, config: DemoConfig, rng: np.random.Generator
) -> np.ndarray:
    prototypes = rng.normal(
        loc=0.0,
        scale=2.0,
        size=(config.num_classes, config.input_dim),
    )
    noise = rng.normal(loc=0.0, scale=0.8, size=(labels.size, config.input_dim))
    features = prototypes[labels] + noise
    return features.astype(np.float32)


def _iid_indices(
    total_samples: int, num_clients: int, rng: np.random.Generator
) -> list[np.ndarray]:
    indices = np.arange(total_samples)
    rng.shuffle(indices)
    return [split.astype(np.int64) for split in np.array_split(indices, num_clients)]


def _dirichlet_label_skew_indices(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    per_client: list[list[int]] = [[] for _ in range(num_clients)]

    for class_id in sorted(np.unique(labels).tolist()):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        cut_points = (np.cumsum(proportions)[:-1] * class_indices.size).astype(int)
        for client_id, split in enumerate(np.split(class_indices, cut_points)):
            per_client[client_id].extend(split.tolist())

    _rebalance_empty_clients(per_client, rng)

    result: list[np.ndarray] = []
    for indices in per_client:
        arr = np.asarray(indices, dtype=np.int64)
        rng.shuffle(arr)
        result.append(arr)
    return result


def _rebalance_empty_clients(
    per_client: list[list[int]], rng: np.random.Generator
) -> None:
    for client_id, indices in enumerate(per_client):
        if indices:
            continue
        donor_id = max(range(len(per_client)), key=lambda idx: len(per_client[idx]))
        donor = per_client[donor_id]
        move_count = max(1, len(donor) // 10)
        rng.shuffle(donor)
        per_client[client_id].extend(donor[:move_count])
        del donor[:move_count]


def _build_client_dataset(
    client_id: int,
    indices: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    config: DemoConfig,
    rng: np.random.Generator,
) -> ClientDataset:
    if indices.size < 2:
        raise ValueError(f"Client {client_id} has too few samples: {indices.size}")

    shuffled = indices.copy()
    rng.shuffle(shuffled)
    val_size = max(1, int(round(shuffled.size * config.val_fraction)))
    val_indices = shuffled[:val_size]
    train_indices = shuffled[val_size:]
    if train_indices.size == 0:
        train_indices = val_indices

    client_labels = labels[indices]
    histogram = {
        int(class_id): int((client_labels == class_id).sum())
        for class_id in range(config.num_classes)
    }

    return ClientDataset(
        client_id=client_id,
        train_x=torch.from_numpy(features[train_indices]),
        train_y=torch.from_numpy(labels[train_indices]).long(),
        val_x=torch.from_numpy(features[val_indices]),
        val_y=torch.from_numpy(labels[val_indices]).long(),
        label_histogram=histogram,
    )
