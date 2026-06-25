"""PyTorch train/evaluate utilities for head-only demo models."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def train_head(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    num_workers: int = 0,
    weight_decay: float = 0.0,
) -> dict[str, Any]:
    model.train()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        [param for param in model.parameters() if param.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    return evaluate_model(model, train_x, train_y)


def evaluate_model(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    positive_class_id: int | None = None,
) -> dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        logits = model(features)
        loss = criterion(logits, labels)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean()
    num_classes = int(logits.shape[1])
    confusion = _confusion_matrix(
        predictions,
        labels,
        num_classes=num_classes,
    )
    metrics = {
        "loss": float(loss.item()),
        "accuracy": float(accuracy.item()),
        "macro_f1": _macro_f1_from_confusion(confusion),
        "confusion_matrix": confusion,
        "per_class": _per_class_metrics(confusion),
    }
    if positive_class_id is not None:
        recall = _recall_for_class_from_confusion(confusion, positive_class_id)
        metrics["unsafe_recall"] = recall
        metrics["false_negative_rate"] = 1.0 - recall
    return metrics


def _confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for actual, predicted in zip(labels.tolist(), predictions.tolist(), strict=True):
        matrix[int(actual)][int(predicted)] += 1
    return matrix


def _per_class_metrics(confusion: list[list[int]]) -> dict[str, dict[str, float]]:
    return {
        str(class_id): _metrics_for_class(confusion, class_id)
        for class_id in range(len(confusion))
    }


def _metrics_for_class(
    confusion: list[list[int]],
    class_id: int,
) -> dict[str, float]:
    tp = confusion[class_id][class_id]
    fp = sum(row[class_id] for index, row in enumerate(confusion) if index != class_id)
    fn = sum(value for index, value in enumerate(confusion[class_id]) if index != class_id)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _macro_f1_from_confusion(confusion: list[list[int]]) -> float:
    scores = [
        _metrics_for_class(confusion, class_id)["f1"]
        for class_id in range(len(confusion))
    ]
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def _recall_for_class_from_confusion(
    confusion: list[list[int]],
    class_id: int,
) -> float:
    tp = confusion[class_id][class_id]
    fn = sum(value for index, value in enumerate(confusion[class_id]) if index != class_id)
    if tp + fn == 0:
        return 0.0
    return float(tp / (tp + fn))
