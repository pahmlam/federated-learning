"""PyTorch train/evaluate utilities for head-only demo models."""

from __future__ import annotations

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
) -> dict[str, float]:
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
) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        logits = model(features)
        loss = criterion(logits, labels)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean()
    metrics = {
        "loss": float(loss.item()),
        "accuracy": float(accuracy.item()),
        "macro_f1": _macro_f1(predictions, labels, num_classes=int(logits.shape[1])),
    }
    if positive_class_id is not None:
        recall = _recall_for_class(predictions, labels, positive_class_id)
        metrics["unsafe_recall"] = recall
        metrics["false_negative_rate"] = 1.0 - recall
    return metrics


def _macro_f1(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> float:
    scores = [
        _f1_for_class(predictions, labels, class_id)
        for class_id in range(num_classes)
    ]
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def _f1_for_class(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    class_id: int,
) -> float:
    predicted = predictions == class_id
    actual = labels == class_id
    tp = int((predicted & actual).sum().item())
    fp = int((predicted & ~actual).sum().item())
    fn = int((~predicted & actual).sum().item())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0.0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _recall_for_class(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    class_id: int,
) -> float:
    actual = labels == class_id
    tp = int(((predictions == class_id) & actual).sum().item())
    fn = int(((predictions != class_id) & actual).sum().item())
    if tp + fn == 0:
        return 0.0
    return float(tp / (tp + fn))
