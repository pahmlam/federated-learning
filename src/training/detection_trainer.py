"""Train/evaluate utilities for the frozen-backbone PPE detector.

Trains only the head (backbone is frozen by ``build_detection_model``). Evaluation
uses torchmetrics ``MeanAveragePrecision`` (mAP@0.5, mAP@0.5:0.95, per-class AP).
Device-aware so the same code runs on CPU (Mac smoke) and CUDA (RTX3060 / Colab).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection import MeanAveragePrecision

from src.data.detection_dataset import detection_collate_fn

# torchvision detectors ignore extra target keys, but the loss/metric only need these.
_TARGET_KEYS = ("boxes", "labels")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    """Run one training epoch; return the mean summed-loss over batches."""

    model.train()
    total_loss = 0.0
    num_batches = 0
    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = [_target_to_device(target, device) for target in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        num_batches += 1
    return total_loss / max(1, num_batches)


def train_detection_head(
    model: nn.Module,
    dataset: Dataset,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    device: str,
    num_workers: int = 0,
    seed: int = 2026,
) -> dict[str, Any]:
    """Train the detector head for ``epochs`` and return the last epoch loss."""

    model.to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
    )
    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(
        trainable, lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    last_loss = 0.0
    for _ in range(epochs):
        last_loss = train_one_epoch(model, loader, optimizer, device)
    return {"train_loss": last_loss}


@torch.no_grad()
def evaluate_detection(
    model: nn.Module,
    dataset: Dataset,
    *,
    batch_size: int,
    device: str,
    num_workers: int = 0,
    score_threshold: float = 0.0,
) -> dict[str, Any]:
    """Evaluate the detector and return JSON-ready mAP metrics."""

    model.to(device)
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
    )
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    for images, targets in loader:
        outputs = model([image.to(device) for image in images])
        preds = [_filter_predictions(output, score_threshold) for output in outputs]
        ground_truth = [
            {key: target[key] for key in _TARGET_KEYS} for target in targets
        ]
        metric.update(preds, ground_truth)
    return _metrics_to_json(metric.compute())


def _target_to_device(target: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: (value.to(device) if torch.is_tensor(value) else value)
        for key, value in target.items()
    }


def _filter_predictions(output: dict[str, Any], score_threshold: float) -> dict[str, Any]:
    keep = output["scores"] >= score_threshold
    return {
        "boxes": output["boxes"][keep].cpu(),
        "scores": output["scores"][keep].cpu(),
        "labels": output["labels"][keep].cpu(),
    }


def _metrics_to_json(result: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name, value in result.items():
        if torch.is_tensor(value):
            metrics[name] = float(value.item()) if value.ndim == 0 else value.cpu().tolist()
        else:
            metrics[name] = value
    return metrics
