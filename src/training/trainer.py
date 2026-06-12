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
) -> dict[str, float]:
    model.train()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
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
    model: nn.Module, features: torch.Tensor, labels: torch.Tensor
) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        logits = model(features)
        loss = criterion(logits, labels)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean()
    return {"loss": float(loss.item()), "accuracy": float(accuracy.item())}
