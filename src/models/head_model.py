"""Frozen backbone classifier with a trainable head."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.utils.config import DemoConfig


class FrozenBackboneClassifier(nn.Module):
    """Small model that mimics frozen-backbone, train-head-only training."""

    def __init__(self, config: DemoConfig) -> None:
        super().__init__()
        torch.manual_seed(config.seed)
        self.backbone = nn.Sequential(
            nn.Linear(config.input_dim, config.embedding_dim),
            nn.ReLU(),
        )
        torch.manual_seed(config.seed + 1)
        self.head = nn.Linear(config.embedding_dim, config.num_classes)

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embeddings = self.backbone(inputs)
        return self.head(embeddings)


def create_model(config: DemoConfig) -> FrozenBackboneClassifier:
    return FrozenBackboneClassifier(config)


def get_head_parameters(model: FrozenBackboneClassifier) -> list[np.ndarray]:
    return [
        model.head.weight.detach().cpu().numpy().copy(),
        model.head.bias.detach().cpu().numpy().copy(),
    ]


def set_head_parameters(
    model: FrozenBackboneClassifier, parameters: list[np.ndarray]
) -> None:
    if len(parameters) != 2:
        raise ValueError(f"Expected 2 head parameter arrays, got {len(parameters)}")
    weight, bias = parameters
    model.head.weight.data = torch.tensor(
        weight, dtype=model.head.weight.dtype, device=model.head.weight.device
    )
    model.head.bias.data = torch.tensor(
        bias, dtype=model.head.bias.dtype, device=model.head.bias.device
    )


def trainable_parameter_names(model: FrozenBackboneClassifier) -> list[str]:
    return [name for name, param in model.named_parameters() if param.requires_grad]
