"""Head-only classifier for precomputed embeddings."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class EmbeddingHeadClassifier(nn.Module):
    """A direct classifier head over precomputed embedding vectors."""

    def __init__(self, embedding_dim: int, num_classes: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(inputs)


def create_embedding_head_model(
    embedding_dim: int,
    num_classes: int,
    seed: int,
) -> EmbeddingHeadClassifier:
    return EmbeddingHeadClassifier(embedding_dim, num_classes, seed)


def get_embedding_head_parameters(model: EmbeddingHeadClassifier) -> list[np.ndarray]:
    return [
        model.head.weight.detach().cpu().numpy().copy(),
        model.head.bias.detach().cpu().numpy().copy(),
    ]


def set_embedding_head_parameters(
    model: EmbeddingHeadClassifier,
    parameters: list[np.ndarray],
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


def embedding_trainable_parameter_names(
    model: EmbeddingHeadClassifier,
) -> list[str]:
    return [name for name, param in model.named_parameters() if param.requires_grad]
