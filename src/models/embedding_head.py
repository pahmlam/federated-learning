"""Head-only classifier for precomputed embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from src.data.embedding import EmbeddingDatasetBundle
    from src.utils.config import DemoConfig


class EmbeddingHeadClassifier(nn.Module):
    """A trainable classifier head over precomputed embedding vectors.

    Two capacity knobs (default off keep EXP-001..008 behavior identical):
    - ``normalize_input``: L2-normalize embeddings before the head (cosine-style).
    - ``hidden_dim``: when set, use a 2-layer MLP head instead of a linear head.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        seed: int,
        *,
        normalize_input: bool = False,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.normalize_input = normalize_input
        if hidden_dim is None:
            self.head: nn.Module = nn.Linear(embedding_dim, num_classes)
        else:
            self.head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            inputs = F.normalize(inputs, p=2.0, dim=1)
        return self.head(inputs)


def create_embedding_head_model(
    embedding_dim: int,
    num_classes: int,
    seed: int,
    *,
    normalize_input: bool = False,
    hidden_dim: int | None = None,
) -> EmbeddingHeadClassifier:
    return EmbeddingHeadClassifier(
        embedding_dim,
        num_classes,
        seed,
        normalize_input=normalize_input,
        hidden_dim=hidden_dim,
    )


def build_embedding_model(
    config: "DemoConfig",
    bundle: "EmbeddingDatasetBundle",
) -> EmbeddingHeadClassifier:
    """Config-aware factory used by every baseline and the FL client.

    Centralizing model construction keeps capacity knobs consistent across
    centralized / local-only / federated modes (single source of truth).
    """

    return create_embedding_head_model(
        embedding_dim=bundle.embedding_dim,
        num_classes=bundle.num_classes,
        seed=config.seed,
        normalize_input=config.normalize_embedding,
        hidden_dim=config.head_hidden_dim,
    )


def get_embedding_head_parameters(model: EmbeddingHeadClassifier) -> list[np.ndarray]:
    return [param.detach().cpu().numpy().copy() for param in model.head.parameters()]


def set_embedding_head_parameters(
    model: EmbeddingHeadClassifier,
    parameters: list[np.ndarray],
) -> None:
    head_params = list(model.head.parameters())
    if len(parameters) != len(head_params):
        raise ValueError(
            f"Expected {len(head_params)} head parameter arrays, got {len(parameters)}"
        )
    for param, array in zip(head_params, parameters, strict=True):
        param.data = torch.tensor(array, dtype=param.dtype, device=param.device)


def embedding_trainable_parameter_names(
    model: EmbeddingHeadClassifier,
) -> list[str]:
    return [name for name, param in model.named_parameters() if param.requires_grad]
