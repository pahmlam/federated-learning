"""Image-backed embedding precompute helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.real_data import (
    EmbeddingArtifact,
    ManifestRecord,
    validate_embedding_artifact,
    validate_manifest,
)


RESNET18_EMBEDDING_DIM = 512


class _ManifestImageDataset(Dataset):
    def __init__(self, records: Sequence[ManifestRecord], transform: Any) -> None:
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> torch.Tensor:
        record = self.records[index]
        image_path = Path(record.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image_module, _ = _import_image_backend_dependencies()
        with image_module.open(image_path) as image:
            image = image.convert("RGB")
            return self.transform(image)


def create_torchvision_resnet18_embedding_artifact(
    records: Sequence[ManifestRecord],
    *,
    weights: str,
    batch_size: int,
    num_workers: int,
    device: str,
) -> EmbeddingArtifact:
    """Create embeddings by running a frozen torchvision ResNet18 over images."""

    validate_manifest(records)
    if weights not in {"imagenet", "none"}:
        raise ValueError("weights must be 'imagenet' or 'none'")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")

    _, models = _import_image_backend_dependencies()
    torch_device = _resolve_torch_device(device)

    transform = models.ResNet18_Weights.DEFAULT.transforms()
    weights_obj = models.ResNet18_Weights.DEFAULT if weights == "imagenet" else None
    model = models.resnet18(weights=weights_obj)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(torch_device)
    feature_extractor.eval()
    for parameter in feature_extractor.parameters():
        parameter.requires_grad = False

    dataset = _ManifestImageDataset(records, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    feature_batches: list[np.ndarray] = []
    with torch.no_grad():
        for images in loader:
            embeddings = feature_extractor(images.to(torch_device))
            embeddings = torch.flatten(embeddings, start_dim=1)
            feature_batches.append(embeddings.cpu().numpy().astype(np.float32, copy=False))

    features = np.concatenate(feature_batches, axis=0).astype(np.float32, copy=False)
    artifact = EmbeddingArtifact(
        features=features,
        labels=np.asarray([record.label for record in records]),
        client_ids=np.asarray([record.client_id for record in records]),
        splits=np.asarray([record.split for record in records]),
        sample_ids=np.asarray([record.sample_id for record in records]),
    )
    validate_embedding_artifact(artifact)
    return artifact


def _resolve_torch_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda is not available")
    if device == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS device requested but torch.backends.mps is not available")
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: cpu, cuda, mps")
    return torch.device(device)


def _import_image_backend_dependencies() -> tuple[Any, Any]:
    try:
        from PIL import Image
        from torchvision import models
    except Exception as exc:  # pragma: no cover - depends on optional install state
        raise ImportError(
            "The torchvision-resnet18 backend requires Pillow and torchvision. "
            "Install project dependencies before using this backend."
        ) from exc
    return Image, models
