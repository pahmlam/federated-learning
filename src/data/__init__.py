"""Data generation, manifest, and client partition helpers."""

from src.data.image_embeddings import (
    RESNET18_EMBEDDING_DIM,
    create_torchvision_resnet18_embedding_artifact,
)
from src.data.real_data import (
    EmbeddingArtifact,
    ManifestRecord,
    create_synthetic_embedding_artifact,
    group_manifest_by_client,
    load_embedding_artifact,
    load_manifest,
    save_embedding_artifact,
    validate_embedding_artifact,
    validate_manifest,
)

__all__ = [
    "EmbeddingArtifact",
    "ManifestRecord",
    "RESNET18_EMBEDDING_DIM",
    "create_synthetic_embedding_artifact",
    "create_torchvision_resnet18_embedding_artifact",
    "group_manifest_by_client",
    "load_embedding_artifact",
    "load_manifest",
    "save_embedding_artifact",
    "validate_embedding_artifact",
    "validate_manifest",
]
