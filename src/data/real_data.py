"""Real-data manifest and embedding-artifact helpers.

These helpers intentionally work on metadata and precomputed vectors only. They
do not decode images, so they are safe to use before wiring a real image
pipeline.
"""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REQUIRED_MANIFEST_COLUMNS = (
    "sample_id",
    "image_path",
    "label",
    "client_id",
    "split",
)
VALID_SPLITS = {"train", "val", "test"}
REQUIRED_EMBEDDING_ARRAYS = (
    "features",
    "labels",
    "client_ids",
    "splits",
    "sample_ids",
)


@dataclass(frozen=True)
class ManifestRecord:
    """One metadata row for a future lazy image dataset."""

    sample_id: str
    image_path: str
    label: str
    client_id: str
    split: str


@dataclass(frozen=True)
class EmbeddingArtifact:
    """Precomputed vectors ready for lightweight head-only training."""

    features: np.ndarray
    labels: np.ndarray
    client_ids: np.ndarray
    splits: np.ndarray
    sample_ids: np.ndarray

    @property
    def num_samples(self) -> int:
        return int(self.features.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.features.shape[1])

    @property
    def label_mapping(self) -> dict[str, int]:
        return build_label_mapping(self.labels)


def load_manifest(
    path: str | Path,
    root_dir: str | Path | None = None,
    require_files: bool = False,
) -> list[ManifestRecord]:
    """Load manifest metadata without opening image files."""

    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [col for col in REQUIRED_MANIFEST_COLUMNS if col not in fieldnames]
        if missing:
            raise ValueError(f"Manifest is missing required columns: {missing}")

        records: list[ManifestRecord] = []
        for row_number, row in enumerate(reader, start=2):
            values = {
                col: (row.get(col) or "").strip()
                for col in REQUIRED_MANIFEST_COLUMNS
            }
            blank = [col for col, value in values.items() if not value]
            if blank:
                raise ValueError(f"Manifest row {row_number} has blank fields: {blank}")

            split = values["split"].lower()
            if split not in VALID_SPLITS:
                raise ValueError(
                    f"Manifest row {row_number} has invalid split '{values['split']}'"
                )

            image_path = _resolve_image_path(values["image_path"], root_dir)
            if require_files and not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            records.append(
                ManifestRecord(
                    sample_id=values["sample_id"],
                    image_path=str(image_path),
                    label=values["label"],
                    client_id=values["client_id"],
                    split=split,
                )
            )

    validate_manifest(records)
    return records


def validate_manifest(records: Sequence[ManifestRecord]) -> None:
    """Validate manifest records and raise ValueError on schema issues."""

    if not records:
        raise ValueError("Manifest must contain at least one row")

    seen_sample_ids: set[str] = set()
    for index, record in enumerate(records):
        for field in REQUIRED_MANIFEST_COLUMNS:
            value = getattr(record, field)
            if not str(value).strip():
                raise ValueError(f"Manifest record {index} has blank field '{field}'")
        if record.split not in VALID_SPLITS:
            raise ValueError(f"Manifest record {index} has invalid split '{record.split}'")
        if record.sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate sample_id in manifest: {record.sample_id}")
        seen_sample_ids.add(record.sample_id)


def group_manifest_by_client(
    records: Iterable[ManifestRecord],
) -> dict[str, list[ManifestRecord]]:
    """Group manifest records by site/client identifier."""

    grouped: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        grouped[record.client_id].append(record)
    return dict(grouped)


def load_embedding_artifact(path: str | Path) -> EmbeddingArtifact:
    """Load a precomputed embedding artifact from an NPZ file."""

    with np.load(Path(path), allow_pickle=False) as data:
        missing = [name for name in REQUIRED_EMBEDDING_ARRAYS if name not in data.files]
        if missing:
            raise ValueError(f"Embedding artifact is missing arrays: {missing}")
        artifact = EmbeddingArtifact(
            features=np.asarray(data["features"]),
            labels=np.asarray(data["labels"]),
            client_ids=np.asarray(data["client_ids"]).astype(str),
            splits=np.asarray(data["splits"]).astype(str),
            sample_ids=np.asarray(data["sample_ids"]).astype(str),
        )
    validate_embedding_artifact(artifact)
    return artifact


def create_synthetic_embedding_artifact(
    records: Sequence[ManifestRecord],
    embedding_dim: int,
    seed: int,
) -> EmbeddingArtifact:
    """Create deterministic synthetic embeddings from manifest metadata."""

    validate_manifest(records)
    if embedding_dim < 1:
        raise ValueError("embedding_dim must be >= 1")

    features = np.stack(
        [
            _synthetic_feature_vector(record, embedding_dim=embedding_dim, seed=seed)
            for record in records
        ],
        axis=0,
    ).astype(np.float32)
    artifact = EmbeddingArtifact(
        features=features,
        labels=np.asarray([record.label for record in records]),
        client_ids=np.asarray([record.client_id for record in records]),
        splits=np.asarray([record.split for record in records]),
        sample_ids=np.asarray([record.sample_id for record in records]),
    )
    validate_embedding_artifact(artifact)
    return artifact


def save_embedding_artifact(path: str | Path, artifact: EmbeddingArtifact) -> None:
    """Save an embedding artifact as an NPZ file after validation."""

    validate_embedding_artifact(artifact)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        features=artifact.features.astype(np.float32, copy=False),
        labels=artifact.labels.astype(str),
        client_ids=artifact.client_ids.astype(str),
        splits=artifact.splits.astype(str),
        sample_ids=artifact.sample_ids.astype(str),
    )


def validate_embedding_artifact(artifact: EmbeddingArtifact) -> None:
    """Validate precomputed embedding arrays."""

    features = artifact.features
    if features.ndim != 2:
        raise ValueError("features must be a 2D array shaped [num_samples, dim]")
    if features.shape[0] == 0:
        raise ValueError("features must contain at least one sample")

    expected = features.shape[0]
    arrays: dict[str, np.ndarray] = {
        "labels": artifact.labels,
        "client_ids": artifact.client_ids,
        "splits": artifact.splits,
        "sample_ids": artifact.sample_ids,
    }
    for name, array in arrays.items():
        if array.ndim != 1:
            raise ValueError(f"{name} must be a 1D array")
        if array.shape[0] != expected:
            raise ValueError(
                f"{name} length {array.shape[0]} does not match features length {expected}"
            )

    splits = {str(split) for split in artifact.splits.tolist()}
    invalid_splits = sorted(splits - VALID_SPLITS)
    if invalid_splits:
        raise ValueError(f"Embedding artifact has invalid splits: {invalid_splits}")
    if "train" not in splits or "val" not in splits:
        raise ValueError("Embedding artifact must contain at least one train and val row")

    if len(set(artifact.sample_ids.astype(str).tolist())) != expected:
        raise ValueError("Embedding artifact sample_ids must be unique")
    sample_ids = [str(sample_id).strip() for sample_id in artifact.sample_ids.tolist()]
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError("Embedding artifact sample_ids must not contain blank values")
    client_ids = [str(client_id).strip() for client_id in artifact.client_ids.tolist()]
    if any(not client_id for client_id in client_ids):
        raise ValueError("Embedding artifact client_ids must not contain blank values")
    if not set(client_ids):
        raise ValueError("Embedding artifact must contain at least one client_id")
    build_label_mapping(artifact.labels)


def build_label_mapping(labels: np.ndarray) -> dict[str, int]:
    """Build a deterministic label-to-id mapping from labels."""

    flat = np.asarray(labels)
    if flat.ndim != 1 or flat.shape[0] == 0:
        raise ValueError("labels must be a non-empty 1D array")

    normalized = [str(label).strip() for label in flat.tolist()]
    if any(not label for label in normalized):
        raise ValueError("labels must not contain blank values")
    return {label: index for index, label in enumerate(sorted(set(normalized)))}


def _resolve_image_path(image_path: str, root_dir: str | Path | None) -> Path:
    path = Path(image_path)
    if root_dir is not None and not path.is_absolute():
        return Path(root_dir) / path
    return path


def _synthetic_feature_vector(
    record: ManifestRecord,
    embedding_dim: int,
    seed: int,
) -> np.ndarray:
    key = "|".join(
        [
            str(seed),
            record.sample_id,
            record.label,
            record.client_id,
            record.split,
        ]
    )
    values: list[float] = []
    counter = 0
    while len(values) < embedding_dim:
        digest = hashlib.sha256(f"{key}|{counter}".encode("utf-8")).digest()
        values.extend(byte / 255.0 for byte in digest)
        counter += 1
    vector = np.asarray(values[:embedding_dim], dtype=np.float32)
    return (vector - 0.5) * 2.0
