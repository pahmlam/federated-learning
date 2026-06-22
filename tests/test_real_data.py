import csv
import subprocess
import sys

import numpy as np
import pytest

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
)


def _write_manifest(path, rows, fieldnames=None):
    columns = fieldnames or ["sample_id", "image_path", "label", "client_id", "split"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _image_backend_image_module():
    try:
        from PIL import Image
        from torchvision import models  # noqa: F401
    except Exception as exc:
        pytest.skip(f"image embedding backend dependencies unavailable: {exc}")
    return Image


def test_load_manifest_reads_metadata_without_requiring_image_files(tmp_path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "sample_id": "s1",
                "image_path": "missing/a.jpg",
                "label": "helmet",
                "client_id": "site-a",
                "split": "train",
            },
            {
                "sample_id": "s2",
                "image_path": "missing/b.jpg",
                "label": "no_helmet",
                "client_id": "site-b",
                "split": "val",
            },
        ],
    )

    records = load_manifest(manifest, root_dir=tmp_path)

    assert len(records) == 2
    assert records[0].sample_id == "s1"
    assert records[0].image_path == str(tmp_path / "missing/a.jpg")
    assert records[1].split == "val"


def test_load_manifest_rejects_missing_required_columns(tmp_path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [{"sample_id": "s1", "image_path": "a.jpg", "label": "helmet"}],
        fieldnames=["sample_id", "image_path", "label"],
    )

    with pytest.raises(ValueError, match="missing required columns"):
        load_manifest(manifest)


def test_load_manifest_rejects_invalid_split(tmp_path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "sample_id": "s1",
                "image_path": "a.jpg",
                "label": "helmet",
                "client_id": "site-a",
                "split": "dev",
            }
        ],
    )

    with pytest.raises(ValueError, match="invalid split"):
        load_manifest(manifest)


def test_group_manifest_by_client(tmp_path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "sample_id": "s1",
                "image_path": "a.jpg",
                "label": "helmet",
                "client_id": "site-a",
                "split": "train",
            },
            {
                "sample_id": "s2",
                "image_path": "b.jpg",
                "label": "vest",
                "client_id": "site-a",
                "split": "val",
            },
            {
                "sample_id": "s3",
                "image_path": "c.jpg",
                "label": "helmet",
                "client_id": "site-b",
                "split": "train",
            },
        ],
    )

    grouped = group_manifest_by_client(load_manifest(manifest))

    assert sorted(grouped) == ["site-a", "site-b"]
    assert [record.sample_id for record in grouped["site-a"]] == ["s1", "s2"]


def test_load_embedding_artifact_accepts_valid_npz(tmp_path):
    path = tmp_path / "embeddings.npz"
    np.savez(
        path,
        features=np.ones((3, 4), dtype=np.float32),
        labels=np.array(["helmet", "vest", "helmet"]),
        client_ids=np.array(["site-a", "site-b", "site-a"]),
        splits=np.array(["train", "val", "train"]),
        sample_ids=np.array(["s1", "s2", "s3"]),
    )

    artifact = load_embedding_artifact(path)

    assert artifact.num_samples == 3
    assert artifact.embedding_dim == 4
    assert artifact.label_mapping == {"helmet": 0, "vest": 1}


def test_validate_embedding_artifact_rejects_length_mismatch():
    artifact = EmbeddingArtifact(
        features=np.ones((2, 4), dtype=np.float32),
        labels=np.array(["helmet"]),
        client_ids=np.array(["site-a", "site-b"]),
        splits=np.array(["train", "val"]),
        sample_ids=np.array(["s1", "s2"]),
    )

    with pytest.raises(ValueError, match="does not match features length"):
        validate_embedding_artifact(artifact)


def test_validate_embedding_artifact_rejects_non_2d_features():
    artifact = EmbeddingArtifact(
        features=np.ones((4,), dtype=np.float32),
        labels=np.array(["helmet"]),
        client_ids=np.array(["site-a"]),
        splits=np.array(["train"]),
        sample_ids=np.array(["s1"]),
    )

    with pytest.raises(ValueError, match="features must be a 2D array"):
        validate_embedding_artifact(artifact)


def test_validate_embedding_artifact_requires_train_and_val():
    artifact = EmbeddingArtifact(
        features=np.ones((2, 4), dtype=np.float32),
        labels=np.array(["helmet", "vest"]),
        client_ids=np.array(["site-a", "site-b"]),
        splits=np.array(["train", "train"]),
        sample_ids=np.array(["s1", "s2"]),
    )

    with pytest.raises(ValueError, match="train and val"):
        validate_embedding_artifact(artifact)


def test_validate_embedding_artifact_rejects_blank_client_ids():
    artifact = EmbeddingArtifact(
        features=np.ones((2, 4), dtype=np.float32),
        labels=np.array(["helmet", "vest"]),
        client_ids=np.array(["site-a", ""]),
        splits=np.array(["train", "val"]),
        sample_ids=np.array(["s1", "s2"]),
    )

    with pytest.raises(ValueError, match="client_ids"):
        validate_embedding_artifact(artifact)


def test_synthetic_precompute_creates_valid_artifact_from_manifest(tmp_path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "sample_id": "s1",
                "image_path": "a.jpg",
                "label": "safe",
                "client_id": "site-a",
                "split": "train",
            },
            {
                "sample_id": "s2",
                "image_path": "b.jpg",
                "label": "unsafe",
                "client_id": "site-b",
                "split": "val",
            },
        ],
    )
    records = load_manifest(manifest)

    artifact = create_synthetic_embedding_artifact(
        records,
        embedding_dim=6,
        seed=2026,
    )

    assert artifact.features.shape == (2, 6)
    assert artifact.features.dtype == np.float32
    assert artifact.labels.tolist() == ["safe", "unsafe"]
    assert artifact.client_ids.tolist() == ["site-a", "site-b"]
    assert artifact.splits.tolist() == ["train", "val"]
    assert artifact.sample_ids.tolist() == ["s1", "s2"]


def test_synthetic_precompute_is_deterministic_for_same_seed(tmp_path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "sample_id": "s1",
                "image_path": "a.jpg",
                "label": "safe",
                "client_id": "site-a",
                "split": "train",
            },
            {
                "sample_id": "s2",
                "image_path": "b.jpg",
                "label": "unsafe",
                "client_id": "site-b",
                "split": "val",
            },
        ],
    )
    records = load_manifest(manifest)

    first = create_synthetic_embedding_artifact(records, embedding_dim=8, seed=7)
    second = create_synthetic_embedding_artifact(records, embedding_dim=8, seed=7)

    np.testing.assert_allclose(first.features, second.features)


def test_synthetic_precompute_changes_with_seed(tmp_path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "sample_id": "s1",
                "image_path": "a.jpg",
                "label": "safe",
                "client_id": "site-a",
                "split": "train",
            },
            {
                "sample_id": "s2",
                "image_path": "b.jpg",
                "label": "unsafe",
                "client_id": "site-b",
                "split": "val",
            },
        ],
    )
    records = load_manifest(manifest)

    first = create_synthetic_embedding_artifact(records, embedding_dim=8, seed=7)
    second = create_synthetic_embedding_artifact(records, embedding_dim=8, seed=8)

    assert not np.allclose(first.features, second.features)


def test_torchvision_resnet18_precompute_creates_real_image_artifact(tmp_path):
    image_module = _image_backend_image_module()
    first_image = tmp_path / "safe.jpg"
    second_image = tmp_path / "unsafe.jpg"
    image_module.new("RGB", (32, 32), color=(20, 120, 200)).save(first_image)
    image_module.new("RGB", (32, 32), color=(200, 80, 20)).save(second_image)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "sample_id": "s1",
                "image_path": first_image.name,
                "label": "safe",
                "client_id": "site-a",
                "split": "train",
            },
            {
                "sample_id": "s2",
                "image_path": second_image.name,
                "label": "unsafe",
                "client_id": "site-b",
                "split": "val",
            },
        ],
    )
    records = load_manifest(manifest, root_dir=tmp_path, require_files=True)

    artifact = create_torchvision_resnet18_embedding_artifact(
        records,
        weights="none",
        batch_size=1,
        num_workers=0,
        device="cpu",
    )

    assert artifact.features.shape == (2, RESNET18_EMBEDDING_DIM)
    assert artifact.features.dtype == np.float32
    assert artifact.labels.tolist() == ["safe", "unsafe"]
    assert artifact.client_ids.tolist() == ["site-a", "site-b"]
    assert artifact.splits.tolist() == ["train", "val"]
    assert artifact.sample_ids.tolist() == ["s1", "s2"]


def test_torchvision_resnet18_precompute_rejects_missing_image(tmp_path):
    _image_backend_image_module()
    records = [
        ManifestRecord(
            sample_id="s1",
            image_path=str(tmp_path / "missing.jpg"),
            label="safe",
            client_id="site-a",
            split="train",
        )
    ]

    with pytest.raises(FileNotFoundError, match="Image file not found"):
        create_torchvision_resnet18_embedding_artifact(
            records,
            weights="none",
            batch_size=1,
            num_workers=0,
            device="cpu",
        )


def test_save_embedding_artifact_round_trip(tmp_path):
    artifact = EmbeddingArtifact(
        features=np.ones((2, 4), dtype=np.float32),
        labels=np.array(["safe", "unsafe"]),
        client_ids=np.array(["site-a", "site-b"]),
        splits=np.array(["train", "val"]),
        sample_ids=np.array(["s1", "s2"]),
    )
    path = tmp_path / "embeddings.npz"

    save_embedding_artifact(path, artifact)
    loaded = load_embedding_artifact(path)

    np.testing.assert_allclose(loaded.features, artifact.features)
    assert loaded.labels.tolist() == ["safe", "unsafe"]


def test_precompute_embeddings_cli_writes_valid_artifact(tmp_path):
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "embeddings.npz"
    _write_manifest(
        manifest,
        [
            {
                "sample_id": "s1",
                "image_path": "a.jpg",
                "label": "safe",
                "client_id": "site-a",
                "split": "train",
            },
            {
                "sample_id": "s2",
                "image_path": "b.jpg",
                "label": "unsafe",
                "client_id": "site-b",
                "split": "val",
            },
        ],
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/precompute_embeddings.py",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--backend",
            "synthetic",
            "--embedding-dim",
            "5",
            "--seed",
            "42",
        ],
        check=True,
    )
    artifact = load_embedding_artifact(output)

    assert artifact.features.shape == (2, 5)
    assert artifact.sample_ids.tolist() == ["s1", "s2"]
