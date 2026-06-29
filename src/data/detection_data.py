"""Load a detection manifest into per-client datasets for baselines and FL.

Reads the CSV written by ``detection_manifest`` and resolves image/annotation
paths under ``root_dir`` into ``PPEDetectionDataset`` objects, grouped by client
and split, plus a pooled dataset for the centralized reference baseline. Mirrors
the embedding bundle (``src/data/embedding.py``) but holds datasets, not tensors.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from src.data.detection_dataset import DetectionRecord, PPEDetectionDataset, read_voc_objects
from src.utils.detection_config import ppe_label_to_index


@dataclass(frozen=True)
class DetectionClientData:
    """One client's train/val detection datasets + PPE class-presence histogram."""

    client_id: int
    client_label: str
    train: PPEDetectionDataset
    val: PPEDetectionDataset
    label_histogram: dict[str, int]


@dataclass(frozen=True)
class DetectionDatasetBundle:
    clients: list[DetectionClientData]
    pooled_train: PPEDetectionDataset
    pooled_val: PPEDetectionDataset
    num_classes: int
    manifest_path: str
    image_size: int


def _read_manifest_records(
    manifest_path: str | Path, root_dir: str | Path
) -> list[tuple[str, str, DetectionRecord]]:
    """Return (client_label, split, record) tuples resolved against root_dir."""

    root = Path(root_dir)
    records: list[tuple[str, str, DetectionRecord]] = []
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            record = DetectionRecord(
                image_path=root / row["image_path"],
                voc_path=root / row["voc_path"],
                client_id=row["client_id"],
                split=row["split"],
            )
            records.append((row["client_id"], row["split"], record))
    return records


def load_detection_bundle(
    manifest_path: str | Path,
    root_dir: str | Path,
    *,
    image_size: int = 512,
) -> DetectionDatasetBundle:
    records = _read_manifest_records(manifest_path, root_dir)
    if not records:
        raise ValueError(f"manifest {manifest_path} produced no records")

    client_labels = sorted({client for client, _, _ in records})
    clients: list[DetectionClientData] = []
    pooled_train: list[DetectionRecord] = []
    pooled_val: list[DetectionRecord] = []

    for client_id, client_label in enumerate(client_labels):
        train_recs = [r for c, s, r in records if c == client_label and s == "train"]
        val_recs = [r for c, s, r in records if c == client_label and s == "val"]
        if not train_recs or not val_recs:
            raise ValueError(
                f"client {client_label} is missing a train or val split"
            )
        pooled_train.extend(train_recs)
        pooled_val.extend(val_recs)
        clients.append(
            DetectionClientData(
                client_id=client_id,
                client_label=client_label,
                train=PPEDetectionDataset(train_recs, image_size=image_size),
                val=PPEDetectionDataset(val_recs, image_size=image_size),
                label_histogram=_presence_histogram(train_recs + val_recs),
            )
        )

    return DetectionDatasetBundle(
        clients=clients,
        pooled_train=PPEDetectionDataset(pooled_train, image_size=image_size),
        pooled_val=PPEDetectionDataset(pooled_val, image_size=image_size),
        num_classes=len(ppe_label_to_index()) + 1,
        manifest_path=str(manifest_path),
        image_size=image_size,
    )


def _presence_histogram(records: list[DetectionRecord]) -> dict[str, int]:
    """Count, per PPE class, how many images contain >= 1 box of that class."""

    label_names = list(ppe_label_to_index().keys())
    histogram = {name: 0 for name in label_names}
    for record in records:
        present = {obj.name.strip().lower() for obj in read_voc_objects(record.voc_path)}
        for name in label_names:
            if name in present:
                histogram[name] += 1
    return histogram
