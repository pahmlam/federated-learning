"""PPE detection dataset: VOC annotations -> torchvision detection targets.

Lazy-loads each image and parses its VOC XML into a torchvision-style target
``{"boxes": [N,4] xyxy float32, "labels": [N] int64}``, keeping only the 8 core
PPE classes (remapped to 1..8; 0 is background). Large images are downscaled so
the longer side is at most ``image_size`` to cap decode/memory; boxes are scaled
to match. Reuses the canonical class order from ``detection_config``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from src.utils.detection_config import ppe_label_to_index


@dataclass(frozen=True)
class VocObject:
    """A single VOC ``<object>``: class name + (xmin, ymin, xmax, ymax)."""

    name: str
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectionRecord:
    """One detection sample: resolved image + annotation paths and partition info."""

    image_path: Path
    voc_path: Path
    client_id: str
    split: str


def read_voc_objects(xml_path: str | Path) -> list[VocObject]:
    """Read all ``<object>`` name + bndbox values from a VOC annotation file."""

    root = ET.parse(xml_path).getroot()
    objects: list[VocObject] = []
    for node in root.findall(".//object"):
        name = (node.findtext("name") or "").strip()
        bndbox = node.find("bndbox")
        if bndbox is None:
            continue
        try:
            xmin = float(bndbox.findtext("xmin"))
            ymin = float(bndbox.findtext("ymin"))
            xmax = float(bndbox.findtext("xmax"))
            ymax = float(bndbox.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        objects.append(VocObject(name=name, box=(xmin, ymin, xmax, ymax)))
    return objects


def voc_to_target(
    objects: Iterable[VocObject],
    label_map: dict[str, int],
    *,
    scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Convert VOC objects to a detection target, filtering to PPE classes.

    Drops objects whose class is not in ``label_map`` and degenerate boxes
    (zero/negative width or height after scaling).
    """

    boxes: list[list[float]] = []
    labels: list[int] = []
    for obj in objects:
        key = obj.name.strip().lower()
        if key not in label_map:
            continue
        xmin, ymin, xmax, ymax = (value * scale for value in obj.box)
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[key])

    if boxes:
        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
    return {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
    }


class PPEDetectionDataset(Dataset):
    """Map detection records to (image_tensor, target) pairs, lazily."""

    def __init__(self, records: Sequence[DetectionRecord], *, image_size: int = 512) -> None:
        self.records = list(records)
        self.image_size = image_size
        self.label_map = ppe_label_to_index()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        width, height = image.size
        scale = min(1.0, self.image_size / max(width, height))
        if scale < 1.0:
            new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            image = image.resize(new_size)
        target = voc_to_target(read_voc_objects(record.voc_path), self.label_map, scale=scale)
        target["image_id"] = torch.tensor([index])
        return to_tensor(image), target


def detection_collate_fn(batch):
    """Collate variable-length detection samples into parallel tuples.

    torchvision detectors expect ``(list[Tensor], list[dict])`` per batch.
    """

    return tuple(zip(*batch, strict=True))
