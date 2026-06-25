"""Reproducible PPE manifest generation from VOC annotations.

This module turns the raw VOC-annotated PPE dataset into a deterministic
classification manifest (``sample_id,image_path,label,client_id,split``) using
an image-level proxy rule. It only reads annotation XML and lists image files;
it never decodes images, so it stays cheap and OOM-safe.

The proxy rule mirrors EXP-004: an image is ``safe`` when its VOC annotation
contains at least one core PPE object, otherwise ``unsafe``.
"""

from __future__ import annotations

import csv
import random
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CORE_PPE: tuple[str, ...] = (
    "helmet",
    "safety-vest",
    "safety-suit",
    "face-mask-medical",
    "gloves",
    "glasses",
    "ear-mufs",
    "face-guard",
)
SAFE_LABEL = "safe"
UNSAFE_LABEL = "unsafe"
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png")
MANIFEST_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "image_path",
    "label",
    "client_id",
    "split",
)
_MIN_GROUP_SIZE = 2  # one train + one val


@dataclass(frozen=True)
class LabeledImage:
    """A VOC-derived sample: annotation stem, resolved image file, proxy label."""

    stem: str
    filename: str
    label: str


@dataclass(frozen=True)
class ManifestRow:
    """One manifest CSV row."""

    sample_id: str
    image_path: str
    label: str
    client_id: str
    split: str


def classify_objects(object_names: Iterable[str], core_ppe: Iterable[str]) -> str:
    """Return ``safe`` when any object is a core PPE class, else ``unsafe``."""

    core = {name.strip().lower() for name in core_ppe}
    for name in object_names:
        if name.strip().lower() in core:
            return SAFE_LABEL
    return UNSAFE_LABEL


def read_voc_object_names(xml_path: Path) -> list[str]:
    """Read all ``<object><name>`` values from a VOC annotation file."""

    root = ET.parse(xml_path).getroot()
    return [(node.text or "").strip() for node in root.findall(".//object/name")]


def build_image_index(images_dir: Path) -> dict[str, Path]:
    """Map each image stem to its file, resolving mixed extensions."""

    index: dict[str, Path] = {}
    for path in sorted(Path(images_dir).iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(path.stem, path)
    return index


def collect_labeled_samples(
    voc_dir: str | Path,
    images_dir: str | Path,
    core_ppe: Iterable[str] = DEFAULT_CORE_PPE,
) -> list[LabeledImage]:
    """Label every annotation that has a matching image file."""

    core = tuple(core_ppe)
    image_index = build_image_index(Path(images_dir))
    samples: list[LabeledImage] = []
    for xml_path in sorted(Path(voc_dir).glob("*.xml")):
        image_path = image_index.get(xml_path.stem)
        if image_path is None:
            continue
        label = classify_objects(read_voc_object_names(xml_path), core)
        samples.append(
            LabeledImage(stem=xml_path.stem, filename=image_path.name, label=label)
        )
    return samples


def generate_manifest_rows(
    samples: Sequence[LabeledImage],
    *,
    sites: Sequence[str],
    safe_ratios: Sequence[float],
    per_site: int,
    val_fraction: float,
    seed: int,
    image_path_prefix: str = "images",
) -> list[ManifestRow]:
    """Build a deterministic, leakage-free manifest with controlled label skew.

    Samples are drawn without replacement from shared safe/unsafe pools, so no
    underlying image appears under more than one site or split.
    """

    if len(sites) != len(safe_ratios):
        raise ValueError("sites and safe_ratios must have equal length")
    if per_site < _MIN_GROUP_SIZE:
        raise ValueError(f"per_site must be >= {_MIN_GROUP_SIZE}")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")

    safe_pool = [s for s in samples if s.label == SAFE_LABEL]
    unsafe_pool = [s for s in samples if s.label == UNSAFE_LABEL]
    rng = random.Random(seed)
    rng.shuffle(safe_pool)
    rng.shuffle(unsafe_pool)

    plan = [(site, round(per_site * ratio), per_site - round(per_site * ratio))
            for site, ratio in zip(sites, safe_ratios)]
    _check_pool_capacity(plan, len(safe_pool), len(unsafe_pool))

    safe_iter = iter(safe_pool)
    unsafe_iter = iter(unsafe_pool)
    rows: list[ManifestRow] = []
    for site, n_safe, n_unsafe in plan:
        for label, count, pool_iter in (
            (SAFE_LABEL, n_safe, safe_iter),
            (UNSAFE_LABEL, n_unsafe, unsafe_iter),
        ):
            if count == 0:
                continue
            if count < _MIN_GROUP_SIZE:
                raise ValueError(
                    f"group {site}/{label} needs >= {_MIN_GROUP_SIZE} samples "
                    f"for a train/val split, got {count}"
                )
            group = [next(pool_iter) for _ in range(count)]
            rows.extend(_rows_for_group(group, site, label, val_fraction, image_path_prefix))
    return rows


def write_manifest(rows: Sequence[ManifestRow], output: str | Path) -> None:
    """Write manifest rows to a CSV file."""

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(MANIFEST_COLUMNS)
        for row in rows:
            writer.writerow(
                [row.sample_id, row.image_path, row.label, row.client_id, row.split]
            )


def summarize_rows(rows: Sequence[ManifestRow]) -> dict[str, object]:
    """Summarize counts by site, label and split for logging/verification."""

    per_site: dict[str, dict[str, int]] = {}
    for row in rows:
        site_summary = per_site.setdefault(
            row.client_id,
            {"total": 0, SAFE_LABEL: 0, UNSAFE_LABEL: 0, "train": 0, "val": 0},
        )
        site_summary["total"] += 1
        site_summary[row.label] += 1
        site_summary[row.split] += 1
    return {"total": len(rows), "per_site": per_site}


def _rows_for_group(
    group: Sequence[LabeledImage],
    site: str,
    label: str,
    val_fraction: float,
    image_path_prefix: str,
) -> list[ManifestRow]:
    train, val = _split_train_val(group, val_fraction)
    rows: list[ManifestRow] = []
    for split_name, items in (("train", train), ("val", val)):
        for item in items:
            rows.append(
                ManifestRow(
                    sample_id=f"{site}_{split_name}_{label}_{item.stem}",
                    image_path=f"{image_path_prefix}/{item.filename}",
                    label=label,
                    client_id=site,
                    split=split_name,
                )
            )
    return rows


def _split_train_val(
    items: Sequence[LabeledImage], val_fraction: float
) -> tuple[list[LabeledImage], list[LabeledImage]]:
    size = len(items)
    n_val = max(1, round(size * val_fraction))
    if n_val >= size:
        n_val = size - 1
    return list(items[n_val:]), list(items[:n_val])


def _check_pool_capacity(
    plan: Sequence[tuple[str, int, int]], safe_available: int, unsafe_available: int
) -> None:
    need_safe = sum(n_safe for _, n_safe, _ in plan)
    need_unsafe = sum(n_unsafe for _, _, n_unsafe in plan)
    if need_safe > safe_available:
        raise ValueError(
            f"not enough safe samples: need {need_safe}, have {safe_available}"
        )
    if need_unsafe > unsafe_available:
        raise ValueError(
            f"not enough unsafe samples: need {need_unsafe}, have {unsafe_available}"
        )
