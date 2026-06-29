"""Reproducible non-IID manifest for the PPE detection track.

Selects PPE-positive images (>= 1 of the 8 core PPE classes) from the VOC dataset
and partitions them across sites with **class-presence skew** (e.g. site-a
helmet-heavy, site-b vest/mask-heavy, site-c balanced), leakage-free and seeded.
Only reads annotation XML + lists image files -- never decodes images, so it stays
cheap. Reuses ``build_image_index`` / ``read_voc_object_names`` from the
classification manifest generator.
"""

from __future__ import annotations

import csv
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.data.manifest_generator import (
    DEFAULT_CORE_PPE,
    build_image_index,
    read_voc_object_names,
)

MANIFEST_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "image_path",
    "voc_path",
    "client_id",
    "split",
)
_MIN_GROUP_SIZE = 2  # one train + one val

# Default non-IID skew: which PPE classes each site is biased toward.
# An empty focus set means "balanced" (takes any remaining PPE-positive image).
DEFAULT_SITE_FOCUS: dict[str, frozenset[str]] = {
    "site-a": frozenset({"helmet"}),
    "site-b": frozenset({"safety-vest", "face-mask-medical"}),
    "site-c": frozenset(),
}


@dataclass(frozen=True)
class DetectionSample:
    """A PPE-positive VOC sample: stem, image/annotation filenames, PPE classes present."""

    stem: str
    image_filename: str
    voc_filename: str
    ppe_classes: frozenset[str]


@dataclass(frozen=True)
class DetectionManifestRow:
    sample_id: str
    image_path: str
    voc_path: str
    client_id: str
    split: str


def collect_detection_samples(
    voc_dir: str | Path,
    images_dir: str | Path,
    core_ppe: Sequence[str] = DEFAULT_CORE_PPE,
) -> list[DetectionSample]:
    """Collect every annotation that has a matching image and >= 1 core PPE class."""

    core = {name.strip().lower() for name in core_ppe}
    image_index = build_image_index(Path(images_dir))
    samples: list[DetectionSample] = []
    for xml_path in sorted(Path(voc_dir).glob("*.xml")):
        image_path = image_index.get(xml_path.stem)
        if image_path is None:
            continue
        names = {name.strip().lower() for name in read_voc_object_names(xml_path)}
        present = names & core
        if not present:
            continue
        samples.append(
            DetectionSample(
                stem=xml_path.stem,
                image_filename=image_path.name,
                voc_filename=xml_path.name,
                ppe_classes=frozenset(present),
            )
        )
    return samples


def generate_detection_manifest_rows(
    samples: Sequence[DetectionSample],
    *,
    sites: Sequence[str],
    per_site: int,
    val_fraction: float,
    seed: int,
    site_focus: Mapping[str, frozenset[str]] | None = None,
    image_prefix: str = "images",
    voc_prefix: str = "voc_labels",
) -> list[DetectionManifestRow]:
    """Partition samples across sites with class-presence skew, leakage-free.

    Focused sites are filled first (preferring images that contain their focus
    classes, falling back to any remaining sample); balanced sites take the rest.
    """

    if per_site < _MIN_GROUP_SIZE:
        raise ValueError(f"per_site must be >= {_MIN_GROUP_SIZE}")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")

    focus = dict(site_focus) if site_focus is not None else DEFAULT_SITE_FOCUS
    rng = random.Random(seed)
    pool = list(samples)
    rng.shuffle(pool)

    # Focused sites first (more specific selection), balanced/empty-focus sites last.
    ordered_sites = sorted(sites, key=lambda site: not focus.get(site))
    assigned: set[str] = set()
    rows: list[DetectionManifestRow] = []
    for site in ordered_sites:
        focus_classes = focus.get(site, frozenset())
        chosen = _select_for_site(pool, assigned, focus_classes, per_site)
        if len(chosen) < _MIN_GROUP_SIZE:
            raise ValueError(
                f"not enough samples for site {site}: got {len(chosen)}, "
                f"need >= {_MIN_GROUP_SIZE}"
            )
        for sample in chosen:
            assigned.add(sample.stem)
        rows.extend(
            _rows_for_site(chosen, site, val_fraction, image_prefix, voc_prefix)
        )
    return rows


def write_detection_manifest(
    rows: Sequence[DetectionManifestRow], output: str | Path
) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(MANIFEST_COLUMNS)
        for row in rows:
            writer.writerow(
                [row.sample_id, row.image_path, row.voc_path, row.client_id, row.split]
            )


def summarize_detection_rows(rows: Sequence[DetectionManifestRow]) -> dict[str, object]:
    per_site: dict[str, dict[str, int]] = {}
    for row in rows:
        summary = per_site.setdefault(row.client_id, {"total": 0, "train": 0, "val": 0})
        summary["total"] += 1
        summary[row.split] += 1
    return {"total": len(rows), "per_site": per_site}


def _select_for_site(
    pool: Sequence[DetectionSample],
    assigned: set[str],
    focus_classes: frozenset[str],
    per_site: int,
) -> list[DetectionSample]:
    available = [s for s in pool if s.stem not in assigned]
    if not focus_classes:
        return available[:per_site]
    preferred = [s for s in available if s.ppe_classes & focus_classes]
    chosen = preferred[:per_site]
    if len(chosen) < per_site:
        chosen_stems = {s.stem for s in chosen}
        fallback = [s for s in available if s.stem not in chosen_stems]
        chosen.extend(fallback[: per_site - len(chosen)])
    return chosen


def _rows_for_site(
    samples: Sequence[DetectionSample],
    site: str,
    val_fraction: float,
    image_prefix: str,
    voc_prefix: str,
) -> list[DetectionManifestRow]:
    n_val = max(1, round(len(samples) * val_fraction))
    if n_val >= len(samples):
        n_val = len(samples) - 1
    val_samples, train_samples = samples[:n_val], samples[n_val:]
    rows: list[DetectionManifestRow] = []
    for split_name, items in (("train", train_samples), ("val", val_samples)):
        for sample in items:
            rows.append(
                DetectionManifestRow(
                    sample_id=f"{site}_{split_name}_{sample.stem}",
                    image_path=f"{image_prefix}/{sample.image_filename}",
                    voc_path=f"{voc_prefix}/{sample.voc_filename}",
                    client_id=site,
                    split=split_name,
                )
            )
    return rows
