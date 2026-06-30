#!/usr/bin/env python3
"""Export per-site PPE detection shards for Flower deployment clients."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.detection_manifest import MANIFEST_COLUMNS
from src.utils.detection_config import DetectionConfig
from src.utils.io import write_json


def export_detection_subsets(
    manifest: str | Path,
    root_dir: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    rows = _read_rows(manifest)
    root = Path(root_dir)
    out = Path(output_dir)
    sites = sorted({row["client_id"] for row in rows})
    summary: dict[str, Any] = {"total": len(rows), "sites": {}}

    out.mkdir(parents=True, exist_ok=True)
    for site in sites:
        site_rows = [row for row in rows if row["client_id"] == site]
        site_dir = out / site
        zip_path = out / f"{site}.zip"
        _prepare_target(site_dir, zip_path, overwrite=overwrite)

        copied: set[Path] = set()
        for row in site_rows:
            for key in ("image_path", "voc_path"):
                source = root / row[key]
                target = site_dir / row[key]
                if target in copied:
                    continue
                if not source.is_file():
                    raise FileNotFoundError(f"missing source file: {source}")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                copied.add(target)

        _write_manifest(site_dir / "manifest.csv", site_rows)
        shutil.make_archive(str(zip_path.with_suffix("")), "zip", site_dir)
        summary["sites"][site] = _summarize_site(site_rows, zip_path)

    write_json(out / "summary.json", summary)
    return summary


def _read_rows(manifest: str | Path) -> list[dict[str, str]]:
    with Path(manifest).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"manifest {manifest} has no rows")
    missing = [column for column in MANIFEST_COLUMNS if column not in rows[0]]
    if missing:
        raise ValueError(f"manifest {manifest} is missing columns: {missing}")
    return rows


def _prepare_target(site_dir: Path, zip_path: Path, *, overwrite: bool) -> None:
    existing = [path for path in (site_dir, zip_path) if path.exists()]
    if existing and not overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing export target(s): {paths}")
    if overwrite:
        if site_dir.exists():
            shutil.rmtree(site_dir)
        if zip_path.exists():
            zip_path.unlink()
    site_dir.mkdir(parents=True, exist_ok=False)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _summarize_site(rows: list[dict[str, str]], zip_path: Path) -> dict[str, Any]:
    splits = {"train": 0, "val": 0}
    for row in rows:
        splits[row["split"]] = splits.get(row["split"], 0) + 1
    return {
        "total": len(rows),
        "train": splits.get("train", 0),
        "val": splits.get("val", 0),
        "zip_path": str(zip_path),
    }


def main() -> None:
    env_config = DetectionConfig.from_env_and_overrides({})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=env_config.manifest_path)
    parser.add_argument("--root-dir", default=env_config.root_dir)
    parser.add_argument("--output-dir", default=str(Path(env_config.output_dir) / "shards"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = export_detection_subsets(
        args.manifest,
        args.root_dir,
        args.output_dir,
        overwrite=args.overwrite,
    )
    print(f"Exported {summary['total']} rows into {args.output_dir}")


if __name__ == "__main__":
    main()
