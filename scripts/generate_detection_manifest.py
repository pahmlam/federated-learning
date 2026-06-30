#!/usr/bin/env python3
"""Generate a reproducible non-IID PPE *detection* manifest from VOC annotations.

Selects PPE-positive images and partitions them across sites with class-presence
skew (default: site-a helmet-heavy, site-b vest/mask-heavy, site-c balanced),
leakage-free and seeded. Only reads XML + lists image files (no image decode).

Example (EXP-011, ~300 images / site)::

    venv/bin/python scripts/generate_detection_manifest.py \
        --voc-dir data/ppe/voc_labels \
        --images-dir data/ppe/images \
        --output configs/datasets/ppe_detection_exp011_manifest.csv \
        --sites site-a,site-b,site-c \
        --per-site 300 \
        --val-fraction 0.2 \
        --seed 2026
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.detection_manifest import (
    collect_detection_samples,
    generate_detection_manifest_rows,
    summarize_detection_rows,
    write_detection_manifest,
)
from src.utils.detection_config import DetectionConfig


def main() -> None:
    env_config = DetectionConfig.from_env_and_overrides({})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voc-dir", default=str(Path(env_config.root_dir) / "voc_labels"))
    parser.add_argument("--images-dir", default=str(Path(env_config.root_dir) / "images"))
    parser.add_argument("--output", default=env_config.manifest_path)
    parser.add_argument("--sites", default="site-a,site-b,site-c")
    parser.add_argument("--per-site", type=int, default=300)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=env_config.seed)
    args = parser.parse_args()

    samples = collect_detection_samples(args.voc_dir, args.images_dir)
    rows = generate_detection_manifest_rows(
        samples,
        sites=[s.strip() for s in args.sites.split(",") if s.strip()],
        per_site=args.per_site,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    write_detection_manifest(rows, args.output)
    summary = summarize_detection_rows(rows)
    print(f"PPE-positive pool: {len(samples)} images")
    print(f"Wrote {summary['total']} manifest rows to {args.output}")
    print(json.dumps(summary["per_site"], indent=2))


if __name__ == "__main__":
    main()
