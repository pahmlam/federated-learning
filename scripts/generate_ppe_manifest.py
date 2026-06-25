#!/usr/bin/env python3
"""Generate a reproducible PPE classification manifest from VOC annotations.

Applies the EXP-004 image-level proxy rule (an image is ``safe`` when its VOC
annotation contains a core PPE object, else ``unsafe``), then samples a
non-IID, leakage-free manifest across sites with controlled label skew.

Example (EXP-006, 480 samples / 3 sites)::

    venv/bin/python scripts/generate_ppe_manifest.py \
        --voc-dir data/ppe/voc_labels \
        --images-dir data/ppe/images \
        --output configs/datasets/ppe_real_exp006_manifest.csv \
        --sites site-a,site-b,site-c \
        --safe-ratios 0.75,0.25,0.5 \
        --per-site 160 \
        --val-fraction 0.25 \
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

from src.data.manifest_generator import (
    DEFAULT_CORE_PPE,
    collect_labeled_samples,
    generate_manifest_rows,
    summarize_rows,
    write_manifest,
)


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(item) for item in _parse_str_list(value)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voc-dir", default="data/ppe/voc_labels")
    parser.add_argument("--images-dir", default="data/ppe/images")
    parser.add_argument(
        "--output", default="configs/datasets/ppe_real_exp006_manifest.csv"
    )
    parser.add_argument("--sites", default="site-a,site-b,site-c")
    parser.add_argument(
        "--safe-ratios",
        default="0.75,0.25,0.5",
        help="Per-site fraction of safe samples, aligned with --sites.",
    )
    parser.add_argument("--per-site", type=int, default=160)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--image-path-prefix",
        default="images",
        help="Prefix prepended to image filenames, relative to precompute --root-dir.",
    )
    parser.add_argument(
        "--core-ppe",
        default=",".join(DEFAULT_CORE_PPE),
        help="Comma-separated core PPE class names defining the safe proxy.",
    )
    args = parser.parse_args()

    samples = collect_labeled_samples(
        args.voc_dir, args.images_dir, core_ppe=_parse_str_list(args.core_ppe)
    )
    rows = generate_manifest_rows(
        samples,
        sites=_parse_str_list(args.sites),
        safe_ratios=_parse_float_list(args.safe_ratios),
        per_site=args.per_site,
        val_fraction=args.val_fraction,
        seed=args.seed,
        image_path_prefix=args.image_path_prefix,
    )
    write_manifest(rows, args.output)

    summary = summarize_rows(rows)
    print(f"Wrote manifest to {args.output} ({summary['total']} rows)")
    print(json.dumps(summary["per_site"], indent=2))


if __name__ == "__main__":
    main()
