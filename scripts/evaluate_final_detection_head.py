#!/usr/bin/env python3
"""Evaluate the final aggregated detection head on a site's local validation shard.

Post-FL handoff: take ``outputs/<exp-id>/final_head.npz`` produced by a Flower
deployment and run it against one site's local data, on the machine that owns
that data (the server never holds raw data). Writes a JSON metrics report.

Example::

    venv/bin/python scripts/evaluate_final_detection_head.py \
        --head-path outputs/EXP-012/final_head.npz \
        --manifest configs/datasets/site_a_manifest.csv \
        --root-dir data/ppe --client-id site-a \
        --output outputs/EXP-012/final_head_site_a_metrics.json \
        --device auto --batch-size 2
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.final_head_eval import evaluate_final_head
from src.utils.detection_config import DetectionConfig
from src.utils.io import write_json


def build_config(args: argparse.Namespace) -> DetectionConfig:
    """Start from defaults+env, then apply explicit CLI overrides."""

    base = DetectionConfig.from_env_and_overrides({})
    return replace(
        base,
        manifest_path=args.manifest,
        root_dir=args.root_dir,
        client_id=args.client_id,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        score_threshold=args.score_threshold,
        pretrained=args.pretrained,
        seed=args.seed,
    ).normalized()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    env_config = DetectionConfig.from_env_and_overrides({})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-path", required=True, help="Path to final_head.npz")
    parser.add_argument("--manifest", default=env_config.manifest_path)
    parser.add_argument("--root-dir", default=env_config.root_dir)
    parser.add_argument(
        "--client-id",
        default=env_config.client_id,
        help="Site to evaluate when the manifest holds multiple clients",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Where to write the metrics JSON report",
    )
    parser.add_argument("--image-size", type=int, default=env_config.image_size)
    parser.add_argument("--batch-size", type=int, default=env_config.batch_size)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default=env_config.device
    )
    parser.add_argument("--num-workers", type=int, default=env_config.num_workers)
    parser.add_argument(
        "--score-threshold", type=float, default=env_config.score_threshold
    )
    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        default=env_config.pretrained,
    )
    pretrained_group.add_argument(
        "--no-pretrained", dest="pretrained", action="store_false"
    )
    parser.add_argument("--seed", type=int, default=env_config.seed)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config(args)
    print(
        f"[final-head] head={args.head_path} manifest={config.manifest_path} "
        f"root_dir={config.root_dir} client_id={config.client_id} "
        f"device={config.device}",
        flush=True,
    )
    report = evaluate_final_head(config, args.head_path, client_id=config.client_id)
    write_json(args.output, report)
    print(
        f"[final-head] client={report['client_label']} "
        f"num_examples={report['num_examples']} "
        f"map={report.get('map', -1.0):.4f} map_50={report.get('map_50', -1.0):.4f}",
        flush=True,
    )
    print(f"Wrote final-head eval metrics to {args.output}", flush=True)


if __name__ == "__main__":
    main()
