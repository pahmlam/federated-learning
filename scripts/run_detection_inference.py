#!/usr/bin/env python3
"""Run PPE detection inference over unlabeled local images with the final head.

Post-FL operational handoff: take ``outputs/<exp-id>/final_head.npz`` produced by
a Flower deployment and run the detector over a site's local *unlabeled* images,
on the machine that owns them (the server never holds raw data). Writes one JSON
file of detections per image, and optionally annotated preview images.

Example::

    venv/bin/python scripts/run_detection_inference.py \
        --head-path outputs/EXP-012-rerun/final_head.npz \
        --input-dir data/ppe/site_b/images \
        --output-dir outputs/EXP-012-rerun/inference_site_b \
        --device auto --score-threshold 0.5 --save-images
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.detection_inference import (
    annotate_image,
    collect_image_paths,
    load_inference_model,
    run_inference_on_image,
)
from src.models.detection_model import resolve_device
from src.utils.detection_config import DetectionConfig
from src.utils.io import write_json


def build_config(args: argparse.Namespace) -> DetectionConfig:
    """Start from defaults+env, then apply explicit CLI overrides."""

    base = DetectionConfig.from_env_and_overrides({})
    return replace(
        base,
        image_size=args.image_size,
        device=args.device,
        score_threshold=args.score_threshold,
        pretrained=args.pretrained,
        seed=args.seed,
    ).normalized()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    env_config = DetectionConfig.from_env_and_overrides({})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-path", required=True, help="Path to final_head.npz")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image", help="Path to a single input image")
    source_group.add_argument("--input-dir", help="Directory of input images")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to write per-image detection JSON (and annotated images)",
    )
    parser.add_argument("--image-size", type=int, default=env_config.image_size)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default=env_config.device
    )
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
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Also write annotated preview images next to the JSON",
    )
    parser.add_argument("--seed", type=int, default=env_config.seed)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config(args)
    device = resolve_device(config.device)
    image_paths = collect_image_paths(image=args.image, input_dir=args.input_dir)
    output_dir = Path(args.output_dir)

    print(
        f"[inference] head={args.head_path} images={len(image_paths)} "
        f"device={device} score_threshold={config.score_threshold}",
        flush=True,
    )
    model = load_inference_model(config, args.head_path)

    total_detections = 0
    for image_path in image_paths:
        report, image = run_inference_on_image(
            model,
            image_path,
            config=config,
            head_path=args.head_path,
            device=device,
        )
        total_detections += report["num_detections"]
        json_path = output_dir / f"{Path(image_path).stem}.json"
        write_json(json_path, report)
        if args.save_images:
            annotated = annotate_image(image, report["detections"])
            annotated_path = output_dir / f"{Path(image_path).stem}_annotated.jpg"
            annotated_path.parent.mkdir(parents=True, exist_ok=True)
            annotated.save(annotated_path)
        print(
            f"[inference] {Path(image_path).name}: "
            f"{report['num_detections']} detection(s) -> {json_path}",
            flush=True,
        )

    print(
        f"Wrote {len(image_paths)} inference JSON file(s) "
        f"({total_detections} total detections) to {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
