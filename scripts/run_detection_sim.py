#!/usr/bin/env python3
"""Run PPE detection baselines (centralized / local-only / federated) in simulation.

Loads a detection manifest into per-client datasets and runs the requested modes,
writing JSON metrics per mode plus a summary. Validate here (CPU smoke on the Mac,
real run on the RTX3060) before real 3-node deployment.

Example::

    venv/bin/python scripts/run_detection_sim.py --mode all \
        --manifest configs/datasets/ppe_detection_exp011_manifest.csv \
        --root-dir data/ppe --output-dir outputs/EXP-011 --exp-id EXP-011 \
        --image-size 512 --batch-size 2 --local-epochs 2 \
        --centralized-epochs 4 --num-rounds 5 --device auto
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.detection_data import load_detection_bundle
from src.fl.detection_federated import run_detection_federated
from src.training.detection_baselines import (
    run_detection_centralized,
    run_detection_local_only,
)
from src.utils.detection_config import DetectionConfig
from src.utils.io import write_json
from src.utils.resources import get_resource_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["centralized", "local-only", "federated", "all"], default="all"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root-dir", default="data/ppe")
    parser.add_argument("--output-dir", default="outputs/EXP-011")
    parser.add_argument("--exp-id", default="EXP-011")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--centralized-epochs", type=int, default=4)
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    print(
        f"[run] loading manifest={args.manifest} root_dir={args.root_dir} "
        f"image_size={args.image_size}",
        flush=True,
    )
    bundle = load_detection_bundle(args.manifest, args.root_dir, image_size=args.image_size)
    config = DetectionConfig(
        exp_id=args.exp_id,
        output_dir=args.output_dir,
        num_clients=len(bundle.clients),
        image_size=args.image_size,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        centralized_epochs=args.centralized_epochs,
        num_rounds=args.num_rounds,
        lr=args.lr,
        device=args.device,
        pretrained=not args.no_pretrained,
        num_workers=args.num_workers,
        seed=args.seed,
    ).normalized()
    output_dir = Path(config.output_dir)
    print(
        f"[run] loaded {len(bundle.clients)} clients, "
        f"train={len(bundle.pooled_train)} val={len(bundle.pooled_val)}, "
        f"device={config.device}, pretrained={config.pretrained}, "
        f"batch_size={config.batch_size}",
        flush=True,
    )

    results: dict[str, Any] = {}
    if args.mode in {"centralized", "all"}:
        print("[run] mode centralized start", flush=True)
        results["centralized"] = run_detection_centralized(config, bundle)
        write_json(output_dir / "centralized_metrics.json", results["centralized"])
        print("[run] mode centralized metrics written", flush=True)
    if args.mode in {"local-only", "all"}:
        print("[run] mode local-only start", flush=True)
        results["local-only"] = run_detection_local_only(config, bundle)
        write_json(output_dir / "local_only_metrics.json", results["local-only"])
        print("[run] mode local-only metrics written", flush=True)
    if args.mode in {"federated", "all"}:
        print("[run] mode federated start", flush=True)
        results["federated"] = run_detection_federated(config, bundle)
        write_json(output_dir / "federated_metrics.json", results["federated"])
        print("[run] mode federated metrics written", flush=True)

    summary = {
        "experiment": config.exp_id,
        "data_source": "detection",
        "manifest_path": str(args.manifest),
        "config": {
            "num_clients": len(bundle.clients),
            "num_classes": bundle.num_classes,
            "image_size": config.image_size,
            "batch_size": config.batch_size,
            "local_epochs": config.local_epochs,
            "centralized_epochs": config.centralized_epochs,
            "num_rounds": config.num_rounds,
            "lr": config.lr,
            "device": config.device,
            "pretrained": config.pretrained,
            "seed": config.seed,
        },
        "modes": {mode: result.get("global") for mode, result in results.items()},
        "resource_snapshot": get_resource_snapshot(),
    }
    write_json(output_dir / "summary.json", summary)
    print(f"Wrote detection sim metrics to {output_dir}")


if __name__ == "__main__":
    main()
