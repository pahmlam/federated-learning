#!/usr/bin/env python3
"""Run synthetic FL demo baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fl.federated import run_federated
from src.training.baselines import run_centralized, run_local_only
from src.utils.config import build_config
from src.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic FL demo baselines.")
    parser.add_argument(
        "--mode",
        choices=["centralized", "local-only", "federated", "all"],
        default="all",
    )
    parser.add_argument("--partition", choices=["iid", "non_iid"], default="non_iid")
    parser.add_argument("--quick", action="store_true", help="Use tiny CPU smoke config.")
    parser.add_argument("--output-dir", default="outputs/EXP-001")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    config = build_config(
        partition=args.partition,
        quick=args.quick,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    output_dir = Path(config.output_dir)

    results: dict[str, Any] = {}
    if args.mode in {"centralized", "all"}:
        results["centralized"] = run_centralized(config)
        write_json(output_dir / "centralized_metrics.json", results["centralized"])

    if args.mode in {"local-only", "all"}:
        results["local-only"] = run_local_only(config)
        write_json(output_dir / "local_only_metrics.json", results["local-only"])

    if args.mode in {"federated", "all"}:
        results["federated"] = run_federated(config)
        write_json(output_dir / "federated_metrics.json", results["federated"])

    summary = {
        "experiment": config.exp_id,
        "partition": config.partition,
        "quick": config.quick,
        "modes": {
            mode: {
                "loss": result["global"]["loss"],
                "accuracy": result["global"]["accuracy"],
                "training_time_sec": result["training_time_sec"],
                "update_size_bytes": result["update_size_bytes"],
                "communication_cost_bytes": result["communication_cost_bytes"],
            }
            for mode, result in results.items()
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(f"Wrote demo metrics to {output_dir}")


if __name__ == "__main__":
    main()
