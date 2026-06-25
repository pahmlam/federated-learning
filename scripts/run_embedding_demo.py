#!/usr/bin/env python3
"""Run baseline modes on a precomputed embedding artifact."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.embedding import load_embedding_dataset_bundle
from src.fl.embedding_federated import run_embedding_federated
from src.training.embedding_baselines import (
    run_embedding_centralized,
    run_embedding_local_only,
)
from src.utils.config import DemoConfig
from src.utils.io import write_json
from src.utils.resources import get_resource_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FL baselines on a precomputed embedding artifact."
    )
    parser.add_argument(
        "--mode",
        choices=["centralized", "local-only", "federated", "all"],
        default="all",
    )
    parser.add_argument(
        "--artifact",
        default="data/processed/ppe_embeddings_oom_safe.npz",
        help="Embedding NPZ created by scripts/precompute_embeddings.py.",
    )
    parser.add_argument(
        "--profile",
        choices=["default", "quick", "oom-safe"],
        default="oom-safe",
    )
    parser.add_argument("--output-dir", default="outputs/EXP-003")
    parser.add_argument(
        "--exp-id",
        default=None,
        help="Experiment ID to write into JSON outputs. Defaults to EXP-* output dir name.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--centralized-epochs", type=int, default=None)
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument(
        "--normalize-embedding",
        action="store_true",
        help="L2-normalize embeddings before the head (cosine-style; EXP-009).",
    )
    parser.add_argument(
        "--head-hidden-dim",
        type=int,
        default=None,
        help="Hidden width for a 2-layer MLP head; omit for a linear head (EXP-010).",
    )
    args = parser.parse_args()

    bundle = load_embedding_dataset_bundle(args.artifact)
    exp_id = _resolve_exp_id(args.output_dir, args.exp_id)
    config = DemoConfig(
        exp_id=exp_id,
        output_dir=args.output_dir,
        profile=args.profile,
        num_clients=len(bundle.clients),
        num_classes=bundle.num_classes,
        input_dim=bundle.embedding_dim,
        embedding_dim=bundle.embedding_dim,
        seed=args.seed,
    ).normalized()
    # Keep artifact-derived dimensions intact even when oom-safe clamps synthetic fields.
    config = replace(
        config,
        exp_id=exp_id,
        output_dir=args.output_dir,
        num_clients=len(bundle.clients),
        num_classes=bundle.num_classes,
        input_dim=bundle.embedding_dim,
        embedding_dim=bundle.embedding_dim,
        local_epochs=(
            args.local_epochs
            if args.local_epochs is not None
            else config.local_epochs
        ),
        centralized_epochs=(
            args.centralized_epochs
            if args.centralized_epochs is not None
            else config.centralized_epochs
        ),
        num_rounds=args.num_rounds if args.num_rounds is not None else config.num_rounds,
        batch_size=args.batch_size if args.batch_size is not None else config.batch_size,
        lr=args.lr if args.lr is not None else config.lr,
        weight_decay=(
            args.weight_decay
            if args.weight_decay is not None
            else config.weight_decay
        ),
        normalize_embedding=args.normalize_embedding or config.normalize_embedding,
        head_hidden_dim=(
            args.head_hidden_dim
            if args.head_hidden_dim is not None
            else config.head_hidden_dim
        ),
    )
    _validate_overrides(config)
    output_dir = Path(config.output_dir)

    results: dict[str, Any] = {}
    if args.mode in {"centralized", "all"}:
        results["centralized"] = run_embedding_centralized(config, bundle)
        write_json(output_dir / "centralized_metrics.json", results["centralized"])

    if args.mode in {"local-only", "all"}:
        results["local-only"] = run_embedding_local_only(config, bundle)
        write_json(output_dir / "local_only_metrics.json", results["local-only"])

    if args.mode in {"federated", "all"}:
        results["federated"] = run_embedding_federated(config, bundle)
        write_json(output_dir / "federated_metrics.json", results["federated"])

    summary = {
        "experiment": config.exp_id,
        "profile": config.profile,
        "data_source": "embedding",
        "artifact_path": bundle.artifact_path,
        "label_mapping": bundle.label_mapping,
        "config": {
            "num_clients": len(bundle.clients),
            "num_classes": bundle.num_classes,
            "embedding_dim": bundle.embedding_dim,
            "batch_size": config.batch_size,
            "local_epochs": config.local_epochs,
            "centralized_epochs": config.centralized_epochs,
            "num_rounds": config.num_rounds,
            "weight_decay": config.weight_decay,
            "normalize_embedding": config.normalize_embedding,
            "head_hidden_dim": config.head_hidden_dim,
            "num_workers": config.num_workers,
            "client_num_cpus": config.client_num_cpus,
            "ray_num_cpus": config.ray_num_cpus,
            "seed": config.seed,
        },
        "modes": {
            mode: _summary_for_mode(result)
            for mode, result in results.items()
        },
        "resource_snapshot": get_resource_snapshot(),
    }
    write_json(output_dir / "summary.json", summary)
    print(f"Wrote embedding demo metrics to {output_dir}")


def _resolve_exp_id(output_dir: str, explicit_exp_id: str | None) -> str:
    if explicit_exp_id:
        return explicit_exp_id
    output_name = Path(output_dir).name
    if output_name.startswith("EXP-"):
        return output_name
    return "EXP-003"


def _validate_overrides(config: DemoConfig) -> None:
    if config.local_epochs < 1:
        raise ValueError("--local-epochs must be >= 1")
    if config.centralized_epochs < 1:
        raise ValueError("--centralized-epochs must be >= 1")
    if config.num_rounds < 1:
        raise ValueError("--num-rounds must be >= 1")
    if config.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if config.lr <= 0:
        raise ValueError("--lr must be > 0")


def _summary_for_mode(result: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        name: float(value)
        for name, value in result["global"].items()
        if isinstance(value, (int, float))
    }
    metrics.update(
        {
            "training_time_sec": result["training_time_sec"],
            "update_size_bytes": result["update_size_bytes"],
            "communication_cost_bytes": result["communication_cost_bytes"],
            "resource_snapshot": result.get("resource_snapshot"),
        }
    )
    return metrics


if __name__ == "__main__":
    main()
