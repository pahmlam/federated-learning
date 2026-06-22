#!/usr/bin/env python3
"""Precompute embedding artifacts from a real-data manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.real_data import (
    create_synthetic_embedding_artifact,
    load_embedding_artifact,
    load_manifest,
    save_embedding_artifact,
)
from src.data.image_embeddings import create_torchvision_resnet18_embedding_artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an embedding NPZ artifact from a manifest."
    )
    parser.add_argument(
        "--manifest",
        default="configs/datasets/ppe_manifest_template.csv",
        help="CSV manifest with sample_id,image_path,label,client_id,split columns.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/ppe_embeddings_oom_safe.npz",
        help="Output NPZ artifact path.",
    )
    parser.add_argument(
        "--backend",
        choices=["synthetic", "torchvision-resnet18"],
        default="synthetic",
        help=(
            "Embedding backend. synthetic does not read image files; "
            "torchvision-resnet18 reads images and uses a frozen ResNet18."
        ),
    )
    parser.add_argument(
        "--weights",
        choices=["imagenet", "none"],
        default="imagenet",
        help="Weights for torchvision-resnet18. Tests can use none to avoid downloads.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--root-dir", default=None)
    parser.add_argument("--require-files", action="store_true")
    args = parser.parse_args()

    records = load_manifest(
        args.manifest,
        root_dir=args.root_dir,
        require_files=args.require_files or args.backend == "torchvision-resnet18",
    )
    if args.backend == "synthetic":
        artifact = create_synthetic_embedding_artifact(
            records,
            embedding_dim=args.embedding_dim,
            seed=args.seed,
        )
    elif args.backend == "torchvision-resnet18":
        artifact = create_torchvision_resnet18_embedding_artifact(
            records,
            weights=args.weights,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
        )
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    save_embedding_artifact(args.output, artifact)
    validated = load_embedding_artifact(args.output)
    print(
        "Wrote embedding artifact "
        f"to {args.output} "
        f"({validated.num_samples} samples, dim={validated.embedding_dim})"
    )


if __name__ == "__main__":
    main()
