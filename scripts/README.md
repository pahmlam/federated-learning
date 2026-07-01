# Scripts

Command-line entrypoints for running experiments.

Current script:

- `run_demo.py`: runs synthetic `centralized`, `local-only`, `federated`, or all modes.
- `precompute_embeddings.py`: creates an embedding `.npz` artifact from a manifest. The `synthetic` backend does not read image files; `torchvision-resnet18` reads images and uses a frozen ResNet18.
- `run_embedding_demo.py`: runs `centralized`, `local-only`, and `federated` baselines on an embedding `.npz` artifact.
- `generate_detection_manifest.py`: creates a seeded PPE detection manifest from VOC labels/images.
- `run_detection_sim.py`: runs PPE detection centralized, local-only, and federated simulation baselines.
- `export_detection_subset.py`: exports per-site detection shards for deployment clients.
- `evaluate_final_detection_head.py`: loads a deployed `final_head.npz` and evaluates it on one site's labeled validation shard.
- `run_detection_inference.py`: loads a deployed `final_head.npz` and runs operational inference on one local image or image directory, writing per-image detection JSON and optional annotated images.

Quick smoke command:

```bash
venv/bin/python scripts/run_demo.py --mode all --quick
```

PPE manifest-to-embedding dry run:

```bash
venv/bin/python scripts/precompute_embeddings.py --backend synthetic
```

PPE real-image frozen-backbone embedding smoke, after a private manifest/image root exists:

```bash
venv/bin/python scripts/precompute_embeddings.py \
  --backend torchvision-resnet18 \
  --weights imagenet \
  --manifest configs/datasets/ppe_real_smoke_manifest.csv \
  --root-dir /path/to/private/ppe_images \
  --output data/processed/ppe_real_embeddings_oom_safe.npz \
  --batch-size 4 \
  --num-workers 0 \
  --device mps
```

PPE embedding baseline dry run:

```bash
venv/bin/python scripts/run_embedding_demo.py --mode all --artifact data/processed/ppe_embeddings_oom_safe.npz --profile oom-safe --output-dir outputs/EXP-003
```

PPE real-image embedding baseline, after the frozen-backbone artifact exists:

```bash
venv/bin/python scripts/run_embedding_demo.py --mode all --artifact data/processed/ppe_real_embeddings_oom_safe.npz --profile oom-safe --output-dir outputs/EXP-004 --exp-id EXP-004
```

Site-side PPE detection inference after deployment:

```bash
venv/bin/python scripts/run_detection_inference.py \
  --head-path outputs/EXP-012-rerun/final_head.npz \
  --input-dir <site-local-images> \
  --output-dir outputs/EXP-012-rerun/inference_site_b \
  --device auto \
  --score-threshold 0.5 \
  --save-images
```
