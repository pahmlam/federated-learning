# FL Camera Research

This repository is a Federated Learning research and demo workspace for camera/vision tasks on edge deployments.

The first target use cases are:

- **Face**: recognition, verification, or embedding adaptation.
- **Clothing/PPE**: uniform, helmet, reflective vest, mask, or PPE compliance.

The goal is to build a clean experimental foundation before proposing new FL methods. Flower is used as the baseline FL framework. The existing `demo/` directory remains a Flower quickstart reference.

Default technical assumption:

```text
pretrained/fixed backbone
    -> freeze or nearly freeze backbone
    -> train only lightweight parts: embedding/head/adapter/LoRA
    -> aggregate lightweight updates when possible
```

This repo does **not** start by training a full vision model from scratch.

## Current Status

Demo v0 is available with synthetic offline data. It is a technical smoke demo, not a real face/PPE experiment yet.

The demo includes:

- `centralized`: pooled-data baseline.
- `local-only`: each simulated client trains its own head.
- `federated`: Flower/FedAvg over 5 synthetic clients.
- IID and non-IID synthetic client partitioning.
- Frozen backbone and trainable classifier head only.
- Per-client metrics, training time, update size, and communication cost.

Latest demo output is stored in:

```text
outputs/EXP-001/
```

Experiment notes are tracked in:

```text
docs/journal/
```

## Directory Structure

```text
.
├── configs/
│   ├── datasets/        # Future dataset configs and label schemas
│   ├── experiments/     # Future experiment configs
│   └── flower/          # Future Flower/server/client configs
├── data/
│   ├── raw/             # Local raw data, not for large/private commits
│   ├── processed/       # Local processed data
│   └── partitions/      # Client split/partition artifacts
├── demo/
│   └── quickstart_numpy/ # Original Flower NumPy quickstart reference
├── docs/
│   ├── journal/         # Experiment registry and per-run reports
│   ├── md/              # Planning, problem statement, research report
│   └── pdfs/            # Local paper PDFs, ignored by Git
├── experiments/
│   ├── centralized/     # Experiment notes/config references for pooled baseline
│   ├── local_only/      # Per-client baseline experiment area
│   ├── federated/       # Flower/FedAvg and later FL experiments
│   └── ablations/       # Later method/PEFT/personalization ablations
├── notebooks/           # Learning notes, analysis notebooks, smoke explorations
├── outputs/
│   ├── EXP-001/         # Current synthetic demo outputs
│   ├── logs/            # Local logs
│   ├── metrics/         # Local metric exports
│   ├── checkpoints/     # Local model checkpoints
│   └── reports/         # Local generated reports
├── scripts/
│   └── run_demo.py      # CLI for centralized/local-only/federated synthetic demo
├── src/
│   ├── data/            # Synthetic data generation and client partitioning
│   ├── evaluation/      # Metrics, update size, communication cost helpers
│   ├── fl/              # Flower ClientApp, ServerApp, FedAvg simulation path
│   ├── models/          # Frozen backbone + trainable head model
│   ├── training/        # Centralized/local-only train/eval loops
│   └── utils/           # Config and JSON output helpers
└── tests/               # Unit tests for partition, model, metrics
```

## Main Training Modes

- `centralized`: train the lightweight trainable part on pooled data when a centralized baseline is possible.
- `local-only`: train/evaluate the lightweight trainable part for each client/site independently.
- `federated`: train the lightweight trainable part with Flower across simulated clients.

Every serious experiment should compare these modes before adding a new method.

## Run The Demo

Install dependencies in the project virtual environment:

```bash
venv/bin/pip install -e .
```

Run all three modes with a quick CPU config:

```bash
venv/bin/python scripts/run_demo.py --mode all --quick
```

Run one mode:

```bash
venv/bin/python scripts/run_demo.py --mode centralized --quick
venv/bin/python scripts/run_demo.py --mode local-only --quick
venv/bin/python scripts/run_demo.py --mode federated --quick
```

Optional Flower-native run:

```bash
PATH="$PWD/venv/bin:$PATH" venv/bin/flwr run . --run-config "quick=true num-server-rounds=2" --federation-config "num-supernodes=5" --stream
```

Run tests:

```bash
venv/bin/python -m pytest
```

## Demo Output

`scripts/run_demo.py` writes JSON files to `outputs/EXP-001/`:

```text
centralized_metrics.json
local_only_metrics.json
federated_metrics.json
summary.json
```

Each metrics file includes:

- global loss and accuracy;
- per-client loss and accuracy;
- per-client label histogram;
- training time;
- update size in bytes;
- communication cost in bytes.

## Research Direction

The repo is intended to grow toward:

- non-IID partitioning by site, camera, identity, clothing/PPE type, lighting, or domain;
- pretrained backbones with frozen or near-frozen weights;
- trainable head, embedding, adapter, or LoRA modules instead of full-model training;
- aggregation of only selected trainable/update parts when possible;
- edge-aware FL experiments for compute, memory, communication cost, and client availability;
- personalized FL and communication-efficient methods for face and clothing/PPE tasks.

## Data And Artifact Rules

Do not commit large or sensitive artifacts:

- raw camera images/videos;
- face images or private datasets;
- heavy model checkpoints;
- local paper PDFs in `docs/pdfs/`;
- large generated logs/outputs.

`docs/pdfs/` is intentionally ignored. Keep papers local unless there is a clear reason to track a lightweight text summary.
