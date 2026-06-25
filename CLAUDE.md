This file provides guidance to Claude sessions when working with this repository.

## Repository Overview

This is a Federated Learning research/prototype repository for camera/vision tasks on edge deployments. The target use cases are:

- **PPE (current focus)**: **object detection** of 8 core PPE classes (`helmet, safety-vest, safety-suit, face-mask-medical, gloves, glasses, ear-mufs, face-guard`). This is the active track.
- **Face**: recognition, verification, or embedding adaptation (still pending, not started).

### Current Project Status (read before working on PPE)

- **PPE is now an object-detection task** (Faster R-CNN MobileNetV3-Large-FPN, freeze backbone+FPN, train detection head only). Do **not** treat PPE as image-level classification.
- **EXP-001 → EXP-010 are an archived stage-1 classification baseline** (binary `safe/unsafe` over a proxy `has_core_ppe` image-level label). That track hit a ~0.60 macro-F1 ceiling that EXP-009 attributed mostly to the proxy label. Keep those journals as historical reference; new work is detection.
- **The project has moved from single-machine simulation to real federated deployment** across heterogeneous machines (see "Deployment Topology" below).

This is a **research workspace**, not a production codebase. Build a clean experimental foundation before proposing new FL methods.

Core technical direction:

```text
pretrained backbone
    -> freeze or nearly freeze backbone
    -> train only embedding/head/adapter/LoRA
    -> aggregate lightweight updates when possible
```

Do **not** assume the project is trying to train a full vision model from scratch.

## Key Components

### Research And Planning Docs

- `docs/md/kehoachthuviec.md`: internship/work plan and milestones.
- `docs/md/PLAN.md`: daily checklist from 10/06/2026 to 31/08/2026.
- `docs/md/REPORT.md`: research report about FL for camera/vision use cases.
- `docs/Federated Learning.md`: FL background notes.
- `docs/journal/`: per-experiment journals (EXP-XXX) + registry in `docs/journal/README.md`.

Search these docs first before answering project-specific research questions.

### Repo Skeleton

- `README.md`: root overview and current assumptions.
- `configs/`: future dataset, experiment, and Flower configs.
- `data/`: local data staging. Do not commit large/private data.
- `experiments/`: centralized, local-only, federated, and ablation experiment organization.
- `src/`: future implementation package.
- `outputs/`: local logs, metrics, checkpoints, and reports. Avoid committing large artifacts.
- `tests/`: future smoke/unit tests.

### Flower Quickstart Reference

- `demo/quickstart_numpy/`: existing Flower NumPy quickstart.
- `demo/README.md`: how to run the quickstart.

Keep `demo/` as a learning/reference app unless the user explicitly asks to modify it.

## Critical Patterns

### FL Framework vs Research Method

Flower is the baseline FL framework for server/client orchestration, simulation, and strategy experiments.

Do not start by building a new FL framework. First build:

1. `centralized` baseline;
2. `local-only` baseline;
3. `federated` baseline with FedAvg;
4. non-IID client partitioning;
5. per-client/site evaluation.

**These baselines are already done** (classification track EXP-001→010). The current stage builds the **PPE detection** equivalents of all three modes, then moves to **real Flower deployment** (not just simulation). Keep using the modern Flower `ClientApp`/`ServerApp` API (see `src/fl/client_app.py`, `src/fl/server_app.py`) so the same app runs in both simulation and deployment.

### Parameter-Efficient Training Is Mandatory By Default

All first-stage experiments should use a pretrained backbone and train only lightweight parts:

- embedding layer;
- classifier/task head;
- adapter;
- LoRA module;
- similarly small trainable component.

Avoid full-model fine-tuning or full training from scratch unless the user explicitly changes direction.

**For PPE detection specifically:** freeze the detector **backbone + FPN**, train only the **RPN head + ROI box predictor**. FedAvg aggregates **only the detection head** parameters. Note: the classification track's precompute-embedding OOM trick does **not** apply to detection (detection needs spatial feature maps, not a single pooled vector) — handle memory via small input size / small batch on the GPU clients instead.

### Client Definition

A client = one site / edge box / NVR / local server — never "one camera = one FL client" (cameras may only run inference; training happens on edge/server hardware).

### Deployment Topology (real, not simulation)

The PPE detection track runs as **real cross-machine federated learning** over a Tailscale mesh VPN:

| Role | Machine | Notes |
| :--- | :--- | :--- |
| **Server / SuperLink** (aggregator) | Mac local (M2, CPU) | Light — only aggregates weights. Holds **no data**. |
| **Client #1 — `site-a`** | Ubuntu + RTX3060 (GPU) | SuperNode. Also runs the centralized pooled reference baseline. |
| **Client #2 — `site-b`** | Google Colab account #1 | SuperNode (GPU). |
| **Client #3 — `site-c`** | Google Colab account #2 | SuperNode (GPU). |

Key consequences:

- **Data stays local per client** (each holds only its own shard). The server never sees raw data → use **distributed evaluation**; the global metric is the weighted aggregate of per-client validation, not a server-side pooled eval.
- Networking is **Tailscale** (Mac gets a stable tailnet IP; Colab/Ubuntu dial the SuperLink). Run `--insecure` only because the tailnet is private.
- Colab clients are **stragglers by nature** (session limits, disconnects) — keep rounds short and make FedAvg tolerant of missing nodes.
- Validate in **simulation first** (`flwr run . local-sim`, runnable on the RTX3060), then deploy (`flwr run . deploy`).

### Required Baselines And Metrics

Every serious experiment must compare `centralized` (if possible), `local-only`, and `federated` modes.
You MUST log every experiment to `docs/journal/` using the format in [template.md](file:///Users/phamtunglam/Documents/VNPT/federated-learning/docs/journal/template.md) and register it in [README.md](file:///Users/phamtunglam/Documents/VNPT/federated-learning/docs/journal/README.md).
Report all metrics specified in the template: global/per-client loss and metrics (Acc/F1/mAP), training time, round count, update size, and communication cost.

For **PPE detection**, the primary metric is **mAP@0.5 and mAP@0.5:0.95 with per-class AP** (via `torchmetrics.detection.MeanAveragePrecision`), reported **per-client** plus a weighted global aggregate. The `centralized` pooled baseline is a reference only (it requires gathering all data on one machine — the Ubuntu GPU — which is outside the FL privacy model).

### Data And Privacy

Face and camera data are sensitive. PPE data can expose people, factory layout, and operational processes.

Do not commit:

- raw camera images/videos;
- face images;
- private datasets;
- model checkpoints;
- large logs or generated outputs.

FL does not remove the need for reliable labels. Expect manual labeling, human-in-the-loop feedback, pseudo-label review, or active learning in real workflows.

## Expected Implementation Shape

When implementation begins, prefer this module split:

- `src/data/`: dataset loading, preprocessing, and client partitioning.
- `src/models/`: pretrained backbone wrappers and trainable head/embedding/adapter/LoRA modules.
- `src/training/`: centralized and local-only loops for lightweight trainable parts.
- `src/fl/`: Flower ClientApp, ServerApp, strategy, aggregation, and client selection.
- `src/evaluation/`: metrics, per-client reports, communication cost, update size.
- `src/utils/`: config, logging, seeding, and path helpers.

Keep modules small and testable. Keep configs separate from code. Avoid hardcoded absolute paths.

## Workflow Best Practices

For research/documentation tasks:

- Read relevant docs first.
- Preserve neutral research tone.
- Avoid claiming a method/framework is new before baselines and bottlenecks are demonstrated.
- Distinguish clearly between FL framework, experiment repo, and research method.

For implementation tasks:

- Start from a minimal runnable pipeline.
- Implement baselines before method improvements.
- Add smoke tests early.
- Verify at least one quick run for the mode being changed.
- Do not modify unrelated docs or `demo/` unless asked.
- Use `rg` for search and `apply_patch` for edits.

For planning tasks:

- Keep plans decision-complete.
- Tie work back to `docs/md/PLAN.md` and `docs/md/kehoachthuviec.md`.
- Prefer practical next steps over broad research speculation.

## Current Commands

Run the test suite:

```bash
venv/bin/python -m pytest
```

Classification track (archived stage-1 baseline, reuses precomputed embeddings):

```bash
venv/bin/python scripts/run_embedding_demo.py --mode all \
  --artifact data/processed/ppe_real_embeddings_exp006.npz --profile oom-safe ...
```

Detection track (current) — once built, runs via the modern Flower API:

```bash
flwr run . local-sim   # validate in simulation (runnable on the RTX3060)
flwr run . deploy       # real 3-node deployment (SuperLink on Mac, SuperNodes over Tailscale)
```

Flower quickstart reference (do not modify unless asked): `cd demo && flwr run .`

## Documentation Standards

- Keep docs focused and concise.
- Prefer relative links between local docs.
- Use tables for structured comparisons.
- Keep heading levels hierarchical.
