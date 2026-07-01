This file provides guidance to Claude sessions when working with this repository.

## Collaboration Role

Claude Code is the implementation worker for this repository.

- Own code and test changes requested by the user or by a Codex handoff prompt.
- Keep implementation changes scoped to the requested feature/fix.
- Run relevant tests and report exactly what passed or failed.
- Do **not** modify Markdown/documentation files by default.
- If behavior changes require docs updates, mention the required updates in the final summary instead of editing docs.
- Only edit docs (`*.md`, journal, planning files, README files) when the user explicitly asks Claude Code to do documentation work.

Codex is the reviewer/planner/documentation partner:

- Codex reviews Claude Code's implementation work.
- Codex owns `PLAN.md`, `FLOW.md`, README/doc updates, journal entries, and experiment interpretation.
- Codex will update documentation after code behavior is reviewed.

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
- `docs/journal/README.md`: single-file experiment journal grouped by date; append all EXP/WIP entries here
- `docs/engineering`: single-file report all error when run and how to fix, append all here

Search these docs first before answering project-specific research questions.

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

The PPE detection track runs as **real cross-machine federated learning** over Tailscale userspace networking:

| Role | Machine | Notes |
| :--- | :--- | :--- |
| **Server / SuperLink** (aggregator) | Mac local (M2, CPU) | Light — only aggregates weights. Holds **no data**. |
| **Client #1 — `site-b`** | Google Colab account #1 | SuperNode (GPU). |
| **Client #2 — `site-c`** | Google Colab account #2 | SuperNode (GPU). |

`site-a` / Ubuntu RTX3060 is **not part of the active deployment target** anymore because it cannot be run/joined under the current access constraints. Keep old `site-a` results only as historical simulation context.

Key consequences:

- **Data stays local per client** (each holds only its own shard). The server never sees raw data → use **distributed evaluation**; the global metric is the weighted aggregate of per-client validation, not a server-side pooled eval.
- Networking is **Tailscale userspace + proxychains** for Colab clients (Mac gets a stable tailnet IP; Colab dials the SuperLink through SOCKS). Run `--insecure` only because the tailnet is private.
- Colab clients are **stragglers by nature** (session limits, disconnects) — keep rounds short and make FedAvg tolerant of missing nodes.
- Validate in **simulation first** (`flwr run . local-sim` or `scripts/run_detection_sim.py`), then deploy (`flwr run . deploy`).

### Required Baselines And Metrics

Every serious experiment must compare `centralized` (if possible), `local-only`, and `federated` modes.
Experiment journal updates are documentation work and are Codex-owned by default. Claude Code should not edit the journal unless explicitly asked. When Claude Code runs an experiment, report the metrics/artifact paths in the final summary so Codex can append `docs/journal/README.md`.
Required metrics: global/per-client loss and metrics (Acc/F1/mAP), training time, round count, update size, and communication cost.

For **PPE detection**, the primary metric is **mAP@0.5 and mAP@0.5:0.95 with per-class AP** (via `torchmetrics.detection.MeanAveragePrecision`), reported **per-client** plus a weighted global aggregate. The `centralized` pooled baseline is a reference only (it requires gathering all data on one machine, which is outside the FL privacy model).

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
- Do not perform documentation edits unless explicitly requested. Report documentation changes needed in your final summary.

For implementation tasks:

- Start from a minimal runnable pipeline.
- Implement baselines before method improvements.
- Add smoke tests early.
- Verify at least one quick run for the mode being changed.
- Do not modify docs, unrelated files, or `demo/` unless asked.
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

Detection track (current) — once built, runs via the modern Flower API:

```bash
flwr run . local-sim   # validate in simulation
flwr run . deploy       # real 2-Colab deployment (SuperLink on Mac, SuperNodes over Tailscale userspace)
```

Flower quickstart reference (do not modify unless asked): `cd demo && flwr run .`
