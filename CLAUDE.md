# AGENTS.md

This file provides guidance to Codex sessions when working with this repository.

## Repository Overview

This is a Federated Learning research/prototype repository for camera/vision tasks on edge deployments. The first target use cases are:

- **Face**: recognition, verification, or embedding adaptation.
- **Clothing/PPE**: uniform, helmet, reflective vest, mask, or PPE compliance.

This is currently a **skeleton and research workspace**, not a production codebase. The goal is to build a clean experimental foundation before proposing new FL methods.

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

- `docs/kehoachthuviec.md`: internship/work plan and milestones.
- `docs/PLAN.md`: daily checklist from 10/06/2026 to 31/08/2026.
- `docs/REPORT.md`: research report about FL for camera/vision use cases.
- `docs/Federated Learning.md`: FL background notes.
- `docs/tonghoppaper.md`: extracted paper notes.

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

Only after baselines work should the project explore method improvements.

### Parameter-Efficient Training Is Mandatory By Default

All first-stage experiments should use a pretrained backbone and train only lightweight parts:

- embedding layer;
- classifier/task head;
- adapter;
- LoRA module;
- similarly small trainable component.

Avoid full-model fine-tuning or full training from scratch unless the user explicitly changes direction.

### Client Definition

In this repo, a client should usually mean:

- one site;
- a camera cluster;
- an edge box;
- an NVR;
- a local server.

Do not default to "one camera = one FL client". Cameras may only run inference; local training is more realistically done on edge/NVR/local server hardware.

### Required Baselines And Metrics

Every serious experiment must compare `centralized` (if possible), `local-only`, and `federated` modes.
You MUST log every experiment to `docs/journal/` using the format in [template.md](file:///Users/phamtunglam/Documents/VNPT/federated-learning/docs/journal/template.md) and register it in [README.md](file:///Users/phamtunglam/Documents/VNPT/federated-learning/docs/journal/README.md).
Report all metrics specified in the template: global/per-client loss and metrics (Acc/F1/mAP), training time, round count, update size, and communication cost.

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
- Tie work back to `docs/PLAN.md` and `docs/kehoachthuviec.md`.
- Prefer practical next steps over broad research speculation.

## Current Commands

There is no root package or test suite yet.

Existing Flower quickstart reference:

```bash
cd demo
flwr run .
```

Useful inspection commands:

```bash
find . -maxdepth 3 -type f | sort
rg "Flower|FedAvg|embedding|adapter|LoRA|face|PPE" .
```

## Documentation Standards

- Keep docs focused and concise.
- Prefer relative links between local docs.
- Use tables for structured comparisons.
- Keep heading levels hierarchical.
- If editing `AGENTS.md`, keep it practical and below roughly 200 lines.

## Answering User Questions

The user is learning FL and Flower. Good answers should:

- explain the real workflow, not just theory;
- say which parts are automated and which parts need human labels;
- avoid implying Flower solves labeling, privacy, edge compute, or novelty by itself;
- connect answers back to face/PPE, edge clients, and parameter-efficient training;
- recommend baseline-first work before proposing method changes.
