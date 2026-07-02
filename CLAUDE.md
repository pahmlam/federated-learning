This file gives Claude Code the minimum repository orientation needed for coding work.
Detailed status, roadmap, and experiment history live in the docs listed below.

## Role

Claude Code is the implementation worker for this repository.

- Own code and test changes requested by the user or by a Codex handoff prompt.
- Keep changes scoped to the requested feature/fix.
- Run relevant tests and report exactly what passed or failed.
- Do not modify Markdown/documentation files by default.
- If behavior changes require docs updates, mention the required updates in the final summary.
- Only edit docs (`*.md`, journal, planning files, README files) when the user explicitly asks Claude Code to do documentation work.

Codex is the reviewer, planner, experiment interpreter, and documentation owner.

- Codex reviews implementation work.
- Codex owns `README.md`, `FLOW.md`, `docs/md/PLAN.md`, `docs/journal/README.md`, and experiment interpretation.
- Codex applies documentation updates after code behavior is reviewed.

## Read First

- `README.md`: current repository overview, setup, commands, and latest usable state.
- `FLOW.md`: implemented system flow and architecture.
- `docs/md/PLAN.md`: roadmap and next priorities.
- `docs/journal/README.md`: single experiment journal.
- `docs/engineering`: run errors and fixes.
- `.env.example`: supported environment variables.

Use these docs as the source of truth instead of duplicating detailed progress in this file.

## Repository Shape

```text
configs/              Dataset and experiment config files
data/                 Local datasets, gitignored/DVC-managed
demo/                 Flower quickstart reference; do not modify by default
docs/                 Planning, research notes, journal, engineering notes
notebooks/            Colab SuperNode notebooks for site-b/site-c
outputs/              Generated metrics, logs, shards, checkpoints; do not commit
scripts/              CLI helpers for manifests, simulation, deployment logs, inference
src/data/             Dataset loading, manifests, client bundles
src/models/           Model builders and trainable-parameter helpers
src/training/         Centralized/local training and evaluation loops
src/fl/               Flower apps, strategies, task seam, EdgeProfile
src/evaluation/       Metrics, parameter/update-size helpers
src/utils/            Config, env, IO, resources, seeding helpers
tests/                Unit and smoke tests
```

## Coding Context

- This is a Federated Learning research/prototype repo for camera/vision edge deployments.
- Active workload: PPE **object detection**, not image classification.
- Active detector: Faster R-CNN MobileNetV3-Large-FPN with frozen backbone + FPN.
- Trainable/aggregated part: detection head only.
- FL framework: Flower modern `ClientApp`/`ServerApp`.
- Main task seam: `src/fl/task.py`, implemented by `DetectionTask` and `EmbeddingClassificationTask`.
- A client means one site/edge box/NVR/local server, not one camera.
- Server should not hold raw client data; deployment evaluation is distributed and weighted by client examples.
- Edge device emulation lives in `src/fl/edge_profile.py` and should stay app/task-level unless a plan says otherwise.

## Implementation Guardrails

- Do not train full vision models from scratch unless the user explicitly changes direction.
- Prefer pretrained backbone + lightweight head/adapter/LoRA style updates.
- Do not build a new FL framework; extend the Flower-based system.
- Keep modules small, testable, and aligned with the existing directory boundaries.
- Avoid hardcoded absolute paths.
- Do not modify docs, unrelated files, or `demo/` unless asked.
- Do not commit raw images/videos, private datasets, checkpoints, large logs, or generated outputs.
- Use `rg` for search and `apply_patch` for manual edits.

## Common Commands

```bash
venv/bin/python -m pytest
venv/bin/python scripts/run_detection_sim.py --mode all --device cuda
venv/bin/flwr run . local-sim --stream
venv/bin/flwr run . deploy --stream
```
