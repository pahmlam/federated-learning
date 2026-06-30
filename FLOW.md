# Current FL System Flow

This document describes the flow implemented in the repository today. The
current validation workload is PPE object detection, but the main target is a
reusable FL system for camera/vision workloads.

## 1. Current Scope

The implemented system supports:

- PPE detection workload with 8 foreground classes plus background.
- Faster R-CNN MobileNetV3-Large-FPN from TorchVision.
- Frozen backbone + FPN.
- Trainable detection head only:
  - RPN head
  - ROI box head/predictor
- FedAvg over detection-head parameters only.
- Centralized, local-only, and federated simulation modes.
- Flower `ClientApp` / `ServerApp` deployment path.
- Distributed evaluation: each client evaluates on its own validation shard.

A deployment run now writes its own server-side artifacts:

- `deployment_summary.json` (run config, server settings, update size, planned +
  completed communication cost, timing, final distributed-eval metrics, output
  paths, and a `status` of `completed` / `partial` / `failed`),
- `final_head.npz` (the final aggregated detection head, written only when
  Flower exposes final aggregated arrays).

This code path is implemented, unit-tested, and verified end-to-end by the
`outputs/EXP-012-rerun/` deployment smoke.

The system now has a site-side command for the first post-FL handoff step:

- loading `final_head.npz` into a fresh detector,
- evaluating the final global head on one site's local validation shard,
- writing a site-local JSON metrics report.

This command is implemented and unit-tested, but has not yet been run on a real
site against `outputs/EXP-012-rerun/final_head.npz`.

The system does not yet implement the final production inference handoff:

- distributing/exporting that final model back to sites for inference,
- per-client deployment metric files and a server log artifact.

## 2. Main Data Flow

```text
PPE dataset
  -> detection manifest
  -> per-client train/val shards
  -> dataset bundle
  -> model build
  -> train/evaluate
  -> metrics/artifacts
```

Important files:

- `configs/datasets/ppe_detection_exp011_manifest.csv`
- `src/data/detection_manifest.py`
- `src/data/detection_dataset.py`
- `src/data/detection_data.py`
- `src/models/detection_model.py`
- `src/training/detection_trainer.py`

## 3. Manifest And Client Shards

The detection manifest is the source of truth for splitting data into clients.

Flow:

```text
data/ppe/images + data/ppe/voc_labels
  -> scripts/generate_detection_manifest.py
  -> configs/datasets/<manifest>.csv
  -> src.data.detection_data.load_detection_bundle(...)
```

The bundle contains:

- `bundle.clients`: one `DetectionClientData` per site.
- `client.train`: local training dataset for that site.
- `client.val`: local validation dataset for that site.
- `bundle.pooled_train`: pooled training data, used only for centralized
  reference simulation.
- `bundle.pooled_val`: pooled validation data, used only for centralized
  reference simulation.

Privacy model:

- In FL deployment, each client should receive only its own shard.
- The server should not hold raw data.
- Centralized pooled evaluation is only a reference baseline, not the privacy
  preserving FL condition.

## 4. Model Flow

Model construction is in `src/models/detection_model.py`.

Flow:

```text
build_detection_model(...)
  -> load Faster R-CNN MobileNetV3-Large-FPN
  -> replace detection heads for 9 classes
  -> freeze backbone + FPN
  -> expose detection-head parameter get/set helpers
```

The important parameter helpers are:

- `get_detection_head_parameters(model)`
- `set_detection_head_parameters(model, arrays)`
- `detection_trainable_parameter_names(model)`

Only these detection-head parameters are serialized and aggregated by FedAvg.
The full model is not sent as the FL update.

## 5. Local Training And Evaluation

Training and evaluation are implemented in:

- `src/training/detection_trainer.py`

Training flow:

```text
model receives current head parameters
  -> train_detection_head(...)
  -> local train dataloader
  -> optimize trainable detection-head parameters only
  -> return train loss
```

Evaluation flow:

```text
model receives current/global head parameters
  -> evaluate_detection(...)
  -> local validation dataloader
  -> torchmetrics MeanAveragePrecision
  -> return map, map_50, map_75, map_per_class
```

For deployment, only scalar metrics are sent to Flower default aggregation:

- `map`
- `map_50`
- `map_75`
- `num-examples`

`map_per_class` is not sent through the default Flower metric aggregator because
clients can have different class presence, which can produce variable-length
lists and crash default aggregation.

## 6. Simulation Flow

Entry point:

- `scripts/run_detection_sim.py`

Modes:

- `centralized`
- `local-only`
- `federated`
- `all`

### 6.1 Centralized Baseline

Implemented in:

- `src/training/detection_baselines.py`

Flow:

```text
load all client shards into pooled_train/pooled_val
  -> build one model
  -> train detection head on pooled_train
  -> evaluate on pooled_val
  -> evaluate same model on each client.val
  -> write centralized_metrics.json
```

This is a reference only. It breaks the FL privacy model because all client data
is pooled on one machine.

### 6.2 Local-Only Baseline

Implemented in:

- `src/training/detection_baselines.py`

Flow:

```text
for each client:
  build model
  train detection head on client.train only
  evaluate on client.val
weighted average per-client metrics
write local_only_metrics.json
```

No communication happens in this mode.

### 6.3 Federated Simulation

Implemented in:

- `src/fl/detection_federated.py`

Flow:

```text
server initializes global detection-head parameters

for each round:
  for each client:
    build model
    load current global head
    train on client.train
    return updated head + train-set size

  server FedAvg:
    weighted average client heads by train-set size

  distributed evaluation:
    for each client:
      build model
      load aggregated global head
      evaluate on client.val
    weighted average per-client validation metrics

final:
  evaluate final global head on each client.val
  write federated_metrics.json
```

Important point:

The simulation does send the aggregated global head back to each client for
evaluation after aggregation. It does not stop at server aggregation.

Simulation artifacts:

```text
outputs/<EXP-ID>/
  centralized_metrics.json
  local_only_metrics.json
  federated_metrics.json
  summary.json
```

The summary includes run config and resource snapshot. Federated metrics include
history, per-client metrics, weighted global metrics, update size, and estimated
communication cost.

## 7. Flower Deployment Flow

Deployment uses the modern Flower app API.

Entry points:

- `src/fl/detection_serverapp.py`
- `src/fl/detection_clientapp.py`
- `pyproject.toml`

Target topology:

| Role | Machine | Flower role | Data |
| --- | --- | --- | --- |
| Server | Mac local | SuperLink + ServerApp submitter | No raw data |
| Client `site-a` | Ubuntu RTX3060 | SuperNode | Own shard |
| Client `site-b` | Colab #1 | SuperNode | Own shard |
| Client `site-c` | Colab #2 | SuperNode | Own shard |

Current deployment smoke that has passed:

- Mac SuperLink
- 2 Colab SuperNodes
- Tailscale userspace networking + proxychains
- 1 FL round
- train and evaluate completed
- deployment artifacts verified in `outputs/EXP-012-rerun/`

Full 3-site deployment is still pending because Ubuntu RTX3060 was not included
in the successful deployment smoke.

### 7.1 ServerApp Flow

Implemented in:

- `src/fl/detection_serverapp.py`

Flow:

```text
ServerApp starts
  -> read run config / .env values
  -> build initial model
  -> extract initial detection-head arrays
  -> measure head update size + capture trainable param names
  -> configure Flower FedAvg
  -> result = strategy.start(...)
  -> derive status from result (completed | partial | failed)
  -> write deployment_summary.json
  -> save final_head.npz (when result exposes final arrays)
```

Server configuration:

- `fraction_train=1.0`
- `fraction_evaluate=1.0`
- `min_train_nodes=config.num_clients`
- `min_evaluate_nodes=config.num_clients`
- `min_available_nodes=config.num_clients`
- `weighted_by_key="num-examples"`

The server does not load a validation dataset. Evaluation is distributed to
clients.

Artifact writing is wrapped in a `finally` block, so a `deployment_summary.json`
is written even when a run fails. The status is **derived from the result**, not
assumed: `finalize_deployment_artifacts(...)` calls `derive_status(...)` so the
run is recorded honestly as `completed`, `partial`, or `failed`. The artifact
helpers live in `src/fl/deployment_artifacts.py`; `derive_status` /
`finalize_deployment_artifacts` are a network-free seam that is unit-tested
without a real Flower run. The Flower `Result` final arrays are keyed by
`detection_trainable_parameter_names(model)` so the saved `.npz` is named
consistently with the model head.

### 7.2 ClientApp Train Flow

Implemented in:

- `src/fl/detection_clientapp.py`

Flow:

```text
client receives train message with global head arrays
  -> load local config and local shard
  -> build model
  -> set model head from received arrays
  -> train on client.train
  -> return updated head arrays
  -> return train_loss and num-examples
```

The returned arrays are the local updated detection head. Flower FedAvg
aggregates these arrays on the server side.

### 7.3 ClientApp Evaluate Flow

Implemented in:

- `src/fl/detection_clientapp.py`

Flow:

```text
client receives evaluate message with aggregated global head arrays
  -> load local config and local shard
  -> build model
  -> set model head from received arrays
  -> evaluate on client.val
  -> return scalar metrics and num-examples
```

Important point:

The deployment path also sends the aggregated global head back to clients for
evaluation. The current pipeline does not stop immediately after aggregation.

## 8. Configuration Flow

Configuration sources:

```text
code defaults < .env < CLI args
pyproject/Flower defaults < .env < explicit flwr run-config / SuperNode node-config
```

Important files:

- `.env.example`
- `src/utils/env.py`
- `src/utils/detection_config.py`
- `pyproject.toml`

Important environment keys:

- `FL_RUN_ID`
- `FL_OUTPUT_DIR`
- `FL_MANIFEST_PATH`
- `FL_DATA_ROOT`
- `FL_CLIENT_ID`
- `FL_NUM_CLIENTS`
- `FL_DEVICE`

Legacy `FL_DET_*` aliases are still accepted, but new configs should use
`FL_*`.

## 9. Current Artifacts

Simulation artifacts are structured JSON files written by
`scripts/run_detection_sim.py`.

Deployment now writes server-side artifacts from `src/fl/detection_serverapp.py`
(via `src/fl/deployment_artifacts.py`):

```text
outputs/<exp-id>/
  deployment_summary.json   # always written (status completed | partial | failed)
  final_head.npz            # written only when Flower exposes final aggregated arrays
```

`deployment_summary.json` is the deployment equivalent of the simulation
`summary.json`: it records run config, server settings (FedAvg, weighted by
`num-examples`), head update size, communication cost, timing, and the final
distributed-evaluation metrics (`train_loss`, `map`, `map_50`, `map_75`).

`status` reflects what Flower actually produced:

- `completed`: at least the expected rounds finished **and** a final aggregated
  head is available.
- `partial`: some rounds finished but the final head is missing, or fewer than
  the expected rounds completed (e.g. stragglers / early stop).
- `failed`: an exception was raised, no result was returned, or zero rounds
  completed.

Communication cost is reported two ways so partial runs are not over-counted:

- `planned_communication_cost_bytes` — based on the configured `num_rounds`.
- `estimated_completed_communication_cost_bytes` — based on `rounds_completed`.

`final_head.npz` exists **only** when final aggregated arrays are available; a
`failed` run (and a `partial` run whose head never aggregated) writes the summary
but no `.npz`.

Still incomplete:

- Per-client deployment metric files and a server log artifact.
- Future deployment runs should continue journaling from `deployment_summary.json`
  instead of only from logstream.

## 10. What Exists Today

Implemented:

- Manifest generation for PPE detection.
- VOC detection dataset loading.
- Per-client train/val bundles.
- Centralized baseline.
- Local-only baseline.
- Manual FedAvg simulation.
- Flower ClientApp/ServerApp deployment path.
- Distributed evaluation on clients.
- Weighted global metric aggregation.
- Update-size and communication-cost reporting in simulation.
- Two-Colab real deployment smoke.
- Deployment artifacts: server writes `deployment_summary.json` and saves the
  final global head as `final_head.npz`.
- Site-side final-head evaluation: `scripts/evaluate_final_detection_head.py`
  loads `final_head.npz`, evaluates on a local site validation shard, and writes a
  JSON metrics report.

Partially implemented:

- Cross-machine deployment: smoke passed with 2 Colab clients, but full target
  topology with Ubuntu RTX3060 is not complete.
- Deployment robustness: Colab straggler behavior is observed, but dropout,
  checkpoint, and resume behavior are not fully tested.
- Post-FL handoff: site-side final-head evaluation is implemented and unit-tested,
  but not yet run on a real site against `outputs/EXP-012-rerun/final_head.npz`.

Missing:

- Per-client deployment metric files and a server log artifact.
- Export/push final model to clients for inference.
- Site-side image/video inference command for operational use.
- Task abstraction layer for plugging in a new workload without touching the
  detection-specific core.
- Additional aggregation strategies beyond FedAvg.

## 11. Recommended Next Flow To Add

Done so far: the ServerApp writes `deployment_summary.json`, saves the final
global detection head as `final_head.npz`, and a site can run a local final-head
evaluation command.

```text
Flower deployment run completes
  -> write outputs/<EXP-ID>/deployment_summary.json   [done]
  -> save final global detection head (final_head.npz) [done, when arrays exposed]
  -> each site can load final head                     [done, unit-tested]
  -> each site can run local eval command              [done, unit-tested]
  -> verify server artifact writing in real deploy      [done: EXP-012-rerun]
  -> run site eval against real final_head.npz          [next]
```

Remaining artifact/handoff work, in suggested order:

```text
outputs/<EXP-ID>/
  deployment_summary.json   # done
  final_head.npz            # done (when Flower exposes final arrays)
  final_head_site_a_metrics.json  # code exists via site-side command
  final_head_site_b_metrics.json
  final_head_site_c_metrics.json
  server_log.txt            # next
  client_site_a_metrics.json  # next: per-client deployment metric files
  client_site_b_metrics.json
  client_site_c_metrics.json
```

Site-side final-head evaluation command:

```bash
venv/bin/python scripts/evaluate_final_detection_head.py \
  --head-path outputs/<EXP-ID>/final_head.npz \
  --manifest <site-manifest.csv> \
  --root-dir <site-data-root> \
  --client-id <site-id> \
  --output outputs/<EXP-ID>/final_head_<site-id>_metrics.json \
  --device auto
```

Next: run the site-side final-head evaluation command on each real site against
`outputs/EXP-012-rerun/final_head.npz`, then capture per-site metric JSON files.
