# FL Camera Research

This repository is a Federated Learning research and demo workspace for
camera/vision tasks on edge deployments.

Current test workload:

- **PPE object detection** with 8 core classes:
  `helmet`, `safety-vest`, `safety-suit`, `face-mask-medical`, `gloves`,
  `glasses`, `ear-mufs`, `face-guard`.
- Model: `fasterrcnn_mobilenet_v3_large_fpn`.
- Training policy: freeze backbone + FPN, train only the detection head
  (RPN head + ROI box predictor).
- Federated policy: FedAvg aggregates only detection-head parameters.
- Metric: mAP@0.5:0.95, mAP@0.5, mAP@0.75, reported per client/site and as a
  weighted distributed aggregate.

The broader goal is to build a reusable FL system first. PPE detection is the
current workload used to validate orchestration, data partitioning, distributed
evaluation, and communication/runtime behavior before going deeper on model
quality or other tasks.

Archived workload:

- EXP-001 to EXP-010 are stage-1 PPE classification baselines over a proxy
  image-level `safe/unsafe` label. They are kept for history, but the active PPE
  work is detection.

Flower is used as the FL orchestration framework. The repo uses the modern
Flower `ServerApp`/`ClientApp` API in:

```text
src/fl/detection_serverapp.py
src/fl/detection_clientapp.py
```

The first workload abstraction seam is in place:

```text
src/fl/task.py             # FederatedTask protocol + RoundOutput
src/fl/detection_task.py   # PPE detection implementation
src/fl/embedding_classification_task.py
                            # lightweight embedding classification implementation
```

`DetectionTask` owns the active PPE model/data/train/evaluate wiring used by the
Flower apps. `EmbeddingClassificationTask` reuses the stage-1 embedding-head
stack as a second lightweight workload, proving the seam is not detection-only.
The reusable orchestration shape is intentionally small: add task
implementations before adding registries or dynamic plugin loading.

## Current Status

Detection simulation is working.

- EXP-011 ran on RTX3060 with 3 simulated sites and real PPE images.
- Centralized reference: mAP `0.1179`, mAP50 `0.2755`.
- Local-only average: mAP `0.0657`, mAP50 `0.1601`.
- Federated head-only FedAvg: mAP `0.0951`, mAP50 `0.2351`.
- Detection-head update size is large: about `55-58 MB` per client per round.

Cross-machine deployment smoke is working.

- EXP-012-smoke ran with Mac SuperLink + 2 Colab SuperNodes over Tailscale
  userspace networking.
- 1 round completed end-to-end with distributed evaluation.
- The active deployment target is now Mac server + 2 Colab clients. Ubuntu
  RTX3060 / `site-a` is retired from deployment because it cannot be run/joined
  under the current access constraints.

Experiment notes are tracked in:

```text
docs/journal/README.md
```

## Directory Structure

```text
.
├── configs/
│   └── datasets/        # Detection manifests and dataset configs
├── data/
│   └── ppe/             # Local PPE dataset materialized by DVC, gitignored
├── demo/                # Flower NumPy quickstart reference, do not modify by default
├── docs/
│   ├── journal/         # Single experiment registry and reports
│   └── md/              # Planning and research docs
├── outputs/             # Local metrics, logs, exported shards, checkpoints
├── scripts/
│   ├── generate_detection_manifest.py
│   ├── run_detection_sim.py
│   └── export_detection_subset.py
├── src/
│   ├── data/            # Detection dataset, manifest, bundle loading
│   ├── evaluation/      # Metric/update-size helpers
│   ├── fl/              # Flower apps plus task abstraction seam
│   ├── models/          # Faster R-CNN setup and head parameter helpers
│   ├── training/        # Detection baselines and trainer
│   └── utils/           # Config, IO, seed helpers
└── tests/
```

## Setup

On every machine that runs the app, clone or pull the same repo revision, then
create or reuse the project virtual environment:

```bash
python -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt
venv/bin/pip install -e .
```

Create a local `.env` on every machine:

```bash
cp .env.example .env
```

Edit `.env` for each machine. The env file is intended to describe the FL
runtime and the current workload inputs, not a single experiment. PPE detection
is only the current test workload. Config precedence is:

```text
code defaults < .env < CLI args
pyproject/Flower defaults < .env < explicit flwr --run-config / SuperNode node-config
```

`FL_DET_*` names are still accepted as legacy aliases, but new machine configs
should use `FL_*`.

Common examples:

```bash
# Mac server / local simulation
FL_RUN_ID=fl-system-smoke
FL_OUTPUT_DIR=outputs/fl-system-smoke
FL_MANIFEST_PATH=configs/datasets/ppe_detection_exp011_manifest.csv
FL_DATA_ROOT=data/ppe
FL_NUM_CLIENTS=2

# Colab site-b/site-c: change client id and shard path per notebook
FL_CLIENT_ID=site-b
FL_MANIFEST_PATH=data/ppe_site_b/manifest.csv
FL_DATA_ROOT=data/ppe_site_b
```

Run tests:

```bash
venv/bin/python -m pytest
```

If this machine needs the full PPE dataset and has access to the configured DVC
remote:

```bash
venv/bin/dvc pull
```

The dataset should materialize at:

```text
data/ppe/
```

Do not commit raw images, annotations, checkpoints, or large outputs.

## Local Detection Simulation

Run this before cross-machine deployment.

```bash
venv/bin/python scripts/run_detection_sim.py --mode all --device cuda
```

For a tiny CPU smoke, override only what differs from `.env`:

```bash
venv/bin/python scripts/run_detection_sim.py \
  --mode federated \
  --output-dir outputs/EXP-011-smoke \
  --exp-id EXP-011-smoke \
  --batch-size 1 \
  --local-epochs 1 \
  --num-rounds 1 \
  --device cpu \
  --num-workers 0
```

## Cross-Machine Deployment

### Target Topology

| Role | Machine | Flower role | Data |
| --- | --- | --- | --- |
| Server | Mac local | SuperLink + ServerApp submitter | No raw data |
| Client `site-b` | Google Colab #1 | SuperNode | Own PPE shard |
| Client `site-c` | Google Colab #2 | SuperNode | Own PPE shard |

Networking is through Tailscale userspace networking and proxychains for Colab.
`--insecure` is used only because the machines are inside a private tailnet.
`site-a` / Ubuntu RTX3060 is not part of the active deployment target.

### 1. Prepare Flower federation config

Flower needs a deployment federation named `deploy`. If your local Flower config
does not already define it, add or uncomment the following in `pyproject.toml`
or your Flower configuration file:

```toml
[tool.flwr.federations]
default = "local-sim"

[tool.flwr.federations.local-sim]
options.num-supernodes = 2
options.backend.client-resources.num-cpus = 1
options.backend.client-resources.num-gpus = 0

[tool.flwr.federations.deploy]
address = "127.0.0.1:9093"
insecure = true
```

The `deploy.address` points to the SuperLink Control API from the machine that
runs `flwr run . deploy`. If submitting from the Mac where SuperLink runs,
`127.0.0.1:9093` is correct.

### 2. Export per-site shards

Run this on the machine that has `data/ppe`:

```bash
venv/bin/python scripts/export_detection_subset.py \
  --output-dir outputs/fl-deploy/shards \
  --overwrite
```

This may create all sites present in the manifest:

```text
outputs/fl-deploy/shards/site-a.zip
outputs/fl-deploy/shards/site-b.zip
outputs/fl-deploy/shards/site-c.zip
```

For the active deployment, copy only `site-b.zip` and `site-c.zip` to the two
Colab clients and unzip each into a local folder:

```bash
mkdir -p data/ppe_site_b
unzip site-b.zip -d data/ppe_site_b
```

After unzip, each client folder must contain:

```text
data/ppe_site_b/manifest.csv
data/ppe_site_b/images/...
data/ppe_site_b/voc_labels/...
```

Use the matching folder/name for `site-c`.

### 3. Start the Mac SuperLink

On the Mac server:

```bash
venv/bin/flower-superlink \
  --insecure \
  --fleet-api-address 0.0.0.0:9092 \
  --control-api-address 0.0.0.0:9093
```

Get the Mac Tailscale IP:

```bash
tailscale ip -4
```

In the examples below, replace `100.100.58.9` with that IP.

### 4. Start Colab SuperNodes

Colab usually cannot create a real Tailscale TUN interface. Use Tailscale
userspace networking with a SOCKS proxy.

Install project dependencies in Colab, then start Tailscale userspace:

```bash
sudo tailscaled \
  --state=mem: \
  --tun=userspace-networking \
  --socks5-server=127.0.0.1:1055 \
  > /tmp/tailscaled.log 2>&1 &

sudo tailscale up --authkey=<TAILSCALE_AUTH_KEY> --hostname=colab-site-b
```

Configure proxychains:

```bash
printf "strict_chain\nproxy_dns\n[ProxyList]\nsocks5 127.0.0.1 1055\n" \
  | sudo tee /etc/proxychains4.conf
```

Start `site-b`:

```bash
proxychains4 venv/bin/flower-supernode \
  --insecure \
  --superlink 100.100.58.9:9092 \
  --node-config 'client-id="site-b" manifest-path="data/ppe_site_b/manifest.csv" root-dir="data/ppe_site_b"'
```

For the second Colab, use a different hostname and `site-c` paths:

```bash
proxychains4 venv/bin/flower-supernode \
  --insecure \
  --superlink 100.100.58.9:9092 \
  --node-config 'client-id="site-c" manifest-path="data/ppe_site_c/manifest.csv" root-dir="data/ppe_site_c"'
```

### 5. Submit a deployment run

From the Mac, in the repo root:

```bash
venv/bin/flwr run . deploy --stream \
  --run-config 'num-clients=2'
```

The deployment baseline is strict by default: train, evaluate, and availability
all require `num-clients` nodes. For a Colab dropout robustness smoke only, you
can relax the minimums:

```bash
venv/bin/flwr run . deploy --stream \
  --run-config 'num-clients=2 min-train-nodes=1 min-evaluate-nodes=1 min-available-nodes=1'
```

Relaxed-min-node runs test system behavior under stragglers; do not compare
their mAP directly with strict two-client baseline runs.

Inspect runs:

```bash
venv/bin/flwr list deploy
venv/bin/flwr log <RUN_ID> deploy --show
```

### 6. Run site-side inference

After a deployment run writes `final_head.npz`, each site can load that final
head and run detection on local unlabeled images. The server still does not need
raw images.

```bash
venv/bin/python scripts/run_detection_inference.py \
  --head-path outputs/EXP-012-rerun/final_head.npz \
  --input-dir <site-local-images> \
  --output-dir outputs/EXP-012-rerun/inference_site_b \
  --device auto \
  --score-threshold 0.5 \
  --save-images
```

The script writes one JSON file per image with class labels, scores, and boxes
in original-image coordinates. With `--save-images`, it also writes annotated
preview images. These outputs are generated artifacts and should stay out of
Git.

### Deployment Notes

- The server does not load raw data and does not run pooled evaluation.
- Deployment metrics are distributed client metrics aggregated by
  `num-examples`.
- The ClientApp reports scalar metrics only (`map`, `map_50`, `map_75`) to avoid
  Flower default aggregation failures with variable-length per-class AP lists.
- Per-class AP should be analyzed from simulation outputs or with a custom
  deployment metric aggregator.
- Detection-head upload/download is large. Start with `num-rounds=1` before
  increasing rounds.
- If networking is slow through Colab userspace SOCKS, debug with smaller
  shards or `pretrained=false` first.
- `deployment_summary.json` records the effective `min_train_nodes`,
  `min_evaluate_nodes`, and `min_available_nodes` thresholds used by ServerApp.
- Structured deployment artifacts are written under `outputs/<EXP-ID>/`:
  `deployment_summary.json`, `round_metrics.json`, and `final_head.npz` when a
  final aggregated head is available.
- Raw Flower/SuperLink/SuperNode text logs belong under `outputs/logs/<EXP-ID>/`.
  The deployment summary records expected log paths and a `flwr log <RUN_ID>
  deploy --show` hint when the run id is available.
- After a deployment run, capture the Flower logstream with:
  ```bash
  venv/bin/python scripts/capture_flower_logs.py \
    --exp-id <EXP-ID> \
    --run-id <FLOWER_RUN_ID> \
    --print-commands
  ```
  This writes `flower_run_log.txt` and `log_capture_summary.json` under
  `outputs/logs/<EXP-ID>/`. SuperLink/SuperNode terminal logs must still be
  redirected with `tee` during the run if you need those raw files.
- `round_metrics.json` contains Flower weighted aggregates by round. Flower
  `Result` does not expose actual client identities in this artifact path, so
  participation fields are recorded as `null` rather than guessed.

## Workload Extension Point

The reusable FL seam is `FederatedTask` in `src/fl/task.py`. A workload provides:

- client context loading,
- model construction,
- global/trainable array get/set,
- one local train round,
- one local evaluate round.

PPE detection implements this as `DetectionTask` in `src/fl/detection_task.py`.
The Flower ClientApp now drives detection through that task wrapper. Server-side
artifact writing, `final_head.npz` naming, and detection simulation modules
remain detection-specific by design.

`EmbeddingClassificationTask` in `src/fl/embedding_classification_task.py` is the
second implementation. It reuses the older embedding-head classification stack
and runs on in-memory embedding tensors, so it is a low-cost stand-in for where a
future face-embedding workload would slot in. It is not wired into the active
Flower deployment path yet; PPE detection remains the deployed workload.

A future face-recognition workload should add a concrete task implementation,
such as `FaceTask`, with the same protocol before changing orchestration code.

## Legacy Synthetic Demo

The old synthetic demo is still available as a framework smoke test:

```bash
venv/bin/python scripts/run_demo.py --mode all --quick
venv/bin/python scripts/run_demo.py --mode all --profile oom-safe
```

Optional Flower-native synthetic run:

```bash
PATH="$PWD/venv/bin:$PATH" venv/bin/flwr run . \
  --run-config "quick=true num-server-rounds=2" \
  --federation-config "num-supernodes=5" \
  --stream
```

## Data And Artifact Rules

Do not commit large or sensitive artifacts:

- raw camera images/videos;
- face images or private datasets;
- PPE images or full annotation exports;
- model checkpoints;
- large logs or generated outputs;
- local paper PDFs in `docs/pdfs/`.

Use DVC or local transfer for datasets. Keep `docs/journal/README.md` as the
single experiment journal, and store raw artifacts under `outputs/EXP-XXX/`.
