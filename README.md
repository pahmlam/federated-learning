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
- The full target deployment is still: Mac server + Ubuntu RTX3060 + 2 Colab
  clients.

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
│   ├── fl/              # Flower detection ClientApp/ServerApp
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
FL_NUM_CLIENTS=3

# Ubuntu site-a
FL_CLIENT_ID=site-a
FL_MANIFEST_PATH=data/ppe_site_a/manifest.csv
FL_DATA_ROOT=data/ppe_site_a
FL_DEVICE=auto

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
| Client `site-a` | Ubuntu + RTX3060 | SuperNode | Own PPE shard |
| Client `site-b` | Google Colab #1 | SuperNode | Own PPE shard |
| Client `site-c` | Google Colab #2 | SuperNode | Own PPE shard |

Networking is through Tailscale. `--insecure` is used only because the machines
are inside a private tailnet.

### 1. Prepare Flower federation config

Flower needs a deployment federation named `deploy`. If your local Flower config
does not already define it, add or uncomment the following in `pyproject.toml`
or your Flower configuration file:

```toml
[tool.flwr.federations]
default = "local-sim"

[tool.flwr.federations.local-sim]
options.num-supernodes = 3
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

This creates:

```text
outputs/fl-deploy/shards/site-a.zip
outputs/fl-deploy/shards/site-b.zip
outputs/fl-deploy/shards/site-c.zip
```

Copy each zip to its matching client and unzip it into a local folder:

```bash
mkdir -p data/ppe_site_a
unzip site-a.zip -d data/ppe_site_a
```

After unzip, each client folder must contain:

```text
data/ppe_site_a/manifest.csv
data/ppe_site_a/images/...
data/ppe_site_a/voc_labels/...
```

Use the matching folder/name for `site-b` and `site-c`.

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

### 4. Start an Ubuntu SuperNode

On Ubuntu, after joining the same tailnet and installing dependencies:

```bash
python -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt
venv/bin/pip install -e .
```

Start `site-a`:

```bash
venv/bin/flower-supernode \
  --insecure \
  --superlink 100.100.58.9:9092
```

If Ubuntu has the full dataset instead of an exported shard, set
`FL_MANIFEST_PATH` and `FL_DATA_ROOT` in `.env`, or override with
`--node-config`.

### 5. Start a Colab SuperNode

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
  --superlink 100.100.58.9:9092
```

For the second Colab, use a different hostname and `site-c` paths:

```bash
proxychains4 venv/bin/flower-supernode \
  --insecure \
  --superlink 100.100.58.9:9092
```

### 6. Submit a deployment run

From the Mac, in the repo root:

```bash
venv/bin/flwr run . deploy --stream \
  --run-config 'num-clients=3'
```

For the 2-Colab smoke setup, use `num-clients=2` and start only `site-b` and
`site-c`:

```bash
venv/bin/flwr run . deploy --stream \
  --run-config 'num-clients=2'
```

Inspect runs:

```bash
venv/bin/flwr list deploy
venv/bin/flwr log <RUN_ID> deploy --show
```

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
