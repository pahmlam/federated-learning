# WIP — PPE Detection Pivot + Federated Deployment (build progress)

> **Đây KHÔNG phải EXP journal** (chưa có metric run). Đây là **log tiến độ build** để resume sau khi
> context compact. EXP-011 journal sẽ tạo riêng khi có kết quả sim run thật.

- **Ngày:** 2026-06-25
- **Trạng thái:** Phase A scaffold gần xong; deployment scaffold đã thêm, còn EXP-011 GPU run
- **Plan đã duyệt:** `/Users/phamtunglam/.claude/plans/t-i-ngh-m-y-mac-mighty-stearns.md`

---

## Context
- **Request:** PPE được chỉ định lại là **object detection** (không còn classification). Build lại track
  detection + chuyển sang **federated deployment thật 3 client**.
- **Goal:** dựng pipeline detection FL (freeze backbone, train head, FedAvg head), validate trong
  simulation, rồi deploy thật → đo mAP per-client.

## Quyết định đã chốt (durable)
- **Detector:** `fasterrcnn_mobilenet_v3_large_fpn`, freeze **backbone+FPN**, chỉ train RPN head + ROI heads. FedAvg chỉ aggregate head.
- **Lớp:** 8 PPE core (`helmet, safety-vest, safety-suit, face-mask-medical, gloves, glasses, ear-mufs, face-guard`) + background = **9 lớp**. Class order = `DEFAULT_CORE_PPE` (single source of truth).
- **Topology:** **Mac local = server (SuperLink, aggregator, không giữ data)**; 3 client = **Ubuntu RTX3060 (site-a)** + **Colab#1 (site-b)** + **Colab#2 (site-c)**. Distributed evaluation (server không có pooled data).
- **Mạng (Pha B):** **Tailscale** (mesh VPN), `--insecure` trong tailnet riêng.
- **Metric:** mAP@0.5 + mAP@0.5:0.95 + per-class AP (torchmetrics), per-client.
- **EXP-001→010 = stage-1 classification đã archive** (trần ~0.60 do nhãn proxy; EXP-009 L2-norm là kết luận chính).

## Đã xong
- **Phase 0 (docs):** `CLAUDE.md` (tracked) + `docs/md/PLAN.md` (local, gitignored) cập nhật đầy đủ. PLAN §4.4.1 có checklist Phase 0/A/B.
- **Phase A bước 1–3 (core pipeline, 83 test pass toàn repo):**
  - `src/utils/detection_config.py` — `DetectionConfig`, `ppe_label_to_index()`, `NUM_DETECTION_CLASSES=9`.
  - `src/data/detection_dataset.py` — `PPEDetectionDataset`, `read_voc_objects`, `voc_to_target`, `detection_collate_fn`, `DetectionRecord`.
  - `src/models/detection_model.py` — `build_detection_model(pretrained=)`, `get/set_detection_head_parameters`, `detection_trainable_parameter_names`, `resolve_device`.
  - `src/training/detection_trainer.py` — `train_detection_head`, `evaluate_detection` (torchmetrics mAP), `train_one_epoch`.
  - Tests: `tests/test_detection_{config,dataset,model,trainer}.py`.
- **Deps cài + khai báo:** `torchmetrics[detection]` 1.9.0, `pycocotools` 2.0.11 (trong `venv`, đã thêm vào `pyproject.toml`).
- **Phase A bước 4–7 (data + baselines + FL sim + scripts, 94 test pass toàn repo):**
  - `src/data/detection_manifest.py` — `collect_detection_samples`, `generate_detection_manifest_rows` (non-IID PPE-skew, leakage-free), write/summarize.
  - `src/data/detection_data.py` — `load_detection_bundle` → `DetectionDatasetBundle` (per-client train/val + pooled + label histogram).
  - `src/training/detection_baselines.py` — `run_detection_centralized`, `run_detection_local_only` (mAP per-client + weighted global).
  - `src/fl/detection_federated.py` — `run_detection_federated` (FedAvg head thủ công, distributed-eval) + `federated_average`.
  - `scripts/generate_detection_manifest.py`, `scripts/run_detection_sim.py`.
  - Tests: `tests/test_detection_{manifest,data,baselines,federated}.py`.
- **Rủi ro retired:** frozen Faster R-CNN train được (loss+backward); mAP end-to-end; head serialize ổn định; **FedAvg detection chạy end-to-end trong sim**; **smoke CPU 12 ảnh THẬT OK** (`outputs/EXP-011-smoke/`, mode federated, mAP=0 đúng kỳ vọng vì random init + 1 round).
- **Phát hiện:** detection head update_size ≈ **58 MB** (so 4 KB head classification) → comm cost rất lớn; cần cân nhắc ở deployment (round ngắn, ít round).
- **Deployment scaffold added:**
  - `src/fl/detection_clientapp.py` — Flower ClientApp detection; hỗ trợ single-shard manifest, `client-id`, `partition-id`, node-level `manifest-path`/`root-dir` override.
  - `src/fl/detection_serverapp.py` — Flower ServerApp FedAvg head-only; server không evaluate pooled data, global metric lấy từ weighted distributed evaluation.
  - `scripts/export_detection_subset.py` — export từng site thành folder + zip (`manifest.csv`, `images/`, `voc_labels/`, `summary.json`).
  - `pyproject.toml` — root Flower app trỏ sang detection app; thêm `[tool.flwr.federations.local-sim]` và `[tool.flwr.federations.deploy]`.
  - Tests mới: `tests/test_detection_flower_apps.py`, `tests/test_export_detection_subset.py`.
  - Verification: `101 passed`; Flower local-sim smoke trên tiny `/private/tmp` manifest pass với `pretrained=false`, CPU, 2 SuperNodes. Lưu ý Flower 1.30 dùng simulation override schema mới (`num-supernodes`, `client-resources-num-cpus`, `client-resources-num-gpus`) và tự migrate legacy federation config khi chạy.

## Còn lại
- [x] `src/fl/detection_clientapp.py` + `detection_serverapp.py` — modern Flower API, load shard theo `node_config(manifest-path/root-dir/data-root, client-id/partition-id)`. **Cho Phase B (deployment).**
- [x] `scripts/export_detection_subset.py` — đóng gói từng shard cho client.
- [x] Thêm deps vào `pyproject.toml`: `torchmetrics[detection]`, `pycocotools`.
- [ ] **Real sim RTX3060** (`--device cuda`, `pretrained=True`, ~300 ảnh/site, vài round) → **journal EXP-011** + registry (cột mAP). *(Không chạy được trên Mac CPU — cần GPU.)*

## Phase B (sau EXP-011)
- Tailscale up trên 4 máy; `[tool.flwr.federations.local-sim|deploy]` trong `pyproject.toml`; SuperLink trên Mac; export 3 shard; notebook supernode Colab; `flwr run . deploy` → **EXP-012**.

## Cách resume nhanh
```bash
venv/bin/python -m pytest -q          # phải 94 passed
cat .claude/plans/2026-06-25_detection-pivot-plan.md   # plan đầy đủ (đã chuyển vào repo)
# Lệnh chạy thật (trên RTX3060):
venv/bin/python scripts/generate_detection_manifest.py --output configs/datasets/ppe_detection_exp011_manifest.csv --per-site 300 --val-fraction 0.2
venv/bin/python scripts/run_detection_sim.py --mode all --manifest configs/datasets/ppe_detection_exp011_manifest.csv --root-dir data/ppe --output-dir outputs/EXP-011 --exp-id EXP-011 --device cuda
```
Việc còn lại chính: **chạy EXP-011 trên GPU** rồi viết journal. Dữ liệu thật: `data/ppe/` (8099 ảnh VOC, **13GB**, gitignored — KHÔNG commit ảnh).

## Lưu ý
- `data/ppe/voc_labels/*.xml` (VOC) và `data/ppe/labels/*.txt` (YOLO) đều có; dùng VOC.
- `docs/md/` bị gitignore → PLAN.md là file local, không vào git.
- Repo chạy trên `venv/bin/python`; không có ruff/black trong venv.
