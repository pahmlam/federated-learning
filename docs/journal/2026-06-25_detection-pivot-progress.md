# WIP — PPE Detection Pivot + Federated Deployment (build progress)

> **Đây KHÔNG phải EXP journal** (chưa có metric run). Đây là **log tiến độ build** để resume sau khi
> context compact. EXP-011 journal sẽ tạo riêng khi có kết quả sim run thật.

- **Ngày:** 2026-06-25
- **Trạng thái:** đang build Phase A (3/8 bước xong, có test)
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
- **Deps cài:** `torchmetrics[detection]` 1.9.0, `pycocotools` 2.0.11 (trong `venv`). **Chưa thêm vào `pyproject.toml`/requirements.**
- **Rủi ro retired:** frozen Faster R-CNN train được (loss+backward), mAP chạy end-to-end, head serialize ổn định.

## Còn lại (Phase A bước 4–8)
- [ ] `src/training/detection_baselines.py` — centralized (pooled) + local-only, mAP per-client (mirror `embedding_baselines.py`).
- [ ] `src/fl/detection_clientapp.py` + `detection_serverapp.py` — modern Flower API (mirror `src/fl/client_app.py`/`server_app.py`), load shard theo `node_config(data-root, partition-id)`, FedAvg distributed-eval.
- [ ] `src/data/detection_manifest.py` + `scripts/generate_detection_manifest.py` — non-IID PPE-skew (site-a helmet-heavy, site-b vest/mask-heavy, site-c balanced), seeded, leakage-free; reuse `build_image_index` + `read_voc_objects`.
- [ ] `scripts/run_detection_sim.py` + `scripts/export_detection_subset.py`.
- [ ] Thêm deps vào `pyproject.toml` (optional group) hoặc `requirements-detection.txt`.
- [ ] **Smoke CPU Mac** 12 ảnh / 1 round / 3 sim client → verify FedAvg detection.
- [ ] **Real sim RTX3060** ~900 ảnh → **journal EXP-011** + registry (cột mAP).

## Phase B (sau EXP-011)
- Tailscale up trên 4 máy; `[tool.flwr.federations.local-sim|deploy]` trong `pyproject.toml`; SuperLink trên Mac; export 3 shard; notebook supernode Colab; `flwr run . deploy` → **EXP-012**.

## Cách resume nhanh
```bash
venv/bin/python -m pytest -q          # phải 83 passed
cat /Users/phamtunglam/.claude/plans/t-i-ngh-m-y-mac-mighty-stearns.md   # plan đầy đủ
```
Tiếp tục từ bước 4 (`detection_baselines.py`). Dữ liệu thật: `data/ppe/` (8099 ảnh VOC, **13GB**, gitignored — phải subset; KHÔNG commit ảnh).

## Lưu ý
- `data/ppe/voc_labels/*.xml` (VOC) và `data/ppe/labels/*.txt` (YOLO) đều có; dùng VOC.
- `docs/md/` bị gitignore → PLAN.md là file local, không vào git.
- Repo chạy trên `venv/bin/python`; không có ruff/black trong venv.
