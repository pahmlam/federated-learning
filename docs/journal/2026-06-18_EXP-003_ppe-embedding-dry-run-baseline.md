# Thử Nghiệm: PPE Embedding Dry-Run Baseline
- **Mã Thử Nghiệm:** EXP-003
- **Ngày Thực Hiện:** 2026-06-18
- **Trạng Thái:** Thành công
- **Git Commit Hash:** `committed`

---

## 1. Mục Tiêu (Objective)
- Kiểm tra luồng từ PPE manifest sang embedding artifact `.npz`, rồi chạy đủ 3 baseline `centralized`, `local-only`, `federated`.
- Xác nhận pipeline FL head-only có thể dùng input dạng embedding thay vì synthetic vectors sinh trực tiếp trong code.
- Chưa đánh giá chất lượng PPE thật vì artifact vẫn được sinh bằng backend `synthetic`, chưa phải frozen pretrained backbone.

## 2. Cấu Hình & Thiết Lập (Configuration)
- **Bài toán:** PPE binary classification dry-run (`safe` / `unsafe`).
- **Dữ Liệu & Phân Chia Client (Data & Partition):**
  - Manifest: `configs/datasets/ppe_manifest_template.csv`
  - Artifact: `data/processed/ppe_embeddings_oom_safe.npz`
  - Backend artifact: `synthetic`
  - Số lượng client mô phỏng: 2 clients (`site-a`, `site-b`)
  - Split dùng cho baseline v1: `train` và `val`; `test` được giữ trong artifact nhưng chưa dùng.
- **Mô Hình & Huấn Luyện Tham Số Hiệu Quả (Model & PEFT):**
  - Input: precomputed embedding vectors, dim `16`.
  - Backbone: không dùng trong dry-run này.
  - Phần trainable: classifier head trực tiếp `embedding_dim -> num_classes`.
- **Siêu Tham Số (Hyperparameters):**
  - Profile: `oom-safe`
  - Learning Rate: `0.05`
  - Batch Size: `4`
  - Local Epochs: `1`
  - Centralized Epochs: `1`
  - FL Rounds: `1`
  - DataLoader Workers: `0`
  - Ray CPUs: `1`
  - Optimizer: `SGD`

## 3. So Sánh Kết Quả Giữa Các Baseline (Baseline Comparison)

| Chế Độ Huấn Luyện (Mode) | Global Loss | Global Metric (Acc/F1/mAP) | Tổng Thời Gian Train | Kích Thước Update (Per Round) | Tổng Chi Phí Truyền Thông (Comm. Cost) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Centralized** (Pooled data) | 0.7036 | Acc 0.5000 | 0.72s | 136 bytes | N/A |
| **Local-Only** (Không collab) | 0.7398 | Acc 0.0000 | 0.00s | 136 bytes | N/A |
| **Federated** (FedAvg Baseline) | 0.7036 | Acc 0.5000 | 3.41s | 136 bytes | 544 bytes |

## 4. Kết Quả Chi Tiết Theo Từng Client (Per-Client Metrics)

Kết quả dưới đây là mode `federated` sau 1 round.

| Client ID / Site | Số Lượng Validation | Local Validation Loss | Local Metric (Acc/F1/mAP) | Ghi Chú Đặc Điểm Dữ Liệu |
| :--- | :---: | :---: | :---: | :--- |
| **site-a** | 1 | 0.7368 | Acc 0.0000 | Label histogram `{0: 2, 1: 1}` |
| **site-b** | 1 | 0.6703 | Acc 1.0000 | Label histogram `{0: 1, 1: 2}` |

## 5. Quan Sát & Phân Tích (Observations & Rationale)
- EXP-003 xác nhận data contract hoạt động: manifest -> `.npz` embedding -> 3 baseline.
- `label_mapping` là `{"safe": 0, "unsafe": 1}`.
- Update size tăng từ 108 bytes ở synthetic head EXP-002 lên 136 bytes vì head trực tiếp có shape `2 x 16` cộng bias `2`.
- Flower/Ray vẫn chạy single-machine simulation và được giới hạn còn `num_cpus=1`; log cho thấy VirtualClientEngine tạo 1 actor.
- Accuracy không có ý nghĩa nghiệp vụ vì dữ liệu chỉ có 6 sample và embedding được sinh synthetic, không phải từ ảnh PPE/backbone thật.

## 6. Tài Liệu Hướng Dẫn Tái Lập (Reproducibility & Artifacts)
- **Lệnh tạo artifact:**
  ```bash
  venv/bin/python scripts/precompute_embeddings.py --backend synthetic --output data/processed/ppe_embeddings_oom_safe.npz
  ```
- **Lệnh chạy baseline:**
  ```bash
  venv/bin/python scripts/run_embedding_demo.py --mode all --artifact data/processed/ppe_embeddings_oom_safe.npz --profile oom-safe --output-dir outputs/EXP-003
  ```
- **Đường dẫn outputs:** `outputs/EXP-003/`
- **File output chính:**
  - `outputs/EXP-003/centralized_metrics.json`
  - `outputs/EXP-003/local_only_metrics.json`
  - `outputs/EXP-003/federated_metrics.json`
  - `outputs/EXP-003/summary.json`

## 7. Bước Tiếp Theo (Next Steps)
- [x] Thay backend `synthetic` bằng backend frozen pretrained backbone thật. *(Đã làm: EXP-004 — torchvision-resnet18.)*
- [x] Tạo manifest từ dataset PPE public hoặc nội bộ. *(EXP-004 manifest `data/ppe`; EXP-006 generator có seed.)*
- [x] Chạy real-data smoke run nhỏ với cùng profile OOM-safe trước khi tăng batch/client/round. *(Đã làm: EXP-004.)*
