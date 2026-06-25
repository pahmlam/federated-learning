# Thử Nghiệm: Synthetic Flower + PyTorch Baseline
- **Mã Thử Nghiệm:** EXP-001
- **Ngày Thực Hiện:** 2026-06-12
- **Trạng Thái:** Thành công
- **Git Commit Hash:** `committed`

---

## 1. Mục Tiêu (Objective)
- Dựng demo tối thiểu cho Federated Learning với 3 baseline: centralized, local-only và federated.
- Kiểm tra luồng freeze backbone, chỉ train classifier head, trước khi chuyển sang face/PPE thật.

## 2. Cấu Hình & Thiết Lập (Configuration)
- **Bài toán:** Toy-task / synthetic classification.
- **Dữ Liệu & Phân Chia Client (Data & Partition):**
  - Dataset: synthetic vectors sinh offline.
  - Số lượng client mô phỏng: 5 clients.
  - Loại phân chia dữ liệu: IID hoặc Non-IID bằng Dirichlet label skew.
- **Mô Hình & Huấn Luyện Tham Số Hiệu Quả (Model & PEFT):**
  - Backbone: frozen linear backbone.
  - Phần trainable: classifier head.
- **Siêu Tham Số (Hyperparameters):**
  - Learning Rate: `0.05`
  - Batch Size: `16`
  - Local Epochs: `2`
  - FL Rounds: `3`
  - Optimizer: `SGD`

## 3. So Sánh Kết Quả Giữa Các Baseline (Baseline Comparison)

| Chế Độ Huấn Luyện (Mode) | Global Loss | Global Metric (Acc/F1/mAP) | Tổng Thời Gian Train | Kích Thước Update (Per Round) | Tổng Chi Phí Truyền Thông (Comm. Cost) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Centralized** (Pooled data) | 0.4948 | Acc 0.9216 | 0.50s | 108 bytes | N/A |
| **Local-Only** (Không collab) | 0.7755 | Acc 0.8039 | 0.01s | 108 bytes | N/A |
| **Federated** (FedAvg Baseline) | 0.7804 | Acc 0.7059 | 3.54s | 108 bytes | 2,160 bytes |

## 4. Kết Quả Chi Tiết Theo Từng Client (Per-Client Metrics)

Kết quả per-client đã được lưu trong `outputs/EXP-001/*_metrics.json`. Mỗi mode có đủ 5 client mô phỏng, gồm số mẫu validation, loss, accuracy và label histogram.

## 5. Quan Sát & Phân Tích (Observations & Rationale)
- Demo này chưa đại diện cho face/PPE thật.
- Mục tiêu là kiểm tra luồng kỹ thuật: chia client, train head-only, aggregation, metric theo client và communication cost.
- Smoke test đã chạy được 3 mode với dữ liệu non-IID synthetic.
- Federated dùng 2 round ở chế độ `--quick`, update head có kích thước 108 bytes, tổng communication cost 2,160 bytes.
- Bước tiếp theo là thay synthetic data bằng task face hoặc PPE đơn giản.

## 6. Tài Liệu Hướng Dẫn Tái Lập (Reproducibility & Artifacts)
- **Lệnh chạy:**
  ```bash
  venv/bin/python scripts/run_demo.py --mode all --quick
  PATH="$PWD/venv/bin:$PATH" venv/bin/flwr run . --run-config "quick=true num-server-rounds=2" --federation-config "num-supernodes=5" --stream
  ```
- **Đường dẫn outputs:** `outputs/EXP-001/`

## 7. Bước Tiếp Theo (Next Steps)
- [x] Chạy unit tests.
- [x] Chạy smoke test centralized/local-only/federated.
- [x] Cập nhật bảng kết quả sau khi có output.
