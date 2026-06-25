# Thử Nghiệm: OOM-Safe Single-Machine Smoke Profile
- **Mã Thử Nghiệm:** EXP-002
- **Ngày Thực Hiện:** 2026-06-18
- **Trạng Thái:** Thành công
- **Git Commit Hash:** `committed`

---

## 1. Mục Tiêu (Objective)
- Kiểm tra profile chạy nhỏ, an toàn bộ nhớ, trước khi thay synthetic data bằng model/dataset thật.
- Xác nhận Flower simulation hiện tại vẫn là single-machine simulation: nhiều client mô phỏng chạy trên cùng một máy.
- Giữ đủ 3 baseline `centralized`, `local-only`, `federated` nhưng giảm client/sample/batch/round để hạn chế OOM.

## 2. Cấu Hình & Thiết Lập (Configuration)
- **Bài toán:** Toy-task / synthetic classification, dùng như smoke test OOM-safe.
- **Dữ Liệu & Phân Chia Client (Data & Partition):**
  - Dataset: synthetic vectors sinh offline.
  - Số lượng client mô phỏng: 3 clients.
  - Loại phân chia dữ liệu: Non-IID bằng Dirichlet label skew.
- **Mô Hình & Huấn Luyện Tham Số Hiệu Quả (Model & PEFT):**
  - Backbone: frozen linear backbone.
  - Phần trainable: classifier head.
- **Siêu Tham Số (Hyperparameters):**
  - Profile: `oom-safe`
  - Learning Rate: `0.05`
  - Batch Size: `4`
  - Samples per Client: `40`
  - Local Epochs: `1`
  - Centralized Epochs: `1`
  - FL Rounds: `1`
  - DataLoader Workers: `0`
  - Ray CPUs: `1`
  - Client CPUs: `1.0`
  - Optimizer: `SGD`

## 3. So Sánh Kết Quả Giữa Các Baseline (Baseline Comparison)

| Chế Độ Huấn Luyện (Mode) | Global Loss | Global Metric (Acc/F1/mAP) | Tổng Thời Gian Train | Kích Thước Update (Per Round) | Tổng Chi Phí Truyền Thông (Comm. Cost) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Centralized** (Pooled data) | 0.8381 | Acc 0.6207 | 0.75s | 108 bytes | N/A |
| **Local-Only** (Không collab) | 0.5791 | Acc 0.9310 | 0.01s | 108 bytes | N/A |
| **Federated** (FedAvg Baseline) | 1.0665 | Acc 0.3448 | 3.83s | 108 bytes | 648 bytes |

## 4. Kết Quả Chi Tiết Theo Từng Client (Per-Client Metrics)

Kết quả dưới đây là mode `federated` sau 1 round.

| Client ID / Site | Số Lượng Validation | Local Validation Loss | Local Metric (Acc/F1/mAP) | Ghi Chú Đặc Điểm Dữ Liệu |
| :--- | :---: | :---: | :---: | :--- |
| **Client 0** | 8 | 1.3323 | Acc 0.0000 | Non-IID label skew, histogram `{0: 0, 1: 0, 2: 33}` |
| **Client 1** | 11 | 1.2029 | Acc 0.0000 | Non-IID label skew, histogram `{0: 0, 1: 39, 2: 6}` |
| **Client 2** | 10 | 0.7036 | Acc 1.0000 | Non-IID label skew, histogram `{0: 40, 1: 1, 2: 1}` |

## 5. Quan Sát & Phân Tích (Observations & Rationale)
- Profile `oom-safe` chạy được đủ 3 mode với 3 client, batch size 4, 1 local epoch và 1 FL round.
- Flower/Ray được giới hạn còn `num_cpus=1`, log cho thấy VirtualClientEngine tạo 1 actor, phù hợp mục tiêu giảm concurrency trên một máy cá nhân.
- Peak RSS đo bằng `resource.getrusage(RUSAGE_SELF)` của process chính khoảng 368.31 MB. Số này là chỉ báo nhẹ, chưa thay thế đo RAM toàn hệ thống hoặc Ray worker riêng.
- Kết quả accuracy không dùng để kết luận FL tốt/xấu vì đây là smoke test nhỏ, chỉ có synthetic label skew và 1 round.
- Trong sandbox Codex, lần chạy đầu bị chặn quyền `sysctl/psutil`; rerun với quyền ngoài sandbox thành công. Khi chạy trực tiếp trong terminal local, lệnh thường có thể chạy bình thường.

## 6. Tài Liệu Hướng Dẫn Tái Lập (Reproducibility & Artifacts)
- **Lệnh chạy:**
  ```bash
  venv/bin/python scripts/run_demo.py --mode all --profile oom-safe
  ```
- **Đường dẫn outputs:** `outputs/EXP-002/`
- **File output chính:**
  - `outputs/EXP-002/centralized_metrics.json`
  - `outputs/EXP-002/local_only_metrics.json`
  - `outputs/EXP-002/federated_metrics.json`
  - `outputs/EXP-002/summary.json`

## 7. Bước Tiếp Theo (Next Steps)
- [ ] Thiết kế dataset loader thật theo hướng lazy-load, không load toàn bộ ảnh/video vào RAM.
- [ ] Thiết kế bước precompute embeddings bằng frozen backbone trước khi train head/adapter trên real data.
- [ ] Khi chuyển sang face/PPE, chạy lại profile nhỏ tương tự trước khi tăng client, batch, round hoặc dataset size.
