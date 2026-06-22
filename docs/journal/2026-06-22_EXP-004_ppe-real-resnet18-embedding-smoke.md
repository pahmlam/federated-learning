# Thử Nghiệm: PPE Real ResNet18 Embedding Smoke
- **Mã Thử Nghiệm:** EXP-004
- **Ngày Thực Hiện:** 2026-06-22
- **Trạng Thái:** Thành công
- **Git Commit Hash:** `chưa commit`

---

## 1. Mục Tiêu (Objective)
- Chạy thử nghiệm PPE đầu tiên trên ảnh thật trong `data/ppe` theo cấu hình OOM-safe.
- Kiểm tra luồng: VOC annotation -> manifest classification proxy -> frozen ResNet18 embedding `.npz` -> 3 baseline `centralized`, `local-only`, `federated`.
- Giữ đúng hướng parameter-efficient: backbone ResNet18 pretrained ImageNet được freeze, chỉ train classifier head trên embedding 512 chiều.

## 2. Cấu Hình & Thiết Lập (Configuration)
- **Bài toán:** PPE binary classification smoke (`safe` / `unsafe`).
- **Dữ Liệu & Phân Chia Client (Data & Partition):**
  - Dataset root local: `data/ppe`
  - Manifest: `configs/datasets/ppe_real_smoke_manifest.csv`
  - Artifact: `data/processed/ppe_real_embeddings_oom_safe.npz`
  - Số lượng mẫu smoke: 240 ảnh
  - Số lượng client mô phỏng: 3 clients (`site-a`, `site-b`, `site-c`)
  - Loại phân chia dữ liệu: non-IID label skew nhẹ.
  - Label proxy:
    - `safe`: ảnh có ít nhất một PPE core trong VOC annotation: `helmet`, `safety-vest`, `safety-suit`, `face-mask-medical`, `gloves`, `glasses`, `ear-mufs`, hoặc `face-guard`.
    - `unsafe`: ảnh không có PPE core theo rule trên.
- **Mô Hình & Huấn Luyện Tham Số Hiệu Quả (Model & PEFT):**
  - Frozen backbone: `torchvision-resnet18` pretrained ImageNet.
  - Input baseline: precomputed embedding vectors, dim `512`.
  - Phần trainable: classifier head trực tiếp `512 -> 2`.
- **Siêu Tham Số (Hyperparameters):**
  - Profile: `oom-safe`
  - Learning Rate: `0.05`
  - Batch Size: `4`
  - Local Epochs: `1`
  - Centralized Epochs: `1`
  - FL Rounds: `1`
  - DataLoader Workers: `0`
  - Ray CPUs: `1`
  - Client CPUs: `1.0`
  - Optimizer: `SGD`

## 3. So Sánh Kết Quả Giữa Các Baseline (Baseline Comparison)

| Chế Độ Huấn Luyện (Mode) | Global Loss | Global Metric (Acc/F1/mAP) | Unsafe Recall / FNR | Tổng Thời Gian Train | Kích Thước Update (Per Round) | Tổng Chi Phí Truyền Thông (Comm. Cost) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized** (Pooled data) | 2.0818 | Acc 0.4667 / Macro-F1 0.4661 | 0.5000 / 0.5000 | 0.82s | 4,104 bytes | N/A |
| **Local-Only** (Không collab) | 7.8129 | Acc 0.5000 / Macro-F1 0.3206 | 1.0000 / 0.0000 | 0.01s | 4,104 bytes | N/A |
| **Federated** (FedAvg Baseline) | 8.0683 | Acc 0.5000 / Macro-F1 0.3333 | 1.0000 / 0.0000 | 3.90s | 4,104 bytes | 24,624 bytes |

## 4. Kết Quả Chi Tiết Theo Từng Client (Per-Client Metrics)

Kết quả dưới đây là mode `federated` sau 1 round.

| Client ID / Site | Số Lượng Validation | Local Validation Loss | Local Metric (Acc/Macro-F1) | Unsafe Recall / FNR | Ghi Chú Đặc Điểm Dữ Liệu |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **site-a** | 20 | 12.0544 | Acc 0.2500 / Macro-F1 0.2000 | 1.0000 / 0.0000 | Label histogram `{safe: 60, unsafe: 20}` |
| **site-b** | 20 | 3.9693 | Acc 0.7500 / Macro-F1 0.4286 | 1.0000 / 0.0000 | Label histogram `{safe: 20, unsafe: 60}` |
| **site-c** | 20 | 8.1814 | Acc 0.5000 / Macro-F1 0.3333 | 1.0000 / 0.0000 | Label histogram `{safe: 40, unsafe: 40}` |

## 5. Quan Sát & Phân Tích (Observations & Rationale)
- EXP-004 xác nhận pipeline ảnh thật chạy được theo hướng OOM-safe: đọc ảnh lazy trong bước precompute, freeze ResNet18, lưu embedding, sau đó chỉ train head nhẹ.
- Kết quả model chưa dùng để kết luận chất lượng PPE vì nhãn `safe/unsafe` là proxy image-level từ object annotation, chưa phải nhãn compliance được review bởi người.
- Federated và local-only đều có xu hướng dự đoán `unsafe` nhiều trong smoke run 1 epoch/1 round, thể hiện qua unsafe recall 1.0 nhưng macro-F1 thấp.
- Update size là 4,104 bytes, phù hợp head `2 x 512` cộng bias `2` ở float32.
- Communication cost của FedAvg là 24,624 bytes, tương ứng gửi/nhận update head cho 3 client trong 1 round.
- Peak RSS của process chính khoảng 353.69 MB theo `resource.getrusage(RUSAGE_SELF)`. Đây là chỉ báo nhẹ, chưa đo toàn bộ Ray worker hoặc peak RAM hệ thống.
- Trong sandbox Codex, Flower/Ray bị chặn `psutil/sysctl`; rerun ngoài sandbox thành công.

## 6. Tài Liệu Hướng Dẫn Tái Lập (Reproducibility & Artifacts)
- **Lệnh tạo artifact:**
  ```bash
  venv/bin/python scripts/precompute_embeddings.py \
    --backend torchvision-resnet18 \
    --weights imagenet \
    --manifest configs/datasets/ppe_real_smoke_manifest.csv \
    --root-dir data/ppe \
    --output data/processed/ppe_real_embeddings_oom_safe.npz \
    --batch-size 4 \
    --num-workers 0 \
    --device cpu
  ```
- **Lệnh chạy baseline:**
  ```bash
  venv/bin/python scripts/run_embedding_demo.py \
    --mode all \
    --artifact data/processed/ppe_real_embeddings_oom_safe.npz \
    --profile oom-safe \
    --output-dir outputs/EXP-004 \
    --exp-id EXP-004
  ```
- **Đường dẫn outputs:** `outputs/EXP-004/`
- **File output chính:**
  - `outputs/EXP-004/centralized_metrics.json`
  - `outputs/EXP-004/local_only_metrics.json`
  - `outputs/EXP-004/federated_metrics.json`
  - `outputs/EXP-004/summary.json`

## 7. Bước Tiếp Theo (Next Steps)
- [ ] Cải thiện rule hoặc tạo nhãn compliance được review bởi người, tránh phụ thuộc hoàn toàn vào proxy `has_core_ppe`.
- [ ] Chạy thêm 2-3 round hoặc tăng epoch nhỏ để xem metric có ổn định hơn không.
- [ ] Thêm confusion matrix hoặc precision/recall theo class cho báo cáo PPE.
- [ ] Nếu ổn định, tăng mẫu/client có kiểm soát và ghi resource/time/update size.
