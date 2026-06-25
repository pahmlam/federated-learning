# Thử Nghiệm: PPE Real ResNet18 Stability Baseline
- **Mã Thử Nghiệm:** EXP-005
- **Ngày Thực Hiện:** 2026-06-22
- **Trạng Thái:** Thành công
- **Git Commit Hash:** `chưa commit`

---

## 1. Mục Tiêu (Objective)
- Củng cố `EXP-004` từ smoke test thành baseline PPE dễ phân tích hơn.
- Giữ nguyên pipeline ảnh thật: VOC annotation proxy -> frozen ResNet18 embedding -> train classifier head.
- Chạy đủ 3 mode `centralized`, `local-only`, `federated` với nhiều epoch/round hơn để kiểm tra độ ổn định ban đầu.
- Thêm metric chẩn đoán: confusion matrix và precision/recall/F1 theo từng class.

## 2. Cấu Hình & Thiết Lập (Configuration)
- **Bài toán:** PPE binary classification smoke (`safe` / `unsafe`).
- **Dữ Liệu & Phân Chia Client (Data & Partition):**
  - Dataset root local: `data/ppe`
  - Manifest: `configs/datasets/ppe_real_smoke_manifest.csv`
  - Artifact: `data/processed/ppe_real_embeddings_oom_safe.npz`
  - Số lượng mẫu smoke: 240 ảnh
  - Số lượng client mô phỏng: 3 clients (`site-a`, `site-b`, `site-c`)
  - Loại phân chia dữ liệu: non-IID label skew nhẹ.
  - Label mapping: `{"safe": 0, "unsafe": 1}`
- **Mô Hình & Huấn Luyện Tham Số Hiệu Quả (Model & PEFT):**
  - Frozen backbone: `torchvision-resnet18` pretrained ImageNet.
  - Input baseline: precomputed embedding vectors, dim `512`.
  - Phần trainable: classifier head trực tiếp `512 -> 2`.
- **Siêu Tham Số (Hyperparameters):**
  - Profile: `oom-safe`
  - Learning Rate: `0.01`
  - Batch Size: `4`
  - Local Epochs: `3`
  - Centralized Epochs: `3`
  - FL Rounds: `3`
  - DataLoader Workers: `0`
  - Ray CPUs: `1`
  - Client CPUs: `1.0`
  - Optimizer: `SGD`

## 3. So Sánh Kết Quả Giữa Các Baseline (Baseline Comparison)

| Chế Độ Huấn Luyện (Mode) | Global Loss | Global Metric (Acc/Macro-F1) | Unsafe Recall / FNR | Tổng Thời Gian Train | Kích Thước Update (Per Round) | Tổng Chi Phí Truyền Thông (Comm. Cost) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized** (Pooled data) | 0.7650 | 0.5167 / 0.5055 | 0.3667 / 0.6333 | 0.80s | 4,104 bytes | N/A |
| **Local-Only** (Không collab) | 0.7450 | 0.6833 / 0.6832 | 0.5667 / 0.4333 | 0.02s | 4,104 bytes | N/A |
| **Federated** (FedAvg Baseline) | 0.6726 | 0.7000 / 0.6997 | 0.7333 / 0.2667 | 3.84s | 4,104 bytes | 73,872 bytes |

## 4. Kết Quả Chi Tiết Theo Từng Client (Per-Client Metrics)

Kết quả dưới đây là mode `federated` sau 3 round.

| Client ID / Site | Số Lượng Validation | Local Validation Loss | Local Metric (Acc/Macro-F1) | Unsafe Recall / FNR | Confusion Matrix `[safe, unsafe]` | Ghi Chú Đặc Điểm Dữ Liệu |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **site-a** | 20 | 0.6871 | 0.7000 / 0.6429 | 0.6000 / 0.4000 | `[[11, 4], [2, 3]]` | Label histogram `{safe: 60, unsafe: 20}` |
| **site-b** | 20 | 0.5584 | 0.8500 / 0.8119 | 0.8667 / 0.1333 | `[[4, 1], [2, 13]]` | Label histogram `{safe: 20, unsafe: 60}` |
| **site-c** | 20 | 0.7724 | 0.5500 / 0.5489 | 0.6000 / 0.4000 | `[[5, 5], [4, 6]]` | Label histogram `{safe: 40, unsafe: 40}` |

## 5. Quan Sát & Phân Tích (Observations & Rationale)
- So với `EXP-004`, FedAvg cải thiện rõ: accuracy từ `0.5000` lên `0.7000`, macro-F1 từ `0.3333` lên `0.6997`.
- Mô hình không còn chỉ nghiêng hẳn về `unsafe`: global federated confusion matrix là `[[20, 10], [8, 22]]`, tương ứng safe recall `0.6667` và unsafe recall `0.7333`.
- `site-b` tốt nhất trong federated (`Acc 0.8500`, `Macro-F1 0.8119`), còn `site-c` thấp nhất (`Acc 0.5500`, `Macro-F1 0.5489`), cho thấy split cân bằng vẫn khó trên subset nhỏ.
- Local-only cũng cải thiện so với `EXP-004`, nhưng FedAvg tốt hơn về macro-F1 và unsafe recall trong cấu hình này.
- Communication cost tăng từ `24,624 bytes` ở `EXP-004` lên `73,872 bytes`, đúng kỳ vọng vì số round tăng từ 1 lên 3 trong khi update size giữ nguyên `4,104 bytes`.
- Peak RSS của process chính khoảng `354.72 MB` theo `resource.getrusage(RUSAGE_SELF)`. Đây vẫn là chỉ báo nhẹ, chưa đo toàn bộ Ray worker hoặc peak RAM hệ thống.
- Kết quả vẫn chưa dùng để kết luận chất lượng PPE thật vì nhãn `safe/unsafe` là proxy image-level từ object annotation, chưa phải nhãn compliance được review bởi người.

## 6. Tài Liệu Hướng Dẫn Tái Lập (Reproducibility & Artifacts)
- **Lệnh chạy baseline:**
  ```bash
  venv/bin/python scripts/run_embedding_demo.py \
    --mode all \
    --artifact data/processed/ppe_real_embeddings_oom_safe.npz \
    --profile oom-safe \
    --output-dir outputs/EXP-005 \
    --exp-id EXP-005 \
    --batch-size 4 \
    --local-epochs 3 \
    --centralized-epochs 3 \
    --num-rounds 3 \
    --lr 0.01
  ```
- **Đường dẫn outputs:** `outputs/EXP-005/`
- **File output chính:**
  - `outputs/EXP-005/centralized_metrics.json`
  - `outputs/EXP-005/local_only_metrics.json`
  - `outputs/EXP-005/federated_metrics.json`
  - `outputs/EXP-005/summary.json`

## 7. Bước Tiếp Theo (Next Steps)
- [ ] Cải thiện định nghĩa nhãn PPE/compliance thay vì chỉ dùng proxy `has_core_ppe`.
- [ ] Chạy thêm một baseline với nhiều mẫu/client hơn nhưng vẫn giữ OOM-safe.
- [ ] Ghi rõ trong báo cáo rằng kết quả hiện tại chứng minh pipeline và baseline, chưa chứng minh chất lượng nghiệp vụ.
- [ ] Sau khi nhãn ổn hơn, mới xét personalized head, adapter/LoRA, FedBN hoặc giảm communication.
