# Thử Nghiệm: PPE Real ResNet18 Scaled Baseline (480 mẫu / 3 site)
- **Mã Thử Nghiệm:** EXP-006
- **Ngày Thực Hiện:** 2026-06-25
- **Trạng Thái:** Thành công
- **Git Commit Hash:** `committed`

---

## 1. Mục Tiêu (Objective)
- Thực hiện Next-step #2 của `EXP-005`: chạy thêm một baseline với **nhiều mẫu hơn** (gấp đôi: 240 → 480) nhưng **vẫn giữ OOM-safe**.
- Khắc phục gap tái lập của `EXP-004`/`EXP-005`: manifest 240 dòng trước đây làm tay. EXP-006 sinh manifest bằng **script có seed cố định** (`scripts/generate_ppe_manifest.py`).
- Giữ nguyên pipeline ảnh thật: VOC annotation proxy -> frozen ResNet18 embedding -> train classifier head, và giữ nguyên cấu trúc 3 site + kiểu non-IID label skew của EXP-005 để so sánh công bằng.

## 2. Cấu Hình & Thiết Lập (Configuration)ßß
- **Bài toán:** PPE binary classification smoke (`safe` / `unsafe`).
- **Dữ Liệu & Phân Chia Client (Data & Partition):**
  - Dataset root local: `data/ppe` (pool 8,099 VOC annotation / 8,077 ảnh; EXP-006 dùng 480, ~5.9%).
  - Manifest: `configs/datasets/ppe_real_exp006_manifest.csv` (sinh tự động, seed 2026).
  - Artifact: `data/processed/ppe_real_embeddings_exp006.npz`
  - Số lượng mẫu: **480 ảnh** (gấp đôi EXP-005).
  - Số lượng client mô phỏng: 3 clients (`site-a`, `site-b`, `site-c`), mỗi site 160 mẫu.
  - Loại phân chia dữ liệu: non-IID label skew (mirror EXP-005 ×2):
    - `site-a`: safe-heavy 0.75 → 120 safe / 40 unsafe (train 120 / val 40)
    - `site-b`: unsafe-heavy 0.25 → 40 safe / 120 unsafe (train 120 / val 40)
    - `site-c`: cân bằng 0.50 → 80 safe / 80 unsafe (train 120 / val 40)
  - Lấy mẫu **không hoàn lại trên pool chung** → không ảnh nào dùng lại giữa các site/split (no leakage).
  - Label proxy (giữ nguyên EXP-004): `safe` nếu VOC có ≥1 core PPE (`helmet, safety-vest, safety-suit, face-mask-medical, gloves, glasses, ear-mufs, face-guard`), ngược lại `unsafe`.
  - Label mapping: `{"safe": 0, "unsafe": 1}`
- **Mô Hình & Huấn Luyện Tham Số Hiệu Quả (Model & PEFT):**
  - Frozen backbone: `torchvision-resnet18` pretrained ImageNet.
  - Input baseline: precomputed embedding vectors, dim `512`.
  - Phần trainable: classifier head trực tiếp `512 -> 2`.
- **Siêu Tham Số (Hyperparameters):** (mirror EXP-005)
  - Profile: `oom-safe` (xác nhận giữ: `num_workers=0`, `ray_num_cpus=1`, `client_num_cpus=1.0`, `batch_size=4`)
  - Learning Rate: `0.01`
  - Batch Size: `4`
  - Local Epochs: `3`
  - Centralized Epochs: `3`
  - FL Rounds: `3`
  - Optimizer: `SGD`

## 3. So Sánh Kết Quả Giữa Các Baseline (Baseline Comparison)

| Chế Độ Huấn Luyện (Mode) | Global Loss | Global Metric (Acc/Macro-F1) | Unsafe Recall / FNR | Tổng Thời Gian Train | Kích Thước Update (Per Round) | Tổng Chi Phí Truyền Thông (Comm. Cost) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized** (Pooled data) | 1.0149 | 0.5583 / 0.5173 | 0.2667 / 0.7333 | 0.74s | 4,104 bytes | N/A |
| **Local-Only** (Không collab) | 0.7437 | 0.6250 / 0.6173 | 0.7222 / 0.2778 | 0.04s | 4,104 bytes | N/A |
| **Federated** (FedAvg Baseline) | 0.8215 | 0.5917 / 0.5812 | 0.7500 / 0.2500 | 3.61s | 4,104 bytes | 73,872 bytes |

## 4. Kết Quả Chi Tiết Theo Từng Client (Per-Client Metrics)

Kết quả dưới đây là mode `federated` sau 3 round.

| Client ID / Site | Số Lượng Validation | Local Validation Loss | Local Metric (Acc/Macro-F1) | Unsafe Recall / FNR | Confusion Matrix `[safe, unsafe]` | Ghi Chú Đặc Điểm Dữ Liệu |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **site-a** | 40 | 0.9325 | 0.5250 / 0.5100 | 0.7000 / 0.3000 | `[[14, 16], [3, 7]]` | Label histogram `{safe: 120, unsafe: 40}` |
| **site-b** | 40 | 0.5919 | 0.7250 / 0.6204 | 0.8333 / 0.1667 | `[[4, 6], [5, 25]]` | Label histogram `{safe: 40, unsafe: 120}` |
| **site-c** | 40 | 0.9400 | 0.5250 / 0.5175 | 0.6500 / 0.3500 | `[[8, 12], [7, 13]]` | Label histogram `{safe: 80, unsafe: 80}` |

Global federated confusion matrix: `[[26, 34], [15, 45]]` → safe recall `0.4333`, unsafe recall `0.7500`.

## 5. Quan Sát & Phân Tích (Observations & Rationale)
- **Tăng mẫu làm metric tổng hợp GIẢM so với EXP-005** (đây là kết quả khách quan, không phải hồi quy về code):
  - Federated: Acc `0.7000 → 0.5917`, macro-F1 `0.6997 → 0.5812`.
  - Local-only: Acc `0.6833 → 0.6250`, macro-F1 `0.6832 → 0.6173`.
  - Centralized gần như giữ nguyên (Acc `0.5167 → 0.5583`).
  - Lý do hợp lý: 480 mẫu được **sample ngẫu nhiên có kiểm soát từ pool 8k ảnh đa dạng**, khó hơn nhiều so với 240 ảnh được chọn tay của EXP-005. Tức một phần kết quả cao của EXP-005 đến từ subset nhỏ/curated; EXP-006 là baseline trung thực hơn.
- **Chưa có "người thắng" tuyệt đối giữa FL và local-only ở quy mô này:** local-only nhỉnh hơn ở macro-F1 (`0.6173` vs `0.5812`) và accuracy, nhưng federated tốt hơn ở **unsafe recall** (`0.7500` vs `0.7222`) và FNR thấp hơn — quan trọng cho mục tiêu an toàn (ít bỏ sót vi phạm). Head tuyến tính + 3 epoch/3 round là chưa đủ; cần nhiều round/epoch hoặc method nhẹ.
- **Non-IID:** site-b (unsafe-heavy) đạt cao nhất trong federated (Acc `0.7250`), còn site-a/site-c thấp (`0.5250`). Head toàn cục bị kéo về thiên dự đoán `unsafe` (safe recall toàn cục chỉ `0.4333`), có lợi cho site lệch unsafe nhưng gây nhiều false alarm ở site cân bằng/lệch safe.
- **Communication cost = 73,872 bytes**, y hệt EXP-005, xác nhận update size **không phụ thuộc số mẫu** — chỉ phụ thuộc head `512×2 + bias` (4,104 bytes/round) × 3 client × 3 round × 2 (gửi+nhận).
- **OOM-safe được giữ nguyên ở 2× dữ liệu:** `summary.json` báo `num_clients=3`, `batch_size=4`, `num_workers=0`, `ray_num_cpus=1`. Peak RSS `406.42 MB` (federated) so với `354.72 MB` của EXP-005 — tăng nhẹ, vẫn an toàn trên máy cá nhân. Precompute 480 ảnh mất ~66s (lazy, batch 4).
- **Tái lập:** manifest giờ sinh bằng `scripts/generate_ppe_manifest.py` (seed cố định) + có unit test `tests/test_generate_ppe_manifest.py`, khắc phục điểm yếu manifest làm tay của EXP-004/005.
- Kết quả vẫn **chưa dùng để kết luận chất lượng PPE nghiệp vụ**: nhãn `safe/unsafe` là proxy image-level từ object annotation, chưa phải nhãn compliance được người review.

## 6. Tài Liệu Hướng Dẫn Tái Lập (Reproducibility & Artifacts)
- **Lệnh sinh manifest:**
  ```bash
  venv/bin/python scripts/generate_ppe_manifest.py \
    --voc-dir data/ppe/voc_labels \
    --images-dir data/ppe/images \
    --output configs/datasets/ppe_real_exp006_manifest.csv \
    --sites site-a,site-b,site-c \
    --safe-ratios 0.75,0.25,0.5 \
    --per-site 160 \
    --val-fraction 0.25 \
    --seed 2026
  ```
- **Lệnh tạo artifact:**
  ```bash
  venv/bin/python scripts/precompute_embeddings.py \
    --backend torchvision-resnet18 --weights imagenet \
    --manifest configs/datasets/ppe_real_exp006_manifest.csv \
    --root-dir data/ppe \
    --output data/processed/ppe_real_embeddings_exp006.npz \
    --batch-size 4 --num-workers 0 --device cpu --require-files
  ```
- **Lệnh chạy baseline:**
  ```bash
  venv/bin/python scripts/run_embedding_demo.py \
    --mode all \
    --artifact data/processed/ppe_real_embeddings_exp006.npz \
    --profile oom-safe \
    --output-dir outputs/EXP-006 \
    --exp-id EXP-006 \
    --batch-size 4 \
    --local-epochs 3 \
    --centralized-epochs 3 \
    --num-rounds 3 \
    --lr 0.01
  ```
- **Đường dẫn outputs:** `outputs/EXP-006/`
- **File output chính:**
  - `outputs/EXP-006/centralized_metrics.json`
  - `outputs/EXP-006/local_only_metrics.json`
  - `outputs/EXP-006/federated_metrics.json`
  - `outputs/EXP-006/summary.json`

## 7. Bước Tiếp Theo (Next Steps)
- [x] Tăng số round/epoch (vẫn OOM-safe) để xem federated có vượt local-only khi train đủ lâu không. *(Đã làm: EXP-007 — 10 round/5 epoch, federated vượt local-only.)*
- [x] Thử head 2 lớp (MLP nhỏ) hoặc chuẩn hóa embedding (L2-normalize) để cải thiện safe recall. *(Đã làm: EXP-009 L2-norm + EXP-010 MLP. L2-norm là lever đúng; MLP không giúp FL.)*
- [ ] Quét nhanh learning rate / số round như một ablation nhỏ trên manifest EXP-006 cố định.
- [ ] Cải thiện định nghĩa nhãn PPE/compliance thay vì proxy `has_core_ppe` trước khi kết luận nghiệp vụ.
- [ ] Sau khi nhãn ổn hơn, mới xét personalized head, adapter/LoRA, FedBN hoặc giảm communication.
