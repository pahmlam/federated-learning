# Thử Nghiệm: PPE Real ResNet18 Train-Longer Baseline (480/3, 10 round)
- **Mã Thử Nghiệm:** EXP-007
- **Ngày Thực Hiện:** 2026-06-25
- **Trạng Thái:** Thành công
- **Git Commit Hash:** `09098db`

---

## 1. Mục Tiêu (Objective)
- Thực hiện Next-step #1 của `EXP-006`: **tăng số round/epoch** (vẫn OOM-safe) để trả lời câu hỏi *"Federated có vượt local-only khi train đủ lâu không?"*.
- Biến kiểm soát: giữ **nguyên dữ liệu (manifest/artifact EXP-006), nguyên lr=0.01, nguyên profile oom-safe**; chỉ thay budget train. Nhờ vậy so sánh EXP-007 vs EXP-006 cô lập đúng tác động của việc train lâu hơn.
- Tái dùng artifact `data/processed/ppe_real_embeddings_exp006.npz` → **bỏ qua hoàn toàn pha precompute** (chỉ train head trên embedding đã lưu).

## 2. Cấu Hình & Thiết Lập (Configuration)
- **Bài toán:** PPE binary classification smoke (`safe` / `unsafe`).
- **Dữ Liệu & Phân Chia Client:** y hệt `EXP-006`.
  - Manifest: `configs/datasets/ppe_real_exp006_manifest.csv` (480 mẫu, seed 2026).
  - Artifact: `data/processed/ppe_real_embeddings_exp006.npz` (frozen ResNet18, dim 512).
  - 3 clients non-IID label skew: site-a (120 safe/40 unsafe), site-b (40/120), site-c (80/80); mỗi site val 40.
  - Label mapping: `{"safe": 0, "unsafe": 1}`.
- **Mô Hình & PEFT:** frozen `torchvision-resnet18` (ImageNet) + classifier head `512 -> 2` (chỉ head trainable).
- **Siêu Tham Số (thay đổi so với EXP-006 in đậm):**
  - Profile: `oom-safe` (giữ: `num_workers=0`, `ray_num_cpus=1`, `client_num_cpus=1.0`, `batch_size=4`)
  - Learning Rate: `0.01` (giữ)
  - Batch Size: `4` (giữ)
  - Local Epochs: **`5`** (EXP-006: 3)
  - Centralized Epochs: **`15`** (EXP-006: 3)
  - FL Rounds: **`10`** (EXP-006: 3)
  - Optimizer: `SGD`

## 3. So Sánh Kết Quả Giữa Các Baseline (Baseline Comparison)

| Chế Độ Huấn Luyện (Mode) | Global Loss | Global Metric (Acc/Macro-F1) | Unsafe Recall / FNR | Tổng Thời Gian Train | Kích Thước Update (Per Round) | Tổng Chi Phí Truyền Thông (Comm. Cost) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized** (Pooled data) | 1.0619 | 0.6000 / 0.5960 | 0.5000 / 0.5000 | 0.94s | 4,104 bytes | N/A |
| **Local-Only** (Không collab) | 0.8200 | 0.6000 / 0.5908 | 0.3889 / 0.6111 | 0.06s | 4,104 bytes | N/A |
| **Federated** (FedAvg Baseline) | 0.9554 | 0.6083 / 0.6081 | 0.5833 / 0.4167 | 4.35s | 4,104 bytes | 246,240 bytes |

### So sánh trực tiếp với EXP-006 (cùng dữ liệu, chỉ khác budget)
| Mode | EXP-006 (3ep/3rd) Acc/Macro-F1 | EXP-007 (5ep/10rd) Acc/Macro-F1 |
| :--- | :---: | :---: |
| Centralized | 0.5583 / 0.5173 | **0.6000 / 0.5960** |
| Local-Only | 0.6250 / **0.6173** | 0.6000 / 0.5908 |
| Federated | 0.5917 / 0.5812 | **0.6083 / 0.6081** |

## 4. Kết Quả Chi Tiết Theo Từng Client (Per-Client Metrics)
Mode `federated` sau 10 round.

| Client ID / Site | Số Lượng Validation | Local Validation Loss | Local Metric (Acc/Macro-F1) | Unsafe Recall / FNR | Confusion Matrix `[safe, unsafe]` | Ghi Chú Đặc Điểm Dữ Liệu |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **site-a** | 40 | 0.9790 | 0.6500 / 0.6154 | 0.7000 / 0.3000 | `[[19, 11], [3, 7]]` | Label histogram `{safe: 120, unsafe: 40}` |
| **site-b** | 40 | 0.8052 | 0.6000 / 0.5604 | 0.6000 / 0.4000 | `[[6, 4], [12, 18]]` | Label histogram `{safe: 40, unsafe: 120}` |
| **site-c** | 40 | 1.0818 | 0.5750 / 0.5726 | 0.5000 / 0.5000 | `[[13, 7], [10, 10]]` | Label histogram `{safe: 80, unsafe: 80}` |

Global federated confusion matrix: `[[38, 22], [25, 35]]` → safe recall `0.6333`, unsafe recall `0.5833`.

### Đường hội tụ Federated (distributed weighted theo round)
| Round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Macro-F1 | 0.498 | 0.547 | 0.560 | **0.588** | 0.579 | 0.564 | 0.571 | 0.583 | 0.583 | 0.583 |
| Val Loss | 0.803 | 0.822 | 0.833 | 0.847 | 0.863 | 0.881 | 0.899 | 0.918 | 0.937 | 0.955 |

## 5. Quan Sát & Phân Tích (Observations & Rationale)
- **Trả lời câu hỏi mở của EXP-006: CÓ — khi train lâu hơn, FedAvg vượt local-only.** Federated macro-F1 `0.5812 → 0.6081` và lần này **cao hơn local-only** (`0.6081` vs `0.5908`), đồng thời unsafe recall tốt hơn rõ (`0.5833` vs `0.3889`). Ở EXP-006 (3 round) thì federated còn thua local-only.
- **Nhưng "train lâu hơn" có giới hạn — hội tụ sớm rồi chững:** macro-F1 federated đạt đỉnh ~round 4 (`0.588`) rồi đi ngang ~`0.583`. Đáng chú ý: **val loss tăng đều mỗi round** (`0.803 → 0.955`) trong khi acc/F1 gần như không đổi → dấu hiệu **head ngày càng "quá tự tin" (overconfident)**, logits bị đẩy mạnh làm cross-entropy trên val xấu đi dù ranh giới quyết định gần như đứng yên. Tức tăng round/epoch quá ngưỡng **không còn cải thiện chất lượng**, chỉ làm loss tệ hơn.
- **FedAvg tỏ ra ổn định hơn local-only khi tăng local epoch:** đẩy local-epochs 3→5 làm **local-only xấu đi** (macro-F1 `0.6173 → 0.5908`, unsafe recall `0.7222 → 0.3889`) vì mỗi client nhỏ overfit cục bộ; trong khi **bước trung bình hóa của FedAvg đóng vai trò regularizer**, giúp federated vẫn nhích lên. Đây là một luận điểm ủng hộ FL khá sạch trên cấu hình này.
- **Centralized** cải thiện theo budget (`0.5173 → 0.5960` macro-F1 với 15 epoch), như kỳ vọng khi train hội tụ hơn.
- **Non-IID:** trong federated, site-a (safe-heavy) giờ tốt nhất (Acc `0.6500`), các site khác `~0.58-0.60`; phân bố cân bằng hơn EXP-006 (global confusion `[[38,22],[25,35]]`, safe recall `0.6333` thay vì `0.4333` ở EXP-006) — head bớt thiên về `unsafe`.
- **Comm cost = 246,240 bytes**, gấp ~3.33× EXP-006 (`73,872`), đúng tỉ lệ round `10/3` (update size không đổi `4,104` bytes/round × 3 client × 10 round × 2). Xác nhận chi phí truyền thông tỉ lệ thuận số round — một đánh đổi cần cân nhắc khi "train lâu hơn".
- **OOM-safe giữ nguyên:** `summary.json` báo `num_clients=3`, `batch_size=4`, `num_workers=0`, `ray_num_cpus=1`, đúng budget `5/15/10`. Peak RSS `441.02 MB` (federated, so `406.42` ở EXP-006) — tăng nhẹ do 10 round, vẫn rất an toàn trên máy Apple M2 16 GB. Wall time federated `4.35s`.
- Vẫn là **proxy label image-level**, chưa kết luận chất lượng PPE nghiệp vụ.

## 6. Tài Liệu Hướng Dẫn Tái Lập (Reproducibility & Artifacts)
- **Lệnh chạy baseline (tái dùng artifact EXP-006):**
  ```bash
  venv/bin/python scripts/run_embedding_demo.py \
    --mode all \
    --artifact data/processed/ppe_real_embeddings_exp006.npz \
    --profile oom-safe \
    --output-dir outputs/EXP-007 \
    --exp-id EXP-007 \
    --batch-size 4 \
    --local-epochs 5 \
    --centralized-epochs 15 \
    --num-rounds 10 \
    --lr 0.01
  ```
  *(Artifact tạo lại từ manifest đã commit bằng các lệnh `generate_ppe_manifest.py` + `precompute_embeddings.py` trong journal EXP-006.)*
- **Đường dẫn outputs:** `outputs/EXP-007/` (`centralized_metrics.json`, `local_only_metrics.json`, `federated_metrics.json`, `summary.json`).

## 7. Bước Tiếp Theo (Next Steps)
- [ ] Vì tăng budget đã chững và val loss diverge: thử **chống overfit/regularize** thay vì train thêm — L2-normalize embedding, weight decay, hoặc lr nhỏ hơn + early stopping theo val loss.
- [ ] Thử **head 2 lớp (MLP nhỏ)** xem có vượt trần ~0.60 macro-F1 không.
- [ ] Ablation nhỏ lr × round trên manifest EXP-006 cố định để xác định điểm dừng tối ưu (tránh round thừa tốn comm).
- [ ] Cải thiện nhãn PPE/compliance (thay proxy `has_core_ppe`) trước khi kết luận nghiệp vụ.
- [ ] Sau khi nhãn ổn: xét personalized head, adapter/LoRA, FedBN — đặc biệt khi đã thấy FedAvg regularize tốt hơn local-only.
