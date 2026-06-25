# Thử Nghiệm: PPE Embedding MLP Head Capacity Sweep (480/3)
- **Mã Thử Nghiệm:** EXP-010
- **Ngày Thực Hiện:** 2026-06-25
- **Trạng Thái:** Thành công
- **Git Commit Hash:** `pending`

---

## 1. Mục Tiêu (Objective)
- Vế thứ hai của câu hỏi mở `EXP-008`: trần ~0.60 có phải do **capacity** (head tuyến tính quá yếu) không?
- Can thiệp: thay head tuyến tính `512 -> 2` bằng **MLP 2 lớp** `512 -> H -> 2` (ReLU), sweep `H ∈ {32, 64, 128}`.
- Ablation 1 biến so `EXP-007` (giữ raw embedding, lr/budget y hệt, chỉ thêm `--head-hidden-dim`). **Headline = H=64.**
- Tái dùng artifact `data/processed/ppe_real_embeddings_exp006.npz`.

## 2. Cấu Hình & Thiết Lập (Configuration)
- **Bài toán & dữ liệu:** y hệt `EXP-006/007` (480 mẫu, 3 site non-IID, frozen ResNet18 dim 512).
- **Mô Hình & PEFT:** frozen ResNet18 + **MLP head** `Linear(512,H) -> ReLU -> Linear(H,2)` (chỉ MLP trainable). Serialize **4 mảng** `[fc1.w, fc1.b, fc2.w, fc2.b]` → FedAvg aggregate positional vẫn đúng, nhưng **update size tăng mạnh**.
- **Siêu tham số (giữ như EXP-007):** profile `oom-safe`, lr `0.01`, batch `4`, local epochs `5`, centralized epochs `15`, FL rounds `10`, SGD, weight_decay `0`.

## 3. So Sánh Kết Quả Giữa Các Baseline (Baseline Comparison)
Headline `H = 64`:

| Chế Độ Huấn Luyện (Mode) | Global Loss | Global Metric (Acc/Macro-F1) | Unsafe Recall / FNR | Tổng Thời Gian Train | Kích Thước Update (Per Round) | Tổng Chi Phí Truyền Thông (Comm. Cost) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized** (Pooled data) | 0.8424 | 0.6333 / 0.6312 | 0.7167 / 0.2833 | ~0.9s | 131,848 bytes | N/A |
| **Local-Only** (Không collab) | 0.6010 | 0.6667 / 0.6643 | 0.4333 / 0.5667 | ~0.06s | 131,848 bytes | N/A |
| **Federated** (FedAvg Baseline) | 0.9617 | 0.6333 / 0.6329 | 0.6000 / 0.4000 | 3.45s | 131,848 bytes | 7,910,880 bytes |

### Sweep hidden_dim (federated, cùng dữ liệu/budget; so EXP-007 = linear/no hidden)
| hidden_dim | Fed Acc/Macro-F1 | Fed unsafe recall | Val loss r1 → r10 | Δloss (diverge?) | Update size |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **linear** (EXP-007) | 0.608 / 0.608 | 0.583 | 0.803 → 0.955 | +0.152 | 4,104 B |
| 32 | 0.592 / 0.591 | 0.633 | 0.654 → 1.006 | **+0.352 (diverge nặng)** | 65,928 B |
| **64** | 0.633 / 0.633 | 0.600 | 0.656 → 0.962 | +0.306 | 131,848 B |
| 128 | 0.608 / 0.608 | 0.600 | 0.662 → 0.976 | +0.314 | 263,688 B |

## 4. Kết Quả Chi Tiết Theo Từng Client (Per-Client Metrics)
Mode `federated`, headline `H = 64`, sau 10 round.

| Client ID / Site | Số Lượng Validation | Local Metric (Acc/Macro-F1) | Unsafe Recall / FNR | Confusion Matrix `[safe, unsafe]` | Ghi Chú Đặc Điểm Dữ Liệu |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **site-a** | 40 | 0.6250 / 0.5943 | 0.7000 / 0.3000 | `[[18, 12], [3, 7]]` | Label histogram `{safe: 120, unsafe: 40}` |
| **site-b** | 40 | 0.7000 / 0.6703 | 0.6667 / 0.3333 | `[[8, 2], [10, 20]]` | Label histogram `{safe: 40, unsafe: 120}` |
| **site-c** | 40 | 0.5750 / 0.5683 | 0.4500 / 0.5500 | `[[14, 6], [11, 9]]` | Label histogram `{safe: 80, unsafe: 80}` |

Global federated confusion matrix: `[[40, 20], [24, 36]]` → safe recall `0.6667`, unsafe recall `0.6000`.

## 5. Quan Sát & Phân Tích (Observations & Rationale)
- **Capacity KHÔNG phải lever đúng cho federated.** Thêm MLP head **không nâng federated macro-F1** vượt mức ý nghĩa (`0.591–0.633` qua mọi H, so EXP-007 `0.608`). Đáng nói hơn: **val loss diverge NẶNG hơn cả EXP-007** (Δ +0.30…+0.35 so +0.152) — MLP overfit mạnh trên dữ liệu nhỏ mỗi client, và bước trung bình hóa FedAvg trên head phi tuyến kém hiệu quả hơn trên head tuyến tính.
- **MLP chỉ giúp local-only fit, không giúp FL:** local-only macro-F1 lên `0.66–0.69` (cao nhất khi H=128), xác nhận capacity giúp **một model trên data gộp/cục bộ**, nhưng lợi ích đó **không sống sót qua aggregation**. Đây là một kết quả âm tính sạch và có giá trị định hướng.
- **Chi phí truyền thông nổ 16×–64×:** update size `65,928 → 263,688` bytes (so `4,104` của head tuyến tính), comm cost lên tới **15.8 MB** ở H=128. Với mục tiêu edge/PEFT, đây là đánh đổi **không chấp nhận được** khi không có lợi ích chất lượng FL.
- **Đối chiếu trực tiếp với EXP-009:** L2-norm nâng centralized `0.596 → 0.623` với **0 byte thêm**; MLP H=64 nâng centralized `0.596 → 0.631` nhưng tốn **32× update size** và làm FL diverge. → **Kết luận: biểu diễn (L2-norm) là lever rẻ và đúng; capacity (MLP) là lever sai cho FL trên cấu hình này.**
- **OOM-safe giữ nguyên:** peak RSS `453.84 MB` (H=64, federated), wall time `3.45s`. Vẫn an toàn dù head lớn hơn.
- Vẫn là **proxy label image-level**.

## 6. Tài Liệu Hướng Dẫn Tái Lập (Reproducibility & Artifacts)
- **Lệnh chạy sweep (tái dùng artifact EXP-006):**
  ```bash
  for H in 32 64 128; do
    venv/bin/python scripts/run_embedding_demo.py --mode all \
      --artifact data/processed/ppe_real_embeddings_exp006.npz \
      --profile oom-safe --output-dir outputs/EXP-010-h$H --exp-id EXP-010 \
      --batch-size 4 --local-epochs 5 --centralized-epochs 15 --num-rounds 10 \
      --lr 0.01 --head-hidden-dim $H
  done
  ```
- **Thay đổi code (mới, chung với EXP-009):** `EmbeddingHeadClassifier` hỗ trợ `hidden_dim` (MLP 2 lớp); get/set head params tổng quát hóa theo `model.head.parameters()` (2 mảng linear / 4 mảng MLP); `DemoConfig.head_hidden_dim`; CLI `--head-hidden-dim`. Test: 63 passed.
- **Đường dẫn outputs:** `outputs/EXP-010-h32/`, `outputs/EXP-010-h64/`, `outputs/EXP-010-h128/`.

## 7. Bước Tiếp Theo (Next Steps)
- [ ] **Không theo đuổi MLP head cho FL** (diverge + comm cost lớn, không lợi ích). Nếu cần capacity, chỉ xét trên centralized/local, không trên update truyền đi.
- [ ] Lever đúng là **biểu diễn (L2-norm, EXP-009)** + **personalization** (FedBN / personalized head) — ưu tiên các hướng này.
- [ ] Cải thiện nhãn PPE/compliance vẫn là trần absolute còn lại.
