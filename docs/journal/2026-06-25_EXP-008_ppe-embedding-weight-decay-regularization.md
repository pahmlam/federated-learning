# Thử Nghiệm: PPE Embedding Weight-Decay Regularization (480/3)
- **Mã Thử Nghiệm:** EXP-008
- **Ngày Thực Hiện:** 2026-06-25
- **Trạng Thái:** Thành công
- **Git Commit Hash:** `d7352ff`

---

## 1. Mục Tiêu (Objective)
- Thực hiện Next-step #1 của `EXP-007`: **chống overfit bằng regularize** thay vì train thêm.
- `EXP-007` cho thấy head tuyến tính chững ~0.60 macro-F1 từ ~round 4 nhưng **val loss tăng đều mỗi round** (0.803 → 0.955) — dấu hiệu overconfidence.
- Can thiệp được chọn: **chỉ thêm `weight_decay`** vào optimizer SGD (giữ head tuyến tính + embedding thô). Ablation 1 biến sạch so EXP-007 (mọi thứ khác giữ y hệt).
- Câu hỏi: weight_decay có **chặn val loss diverge** / cải thiện calibration / nâng macro-F1 không?

## 2. Cấu Hình & Thiết Lập (Configuration)
- **Bài toán & dữ liệu:** y hệt `EXP-006`/`EXP-007`. Tái dùng artifact `data/processed/ppe_real_embeddings_exp006.npz` (480 mẫu, 3 site non-IID, frozen ResNet18 dim 512). **Không precompute lại.**
- **Mô Hình & PEFT:** frozen ResNet18 + classifier head tuyến tính `512 -> 2` (chỉ head trainable). Serialize tham số `[head.weight, head.bias]` không đổi → FedAvg aggregation giữ nguyên.
- **Siêu tham số (giữ như EXP-007, chỉ thêm weight_decay):**
  - Profile: `oom-safe` (`num_workers=0`, `ray_num_cpus=1`, `batch_size=4`)
  - lr `0.01`, batch `4`, local epochs `5`, centralized epochs `15`, FL rounds `10`, optimizer `SGD`
  - **`weight_decay` ∈ {1e-3, 1e-2, 1e-1}** (sweep). EXP-007 = weight_decay `0`.
  - **Headline = `weight_decay = 1e-1`** (giá trị duy nhất triệt tiêu được hiện tượng diverge).

## 3. So Sánh Kết Quả Giữa Các Baseline (Baseline Comparison)
Headline `weight_decay = 1e-1`:

| Chế Độ Huấn Luyện (Mode) | Global Loss | Global Metric (Acc/Macro-F1) | Unsafe Recall / FNR | Tổng Thời Gian Train | Kích Thước Update (Per Round) | Tổng Chi Phí Truyền Thông (Comm. Cost) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized** (Pooled data) | 0.8769 | 0.5750 / 0.5616 | 0.4000 / 0.6000 | 0.74s | 4,104 bytes | N/A |
| **Local-Only** (Không collab) | 0.8082 | 0.6083 / 0.5960 | 0.3778 / 0.6222 | 0.06s | 4,104 bytes | N/A |
| **Federated** (FedAvg Baseline) | 0.7941 | 0.6167 / 0.6114 | 0.5000 / 0.5000 | 3.08s | 4,104 bytes | 246,240 bytes |

### Sweep weight_decay (federated, cùng dữ liệu/budget; wd=0 là EXP-007)
| weight_decay | Fed Acc/Macro-F1 | Fed unsafe recall | Val loss r1 → r10 | Δloss (diverge?) |
| :---: | :---: | :---: | :---: | :---: |
| **0** (EXP-007) | 0.5917 / 0.5812 | 0.7500 | 0.803 → 0.955 | **+0.152 (diverge)** |
| 1e-3 | 0.6083 / 0.6081 | 0.5833 | 0.803 → 0.952 | +0.149 (vẫn diverge) |
| 1e-2 | 0.6000 / 0.5996 | 0.5667 | 0.802 → 0.924 | +0.122 (giảm nhẹ) |
| **1e-1** | 0.6167 / 0.6114 | 0.5000 | 0.794 → 0.794 | **+0.000 (hết diverge)** |

## 4. Kết Quả Chi Tiết Theo Từng Client (Per-Client Metrics)
Mode `federated`, headline `weight_decay = 1e-1`, sau 10 round.

| Client ID / Site | Số Lượng Validation | Local Validation Loss | Local Metric (Acc/Macro-F1) | Unsafe Recall / FNR | Confusion Matrix `[safe, unsafe]` | Ghi Chú Đặc Điểm Dữ Liệu |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **site-a** | 40 | 0.6655 | 0.7250 / 0.6800 | 0.7000 / 0.3000 | `[[22, 8], [3, 7]]` | Label histogram `{safe: 120, unsafe: 40}` |
| **site-b** | 40 | 0.8048 | 0.6250 / 0.6050 | 0.5667 / 0.4333 | `[[8, 2], [13, 17]]` | Label histogram `{safe: 40, unsafe: 120}` |
| **site-c** | 40 | 0.9119 | 0.5000 / 0.4792 | 0.3000 / 0.7000 | `[[14, 6], [14, 6]]` | Label histogram `{safe: 80, unsafe: 80}` |

Global federated confusion matrix: `[[44, 16], [30, 30]]` → safe recall `0.7333`, unsafe recall `0.5000`.

## 5. Quan Sát & Phân Tích (Observations & Rationale)
- **weight_decay KHÔNG nâng được trần macro-F1:** mọi giá trị wd cho federated macro-F1 ~`0.60-0.61`, ngang EXP-007 (`0.5812`). → Trần ~0.60 là **vấn đề cấu trúc** (head tuyến tính + nhãn proxy), không phải do thiếu regularization. Đây là kết luận quan trọng để định hướng EXP sau.
- **weight_decay CÓ sửa được overconfidence/diverge — nhưng cần đủ mạnh:** chỉ `wd=1e-1` mới làm **val loss đi ngang** (Δ +0.000, từ 0.794 → 0.794) thay vì tăng dốc như EXP-007 (Δ +0.152). `wd=1e-3`/`1e-2` gần như không đủ. Tức mô hình hết "tự đẩy logit", **calibration tốt hơn** ở wd cao.
- **Đánh đổi an toàn (quan trọng cho PPE):** wd càng lớn, dự đoán càng dịch về cân bằng/`safe` → **unsafe recall giảm** (`0.75 @ wd=0 → 0.50 @ wd=1e-1`). Với mục tiêu bắt vi phạm (giảm bỏ sót), wd cao **làm xấu unsafe recall** dù calibration tốt hơn. Đây là trade-off thật, không có lựa chọn thắng tuyệt đối.
- **Federated vẫn ≥ local-only** ở mọi wd (headline: macro-F1 `0.6114` vs `0.5960`; unsafe recall `0.500` vs `0.378`), củng cố quan sát EXP-007 rằng FedAvg regularize tốt hơn train cục bộ.
- **Centralized hơi underfit ở wd=1e-1** (macro-F1 `0.5616`, thấp hơn `0.596` ở wd nhỏ) — pooled data + wd mạnh kéo model về đơn giản hơn.
- **Comm cost = 246,240 bytes**, **không đổi** so EXP-007 — weight_decay không ảnh hưởng kích thước update (vẫn `4,104` bytes/round × 3 × 10 × 2).
- **OOM-safe giữ nguyên:** config `num_clients=3`, `batch_size=4`, `num_workers=0`, `ray_num_cpus=1`. Peak RSS `451-460 MB`, federated ~3s/run. 3 run sweep tổng vài giây.
- Vẫn là **proxy label image-level**, chưa kết luận chất lượng PPE nghiệp vụ.

## 6. Tài Liệu Hướng Dẫn Tái Lập (Reproducibility & Artifacts)
- **Lệnh chạy sweep (tái dùng artifact EXP-006):**
  ```bash
  for WD in 1e-3 1e-2 1e-1; do
    venv/bin/python scripts/run_embedding_demo.py --mode all \
      --artifact data/processed/ppe_real_embeddings_exp006.npz \
      --profile oom-safe --output-dir outputs/EXP-008-wd$WD --exp-id EXP-008 \
      --batch-size 4 --local-epochs 5 --centralized-epochs 15 --num-rounds 10 \
      --lr 0.01 --weight-decay $WD
  done
  ```
- **Thay đổi code (mới):** thêm `weight_decay` xuyên suốt — `DemoConfig` (+validate), `train_head` (SGD), 2 baseline, FL client `fit` + `on_fit_config_fn`, CLI `--weight-decay` + summary. Test: `tests/test_config.py`, `tests/test_embedding_pipeline.py` (54 passed).
- **Đường dẫn outputs:** `outputs/EXP-008-wd1e-3/`, `outputs/EXP-008-wd1e-2/`, `outputs/EXP-008-wd1e-1/` (mỗi cái có 4 JSON).

## 7. Bước Tiếp Theo (Next Steps)
- [x] Vì regularize không nâng trần ~0.60: thử **đổi biểu diễn/capacity** — L2-normalize embedding (cosine-style, EXP-009) hoặc MLP head 2 lớp (EXP-010). *(Đã làm: EXP-009 L2-norm **vượt trần** centralized 0.596→0.623 + chặn diverge miễn phí + unsafe recall 0.717; EXP-010 MLP **không** giúp FL, diverge nặng + comm nổ 16-64×. → Trần là biểu diễn, không phải capacity.)*
- [ ] Nếu cần ưu tiên unsafe recall: cân nhắc class weight / threshold tuning thay vì wd mạnh (vì wd cao làm tụt unsafe recall). *(EXP-009 đã đạt unsafe recall 0.717 không cần wd — ưu tiên thấp hơn giờ.)*
- [ ] Cải thiện nhãn PPE/compliance (thay proxy `has_core_ppe`) — nhiều khả năng đây mới là trần thật.
- [ ] Khi nhãn ổn: personalized head / adapter-LoRA / FedBN.
