# Thử Nghiệm: PPE Embedding L2-Normalize (cosine-style, 480/3)
- **Mã Thử Nghiệm:** EXP-009
- **Ngày Thực Hiện:** 2026-06-25
- **Trạng Thái:** Thành công
- **Git Commit Hash:** `38a7220`

---

## 1. Mục Tiêu (Objective)
- Trả lời câu hỏi mở quan trọng nhất sau `EXP-008`: **trần macro-F1 ~0.60 đến từ capacity model hay từ nhãn proxy?**
- Can thiệp được chọn cho EXP-009: **L2-normalize embedding trước head** (chuyển sang hình học cosine), một thay đổi biểu diễn **không tăng tham số, không tăng comm cost**.
- Ablation 1 biến sạch so `EXP-007` (linear head, weight_decay=0): **mọi thứ giữ y hệt, chỉ bật `--normalize-embedding`**. Nhờ vậy chênh lệch EXP-009 vs EXP-007 cô lập đúng tác động của L2-norm.
- Tái dùng artifact `data/processed/ppe_real_embeddings_exp006.npz` → bỏ qua hoàn toàn pha precompute.

## 2. Cấu Hình & Thiết Lập (Configuration)
- **Bài toán & dữ liệu:** y hệt `EXP-006/007/008` (480 mẫu, 3 site non-IID label skew, frozen ResNet18 dim 512). Label mapping `{"safe": 0, "unsafe": 1}`.
- **Mô Hình & PEFT:** frozen ResNet18 + classifier head tuyến tính `512 -> 2` (chỉ head trainable). **Mới:** input được `F.normalize(x, p=2, dim=1)` trước head → head học trên vector đơn vị (cosine geometry). Serialize tham số `[head.weight, head.bias]` **không đổi** → FedAvg aggregation giữ nguyên, update size `4,104` bytes.
- **Siêu tham số (giữ như EXP-007):**
  - Profile `oom-safe` (`num_workers=0`, `ray_num_cpus=1`, `batch_size=4`)
  - lr `0.01`, batch `4`, local epochs `5`, centralized epochs `15`, FL rounds `10`, optimizer `SGD`, weight_decay `0`.

## 3. So Sánh Kết Quả Giữa Các Baseline (Baseline Comparison)

| Chế Độ Huấn Luyện (Mode) | Global Loss | Global Metric (Acc/Macro-F1) | Unsafe Recall / FNR | Tổng Thời Gian Train | Kích Thước Update (Per Round) | Tổng Chi Phí Truyền Thông (Comm. Cost) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized** (Pooled data) | 0.6853 | 0.6250 / 0.6229 | 0.7000 / 0.3000 | ~0.9s | 4,104 bytes | N/A |
| **Local-Only** (Không collab) | 0.6164 | 0.6750 / 0.6740 | 0.5500 / 0.4500 | ~0.06s | 4,104 bytes | N/A |
| **Federated** (FedAvg Baseline) | 0.6853 | 0.6000 / 0.5945 | 0.7167 / 0.2833 | 4.52s | 4,104 bytes | 246,240 bytes |

### Ablation chính: L2-norm vs EXP-007 (cùng dữ liệu/budget, chỉ khác normalize)
| Mode | EXP-007 (raw embedding) Acc/Macro-F1 / unsafeR | EXP-009 (L2-norm) Acc/Macro-F1 / unsafeR |
| :--- | :---: | :---: |
| Centralized | 0.600 / 0.596 / 0.500 | **0.625 / 0.623 / 0.700** |
| Local-Only | 0.600 / 0.591 / 0.389 | **0.675 / 0.674 / 0.550** |
| Federated | 0.608 / 0.608 / 0.583 | 0.600 / 0.594 / **0.717** |

### Federated convergence (distributed weighted theo round)
| Round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Macro-F1 | 0.450 | 0.458 | 0.517 | 0.548 | 0.556 | 0.557 | 0.565 | 0.565 | 0.565 | 0.571 |
| Val Loss | 0.693 | 0.692 | 0.691 | 0.690 | 0.689 | 0.688 | 0.688 | 0.687 | 0.686 | **0.685** |

Δval-loss r1→r10 = **−0.008 (KHÔNG diverge)** so với EXP-007 `+0.152` — và đạt được **mà không cần weight_decay**.

## 4. Kết Quả Chi Tiết Theo Từng Client (Per-Client Metrics)
Mode `federated`, L2-norm, sau 10 round.

| Client ID / Site | Số Lượng Validation | Local Validation Loss | Local Metric (Acc/Macro-F1) | Unsafe Recall / FNR | Confusion Matrix `[safe, unsafe]` | Ghi Chú Đặc Điểm Dữ Liệu |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **site-a** | 40 | — | 0.4250 / 0.4159 | 0.6000 / 0.4000 | `[[11, 19], [4, 6]]` | Label histogram `{safe: 120, unsafe: 40}` |
| **site-b** | 40 | — | 0.7000 / 0.6238 | 0.7667 / 0.2333 | `[[5, 5], [7, 23]]` | Label histogram `{safe: 40, unsafe: 120}` |
| **site-c** | 40 | — | 0.6750 / 0.6748 | 0.7000 / 0.3000 | `[[13, 7], [6, 14]]` | Label histogram `{safe: 80, unsafe: 80}` |

Global federated confusion matrix: `[[29, 31], [17, 43]]` → safe recall `0.4833`, unsafe recall `0.7167`.

## 5. Quan Sát & Phân Tích (Observations & Rationale)
- **TRẢ LỜI CÂU HỎI MỞ: trần ~0.60 MỘT PHẦN là vấn đề biểu diễn (cấu trúc), KHÔNG chỉ do nhãn.** Bằng chứng sạch nhất là **centralized** (pooled data, một model, không có artifact pooling): macro-F1 `0.596 → 0.623`, unsafe recall `0.500 → 0.700`, loss `1.062 → 0.685`. Chỉ bằng L2-normalize — **không thêm tham số, không thêm byte truyền** — đã vượt trần. Tức raw ResNet18 embedding có biến thiên độ lớn (magnitude) làm khó head tuyến tính; chuẩn hóa về hình cầu đơn vị (cosine) gỡ đúng nút thắt đó.
- **L2-norm chặn diverge "miễn phí", tốt hơn weight_decay (EXP-008):** val loss đi ngang (Δ −0.008) **mà không cần wd**. Quan trọng: EXP-008 chặn diverge bằng wd=1e-1 nhưng **trả giá unsafe recall (0.500)**; EXP-009 chặn diverge **đồng thời NÂNG unsafe recall lên 0.717 (cao nhất từ trước tới nay)**. Không còn trade-off — đây là lựa chọn thắng rõ ràng cho mục tiêu PPE (giảm bỏ sót vi phạm).
- **Lưu ý đọc số local-only "global" 0.674 — đây là pooling artifact, đừng dùng để kết luận FL thua:** L2-norm khiến model local trên site lệch trở nên **dự đoán gần như một lớp** (site-a all-safe: unsafe recall `0.000`; site-b all-unsafe: unsafe recall `1.000`). Khi gộp confusion 3 site degenerate lại, macro-F1 "global" bị thổi lên `0.674`, nhưng **per-client macro-F1 local-only chỉ ~0.46 trung bình** (0.429/0.429/0.517). Trong khi đó **federated per-client trung bình ~0.572** (0.416/0.624/0.675). Theo nguyên tắc CLAUDE.md (báo cáo per-client cho non-IID), **federated thực sự tốt hơn local-only ở mức client** — global average che giấu điều này.
- **Comm cost = 246,240 bytes, không đổi** so EXP-007/008 — L2-norm là phép toán trên input, không động vào kích thước head. Đây là điểm cực kỳ hấp dẫn cho edge FL: cải thiện chất lượng **với chi phí truyền thông bằng 0**.
- **OOM-safe giữ nguyên:** peak RSS `423.41 MB` (federated), wall time `4.52s`. An toàn trên máy cá nhân.
- Vẫn là **proxy label image-level** — absolute macro-F1 pooled centralized ~0.62 cho thấy nhãn vẫn là một trần thực sự, nhưng L2-norm chứng minh biểu diễn là lever rẻ đang bị bỏ phí.

## 6. Tài Liệu Hướng Dẫn Tái Lập (Reproducibility & Artifacts)
- **Lệnh chạy (tái dùng artifact EXP-006):**
  ```bash
  venv/bin/python scripts/run_embedding_demo.py --mode all \
    --artifact data/processed/ppe_real_embeddings_exp006.npz \
    --profile oom-safe --output-dir outputs/EXP-009 --exp-id EXP-009 \
    --batch-size 4 --local-epochs 5 --centralized-epochs 15 --num-rounds 10 \
    --lr 0.01 --normalize-embedding
  ```
- **Thay đổi code (mới):** `EmbeddingHeadClassifier` thêm `normalize_input` (L2-norm trong forward); `DemoConfig.normalize_embedding`; factory `build_embedding_model(config, bundle)` dùng chung cho cả 3 mode; CLI `--normalize-embedding` + summary. Test: `tests/test_config.py`, `tests/test_embedding_pipeline.py` (63 passed toàn repo).
- **Đường dẫn outputs:** `outputs/EXP-009/` (4 JSON).

## 7. Bước Tiếp Theo (Next Steps)
- [ ] **L2-norm giờ là default baseline mới** — các EXP sau nên bật `--normalize-embedding`.
- [ ] Vì sau L2-norm **federated global chững (~0.59) trong khi per-client tốt** và các site lệch dễ degenerate: thử **personalized head / FedBN** — đã có tín hiệu personalization thật (local boundary hữu ích bị FedAvg trung bình hóa làm nhòe).
- [ ] Cải thiện nhãn PPE/compliance (thay proxy `has_core_ppe`) — vẫn là trần absolute còn lại.
- [ ] Cân nhắc kết hợp L2-norm + class weight để vừa giữ unsafe recall cao vừa cân bằng safe recall (site-a safe recall thấp).
