# Thử Nghiệm: [Tên Thử Nghiệm Rõ Ràng]
- **Mã Thử Nghiệm:** EXP-XXX (Ví dụ: EXP-001)
- **Ngày Thực Hiện:** YYYY-MM-DD
- **Trạng Thái:** [Thành công / Thất bại / Đang chạy]
- **Git Commit Hash:** `[Nhập commit hash]`

---

## 1. Mục Tiêu (Objective)
- *Mô tả ngắn gọn mục tiêu của thử nghiệm này (Ví dụ: Đánh giá hiệu năng của thuật toán FedAvg so với Local-only trên dữ liệu PPE bị lệch nhãn giữa các site).*

## 2. Cấu Hình & Thiết Lập (Configuration)
- **Bài toán:** [Face Recognition / PPE Compliance / Toy-task / ...]
- **Dữ Liệu & Phân Chia Client (Data & Partition):**
  - Tên dataset: [Ví dụ: CASIA-WebFace, LFW, hoặc dữ liệu PPE nội bộ]
  - Số lượng client mô phỏng: [Ví dụ: 5 clients]
  - Loại phân chia dữ liệu: [IID / Non-IID (Feature skew, Label skew, Quantity skew)]
  - Mô tả phân phối cụ thể: [Ví dụ: Client 1 chiếm 80% nhãn A, Client 2 chiếm 90% nhãn B...]
- **Mô Hình & Huấn Luyện Tham Số Hiệu Quả (Model & PEFT):**
  - Pretrained Backbone: [Ví dụ: MobileNetV3 (Frozen)]
  - Phần Trainable (Trainable Head/Embedding/Adapter/LoRA): [Ví dụ: Classifier Task Head]
- **Siêu Tham Số (Hyperparameters):**
  - Learning Rate: `0.001`
  - Batch Size: `32`
  - Local Epochs: `5`
  - FL Rounds (Số round): `10`
  - Optimizer: `Adam`

## 3. So Sánh Kết Quả Giữa Các Baseline (Baseline Comparison)
*Bắt buộc phải báo cáo và đối chiếu giữa 3 baseline theo yêu cầu của dự án.*

| Chế Độ Huấn Luyện (Mode) | Global Loss | Global Metric (Acc/F1/mAP) | Tổng Thời Gian Train | Kích Thước Update (Per Round) | Tổng Chi Phí Truyền Thông (Comm. Cost) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Centralized** (Pooled data) | | | | | *N/A (Không truyền thông)* |
| **Local-Only** (Không collab) | *N/A* | *(Trung bình các client)* | | *N/A (Không truyền thông)* | *N/A (Không truyền thông)* |
| **Federated** (FedAvg Baseline) | | | | | |
| **Federated + Method Mới** (Nếu có) | | | | | |

## 4. Kết Quả Chi Tiết Theo Từng Client (Per-Client Metrics)
*Báo cáo kết quả của chế độ Federated (hoặc Federated + Method mới) trên từng client/site.*

| Client ID / Site | Số Lượng Dữ Liệu | Local Validation Loss | Local Metric (Acc/F1/mAP) | Ghi Chú Đặc Điểm Dữ Liệu (Ví dụ: ánh sáng yếu, góc quay cao...) |
| :--- | :---: | :---: | :---: | :--- |
| **Client 1** | | | | |
| **Client 2** | | | | |
| **Client 3** | | | | |
| **Client 4** | | | | |
| **Client 5** | | | | |

## 5. Quan Sát & Phân Tích (Observations & Rationale)
- **Hội tụ:** [Mô hình có hội tụ ổn định không? Cần bao nhiêu round để đạt độ chính xác tối ưu?]
- **Hiệu quả PEFT:** [Việc freeze backbone và chỉ truyền tham số head/adapter giúp tiết kiệm dung lượng truyền thông bao nhiêu % so với truyền full model?]
- **Hiện tượng Non-IID:** [Sự khác biệt về dữ liệu giữa các client ảnh hưởng thế nào đến độ chính xác cục bộ (local metrics)?]
- **Lỗi phát sinh (nếu có):** [Ví dụ: Out of memory, mất kết nối client, v.v.]

## 6. Tài Liệu Hướng Dẫn Tái Lập (Reproducibility & Artifacts)
- **Lệnh chạy:**
  ```bash
  # Ví dụ: python experiments/run_exp.py --config configs/exp_xxx.yaml
  ```
- **Đường dẫn outputs:** `outputs/EXP-XXX/` (Chứa log Tensorboard, file JSON kết quả, checkpoint mô hình).

## 7. Bước Tiếp Theo (Next Steps)
- [ ] *Ghi chú các ý tưởng cải tiến tiếp theo từ kết quả của thử nghiệm này.*
