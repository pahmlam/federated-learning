# Nhật ký thử nghiệm Federated Learning

Thư mục này dùng **một file duy nhất** (`docs/journal/README.md`) để ghi toàn bộ nhật ký thử nghiệm và log tiến độ theo ngày. Raw artifacts vẫn được lưu riêng trong `outputs/EXP-XXX/`; file này chỉ lưu mục tiêu, cấu hình, metric chính, phân tích, lệnh tái lập và bước tiếp theo.

## Quy trình ghi journal

1. Xác định mã thử nghiệm tiếp theo trong bảng [Registry tổng hợp](#registry-tong-hop).
2. Chạy thử nghiệm và lưu raw outputs vào `outputs/EXP-XXX/`.
3. Append nội dung mới vào **ngày tương ứng** trong file này; nếu ngày chưa có thì tạo heading `## YYYY-MM-DD`.
4. Nếu một ngày có nhiều EXP hoặc WIP log, để chung dưới cùng ngày, theo thứ tự thời gian hoặc EXP ID.
5. Không tạo file `YYYY-MM-DD_EXP-XXX_*.md` mới và không dùng template rời.

## Nguyên tắc bắt buộc

- Mọi thử nghiệm nghiêm túc phải so sánh `centralized` nếu khả thi, `local-only`, và `federated`.
- Luôn báo cáo metric theo từng client/site, không chỉ trung bình global.
- Ghi đầy đủ global/per-client loss và metric chính (Acc/F1/mAP), training time, số round, update size và communication cost.
- Ghi lệnh tái lập và đường dẫn raw outputs trong `outputs/EXP-XXX/`.

## Registry tổng hợp

| ID | Ngày chạy | Tên thử nghiệm | Trạng thái | Centralized (Acc/F1) | Local-Only Avg (Acc/F1) | Federated (Acc/F1) | Chi tiết |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **EXP-001** | 2026-06-12 | Synthetic Flower + PyTorch baseline | Thành công | 0.9216 | 0.8039 | 0.7059 | [Chi tiết](#exp-001) |
| **EXP-002** | 2026-06-18 | OOM-safe single-machine smoke profile | Thành công | 0.6207 | 0.9310 | 0.3448 | [Chi tiết](#exp-002) |
| **EXP-003** | 2026-06-18 | PPE embedding dry-run baseline | Thành công | 0.5000 | 0.0000 | 0.5000 | [Chi tiết](#exp-003) |
| **EXP-004** | 2026-06-22 | PPE real ResNet18 embedding smoke | Thành công | 0.4667/0.4661 | 0.5000/0.3206 | 0.5000/0.3333 | [Chi tiết](#exp-004) |
| **EXP-005** | 2026-06-22 | PPE real ResNet18 stability baseline | Thành công | 0.5167/0.5055 | 0.6833/0.6832 | 0.7000/0.6997 | [Chi tiết](#exp-005) |
| **EXP-006** | 2026-06-25 | PPE real ResNet18 scaled baseline (480/3, manifest sinh tự động) | Thành công | 0.5583/0.5173 | 0.6250/0.6173 | 0.5917/0.5812 | [Chi tiết](#exp-006) |
| **EXP-007** | 2026-06-25 | PPE real ResNet18 train-longer baseline (480/3, 10 round) | Thành công | 0.6000/0.5960 | 0.6000/0.5908 | 0.6083/0.6081 | [Chi tiết](#exp-007) |
| **EXP-008** | 2026-06-25 | PPE embedding weight-decay regularization (480/3, wd=1e-1) | Thành công | 0.5750/0.5616 | 0.6083/0.5960 | 0.6167/0.6114 | [Chi tiết](#exp-008) |
| **EXP-009** | 2026-06-25 | PPE embedding L2-normalize (cosine-style, 480/3) | Thành công | 0.6250/0.6229 | 0.6750/0.6740 | 0.6000/0.5945 | [Chi tiết](#exp-009) |
| **EXP-010** | 2026-06-25 | PPE embedding MLP head capacity sweep (480/3, H=64 headline) | Thành công | 0.6333/0.6312 | 0.6667/0.6643 | 0.6333/0.6329 | [Chi tiết](#exp-010) |
| **WIP** | 2026-06-25 | PPE detection pivot + federated deployment + DVC sync | Đang làm | N/A | N/A | N/A | [Chi tiết](#wip-detection-pivot) |

## 2026-06-12

<a id="exp-001"></a>
### EXP-001 — Synthetic Flower + PyTorch baseline
- **Mã Thử Nghiệm:** EXP-001
- **Ngày Thực Hiện:** 2026-06-12
- **Trạng Thái:** Thành công
- **Git Commit Hash:** `committed`

---

#### 1. Mục Tiêu (Objective)
- Dựng demo tối thiểu cho Federated Learning với 3 baseline: centralized, local-only và federated.
- Kiểm tra luồng freeze backbone, chỉ train classifier head, trước khi chuyển sang face/PPE thật.

#### 2. Cấu Hình & Thiết Lập (Configuration)
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

#### 3. So Sánh Kết Quả Giữa Các Baseline (Baseline Comparison)

| Chế Độ Huấn Luyện (Mode) | Global Loss | Global Metric (Acc/F1/mAP) | Tổng Thời Gian Train | Kích Thước Update (Per Round) | Tổng Chi Phí Truyền Thông (Comm. Cost) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Centralized** (Pooled data) | 0.4948 | Acc 0.9216 | 0.50s | 108 bytes | N/A |
| **Local-Only** (Không collab) | 0.7755 | Acc 0.8039 | 0.01s | 108 bytes | N/A |
| **Federated** (FedAvg Baseline) | 0.7804 | Acc 0.7059 | 3.54s | 108 bytes | 2,160 bytes |

#### 4. Kết Quả Chi Tiết Theo Từng Client (Per-Client Metrics)

Kết quả per-client đã được lưu trong `outputs/EXP-001/*_metrics.json`. Mỗi mode có đủ 5 client mô phỏng, gồm số mẫu validation, loss, accuracy và label histogram.

#### 5. Quan Sát & Phân Tích (Observations & Rationale)
- Demo này chưa đại diện cho face/PPE thật.
- Mục tiêu là kiểm tra luồng kỹ thuật: chia client, train head-only, aggregation, metric theo client và communication cost.
- Smoke test đã chạy được 3 mode với dữ liệu non-IID synthetic.
- Federated dùng 2 round ở chế độ `--quick`, update head có kích thước 108 bytes, tổng communication cost 2,160 bytes.
- Bước tiếp theo là thay synthetic data bằng task face hoặc PPE đơn giản.

#### 6. Tài Liệu Hướng Dẫn Tái Lập (Reproducibility & Artifacts)
- **Lệnh chạy:**
  ```bash
  venv/bin/python scripts/run_demo.py --mode all --quick
  PATH="$PWD/venv/bin:$PATH" venv/bin/flwr run . --run-config "quick=true num-server-rounds=2" --federation-config "num-supernodes=5" --stream
  ```
- **Đường dẫn outputs:** `outputs/EXP-001/`

#### 7. Bước Tiếp Theo (Next Steps)
- [x] Chạy unit tests.
- [x] Chạy smoke test centralized/local-only/federated.
- [x] Cập nhật bảng kết quả sau khi có output.

## 2026-06-18

<a id="exp-002"></a>
### EXP-002 — OOM-safe single-machine smoke profile
- **Mã Thử Nghiệm:** EXP-002
- **Ngày Thực Hiện:** 2026-06-18
- **Trạng Thái:** Thành công
- **Git Commit Hash:** `committed`

---

#### 1. Mục Tiêu (Objective)
- Kiểm tra profile chạy nhỏ, an toàn bộ nhớ, trước khi thay synthetic data bằng model/dataset thật.
- Xác nhận Flower simulation hiện tại vẫn là single-machine simulation: nhiều client mô phỏng chạy trên cùng một máy.
- Giữ đủ 3 baseline `centralized`, `local-only`, `federated` nhưng giảm client/sample/batch/round để hạn chế OOM.

#### 2. Cấu Hình & Thiết Lập (Configuration)
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

#### 3. So Sánh Kết Quả Giữa Các Baseline (Baseline Comparison)

| Chế Độ Huấn Luyện (Mode) | Global Loss | Global Metric (Acc/F1/mAP) | Tổng Thời Gian Train | Kích Thước Update (Per Round) | Tổng Chi Phí Truyền Thông (Comm. Cost) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Centralized** (Pooled data) | 0.8381 | Acc 0.6207 | 0.75s | 108 bytes | N/A |
| **Local-Only** (Không collab) | 0.5791 | Acc 0.9310 | 0.01s | 108 bytes | N/A |
| **Federated** (FedAvg Baseline) | 1.0665 | Acc 0.3448 | 3.83s | 108 bytes | 648 bytes |

#### 4. Kết Quả Chi Tiết Theo Từng Client (Per-Client Metrics)

Kết quả dưới đây là mode `federated` sau 1 round.

| Client ID / Site | Số Lượng Validation | Local Validation Loss | Local Metric (Acc/F1/mAP) | Ghi Chú Đặc Điểm Dữ Liệu |
| :--- | :---: | :---: | :---: | :--- |
| **Client 0** | 8 | 1.3323 | Acc 0.0000 | Non-IID label skew, histogram `{0: 0, 1: 0, 2: 33}` |
| **Client 1** | 11 | 1.2029 | Acc 0.0000 | Non-IID label skew, histogram `{0: 0, 1: 39, 2: 6}` |
| **Client 2** | 10 | 0.7036 | Acc 1.0000 | Non-IID label skew, histogram `{0: 40, 1: 1, 2: 1}` |

#### 5. Quan Sát & Phân Tích (Observations & Rationale)
- Profile `oom-safe` chạy được đủ 3 mode với 3 client, batch size 4, 1 local epoch và 1 FL round.
- Flower/Ray được giới hạn còn `num_cpus=1`, log cho thấy VirtualClientEngine tạo 1 actor, phù hợp mục tiêu giảm concurrency trên một máy cá nhân.
- Peak RSS đo bằng `resource.getrusage(RUSAGE_SELF)` của process chính khoảng 368.31 MB. Số này là chỉ báo nhẹ, chưa thay thế đo RAM toàn hệ thống hoặc Ray worker riêng.
- Kết quả accuracy không dùng để kết luận FL tốt/xấu vì đây là smoke test nhỏ, chỉ có synthetic label skew và 1 round.
- Trong sandbox Codex, lần chạy đầu bị chặn quyền `sysctl/psutil`; rerun với quyền ngoài sandbox thành công. Khi chạy trực tiếp trong terminal local, lệnh thường có thể chạy bình thường.

#### 6. Tài Liệu Hướng Dẫn Tái Lập (Reproducibility & Artifacts)
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

#### 7. Bước Tiếp Theo (Next Steps)
- [x] Thiết kế dataset loader thật theo hướng lazy-load, không load toàn bộ ảnh/video vào RAM. *(Đã làm: manifest đọc metadata-only + precompute đọc ảnh theo batch — EXP-004.)*
- [x] Thiết kế bước precompute embeddings bằng frozen backbone trước khi train head/adapter trên real data. *(Đã làm: `scripts/precompute_embeddings.py` frozen ResNet18 — EXP-004.)*
- [x] Khi chuyển sang face/PPE, chạy lại profile nhỏ tương tự trước khi tăng client, batch, round hoặc dataset size. *(PPE: EXP-004 smoke trước khi scale EXP-006; face: chưa bắt đầu.)*

<a id="exp-003"></a>
### EXP-003 — PPE embedding dry-run baseline
- **Mã Thử Nghiệm:** EXP-003
- **Ngày Thực Hiện:** 2026-06-18
- **Trạng Thái:** Thành công
- **Git Commit Hash:** `committed`

---

#### 1. Mục Tiêu (Objective)
- Kiểm tra luồng từ PPE manifest sang embedding artifact `.npz`, rồi chạy đủ 3 baseline `centralized`, `local-only`, `federated`.
- Xác nhận pipeline FL head-only có thể dùng input dạng embedding thay vì synthetic vectors sinh trực tiếp trong code.
- Chưa đánh giá chất lượng PPE thật vì artifact vẫn được sinh bằng backend `synthetic`, chưa phải frozen pretrained backbone.

#### 2. Cấu Hình & Thiết Lập (Configuration)
- **Bài toán:** PPE binary classification dry-run (`safe` / `unsafe`).
- **Dữ Liệu & Phân Chia Client (Data & Partition):**
  - Manifest: `configs/datasets/ppe_manifest_template.csv`
  - Artifact: `data/processed/ppe_embeddings_oom_safe.npz`
  - Backend artifact: `synthetic`
  - Số lượng client mô phỏng: 2 clients (`site-a`, `site-b`)
  - Split dùng cho baseline v1: `train` và `val`; `test` được giữ trong artifact nhưng chưa dùng.
- **Mô Hình & Huấn Luyện Tham Số Hiệu Quả (Model & PEFT):**
  - Input: precomputed embedding vectors, dim `16`.
  - Backbone: không dùng trong dry-run này.
  - Phần trainable: classifier head trực tiếp `embedding_dim -> num_classes`.
- **Siêu Tham Số (Hyperparameters):**
  - Profile: `oom-safe`
  - Learning Rate: `0.05`
  - Batch Size: `4`
  - Local Epochs: `1`
  - Centralized Epochs: `1`
  - FL Rounds: `1`
  - DataLoader Workers: `0`
  - Ray CPUs: `1`
  - Optimizer: `SGD`

#### 3. So Sánh Kết Quả Giữa Các Baseline (Baseline Comparison)

| Chế Độ Huấn Luyện (Mode) | Global Loss | Global Metric (Acc/F1/mAP) | Tổng Thời Gian Train | Kích Thước Update (Per Round) | Tổng Chi Phí Truyền Thông (Comm. Cost) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Centralized** (Pooled data) | 0.7036 | Acc 0.5000 | 0.72s | 136 bytes | N/A |
| **Local-Only** (Không collab) | 0.7398 | Acc 0.0000 | 0.00s | 136 bytes | N/A |
| **Federated** (FedAvg Baseline) | 0.7036 | Acc 0.5000 | 3.41s | 136 bytes | 544 bytes |

#### 4. Kết Quả Chi Tiết Theo Từng Client (Per-Client Metrics)

Kết quả dưới đây là mode `federated` sau 1 round.

| Client ID / Site | Số Lượng Validation | Local Validation Loss | Local Metric (Acc/F1/mAP) | Ghi Chú Đặc Điểm Dữ Liệu |
| :--- | :---: | :---: | :---: | :--- |
| **site-a** | 1 | 0.7368 | Acc 0.0000 | Label histogram `{0: 2, 1: 1}` |
| **site-b** | 1 | 0.6703 | Acc 1.0000 | Label histogram `{0: 1, 1: 2}` |

#### 5. Quan Sát & Phân Tích (Observations & Rationale)
- EXP-003 xác nhận data contract hoạt động: manifest -> `.npz` embedding -> 3 baseline.
- `label_mapping` là `{"safe": 0, "unsafe": 1}`.
- Update size tăng từ 108 bytes ở synthetic head EXP-002 lên 136 bytes vì head trực tiếp có shape `2 x 16` cộng bias `2`.
- Flower/Ray vẫn chạy single-machine simulation và được giới hạn còn `num_cpus=1`; log cho thấy VirtualClientEngine tạo 1 actor.
- Accuracy không có ý nghĩa nghiệp vụ vì dữ liệu chỉ có 6 sample và embedding được sinh synthetic, không phải từ ảnh PPE/backbone thật.

#### 6. Tài Liệu Hướng Dẫn Tái Lập (Reproducibility & Artifacts)
- **Lệnh tạo artifact:**
  ```bash
  venv/bin/python scripts/precompute_embeddings.py --backend synthetic --output data/processed/ppe_embeddings_oom_safe.npz
  ```
- **Lệnh chạy baseline:**
  ```bash
  venv/bin/python scripts/run_embedding_demo.py --mode all --artifact data/processed/ppe_embeddings_oom_safe.npz --profile oom-safe --output-dir outputs/EXP-003
  ```
- **Đường dẫn outputs:** `outputs/EXP-003/`
- **File output chính:**
  - `outputs/EXP-003/centralized_metrics.json`
  - `outputs/EXP-003/local_only_metrics.json`
  - `outputs/EXP-003/federated_metrics.json`
  - `outputs/EXP-003/summary.json`

#### 7. Bước Tiếp Theo (Next Steps)
- [x] Thay backend `synthetic` bằng backend frozen pretrained backbone thật. *(Đã làm: EXP-004 — torchvision-resnet18.)*
- [x] Tạo manifest từ dataset PPE public hoặc nội bộ. *(EXP-004 manifest `data/ppe`; EXP-006 generator có seed.)*
- [x] Chạy real-data smoke run nhỏ với cùng profile OOM-safe trước khi tăng batch/client/round. *(Đã làm: EXP-004.)*

## 2026-06-22

<a id="exp-004"></a>
### EXP-004 — PPE real ResNet18 embedding smoke
- **Mã Thử Nghiệm:** EXP-004
- **Ngày Thực Hiện:** 2026-06-22
- **Trạng Thái:** Thành công
- **Git Commit Hash:** `committed`

---

#### 1. Mục Tiêu (Objective)
- Chạy thử nghiệm PPE đầu tiên trên ảnh thật trong `data/ppe` theo cấu hình OOM-safe.
- Kiểm tra luồng: VOC annotation -> manifest classification proxy -> frozen ResNet18 embedding `.npz` -> 3 baseline `centralized`, `local-only`, `federated`.
- Giữ đúng hướng parameter-efficient: backbone ResNet18 pretrained ImageNet được freeze, chỉ train classifier head trên embedding 512 chiều.

#### 2. Cấu Hình & Thiết Lập (Configuration)
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

#### 3. So Sánh Kết Quả Giữa Các Baseline (Baseline Comparison)

| Chế Độ Huấn Luyện (Mode) | Global Loss | Global Metric (Acc/F1/mAP) | Unsafe Recall / FNR | Tổng Thời Gian Train | Kích Thước Update (Per Round) | Tổng Chi Phí Truyền Thông (Comm. Cost) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized** (Pooled data) | 2.0818 | Acc 0.4667 / Macro-F1 0.4661 | 0.5000 / 0.5000 | 0.82s | 4,104 bytes | N/A |
| **Local-Only** (Không collab) | 7.8129 | Acc 0.5000 / Macro-F1 0.3206 | 1.0000 / 0.0000 | 0.01s | 4,104 bytes | N/A |
| **Federated** (FedAvg Baseline) | 8.0683 | Acc 0.5000 / Macro-F1 0.3333 | 1.0000 / 0.0000 | 3.90s | 4,104 bytes | 24,624 bytes |

#### 4. Kết Quả Chi Tiết Theo Từng Client (Per-Client Metrics)

Kết quả dưới đây là mode `federated` sau 1 round.

| Client ID / Site | Số Lượng Validation | Local Validation Loss | Local Metric (Acc/Macro-F1) | Unsafe Recall / FNR | Ghi Chú Đặc Điểm Dữ Liệu |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **site-a** | 20 | 12.0544 | Acc 0.2500 / Macro-F1 0.2000 | 1.0000 / 0.0000 | Label histogram `{safe: 60, unsafe: 20}` |
| **site-b** | 20 | 3.9693 | Acc 0.7500 / Macro-F1 0.4286 | 1.0000 / 0.0000 | Label histogram `{safe: 20, unsafe: 60}` |
| **site-c** | 20 | 8.1814 | Acc 0.5000 / Macro-F1 0.3333 | 1.0000 / 0.0000 | Label histogram `{safe: 40, unsafe: 40}` |

#### 5. Quan Sát & Phân Tích (Observations & Rationale)
- EXP-004 xác nhận pipeline ảnh thật chạy được theo hướng OOM-safe: đọc ảnh lazy trong bước precompute, freeze ResNet18, lưu embedding, sau đó chỉ train head nhẹ.
- Kết quả model chưa dùng để kết luận chất lượng PPE vì nhãn `safe/unsafe` là proxy image-level từ object annotation, chưa phải nhãn compliance được review bởi người.
- Federated và local-only đều có xu hướng dự đoán `unsafe` nhiều trong smoke run 1 epoch/1 round, thể hiện qua unsafe recall 1.0 nhưng macro-F1 thấp.
- Update size là 4,104 bytes, phù hợp head `2 x 512` cộng bias `2` ở float32.
- Communication cost của FedAvg là 24,624 bytes, tương ứng gửi/nhận update head cho 3 client trong 1 round.
- Peak RSS của process chính khoảng 353.69 MB theo `resource.getrusage(RUSAGE_SELF)`. Đây là chỉ báo nhẹ, chưa đo toàn bộ Ray worker hoặc peak RAM hệ thống.
- Trong sandbox Codex, Flower/Ray bị chặn `psutil/sysctl`; rerun ngoài sandbox thành công.

#### 6. Tài Liệu Hướng Dẫn Tái Lập (Reproducibility & Artifacts)
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

#### 7. Bước Tiếp Theo (Next Steps)
- [ ] Cải thiện rule hoặc tạo nhãn compliance được review bởi người, tránh phụ thuộc hoàn toàn vào proxy `has_core_ppe`.
- [x] Chạy thêm 2-3 round hoặc tăng epoch nhỏ để xem metric có ổn định hơn không. *(EXP-005: 3 round/3 epoch; EXP-007: 10 round.)*
- [x] Thêm confusion matrix hoặc precision/recall theo class cho báo cáo PPE. *(EXP-005 trở đi.)*
- [x] Nếu ổn định, tăng mẫu/client có kiểm soát và ghi resource/time/update size. *(EXP-006: 480 mẫu, ghi resource/time/update.)*

<a id="exp-005"></a>
### EXP-005 — PPE real ResNet18 stability baseline
- **Mã Thử Nghiệm:** EXP-005
- **Ngày Thực Hiện:** 2026-06-22
- **Trạng Thái:** Thành công
- **Git Commit Hash:** `committed`

---

#### 1. Mục Tiêu (Objective)
- Củng cố `EXP-004` từ smoke test thành baseline PPE dễ phân tích hơn.
- Giữ nguyên pipeline ảnh thật: VOC annotation proxy -> frozen ResNet18 embedding -> train classifier head.
- Chạy đủ 3 mode `centralized`, `local-only`, `federated` với nhiều epoch/round hơn để kiểm tra độ ổn định ban đầu.
- Thêm metric chẩn đoán: confusion matrix và precision/recall/F1 theo từng class.

#### 2. Cấu Hình & Thiết Lập (Configuration)
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

#### 3. So Sánh Kết Quả Giữa Các Baseline (Baseline Comparison)

| Chế Độ Huấn Luyện (Mode) | Global Loss | Global Metric (Acc/Macro-F1) | Unsafe Recall / FNR | Tổng Thời Gian Train | Kích Thước Update (Per Round) | Tổng Chi Phí Truyền Thông (Comm. Cost) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized** (Pooled data) | 0.7650 | 0.5167 / 0.5055 | 0.3667 / 0.6333 | 0.80s | 4,104 bytes | N/A |
| **Local-Only** (Không collab) | 0.7450 | 0.6833 / 0.6832 | 0.5667 / 0.4333 | 0.02s | 4,104 bytes | N/A |
| **Federated** (FedAvg Baseline) | 0.6726 | 0.7000 / 0.6997 | 0.7333 / 0.2667 | 3.84s | 4,104 bytes | 73,872 bytes |

#### 4. Kết Quả Chi Tiết Theo Từng Client (Per-Client Metrics)

Kết quả dưới đây là mode `federated` sau 3 round.

| Client ID / Site | Số Lượng Validation | Local Validation Loss | Local Metric (Acc/Macro-F1) | Unsafe Recall / FNR | Confusion Matrix `[safe, unsafe]` | Ghi Chú Đặc Điểm Dữ Liệu |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **site-a** | 20 | 0.6871 | 0.7000 / 0.6429 | 0.6000 / 0.4000 | `[[11, 4], [2, 3]]` | Label histogram `{safe: 60, unsafe: 20}` |
| **site-b** | 20 | 0.5584 | 0.8500 / 0.8119 | 0.8667 / 0.1333 | `[[4, 1], [2, 13]]` | Label histogram `{safe: 20, unsafe: 60}` |
| **site-c** | 20 | 0.7724 | 0.5500 / 0.5489 | 0.6000 / 0.4000 | `[[5, 5], [4, 6]]` | Label histogram `{safe: 40, unsafe: 40}` |

#### 5. Quan Sát & Phân Tích (Observations & Rationale)
- So với `EXP-004`, FedAvg cải thiện rõ: accuracy từ `0.5000` lên `0.7000`, macro-F1 từ `0.3333` lên `0.6997`.
- Mô hình không còn chỉ nghiêng hẳn về `unsafe`: global federated confusion matrix là `[[20, 10], [8, 22]]`, tương ứng safe recall `0.6667` và unsafe recall `0.7333`.
- `site-b` tốt nhất trong federated (`Acc 0.8500`, `Macro-F1 0.8119`), còn `site-c` thấp nhất (`Acc 0.5500`, `Macro-F1 0.5489`), cho thấy split cân bằng vẫn khó trên subset nhỏ.
- Local-only cũng cải thiện so với `EXP-004`, nhưng FedAvg tốt hơn về macro-F1 và unsafe recall trong cấu hình này.
- Communication cost tăng từ `24,624 bytes` ở `EXP-004` lên `73,872 bytes`, đúng kỳ vọng vì số round tăng từ 1 lên 3 trong khi update size giữ nguyên `4,104 bytes`.
- Peak RSS của process chính khoảng `354.72 MB` theo `resource.getrusage(RUSAGE_SELF)`. Đây vẫn là chỉ báo nhẹ, chưa đo toàn bộ Ray worker hoặc peak RAM hệ thống.
- Kết quả vẫn chưa dùng để kết luận chất lượng PPE thật vì nhãn `safe/unsafe` là proxy image-level từ object annotation, chưa phải nhãn compliance được review bởi người.

#### 6. Tài Liệu Hướng Dẫn Tái Lập (Reproducibility & Artifacts)
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

#### 7. Bước Tiếp Theo (Next Steps)
- [ ] Cải thiện định nghĩa nhãn PPE/compliance thay vì chỉ dùng proxy `has_core_ppe`.
- [x] Chạy thêm một baseline với nhiều mẫu/client hơn nhưng vẫn giữ OOM-safe. *(Đã làm: EXP-006, 480 mẫu/3 site.)*
- [ ] Ghi rõ trong báo cáo rằng kết quả hiện tại chứng minh pipeline và baseline, chưa chứng minh chất lượng nghiệp vụ.
- [ ] Sau khi nhãn ổn hơn, mới xét personalized head, adapter/LoRA, FedBN hoặc giảm communication.

## 2026-06-25

<a id="exp-006"></a>
### EXP-006 — PPE real ResNet18 scaled baseline (480/3, manifest sinh tự động)
- **Mã Thử Nghiệm:** EXP-006
- **Ngày Thực Hiện:** 2026-06-25
- **Trạng Thái:** Thành công
- **Git Commit Hash:** `committed`

---

#### 1. Mục Tiêu (Objective)
- Thực hiện Next-step #2 của `EXP-005`: chạy thêm một baseline với **nhiều mẫu hơn** (gấp đôi: 240 → 480) nhưng **vẫn giữ OOM-safe**.
- Khắc phục gap tái lập của `EXP-004`/`EXP-005`: manifest 240 dòng trước đây làm tay. EXP-006 sinh manifest bằng **script có seed cố định** (`scripts/generate_ppe_manifest.py`).
- Giữ nguyên pipeline ảnh thật: VOC annotation proxy -> frozen ResNet18 embedding -> train classifier head, và giữ nguyên cấu trúc 3 site + kiểu non-IID label skew của EXP-005 để so sánh công bằng.

#### 2. Cấu Hình & Thiết Lập (Configuration)ßß
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

#### 3. So Sánh Kết Quả Giữa Các Baseline (Baseline Comparison)

| Chế Độ Huấn Luyện (Mode) | Global Loss | Global Metric (Acc/Macro-F1) | Unsafe Recall / FNR | Tổng Thời Gian Train | Kích Thước Update (Per Round) | Tổng Chi Phí Truyền Thông (Comm. Cost) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized** (Pooled data) | 1.0149 | 0.5583 / 0.5173 | 0.2667 / 0.7333 | 0.74s | 4,104 bytes | N/A |
| **Local-Only** (Không collab) | 0.7437 | 0.6250 / 0.6173 | 0.7222 / 0.2778 | 0.04s | 4,104 bytes | N/A |
| **Federated** (FedAvg Baseline) | 0.8215 | 0.5917 / 0.5812 | 0.7500 / 0.2500 | 3.61s | 4,104 bytes | 73,872 bytes |

#### 4. Kết Quả Chi Tiết Theo Từng Client (Per-Client Metrics)

Kết quả dưới đây là mode `federated` sau 3 round.

| Client ID / Site | Số Lượng Validation | Local Validation Loss | Local Metric (Acc/Macro-F1) | Unsafe Recall / FNR | Confusion Matrix `[safe, unsafe]` | Ghi Chú Đặc Điểm Dữ Liệu |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **site-a** | 40 | 0.9325 | 0.5250 / 0.5100 | 0.7000 / 0.3000 | `[[14, 16], [3, 7]]` | Label histogram `{safe: 120, unsafe: 40}` |
| **site-b** | 40 | 0.5919 | 0.7250 / 0.6204 | 0.8333 / 0.1667 | `[[4, 6], [5, 25]]` | Label histogram `{safe: 40, unsafe: 120}` |
| **site-c** | 40 | 0.9400 | 0.5250 / 0.5175 | 0.6500 / 0.3500 | `[[8, 12], [7, 13]]` | Label histogram `{safe: 80, unsafe: 80}` |

Global federated confusion matrix: `[[26, 34], [15, 45]]` → safe recall `0.4333`, unsafe recall `0.7500`.

#### 5. Quan Sát & Phân Tích (Observations & Rationale)
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

#### 6. Tài Liệu Hướng Dẫn Tái Lập (Reproducibility & Artifacts)
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

#### 7. Bước Tiếp Theo (Next Steps)
- [x] Tăng số round/epoch (vẫn OOM-safe) để xem federated có vượt local-only khi train đủ lâu không. *(Đã làm: EXP-007 — 10 round/5 epoch, federated vượt local-only.)*
- [x] Thử head 2 lớp (MLP nhỏ) hoặc chuẩn hóa embedding (L2-normalize) để cải thiện safe recall. *(Đã làm: EXP-009 L2-norm + EXP-010 MLP. L2-norm là lever đúng; MLP không giúp FL.)*
- [ ] Quét nhanh learning rate / số round như một ablation nhỏ trên manifest EXP-006 cố định.
- [ ] Cải thiện định nghĩa nhãn PPE/compliance thay vì proxy `has_core_ppe` trước khi kết luận nghiệp vụ.
- [ ] Sau khi nhãn ổn hơn, mới xét personalized head, adapter/LoRA, FedBN hoặc giảm communication.

<a id="exp-007"></a>
### EXP-007 — PPE real ResNet18 train-longer baseline (480/3, 10 round)
- **Mã Thử Nghiệm:** EXP-007
- **Ngày Thực Hiện:** 2026-06-25
- **Trạng Thái:** Thành công
- **Git Commit Hash:** `c77b5a6`

---

#### 1. Mục Tiêu (Objective)
- Thực hiện Next-step #1 của `EXP-006`: **tăng số round/epoch** (vẫn OOM-safe) để trả lời câu hỏi *"Federated có vượt local-only khi train đủ lâu không?"*.
- Biến kiểm soát: giữ **nguyên dữ liệu (manifest/artifact EXP-006), nguyên lr=0.01, nguyên profile oom-safe**; chỉ thay budget train. Nhờ vậy so sánh EXP-007 vs EXP-006 cô lập đúng tác động của việc train lâu hơn.
- Tái dùng artifact `data/processed/ppe_real_embeddings_exp006.npz` → **bỏ qua hoàn toàn pha precompute** (chỉ train head trên embedding đã lưu).

#### 2. Cấu Hình & Thiết Lập (Configuration)
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

#### 3. So Sánh Kết Quả Giữa Các Baseline (Baseline Comparison)

| Chế Độ Huấn Luyện (Mode) | Global Loss | Global Metric (Acc/Macro-F1) | Unsafe Recall / FNR | Tổng Thời Gian Train | Kích Thước Update (Per Round) | Tổng Chi Phí Truyền Thông (Comm. Cost) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized** (Pooled data) | 1.0619 | 0.6000 / 0.5960 | 0.5000 / 0.5000 | 0.94s | 4,104 bytes | N/A |
| **Local-Only** (Không collab) | 0.8200 | 0.6000 / 0.5908 | 0.3889 / 0.6111 | 0.06s | 4,104 bytes | N/A |
| **Federated** (FedAvg Baseline) | 0.9554 | 0.6083 / 0.6081 | 0.5833 / 0.4167 | 4.35s | 4,104 bytes | 246,240 bytes |

##### So sánh trực tiếp với EXP-006 (cùng dữ liệu, chỉ khác budget)
| Mode | EXP-006 (3ep/3rd) Acc/Macro-F1 | EXP-007 (5ep/10rd) Acc/Macro-F1 |
| :--- | :---: | :---: |
| Centralized | 0.5583 / 0.5173 | **0.6000 / 0.5960** |
| Local-Only | 0.6250 / **0.6173** | 0.6000 / 0.5908 |
| Federated | 0.5917 / 0.5812 | **0.6083 / 0.6081** |

#### 4. Kết Quả Chi Tiết Theo Từng Client (Per-Client Metrics)
Mode `federated` sau 10 round.

| Client ID / Site | Số Lượng Validation | Local Validation Loss | Local Metric (Acc/Macro-F1) | Unsafe Recall / FNR | Confusion Matrix `[safe, unsafe]` | Ghi Chú Đặc Điểm Dữ Liệu |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **site-a** | 40 | 0.9790 | 0.6500 / 0.6154 | 0.7000 / 0.3000 | `[[19, 11], [3, 7]]` | Label histogram `{safe: 120, unsafe: 40}` |
| **site-b** | 40 | 0.8052 | 0.6000 / 0.5604 | 0.6000 / 0.4000 | `[[6, 4], [12, 18]]` | Label histogram `{safe: 40, unsafe: 120}` |
| **site-c** | 40 | 1.0818 | 0.5750 / 0.5726 | 0.5000 / 0.5000 | `[[13, 7], [10, 10]]` | Label histogram `{safe: 80, unsafe: 80}` |

Global federated confusion matrix: `[[38, 22], [25, 35]]` → safe recall `0.6333`, unsafe recall `0.5833`.

##### Đường hội tụ Federated (distributed weighted theo round)
| Round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Macro-F1 | 0.498 | 0.547 | 0.560 | **0.588** | 0.579 | 0.564 | 0.571 | 0.583 | 0.583 | 0.583 |
| Val Loss | 0.803 | 0.822 | 0.833 | 0.847 | 0.863 | 0.881 | 0.899 | 0.918 | 0.937 | 0.955 |

#### 5. Quan Sát & Phân Tích (Observations & Rationale)
- **Trả lời câu hỏi mở của EXP-006: CÓ — khi train lâu hơn, FedAvg vượt local-only.** Federated macro-F1 `0.5812 → 0.6081` và lần này **cao hơn local-only** (`0.6081` vs `0.5908`), đồng thời unsafe recall tốt hơn rõ (`0.5833` vs `0.3889`). Ở EXP-006 (3 round) thì federated còn thua local-only.
- **Nhưng "train lâu hơn" có giới hạn — hội tụ sớm rồi chững:** macro-F1 federated đạt đỉnh ~round 4 (`0.588`) rồi đi ngang ~`0.583`. Đáng chú ý: **val loss tăng đều mỗi round** (`0.803 → 0.955`) trong khi acc/F1 gần như không đổi → dấu hiệu **head ngày càng "quá tự tin" (overconfident)**, logits bị đẩy mạnh làm cross-entropy trên val xấu đi dù ranh giới quyết định gần như đứng yên. Tức tăng round/epoch quá ngưỡng **không còn cải thiện chất lượng**, chỉ làm loss tệ hơn.
- **FedAvg tỏ ra ổn định hơn local-only khi tăng local epoch:** đẩy local-epochs 3→5 làm **local-only xấu đi** (macro-F1 `0.6173 → 0.5908`, unsafe recall `0.7222 → 0.3889`) vì mỗi client nhỏ overfit cục bộ; trong khi **bước trung bình hóa của FedAvg đóng vai trò regularizer**, giúp federated vẫn nhích lên. Đây là một luận điểm ủng hộ FL khá sạch trên cấu hình này.
- **Centralized** cải thiện theo budget (`0.5173 → 0.5960` macro-F1 với 15 epoch), như kỳ vọng khi train hội tụ hơn.
- **Non-IID:** trong federated, site-a (safe-heavy) giờ tốt nhất (Acc `0.6500`), các site khác `~0.58-0.60`; phân bố cân bằng hơn EXP-006 (global confusion `[[38,22],[25,35]]`, safe recall `0.6333` thay vì `0.4333` ở EXP-006) — head bớt thiên về `unsafe`.
- **Comm cost = 246,240 bytes**, gấp ~3.33× EXP-006 (`73,872`), đúng tỉ lệ round `10/3` (update size không đổi `4,104` bytes/round × 3 client × 10 round × 2). Xác nhận chi phí truyền thông tỉ lệ thuận số round — một đánh đổi cần cân nhắc khi "train lâu hơn".
- **OOM-safe giữ nguyên:** `summary.json` báo `num_clients=3`, `batch_size=4`, `num_workers=0`, `ray_num_cpus=1`, đúng budget `5/15/10`. Peak RSS `441.02 MB` (federated, so `406.42` ở EXP-006) — tăng nhẹ do 10 round, vẫn rất an toàn trên máy Apple M2 16 GB. Wall time federated `4.35s`.
- Vẫn là **proxy label image-level**, chưa kết luận chất lượng PPE nghiệp vụ.

#### 6. Tài Liệu Hướng Dẫn Tái Lập (Reproducibility & Artifacts)
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

#### 7. Bước Tiếp Theo (Next Steps)
- [x] Vì tăng budget đã chững và val loss diverge: thử **chống overfit/regularize** thay vì train thêm — L2-normalize embedding, weight decay, hoặc lr nhỏ hơn + early stopping theo val loss. *(Đã làm: EXP-008 — weight_decay, hết diverge ở wd=1e-1; L2-norm/MLP/early-stopping vẫn pending.)*
- [x] Thử **head 2 lớp (MLP nhỏ)** xem có vượt trần ~0.60 macro-F1 không. *(Đã làm: EXP-010 — MLP H∈{32,64,128} không nâng federated macro-F1 và làm val loss diverge nặng + comm cost nổ 16-64×. Lever đúng là L2-norm/EXP-009, không phải capacity.)*
- [ ] Ablation nhỏ lr × round trên manifest EXP-006 cố định để xác định điểm dừng tối ưu (tránh round thừa tốn comm).
- [ ] Cải thiện nhãn PPE/compliance (thay proxy `has_core_ppe`) trước khi kết luận nghiệp vụ.
- [ ] Sau khi nhãn ổn: xét personalized head, adapter/LoRA, FedBN — đặc biệt khi đã thấy FedAvg regularize tốt hơn local-only.

<a id="exp-008"></a>
### EXP-008 — PPE embedding weight-decay regularization (480/3, wd=1e-1)
- **Mã Thử Nghiệm:** EXP-008
- **Ngày Thực Hiện:** 2026-06-25
- **Trạng Thái:** Thành công
- **Git Commit Hash:** `d7352ff`

---

#### 1. Mục Tiêu (Objective)
- Thực hiện Next-step #1 của `EXP-007`: **chống overfit bằng regularize** thay vì train thêm.
- `EXP-007` cho thấy head tuyến tính chững ~0.60 macro-F1 từ ~round 4 nhưng **val loss tăng đều mỗi round** (0.803 → 0.955) — dấu hiệu overconfidence.
- Can thiệp được chọn: **chỉ thêm `weight_decay`** vào optimizer SGD (giữ head tuyến tính + embedding thô). Ablation 1 biến sạch so EXP-007 (mọi thứ khác giữ y hệt).
- Câu hỏi: weight_decay có **chặn val loss diverge** / cải thiện calibration / nâng macro-F1 không?

#### 2. Cấu Hình & Thiết Lập (Configuration)
- **Bài toán & dữ liệu:** y hệt `EXP-006`/`EXP-007`. Tái dùng artifact `data/processed/ppe_real_embeddings_exp006.npz` (480 mẫu, 3 site non-IID, frozen ResNet18 dim 512). **Không precompute lại.**
- **Mô Hình & PEFT:** frozen ResNet18 + classifier head tuyến tính `512 -> 2` (chỉ head trainable). Serialize tham số `[head.weight, head.bias]` không đổi → FedAvg aggregation giữ nguyên.
- **Siêu tham số (giữ như EXP-007, chỉ thêm weight_decay):**
  - Profile: `oom-safe` (`num_workers=0`, `ray_num_cpus=1`, `batch_size=4`)
  - lr `0.01`, batch `4`, local epochs `5`, centralized epochs `15`, FL rounds `10`, optimizer `SGD`
  - **`weight_decay` ∈ {1e-3, 1e-2, 1e-1}** (sweep). EXP-007 = weight_decay `0`.
  - **Headline = `weight_decay = 1e-1`** (giá trị duy nhất triệt tiêu được hiện tượng diverge).

#### 3. So Sánh Kết Quả Giữa Các Baseline (Baseline Comparison)
Headline `weight_decay = 1e-1`:

| Chế Độ Huấn Luyện (Mode) | Global Loss | Global Metric (Acc/Macro-F1) | Unsafe Recall / FNR | Tổng Thời Gian Train | Kích Thước Update (Per Round) | Tổng Chi Phí Truyền Thông (Comm. Cost) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized** (Pooled data) | 0.8769 | 0.5750 / 0.5616 | 0.4000 / 0.6000 | 0.74s | 4,104 bytes | N/A |
| **Local-Only** (Không collab) | 0.8082 | 0.6083 / 0.5960 | 0.3778 / 0.6222 | 0.06s | 4,104 bytes | N/A |
| **Federated** (FedAvg Baseline) | 0.7941 | 0.6167 / 0.6114 | 0.5000 / 0.5000 | 3.08s | 4,104 bytes | 246,240 bytes |

##### Sweep weight_decay (federated, cùng dữ liệu/budget; wd=0 là EXP-007)
| weight_decay | Fed Acc/Macro-F1 | Fed unsafe recall | Val loss r1 → r10 | Δloss (diverge?) |
| :---: | :---: | :---: | :---: | :---: |
| **0** (EXP-007) | 0.5917 / 0.5812 | 0.7500 | 0.803 → 0.955 | **+0.152 (diverge)** |
| 1e-3 | 0.6083 / 0.6081 | 0.5833 | 0.803 → 0.952 | +0.149 (vẫn diverge) |
| 1e-2 | 0.6000 / 0.5996 | 0.5667 | 0.802 → 0.924 | +0.122 (giảm nhẹ) |
| **1e-1** | 0.6167 / 0.6114 | 0.5000 | 0.794 → 0.794 | **+0.000 (hết diverge)** |

#### 4. Kết Quả Chi Tiết Theo Từng Client (Per-Client Metrics)
Mode `federated`, headline `weight_decay = 1e-1`, sau 10 round.

| Client ID / Site | Số Lượng Validation | Local Validation Loss | Local Metric (Acc/Macro-F1) | Unsafe Recall / FNR | Confusion Matrix `[safe, unsafe]` | Ghi Chú Đặc Điểm Dữ Liệu |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **site-a** | 40 | 0.6655 | 0.7250 / 0.6800 | 0.7000 / 0.3000 | `[[22, 8], [3, 7]]` | Label histogram `{safe: 120, unsafe: 40}` |
| **site-b** | 40 | 0.8048 | 0.6250 / 0.6050 | 0.5667 / 0.4333 | `[[8, 2], [13, 17]]` | Label histogram `{safe: 40, unsafe: 120}` |
| **site-c** | 40 | 0.9119 | 0.5000 / 0.4792 | 0.3000 / 0.7000 | `[[14, 6], [14, 6]]` | Label histogram `{safe: 80, unsafe: 80}` |

Global federated confusion matrix: `[[44, 16], [30, 30]]` → safe recall `0.7333`, unsafe recall `0.5000`.

#### 5. Quan Sát & Phân Tích (Observations & Rationale)
- **weight_decay KHÔNG nâng được trần macro-F1:** mọi giá trị wd cho federated macro-F1 ~`0.60-0.61`, ngang EXP-007 (`0.5812`). → Trần ~0.60 là **vấn đề cấu trúc** (head tuyến tính + nhãn proxy), không phải do thiếu regularization. Đây là kết luận quan trọng để định hướng EXP sau.
- **weight_decay CÓ sửa được overconfidence/diverge — nhưng cần đủ mạnh:** chỉ `wd=1e-1` mới làm **val loss đi ngang** (Δ +0.000, từ 0.794 → 0.794) thay vì tăng dốc như EXP-007 (Δ +0.152). `wd=1e-3`/`1e-2` gần như không đủ. Tức mô hình hết "tự đẩy logit", **calibration tốt hơn** ở wd cao.
- **Đánh đổi an toàn (quan trọng cho PPE):** wd càng lớn, dự đoán càng dịch về cân bằng/`safe` → **unsafe recall giảm** (`0.75 @ wd=0 → 0.50 @ wd=1e-1`). Với mục tiêu bắt vi phạm (giảm bỏ sót), wd cao **làm xấu unsafe recall** dù calibration tốt hơn. Đây là trade-off thật, không có lựa chọn thắng tuyệt đối.
- **Federated vẫn ≥ local-only** ở mọi wd (headline: macro-F1 `0.6114` vs `0.5960`; unsafe recall `0.500` vs `0.378`), củng cố quan sát EXP-007 rằng FedAvg regularize tốt hơn train cục bộ.
- **Centralized hơi underfit ở wd=1e-1** (macro-F1 `0.5616`, thấp hơn `0.596` ở wd nhỏ) — pooled data + wd mạnh kéo model về đơn giản hơn.
- **Comm cost = 246,240 bytes**, **không đổi** so EXP-007 — weight_decay không ảnh hưởng kích thước update (vẫn `4,104` bytes/round × 3 × 10 × 2).
- **OOM-safe giữ nguyên:** config `num_clients=3`, `batch_size=4`, `num_workers=0`, `ray_num_cpus=1`. Peak RSS `451-460 MB`, federated ~3s/run. 3 run sweep tổng vài giây.
- Vẫn là **proxy label image-level**, chưa kết luận chất lượng PPE nghiệp vụ.

#### 6. Tài Liệu Hướng Dẫn Tái Lập (Reproducibility & Artifacts)
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

#### 7. Bước Tiếp Theo (Next Steps)
- [x] Vì regularize không nâng trần ~0.60: thử **đổi biểu diễn/capacity** — L2-normalize embedding (cosine-style, EXP-009) hoặc MLP head 2 lớp (EXP-010). *(Đã làm: EXP-009 L2-norm **vượt trần** centralized 0.596→0.623 + chặn diverge miễn phí + unsafe recall 0.717; EXP-010 MLP **không** giúp FL, diverge nặng + comm nổ 16-64×. → Trần là biểu diễn, không phải capacity.)*
- [ ] Nếu cần ưu tiên unsafe recall: cân nhắc class weight / threshold tuning thay vì wd mạnh (vì wd cao làm tụt unsafe recall). *(EXP-009 đã đạt unsafe recall 0.717 không cần wd — ưu tiên thấp hơn giờ.)*
- [ ] Cải thiện nhãn PPE/compliance (thay proxy `has_core_ppe`) — nhiều khả năng đây mới là trần thật.
- [ ] Khi nhãn ổn: personalized head / adapter-LoRA / FedBN.

<a id="exp-009"></a>
### EXP-009 — PPE embedding L2-normalize (cosine-style, 480/3)
- **Mã Thử Nghiệm:** EXP-009
- **Ngày Thực Hiện:** 2026-06-25
- **Trạng Thái:** Thành công
- **Git Commit Hash:** `38a7220`

---

#### 1. Mục Tiêu (Objective)
- Trả lời câu hỏi mở quan trọng nhất sau `EXP-008`: **trần macro-F1 ~0.60 đến từ capacity model hay từ nhãn proxy?**
- Can thiệp được chọn cho EXP-009: **L2-normalize embedding trước head** (chuyển sang hình học cosine), một thay đổi biểu diễn **không tăng tham số, không tăng comm cost**.
- Ablation 1 biến sạch so `EXP-007` (linear head, weight_decay=0): **mọi thứ giữ y hệt, chỉ bật `--normalize-embedding`**. Nhờ vậy chênh lệch EXP-009 vs EXP-007 cô lập đúng tác động của L2-norm.
- Tái dùng artifact `data/processed/ppe_real_embeddings_exp006.npz` → bỏ qua hoàn toàn pha precompute.

#### 2. Cấu Hình & Thiết Lập (Configuration)
- **Bài toán & dữ liệu:** y hệt `EXP-006/007/008` (480 mẫu, 3 site non-IID label skew, frozen ResNet18 dim 512). Label mapping `{"safe": 0, "unsafe": 1}`.
- **Mô Hình & PEFT:** frozen ResNet18 + classifier head tuyến tính `512 -> 2` (chỉ head trainable). **Mới:** input được `F.normalize(x, p=2, dim=1)` trước head → head học trên vector đơn vị (cosine geometry). Serialize tham số `[head.weight, head.bias]` **không đổi** → FedAvg aggregation giữ nguyên, update size `4,104` bytes.
- **Siêu tham số (giữ như EXP-007):**
  - Profile `oom-safe` (`num_workers=0`, `ray_num_cpus=1`, `batch_size=4`)
  - lr `0.01`, batch `4`, local epochs `5`, centralized epochs `15`, FL rounds `10`, optimizer `SGD`, weight_decay `0`.

#### 3. So Sánh Kết Quả Giữa Các Baseline (Baseline Comparison)

| Chế Độ Huấn Luyện (Mode) | Global Loss | Global Metric (Acc/Macro-F1) | Unsafe Recall / FNR | Tổng Thời Gian Train | Kích Thước Update (Per Round) | Tổng Chi Phí Truyền Thông (Comm. Cost) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized** (Pooled data) | 0.6853 | 0.6250 / 0.6229 | 0.7000 / 0.3000 | ~0.9s | 4,104 bytes | N/A |
| **Local-Only** (Không collab) | 0.6164 | 0.6750 / 0.6740 | 0.5500 / 0.4500 | ~0.06s | 4,104 bytes | N/A |
| **Federated** (FedAvg Baseline) | 0.6853 | 0.6000 / 0.5945 | 0.7167 / 0.2833 | 4.52s | 4,104 bytes | 246,240 bytes |

##### Ablation chính: L2-norm vs EXP-007 (cùng dữ liệu/budget, chỉ khác normalize)
| Mode | EXP-007 (raw embedding) Acc/Macro-F1 / unsafeR | EXP-009 (L2-norm) Acc/Macro-F1 / unsafeR |
| :--- | :---: | :---: |
| Centralized | 0.600 / 0.596 / 0.500 | **0.625 / 0.623 / 0.700** |
| Local-Only | 0.600 / 0.591 / 0.389 | **0.675 / 0.674 / 0.550** |
| Federated | 0.608 / 0.608 / 0.583 | 0.600 / 0.594 / **0.717** |

##### Federated convergence (distributed weighted theo round)
| Round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Macro-F1 | 0.450 | 0.458 | 0.517 | 0.548 | 0.556 | 0.557 | 0.565 | 0.565 | 0.565 | 0.571 |
| Val Loss | 0.693 | 0.692 | 0.691 | 0.690 | 0.689 | 0.688 | 0.688 | 0.687 | 0.686 | **0.685** |

Δval-loss r1→r10 = **−0.008 (KHÔNG diverge)** so với EXP-007 `+0.152` — và đạt được **mà không cần weight_decay**.

#### 4. Kết Quả Chi Tiết Theo Từng Client (Per-Client Metrics)
Mode `federated`, L2-norm, sau 10 round.

| Client ID / Site | Số Lượng Validation | Local Validation Loss | Local Metric (Acc/Macro-F1) | Unsafe Recall / FNR | Confusion Matrix `[safe, unsafe]` | Ghi Chú Đặc Điểm Dữ Liệu |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **site-a** | 40 | — | 0.4250 / 0.4159 | 0.6000 / 0.4000 | `[[11, 19], [4, 6]]` | Label histogram `{safe: 120, unsafe: 40}` |
| **site-b** | 40 | — | 0.7000 / 0.6238 | 0.7667 / 0.2333 | `[[5, 5], [7, 23]]` | Label histogram `{safe: 40, unsafe: 120}` |
| **site-c** | 40 | — | 0.6750 / 0.6748 | 0.7000 / 0.3000 | `[[13, 7], [6, 14]]` | Label histogram `{safe: 80, unsafe: 80}` |

Global federated confusion matrix: `[[29, 31], [17, 43]]` → safe recall `0.4833`, unsafe recall `0.7167`.

#### 5. Quan Sát & Phân Tích (Observations & Rationale)
- **TRẢ LỜI CÂU HỎI MỞ: trần ~0.60 MỘT PHẦN là vấn đề biểu diễn (cấu trúc), KHÔNG chỉ do nhãn.** Bằng chứng sạch nhất là **centralized** (pooled data, một model, không có artifact pooling): macro-F1 `0.596 → 0.623`, unsafe recall `0.500 → 0.700`, loss `1.062 → 0.685`. Chỉ bằng L2-normalize — **không thêm tham số, không thêm byte truyền** — đã vượt trần. Tức raw ResNet18 embedding có biến thiên độ lớn (magnitude) làm khó head tuyến tính; chuẩn hóa về hình cầu đơn vị (cosine) gỡ đúng nút thắt đó.
- **L2-norm chặn diverge "miễn phí", tốt hơn weight_decay (EXP-008):** val loss đi ngang (Δ −0.008) **mà không cần wd**. Quan trọng: EXP-008 chặn diverge bằng wd=1e-1 nhưng **trả giá unsafe recall (0.500)**; EXP-009 chặn diverge **đồng thời NÂNG unsafe recall lên 0.717 (cao nhất từ trước tới nay)**. Không còn trade-off — đây là lựa chọn thắng rõ ràng cho mục tiêu PPE (giảm bỏ sót vi phạm).
- **Lưu ý đọc số local-only "global" 0.674 — đây là pooling artifact, đừng dùng để kết luận FL thua:** L2-norm khiến model local trên site lệch trở nên **dự đoán gần như một lớp** (site-a all-safe: unsafe recall `0.000`; site-b all-unsafe: unsafe recall `1.000`). Khi gộp confusion 3 site degenerate lại, macro-F1 "global" bị thổi lên `0.674`, nhưng **per-client macro-F1 local-only chỉ ~0.46 trung bình** (0.429/0.429/0.517). Trong khi đó **federated per-client trung bình ~0.572** (0.416/0.624/0.675). Theo nguyên tắc CLAUDE.md (báo cáo per-client cho non-IID), **federated thực sự tốt hơn local-only ở mức client** — global average che giấu điều này.
- **Comm cost = 246,240 bytes, không đổi** so EXP-007/008 — L2-norm là phép toán trên input, không động vào kích thước head. Đây là điểm cực kỳ hấp dẫn cho edge FL: cải thiện chất lượng **với chi phí truyền thông bằng 0**.
- **OOM-safe giữ nguyên:** peak RSS `423.41 MB` (federated), wall time `4.52s`. An toàn trên máy cá nhân.
- Vẫn là **proxy label image-level** — absolute macro-F1 pooled centralized ~0.62 cho thấy nhãn vẫn là một trần thực sự, nhưng L2-norm chứng minh biểu diễn là lever rẻ đang bị bỏ phí.

#### 6. Tài Liệu Hướng Dẫn Tái Lập (Reproducibility & Artifacts)
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

#### 7. Bước Tiếp Theo (Next Steps)
- [ ] **L2-norm giờ là default baseline mới** — các EXP sau nên bật `--normalize-embedding`.
- [ ] Vì sau L2-norm **federated global chững (~0.59) trong khi per-client tốt** và các site lệch dễ degenerate: thử **personalized head / FedBN** — đã có tín hiệu personalization thật (local boundary hữu ích bị FedAvg trung bình hóa làm nhòe).
- [ ] Cải thiện nhãn PPE/compliance (thay proxy `has_core_ppe`) — vẫn là trần absolute còn lại.
- [ ] Cân nhắc kết hợp L2-norm + class weight để vừa giữ unsafe recall cao vừa cân bằng safe recall (site-a safe recall thấp).

<a id="exp-010"></a>
### EXP-010 — PPE embedding MLP head capacity sweep (480/3, H=64 headline)
- **Mã Thử Nghiệm:** EXP-010
- **Ngày Thực Hiện:** 2026-06-25
- **Trạng Thái:** Thành công
- **Git Commit Hash:** `38a7220`

---

#### 1. Mục Tiêu (Objective)
- Vế thứ hai của câu hỏi mở `EXP-008`: trần ~0.60 có phải do **capacity** (head tuyến tính quá yếu) không?
- Can thiệp: thay head tuyến tính `512 -> 2` bằng **MLP 2 lớp** `512 -> H -> 2` (ReLU), sweep `H ∈ {32, 64, 128}`.
- Ablation 1 biến so `EXP-007` (giữ raw embedding, lr/budget y hệt, chỉ thêm `--head-hidden-dim`). **Headline = H=64.**
- Tái dùng artifact `data/processed/ppe_real_embeddings_exp006.npz`.

#### 2. Cấu Hình & Thiết Lập (Configuration)
- **Bài toán & dữ liệu:** y hệt `EXP-006/007` (480 mẫu, 3 site non-IID, frozen ResNet18 dim 512).
- **Mô Hình & PEFT:** frozen ResNet18 + **MLP head** `Linear(512,H) -> ReLU -> Linear(H,2)` (chỉ MLP trainable). Serialize **4 mảng** `[fc1.w, fc1.b, fc2.w, fc2.b]` → FedAvg aggregate positional vẫn đúng, nhưng **update size tăng mạnh**.
- **Siêu tham số (giữ như EXP-007):** profile `oom-safe`, lr `0.01`, batch `4`, local epochs `5`, centralized epochs `15`, FL rounds `10`, SGD, weight_decay `0`.

#### 3. So Sánh Kết Quả Giữa Các Baseline (Baseline Comparison)
Headline `H = 64`:

| Chế Độ Huấn Luyện (Mode) | Global Loss | Global Metric (Acc/Macro-F1) | Unsafe Recall / FNR | Tổng Thời Gian Train | Kích Thước Update (Per Round) | Tổng Chi Phí Truyền Thông (Comm. Cost) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized** (Pooled data) | 0.8424 | 0.6333 / 0.6312 | 0.7167 / 0.2833 | ~0.9s | 131,848 bytes | N/A |
| **Local-Only** (Không collab) | 0.6010 | 0.6667 / 0.6643 | 0.4333 / 0.5667 | ~0.06s | 131,848 bytes | N/A |
| **Federated** (FedAvg Baseline) | 0.9617 | 0.6333 / 0.6329 | 0.6000 / 0.4000 | 3.45s | 131,848 bytes | 7,910,880 bytes |

##### Sweep hidden_dim (federated, cùng dữ liệu/budget; so EXP-007 = linear/no hidden)
| hidden_dim | Fed Acc/Macro-F1 | Fed unsafe recall | Val loss r1 → r10 | Δloss (diverge?) | Update size |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **linear** (EXP-007) | 0.608 / 0.608 | 0.583 | 0.803 → 0.955 | +0.152 | 4,104 B |
| 32 | 0.592 / 0.591 | 0.633 | 0.654 → 1.006 | **+0.352 (diverge nặng)** | 65,928 B |
| **64** | 0.633 / 0.633 | 0.600 | 0.656 → 0.962 | +0.306 | 131,848 B |
| 128 | 0.608 / 0.608 | 0.600 | 0.662 → 0.976 | +0.314 | 263,688 B |

#### 4. Kết Quả Chi Tiết Theo Từng Client (Per-Client Metrics)
Mode `federated`, headline `H = 64`, sau 10 round.

| Client ID / Site | Số Lượng Validation | Local Metric (Acc/Macro-F1) | Unsafe Recall / FNR | Confusion Matrix `[safe, unsafe]` | Ghi Chú Đặc Điểm Dữ Liệu |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **site-a** | 40 | 0.6250 / 0.5943 | 0.7000 / 0.3000 | `[[18, 12], [3, 7]]` | Label histogram `{safe: 120, unsafe: 40}` |
| **site-b** | 40 | 0.7000 / 0.6703 | 0.6667 / 0.3333 | `[[8, 2], [10, 20]]` | Label histogram `{safe: 40, unsafe: 120}` |
| **site-c** | 40 | 0.5750 / 0.5683 | 0.4500 / 0.5500 | `[[14, 6], [11, 9]]` | Label histogram `{safe: 80, unsafe: 80}` |

Global federated confusion matrix: `[[40, 20], [24, 36]]` → safe recall `0.6667`, unsafe recall `0.6000`.

#### 5. Quan Sát & Phân Tích (Observations & Rationale)
- **Capacity KHÔNG phải lever đúng cho federated.** Thêm MLP head **không nâng federated macro-F1** vượt mức ý nghĩa (`0.591–0.633` qua mọi H, so EXP-007 `0.608`). Đáng nói hơn: **val loss diverge NẶNG hơn cả EXP-007** (Δ +0.30…+0.35 so +0.152) — MLP overfit mạnh trên dữ liệu nhỏ mỗi client, và bước trung bình hóa FedAvg trên head phi tuyến kém hiệu quả hơn trên head tuyến tính.
- **MLP chỉ giúp local-only fit, không giúp FL:** local-only macro-F1 lên `0.66–0.69` (cao nhất khi H=128), xác nhận capacity giúp **một model trên data gộp/cục bộ**, nhưng lợi ích đó **không sống sót qua aggregation**. Đây là một kết quả âm tính sạch và có giá trị định hướng.
- **Chi phí truyền thông nổ 16×–64×:** update size `65,928 → 263,688` bytes (so `4,104` của head tuyến tính), comm cost lên tới **15.8 MB** ở H=128. Với mục tiêu edge/PEFT, đây là đánh đổi **không chấp nhận được** khi không có lợi ích chất lượng FL.
- **Đối chiếu trực tiếp với EXP-009:** L2-norm nâng centralized `0.596 → 0.623` với **0 byte thêm**; MLP H=64 nâng centralized `0.596 → 0.631` nhưng tốn **32× update size** và làm FL diverge. → **Kết luận: biểu diễn (L2-norm) là lever rẻ và đúng; capacity (MLP) là lever sai cho FL trên cấu hình này.**
- **OOM-safe giữ nguyên:** peak RSS `453.84 MB` (H=64, federated), wall time `3.45s`. Vẫn an toàn dù head lớn hơn.
- Vẫn là **proxy label image-level**.

#### 6. Tài Liệu Hướng Dẫn Tái Lập (Reproducibility & Artifacts)
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

#### 7. Bước Tiếp Theo (Next Steps)
- [ ] **Không theo đuổi MLP head cho FL** (diverge + comm cost lớn, không lợi ích). Nếu cần capacity, chỉ xét trên centralized/local, không trên update truyền đi.
- [ ] Lever đúng là **biểu diễn (L2-norm, EXP-009)** + **personalization** (FedBN / personalized head) — ưu tiên các hướng này.
- [ ] Cải thiện nhãn PPE/compliance vẫn là trần absolute còn lại.

<a id="wip-detection-pivot"></a>
### WIP — PPE Detection Pivot + Federated Deployment

> **Đây KHÔNG phải EXP journal** (chưa có metric run). Đây là **log tiến độ build** để resume sau khi
> context compact. EXP-011 journal sẽ tạo riêng khi có kết quả sim run thật.

- **Ngày:** 2026-06-25
- **Trạng thái:** Phase A scaffold gần xong; deployment scaffold đã thêm, còn EXP-011 GPU run
- **Plan đã duyệt:** `/Users/phamtunglam/.claude/plans/t-i-ngh-m-y-mac-mighty-stearns.md`

---

#### Context
- **Request:** PPE được chỉ định lại là **object detection** (không còn classification). Build lại track
  detection + chuyển sang **federated deployment thật 3 client**.
- **Goal:** dựng pipeline detection FL (freeze backbone, train head, FedAvg head), validate trong
  simulation, rồi deploy thật → đo mAP per-client.

#### Quyết định đã chốt (durable)
- **Detector:** `fasterrcnn_mobilenet_v3_large_fpn`, freeze **backbone+FPN**, chỉ train RPN head + ROI heads. FedAvg chỉ aggregate head.
- **Lớp:** 8 PPE core (`helmet, safety-vest, safety-suit, face-mask-medical, gloves, glasses, ear-mufs, face-guard`) + background = **9 lớp**. Class order = `DEFAULT_CORE_PPE` (single source of truth).
- **Topology:** **Mac local = server (SuperLink, aggregator, không giữ data)**; 3 client = **Ubuntu RTX3060 (site-a)** + **Colab#1 (site-b)** + **Colab#2 (site-c)**. Distributed evaluation (server không có pooled data).
- **Mạng (Pha B):** **Tailscale** (mesh VPN), `--insecure` trong tailnet riêng.
- **Metric:** mAP@0.5 + mAP@0.5:0.95 + per-class AP (torchmetrics), per-client.
- **EXP-001→010 = stage-1 classification đã archive** (trần ~0.60 do nhãn proxy; EXP-009 L2-norm là kết luận chính).

#### Đã xong
- **Phase 0 (docs):** `CLAUDE.md` (tracked) + `docs/md/PLAN.md` (local, gitignored) cập nhật đầy đủ. PLAN §4.4.1 có checklist Phase 0/A/B.
- **Phase A bước 1–3 (core pipeline, 83 test pass toàn repo):**
  - `src/utils/detection_config.py` — `DetectionConfig`, `ppe_label_to_index()`, `NUM_DETECTION_CLASSES=9`.
  - `src/data/detection_dataset.py` — `PPEDetectionDataset`, `read_voc_objects`, `voc_to_target`, `detection_collate_fn`, `DetectionRecord`.
  - `src/models/detection_model.py` — `build_detection_model(pretrained=)`, `get/set_detection_head_parameters`, `detection_trainable_parameter_names`, `resolve_device`.
  - `src/training/detection_trainer.py` — `train_detection_head`, `evaluate_detection` (torchmetrics mAP), `train_one_epoch`.
  - Tests: `tests/test_detection_{config,dataset,model,trainer}.py`.
- **Deps cài + khai báo:** `torchmetrics[detection]` 1.9.0, `pycocotools` 2.0.11 (trong `venv`, đã thêm vào `pyproject.toml`).
- **Phase A bước 4–7 (data + baselines + FL sim + scripts, 94 test pass toàn repo):**
  - `src/data/detection_manifest.py` — `collect_detection_samples`, `generate_detection_manifest_rows` (non-IID PPE-skew, leakage-free), write/summarize.
  - `src/data/detection_data.py` — `load_detection_bundle` → `DetectionDatasetBundle` (per-client train/val + pooled + label histogram).
  - `src/training/detection_baselines.py` — `run_detection_centralized`, `run_detection_local_only` (mAP per-client + weighted global).
  - `src/fl/detection_federated.py` — `run_detection_federated` (FedAvg head thủ công, distributed-eval) + `federated_average`.
  - `scripts/generate_detection_manifest.py`, `scripts/run_detection_sim.py`.
  - Tests: `tests/test_detection_{manifest,data,baselines,federated}.py`.
- **Rủi ro retired:** frozen Faster R-CNN train được (loss+backward); mAP end-to-end; head serialize ổn định; **FedAvg detection chạy end-to-end trong sim**; **smoke CPU 12 ảnh THẬT OK** (`outputs/EXP-011-smoke/`, mode federated, mAP=0 đúng kỳ vọng vì random init + 1 round).
- **Phát hiện:** detection head update_size ≈ **58 MB** (so 4 KB head classification) → comm cost rất lớn; cần cân nhắc ở deployment (round ngắn, ít round).
- **Deployment scaffold added:**
  - `src/fl/detection_clientapp.py` — Flower ClientApp detection; hỗ trợ single-shard manifest, `client-id`, `partition-id`, node-level `manifest-path`/`root-dir` override.
  - `src/fl/detection_serverapp.py` — Flower ServerApp FedAvg head-only; server không evaluate pooled data, global metric lấy từ weighted distributed evaluation.
  - `scripts/export_detection_subset.py` — export từng site thành folder + zip (`manifest.csv`, `images/`, `voc_labels/`, `summary.json`).
  - `pyproject.toml` — root Flower app trỏ sang detection app; thêm `[tool.flwr.federations.local-sim]` và `[tool.flwr.federations.deploy]`.
  - Tests mới: `tests/test_detection_flower_apps.py`, `tests/test_export_detection_subset.py`.
  - Verification: `101 passed`; Flower local-sim smoke trên tiny `/private/tmp` manifest pass với `pretrained=false`, CPU, 2 SuperNodes. Lưu ý Flower 1.30 dùng simulation override schema mới (`num-supernodes`, `client-resources-num-cpus`, `client-resources-num-gpus`) và tự migrate legacy federation config khi chạy.
- **DVC dataset sync added (chuẩn bị EXP-011 trên RTX3060):**
  - `data/ppe.dvc` track dataset PPE thật tại workspace path `data/ppe` (`images/`, `voc_labels/`, `labels/`, `meta-data/`).
  - Snapshot hiện tại: ~14.24 GB, 32,399 files (`md5=b7756a42321c6f53a04909179f1402b6.dir`).
  - DVC remote mặc định: `itf-storage` → `ssh://ITF-Server/mnt/data/user_data/lampt/FL/data/dvc-storage/federated-learning`.
  - Lưu ý: remote path là **DVC cache/hash storage**, không phải folder dataset mà training code đọc trực tiếp. Sau `dvc pull`, dataset được materialize lại ở `data/ppe`, và các script vẫn dùng `--root-dir data/ppe`.
  - `requirements.txt` thêm dependency runtime + `dvc[ssh]` để máy RTX3060/Colab có thể cài nhanh.

#### Còn lại
- [x] `src/fl/detection_clientapp.py` + `detection_serverapp.py` — modern Flower API, load shard theo `node_config(manifest-path/root-dir/data-root, client-id/partition-id)`. **Cho Phase B (deployment).**
- [x] `scripts/export_detection_subset.py` — đóng gói từng shard cho client.
- [x] Thêm deps vào `pyproject.toml`: `torchmetrics[detection]`, `pycocotools`.
- [x] DVC track + remote cho `data/ppe` để sync dataset lên RTX3060/ITF-Server mà không commit ảnh vào Git.
- [ ] **Real sim RTX3060** (`--device cuda`, `pretrained=True`, ~300 ảnh/site, vài round) → **journal EXP-011** + registry (cột mAP). *(Không chạy được trên Mac CPU — cần GPU.)*

#### Phase B (sau EXP-011)
- Tailscale up trên 4 máy; `[tool.flwr.federations.local-sim|deploy]` trong `pyproject.toml`; SuperLink trên Mac; export 3 shard; notebook supernode Colab; `flwr run . deploy` → **EXP-012**.

#### Cách resume nhanh
```bash
venv/bin/python -m pytest -q          # phải 94 passed
cat .claude/plans/2026-06-25_detection-pivot-plan.md   # plan đầy đủ (đã chuyển vào repo)
#### Trên máy mới/RTX3060 sau khi clone/pull code:
dvc pull                              # materialize data/ppe từ remote itf-storage
#### Lệnh chạy thật (trên RTX3060):
venv/bin/python scripts/generate_detection_manifest.py --output configs/datasets/ppe_detection_exp011_manifest.csv --per-site 300 --val-fraction 0.2
venv/bin/python scripts/run_detection_sim.py --mode all --manifest configs/datasets/ppe_detection_exp011_manifest.csv --root-dir data/ppe --output-dir outputs/EXP-011 --exp-id EXP-011 --device cuda
```
Việc còn lại chính: **chạy EXP-011 trên GPU** rồi viết journal. Dữ liệu thật: `data/ppe/` (8099 ảnh VOC, **13GB**, gitignored — KHÔNG commit ảnh).

#### Lưu ý
- `data/ppe/voc_labels/*.xml` (VOC) và `data/ppe/labels/*.txt` (YOLO) đều có; dùng VOC.
- Dataset PPE được version bằng DVC: Git chỉ track `data/ppe.dvc` + config DVC, còn ảnh/annotation thật đi qua `dvc push/pull`.
- Máy chạy `dvc pull` cần SSH config có alias `ITF-Server` trỏ tới `itf.viu.edu.vn`, user `lampt`, port `20001`.
- `docs/md/` bị gitignore → PLAN.md là file local, không vào git.
- Repo chạy trên `venv/bin/python`; không có ruff/black trong venv.
