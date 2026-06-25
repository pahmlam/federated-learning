# Nhật Ký Thử Nghiệm Federated Learning (Experiment Journal)

Thư mục này dùng để lưu trữ và theo dõi toàn bộ các thử nghiệm trong dự án Nghiên cứu Federated Learning. Mỗi thử nghiệm được ghi nhận độc lập dưới dạng một file Markdown và đăng ký tại trang này để phục vụ việc so sánh, đánh giá tiến độ.

---

## 1. Quy Trình Thực Hiện & Ghi Nhận (Workflow Flow)

Dành cho cả **Codex**, **Claude Code** (hoặc các AI Coding Agent tương tự) và **Lập Trình Viên**:

1. **Xác định mã thử nghiệm tiếp theo**: 
   - Kiểm tra bảng [Danh sách thử nghiệm](#2-danh-sách-thử-nghiệm-registry-table) bên dưới để tìm mã thử nghiệm (ID) tiếp theo dạng `EXP-XXX` (ví dụ: `EXP-001`).
2. **Thiết lập & Chạy thử nghiệm**:
   - Viết/cập nhật config tương ứng trong `configs/`.
   - Chạy các kịch bản thử nghiệm: `centralized`, `local-only`, và `federated`.
   - Lưu kết quả raw (json, csv, checkpoints) vào thư mục `outputs/EXP-XXX/`.
3. **Tạo file nhật ký thử nghiệm**:
   - Copy nội dung từ file mẫu [template.md](file:///Users/phamtunglam/Documents/VNPT/federated-learning/docs/journal/template.md).
   - Tạo file mới trong thư mục `docs/journal/` đặt tên theo định dạng:  
     `docs/journal/YYYY-MM-DD_EXP-XXX_tieu-de-viet-tat.md`  
     *(Ví dụ: `docs/journal/2026-06-12_EXP-001_baseline-fedavg.md`)*.
   - Điền đầy đủ thông tin kỹ thuật, kết quả baseline, kết quả chi tiết theo client và phân tích.
4. **Cập nhật bảng mục lục**:
   - Thêm dòng thông tin của thử nghiệm vừa thực hiện vào bảng [Danh sách thử nghiệm](#2-danh-sách-thử-nghiệm-registry-table) dưới đây để người quản lý và các phiên làm việc tiếp theo dễ dàng theo dõi.

---

## 2. Danh Sách Thử Nghiệm (Registry Table)

| ID | Ngày chạy | Tên thử nghiệm | Trạng thái | Centralized (Acc/F1) | Local-Only Avg (Acc/F1) | Federated (Acc/F1) | Chi tiết |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **EXP-001** | 2026-06-12 | Synthetic Flower + PyTorch baseline | Thành công | 0.9216 | 0.8039 | 0.7059 | [Chi tiết](file:///Users/phamtunglam/Documents/VNPT/federated-learning/docs/journal/2026-06-12_EXP-001_synthetic-flower-pytorch-baseline.md) |
| **EXP-002** | 2026-06-18 | OOM-safe single-machine smoke profile | Thành công | 0.6207 | 0.9310 | 0.3448 | [Chi tiết](file:///Users/phamtunglam/Documents/VNPT/federated-learning/docs/journal/2026-06-18_EXP-002_oom-safe-single-machine-smoke.md) |
| **EXP-003** | 2026-06-18 | PPE embedding dry-run baseline | Thành công | 0.5000 | 0.0000 | 0.5000 | [Chi tiết](file:///Users/phamtunglam/Documents/VNPT/federated-learning/docs/journal/2026-06-18_EXP-003_ppe-embedding-dry-run-baseline.md) |
| **EXP-004** | 2026-06-22 | PPE real ResNet18 embedding smoke | Thành công | 0.4667/0.4661 | 0.5000/0.3206 | 0.5000/0.3333 | [Chi tiết](file:///Users/phamtunglam/Documents/VNPT/federated-learning/docs/journal/2026-06-22_EXP-004_ppe-real-resnet18-embedding-smoke.md) |
| **EXP-005** | 2026-06-22 | PPE real ResNet18 stability baseline | Thành công | 0.5167/0.5055 | 0.6833/0.6832 | 0.7000/0.6997 | [Chi tiết](file:///Users/phamtunglam/Documents/VNPT/federated-learning/docs/journal/2026-06-22_EXP-005_ppe-real-resnet18-stability-baseline.md) |

*(Khi chạy thử nghiệm thực tế, hãy thay thế hoặc thêm dòng mới bên dưới dòng mẫu này).*

---

## 3. Một Số Nguyên Tắc Quan Trọng (Critical Rules)

- **Không bỏ qua Baseline**: Mọi thử nghiệm đều phải có số liệu đối chứng từ Centralized (nếu khả thi) và Local-Only.
- **Báo cáo chi tiết theo Client**: Do đặc thù dữ liệu biên là Non-IID, báo cáo global trung bình là chưa đủ; cần phải có bảng phân tách kết quả của từng client để đánh giá mức độ bị ảnh hưởng bởi Domain Shift.
- **Đo chi phí truyền thông & thời gian**: Ghi nhận thời gian huấn luyện local, kích thước file update truyền đi (head/adapter/LoRA), và tổng dung lượng trao đổi qua mạng.
- **Đảm bảo khả năng tái lập (Reproducibility)**: Luôn lưu lại Git Commit Hash và cấu hình chính xác (command line) được sử dụng để thực thi.
