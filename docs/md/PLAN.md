# Kế hoạch nghiên cứu và demo FL

Thời gian: **10/06/2026 - 31/08/2026**  
Trọng tâm: **Federated Learning cho camera/vision trên edge**, trước mắt tập trung **face** và **quần áo/PPE**.  
Mặc định kỹ thuật: **pretrained backbone, freeze backbone, chỉ train embedding/head/adapter/LoRA**.

---

## 1. Mục tiêu

- [x] Hiểu nền tảng Federated Learning, đặc biệt client/server, update, aggregation, non-IID, communication cost và edge constraints.
- [x] Học Flower đủ để chạy simulation, hiểu ClientApp, ServerApp, strategy, FedAvg và logging metric.
- [x] Xây repo/demo FL chuẩn với 3 mode: `centralized`, `local-only`, `federated`.
- [x] Dựng demo tối thiểu Flower + PyTorch với nhiều client mô phỏng.
- [ ] Dựng demo face theo hướng freeze backbone, train embedding/head.
- [ ] Dựng demo quần áo/PPE theo hướng freeze backbone, train head/adapter.
- [x] Đo metric theo client/site, thời gian train, số round, kích thước update và communication cost.
- [ ] Từ baseline, xác định hướng cải tiến method nhẹ như personalized head, adapter/LoRA, FedBN hoặc giảm communication.

---

## 2. Nguyên tắc kỹ thuật

- Không train full model từ đầu trong giai đoạn này.
- Không bắt đầu bằng việc tự viết framework FL mới.
- Flower là baseline framework để dựng luồng FL client-server và simulation.
- FedAvg là baseline aggregation đầu tiên.
- Mọi thử nghiệm cần có ít nhất một baseline so sánh: `local-only`, `centralized` nếu có thể, và `federated`.
- Luôn báo cáo metric theo từng client/site, không chỉ trung bình global.
- Ưu tiên task đơn giản chạy được trước, mở rộng độ khó sau.
- Demo trước, method sau.

---

## 3. Milestone tổng quan

| Giai đoạn | Thời gian | Trọng tâm | Ngõ ra chính |
| --- | --- | --- | --- |
| 1 | 10/06 - 20/06 | FL nền tảng, Flower, repo skeleton | Note Flower, repo skeleton, hiểu luồng FL |
| 2 | 21/06 - 30/06 | Demo tối thiểu Flower + PyTorch | Chạy được 3 mode cơ bản trên toy/simple dataset |
| 3 | 01/07 - 10/07 | Demo face | Freeze backbone, train embedding/head, chia client non-IID |
| 4 | 11/07 - 21/07 | Demo quần áo/PPE | Train head/adapter, so sánh baseline theo client |
| 5 | 22/07 - 31/07 | Edge simulation | Đo thời gian, RAM/CPU/GPU, update size, round |
| 6 | 01/08 - 15/08 | Method nhẹ | Thử adapter/LoRA, personalized head, FedBN hoặc giảm communication |
| 7 | 16/08 - 31/08 | Tổng hợp | Báo cáo kỹ thuật, kết luận, slide trình bày |

---

## 4. Checklist theo ngày

### 4.1. 10/06 - 20/06: FL nền tảng, Flower, repo skeleton

| Ngày | Trọng tâm | Checklist việc cần làm | Deliverable | Trạng thái |
| --- | --- | --- | --- | --- |
| 10/06/2026 | Khởi động hướng nghiên cứu | - [x] Đọc lại yêu cầu thử việc<br>- [x] Tóm tắt mục tiêu FL<br>- [x] Xác định phạm vi face và quần áo/PPE | Mục tiêu FL: xây dựng demo/repo thử nghiệm FL cho camera edge, trước mắt với face và quần áo/PPE. Hệ thống dùng pretrained backbone, freeze backbone, chỉ train embedding/head/adapter tại từng client/site. Server tổng hợp update bằng Flower/FedAvg. Cần so sánh local-only, centralized và federated trên dữ liệu non-IID, đồng thời đo metric theo client, thời gian train và communication cost. | [x] |
| 11/06/2026 | Bối cảnh FL cho camera | - [x] Đọc lại note FL nền tảng<br>- [x] Làm rõ client là site, edge box, NVR hoặc server local<br>- [x] Ghi lại luồng global model, local train, aggregate | Note luồng FL camera | [x] |
| 12/06/2026 | Flower cơ bản | - [x] Đọc Flower quickstart<br>- [x] Chạy hoặc đọc kỹ demo NumPy hiện có<br>- [x] Ghi lại ServerApp, ClientApp, strategy | Note Flower quickstart | [x] |
| 13/06/2026 | Repo skeleton | - [x] Review cấu trúc repo đã dựng<br>- [ ] Ghi vai trò `configs`, `data`, `src`, `experiments`, `outputs`<br>- [ ] Xác định nơi đặt demo sau này | Cấu trúc repo rõ ràng | [ ] |
| 14/06/2026 | Đệm cuối tuần | - [ ] Tổng hợp lại câu hỏi chưa rõ<br>- [ ] Đọc nhẹ về FedAvg và non-IID<br>- [ ] Không code nặng | Note cuối tuần | [ ] |
| 15/06/2026 | Non-IID trong camera | - [ ] Đọc feature skew, label skew, quantity skew<br>- [ ] Liên hệ với face và PPE<br>- [ ] Viết ví dụ chia client non-IID | Note non-IID | [ ] |
| 16/06/2026 | Baseline cần có | - [ ] Xác định `centralized`, `local-only`, `federated` khác nhau thế nào<br>- [ ] Ghi metric cần đo<br>- [ ] Chốt không train full model | Checklist baseline | [ ] |
| 17/06/2026 | Demo tối thiểu | - [x] Chọn task đơn giản để bắt đầu<br>- [x] Chọn dataset tạm thời nếu chưa có dữ liệu nội bộ<br>- [x] Xác định 3-5 client mô phỏng | Thiết kế demo tối thiểu | [x] |
| 18/06/2026 | Freeze backbone | - [ ] Đọc lại khái niệm pretrained backbone<br>- [ ] Xác định phần trainable: embedding/head/adapter<br>- [ ] Ghi cách đo update size | Note parameter-efficient FL | [ ] |
| 19/06/2026 | Flower simulation | - [x] Đọc cách chạy simulation nhiều client<br>- [x] Ghi cách truyền config qua Flower<br>- [x] Ghi cách log metric theo round | Note Flower simulation | [x] |
| 20/06/2026 | Tổng kết giai đoạn 1 | - [ ] Tổng hợp kiến thức FL + Flower<br>- [ ] Cập nhật câu hỏi cần hỏi quản lý nếu có<br>- [ ] Chốt việc tuần sau | Báo cáo tuần 1 ngắn | [ ] |

### 4.2. 21/06 - 30/06: Demo tối thiểu Flower + PyTorch

| Ngày | Trọng tâm | Checklist việc cần làm | Deliverable | Trạng thái |
| --- | --- | --- | --- | --- |
| 21/06/2026 | Đệm thiết kế | - [ ] Review skeleton repo<br>- [ ] Viết outline module tương lai<br>- [ ] Nghỉ/đệm nếu cần | Outline module | [ ] |
| 22/06/2026 | Config demo | - [x] Chốt config tối thiểu cho dataset, model, train<br>- [x] Chốt số client, round, epoch local<br>- [x] Chốt metric demo | Spec config demo | [x] |
| 23/06/2026 | Data partition | - [x] Thiết kế IID partition<br>- [x] Thiết kế non-IID partition<br>- [x] Ghi format partition file | Spec partition | [x] |
| 24/06/2026 | Centralized baseline | - [x] Thiết kế luồng centralized<br>- [x] Xác định input/output cần log<br>- [x] Chốt metric report | Spec centralized | [x] |
| 25/06/2026 | Local-only baseline | - [x] Thiết kế luồng local-only<br>- [x] Xác định cách evaluate từng client<br>- [x] Chốt cách so sánh với FL | Spec local-only | [x] |
| 26/06/2026 | Federated baseline | - [x] Thiết kế luồng Flower FedAvg<br>- [x] Xác định phần update được gửi<br>- [x] Chốt cách log round metric | Spec federated | [x] |
| 27/06/2026 | Đệm cuối tuần | - [ ] Review spec 3 baseline<br>- [ ] Ghi rủi ro triển khai<br>- [ ] Không mở rộng scope | Note rủi ro | [ ] |
| 28/06/2026 | Đọc thêm Flower | - [x] Đọc strategy customization<br>- [x] Ghi cách custom aggregation sau này<br>- [x] Đọc logging/result object | Note strategy | [x] |
| 29/06/2026 | Smoke test plan | - [x] Xác định smoke test cần có<br>- [x] Xác định command chạy demo sau này<br>- [x] Ghi expected outputs | Smoke test spec | [x] |
| 30/06/2026 | Tổng kết demo tối thiểu | - [ ] Tổng hợp spec demo tối thiểu<br>- [ ] Check đủ centralized, local-only, federated<br>- [ ] Chuẩn bị chuyển sang face | Báo cáo tuần 2 | [ ] |

### 4.3. 01/07 - 10/07: Demo face với freeze backbone, train embedding/head

| Ngày | Trọng tâm | Checklist việc cần làm | Deliverable | Trạng thái |
| --- | --- | --- | --- | --- |
| 01/07/2026 | Chọn task face | - [ ] Chọn identification, verification hoặc embedding adaptation<br>- [ ] Xác định metric tương ứng<br>- [ ] Ghi ràng buộc privacy | Spec task face | [ ] |
| 02/07/2026 | Dataset face | - [ ] Tìm dataset public phù hợp nếu chưa có nội bộ<br>- [ ] Kiểm tra license/usage<br>- [ ] Ghi phương án fallback | Note dataset face | [ ] |
| 03/07/2026 | Pipeline face | - [ ] Xác định detect/crop face có cần không<br>- [ ] Xác định backbone pretrained<br>- [ ] Xác định head/embedding trainable | Spec pipeline face | [ ] |
| 04/07/2026 | Đệm cuối tuần | - [ ] Đọc face embedding basics<br>- [ ] Ghi metric verification nếu dùng<br>- [ ] Review privacy note | Note face metric | [ ] |
| 05/07/2026 | Đệm cuối tuần | - [ ] Review chia client theo identity/site/domain<br>- [ ] Ghi các case non-IID face<br>- [ ] Nghỉ/đệm tiến độ | Note non-IID face | [ ] |
| 06/07/2026 | Partition face | - [ ] Thiết kế client split theo identity/domain<br>- [ ] Xác định train/val/test per client<br>- [ ] Ghi cách tránh leakage | Spec split face | [ ] |
| 07/07/2026 | Baseline face | - [ ] Thiết kế centralized face baseline<br>- [ ] Thiết kế local-only face baseline<br>- [ ] Thiết kế FL face baseline | Spec baseline face | [ ] |
| 08/07/2026 | Logging face | - [ ] Chốt metric per client<br>- [ ] Chốt metric global<br>- [ ] Chốt cách log update size và time | Spec logging face | [ ] |
| 09/07/2026 | Review face demo | - [ ] Review toàn bộ thiết kế face<br>- [ ] Ghi rủi ro dataset/privacy<br>- [ ] Chốt việc cần code khi triển khai | Checklist triển khai face | [ ] |
| 10/07/2026 | Tổng kết face | - [ ] Viết tóm tắt demo face<br>- [ ] Soát lại giả định freeze backbone<br>- [ ] Chuẩn bị chuyển sang PPE | Báo cáo face ngắn | [ ] |

### 4.4. 11/07 - 21/07: Demo quần áo/PPE với train head/adapter

| Ngày | Trọng tâm | Checklist việc cần làm | Deliverable | Trạng thái |
| --- | --- | --- | --- | --- |
| 11/07/2026 | Chọn task PPE | - [ ] Chọn classification hay detection đơn giản<br>- [ ] Chốt nhãn: mũ, áo phản quang, đồng phục hoặc compliance<br>- [ ] Ghi metric chính | Spec task PPE | [ ] |
| 12/07/2026 | Dataset PPE | - [ ] Tìm dataset public hoặc phương án nội bộ<br>- [ ] Kiểm tra nhãn có phù hợp không<br>- [ ] Ghi phương án fallback classification | Note dataset PPE | [ ] |
| 13/07/2026 | Pipeline PPE | - [ ] Xác định backbone pretrained<br>- [ ] Xác định head/adapter trainable<br>- [ ] Xác định input/output model | Spec pipeline PPE | [ ] |
| 14/07/2026 | Non-IID PPE | - [ ] Thiết kế split theo màu mũ/áo/site/ánh sáng<br>- [ ] Xác định client đại diện<br>- [ ] Ghi expected difficulty | Spec split PPE | [ ] |
| 15/07/2026 | Baseline PPE | - [ ] Thiết kế centralized baseline<br>- [ ] Thiết kế local-only baseline<br>- [ ] Thiết kế FL FedAvg baseline | Spec baseline PPE | [ ] |
| 16/07/2026 | Metric PPE | - [ ] Chốt F1/mAP/recall vi phạm<br>- [ ] Chốt false alarm và false negative nếu có<br>- [ ] Chốt report theo client | Spec metric PPE | [ ] |
| 17/07/2026 | Human-in-the-loop | - [ ] Ghi cách dữ liệu được lấy mẫu ở site<br>- [ ] Ghi cách gắn nhãn bán tự động<br>- [ ] Ghi vai trò con người trong loop | Note labeling PPE | [ ] |
| 18/07/2026 | Đệm cuối tuần | - [ ] Review spec PPE<br>- [ ] So sánh PPE với face<br>- [ ] Ghi ưu/nhược điểm từng demo | Note so sánh face/PPE | [ ] |
| 19/07/2026 | Đệm cuối tuần | - [ ] Đọc thêm về adapter/LoRA nếu cần<br>- [ ] Ghi ý tưởng train phần nhẹ<br>- [ ] Nghỉ/đệm tiến độ | Note adapter | [ ] |
| 20/07/2026 | Review toàn bộ baseline | - [ ] Soát đủ 3 mode cho face và PPE<br>- [ ] Soát log metric/time/update size<br>- [ ] Ghi checklist triển khai | Checklist baseline tổng | [ ] |
| 21/07/2026 | Tổng kết giai đoạn 4 | - [ ] Viết tóm tắt demo PPE<br>- [ ] Chốt demo ưu tiên nếu thiếu thời gian<br>- [ ] Chuẩn bị edge simulation | Báo cáo tuần 4 | [ ] |

### 4.5. 22/07 - 31/07: Edge simulation và đo tài nguyên

| Ngày | Trọng tâm | Checklist việc cần làm | Deliverable | Trạng thái |
| --- | --- | --- | --- | --- |
| 22/07/2026 | Xác định edge setting | - [ ] Xác định client giả lập là laptop CPU, edge box hay server local<br>- [ ] Chốt tài nguyên cần đo<br>- [ ] Ghi giới hạn thiết bị | Spec edge setting | [ ] |
| 23/07/2026 | Runtime metric | - [ ] Chốt đo train time local<br>- [ ] Chốt đo RAM/CPU/GPU nếu có<br>- [ ] Chốt đo update size | Spec runtime metric | [ ] |
| 24/07/2026 | Communication cost | - [ ] Xác định cách tính bytes per round<br>- [ ] So sánh full model với head/adapter update trên lý thuyết<br>- [ ] Ghi công thức báo cáo | Note communication | [ ] |
| 25/07/2026 | Đệm cuối tuần | - [ ] Review edge constraints<br>- [ ] Đọc client availability/straggler<br>- [ ] Nghỉ/đệm tiến độ | Note edge constraints | [ ] |
| 26/07/2026 | Đệm cuối tuần | - [ ] Ghi các rủi ro khi train trên edge<br>- [ ] Review cách đặt lịch train local<br>- [ ] Không mở rộng scope | Note edge risk | [ ] |
| 27/07/2026 | Flower deployment concept | - [ ] Đọc SuperLink/SuperNode ở mức khái niệm<br>- [ ] Ghi khác biệt simulation và deployment<br>- [ ] Xác định chưa cần deploy thật nếu thiếu thiết bị | Note deployment Flower | [ ] |
| 28/07/2026 | Edge experiment design | - [ ] Thiết kế experiment CPU-only<br>- [ ] Thiết kế experiment giảm batch/epoch<br>- [ ] Chốt expected outputs | Spec edge experiment | [ ] |
| 29/07/2026 | Resource report | - [ ] Chốt format bảng tài nguyên<br>- [ ] Chốt biểu đồ time/update size<br>- [ ] Ghi cách so sánh face/PPE | Template resource report | [ ] |
| 30/07/2026 | Review edge scope | - [ ] Kiểm tra edge scope có thực tế không<br>- [ ] Ghi fallback nếu không có thiết bị<br>- [ ] Chuẩn bị method phase | Checklist edge | [ ] |
| 31/07/2026 | Tổng kết edge | - [ ] Viết tóm tắt edge simulation<br>- [ ] Chốt bottleneck chính<br>- [ ] Chọn 2-3 method candidate | Báo cáo edge ngắn | [ ] |

### 4.6. 01/08 - 15/08: Thử hướng method nhẹ

| Ngày | Trọng tâm | Checklist việc cần làm | Deliverable | Trạng thái |
| --- | --- | --- | --- | --- |
| 01/08/2026 | Chọn candidate method | - [ ] Review bottleneck baseline<br>- [ ] Chọn adapter/LoRA, personalized head, FedBN hoặc giảm communication<br>- [ ] Chốt tiêu chí chọn | Danh sách method candidate | [ ] |
| 02/08/2026 | Đệm cuối tuần | - [ ] Đọc paper/note liên quan method đã chọn<br>- [ ] Ghi điểm mới có thể thử<br>- [ ] Nghỉ/đệm tiến độ | Note method | [ ] |
| 03/08/2026 | Method 1 spec | - [ ] Viết giả thuyết method 1<br>- [ ] Xác định baseline so sánh<br>- [ ] Xác định metric thắng/thua | Spec method 1 | [ ] |
| 04/08/2026 | Method 1 experiment | - [ ] Thiết kế experiment method 1<br>- [ ] Chốt config thay đổi<br>- [ ] Chốt expected output | Experiment method 1 | [ ] |
| 05/08/2026 | Method 1 review | - [ ] Review rủi ro method 1<br>- [ ] Ghi cách tránh so sánh không công bằng<br>- [ ] Chuẩn bị method 2 | Review method 1 | [ ] |
| 06/08/2026 | Method 2 spec | - [ ] Viết giả thuyết method 2<br>- [ ] Xác định baseline so sánh<br>- [ ] Xác định metric thắng/thua | Spec method 2 | [ ] |
| 07/08/2026 | Method 2 experiment | - [ ] Thiết kế experiment method 2<br>- [ ] Chốt config thay đổi<br>- [ ] Chốt expected output | Experiment method 2 | [ ] |
| 08/08/2026 | Đệm cuối tuần | - [ ] Review method 1 và method 2<br>- [ ] Ghi câu hỏi nghiên cứu sắc hơn<br>- [ ] Nghỉ/đệm tiến độ | Note research question | [ ] |
| 09/08/2026 | Đệm cuối tuần | - [ ] Đọc lại personalized FL/FedBN nếu cần<br>- [ ] Ghi liên hệ với face/PPE<br>- [ ] Không mở rộng quá nhiều | Note pFL/FedBN | [ ] |
| 10/08/2026 | So sánh method | - [ ] Thiết kế bảng so sánh method<br>- [ ] Chốt metric model quality<br>- [ ] Chốt metric communication/compute | Template so sánh method | [ ] |
| 11/08/2026 | Fair comparison | - [ ] Chốt cùng dataset/split/round cho baseline và method<br>- [ ] Ghi random seed cần cố định<br>- [ ] Ghi điều kiện dừng | Checklist fair comparison | [ ] |
| 12/08/2026 | Ablation plan | - [ ] Thiết kế ablation nhỏ<br>- [ ] Chọn biến số: rank LoRA, head local/global, FedBN on/off<br>- [ ] Chốt ngõ ra | Spec ablation | [ ] |
| 13/08/2026 | Edge-aware angle | - [ ] Liên hệ method với edge constraints<br>- [ ] Ghi giảm compute/communication kỳ vọng<br>- [ ] Chốt cách báo cáo | Note edge-aware method | [ ] |
| 14/08/2026 | Tổng hợp method | - [ ] Chọn method có tín hiệu tốt nhất để kể trong báo cáo<br>- [ ] Ghi hạn chế<br>- [ ] Ghi hướng paper nếu có | Method summary | [ ] |
| 15/08/2026 | Chốt giai đoạn method | - [ ] Tổng kết các thử nghiệm method<br>- [ ] Xác định còn thiếu số liệu gì<br>- [ ] Chuẩn bị viết báo cáo cuối | Báo cáo method ngắn | [ ] |

### 4.7. 16/08 - 31/08: Tổng hợp kết quả, báo cáo, slide

| Ngày | Trọng tâm | Checklist việc cần làm | Deliverable | Trạng thái |
| --- | --- | --- | --- | --- |
| 16/08/2026 | Đệm tổng hợp | - [ ] Gom note các giai đoạn<br>- [ ] Gom bảng metric cần có<br>- [ ] Lập outline báo cáo cuối | Outline báo cáo | [ ] |
| 17/08/2026 | Viết phần bối cảnh | - [ ] Viết mục FL cho camera edge<br>- [ ] Viết lý do face/PPE<br>- [ ] Viết nguyên tắc freeze backbone | Draft bối cảnh | [ ] |
| 18/08/2026 | Viết phần repo/demo | - [ ] Mô tả cấu trúc repo<br>- [ ] Mô tả 3 mode baseline<br>- [ ] Mô tả Flower dùng ở đâu | Draft repo/demo | [ ] |
| 19/08/2026 | Viết phần dữ liệu | - [ ] Mô tả dataset/split dự kiến hoặc đã dùng<br>- [ ] Mô tả IID/non-IID<br>- [ ] Mô tả privacy và labeling | Draft dữ liệu | [ ] |
| 20/08/2026 | Viết phần face | - [ ] Mô tả task face<br>- [ ] Mô tả backbone/head<br>- [ ] Mô tả metric và kết quả nếu có | Draft face | [ ] |
| 21/08/2026 | Viết phần PPE | - [ ] Mô tả task PPE<br>- [ ] Mô tả head/adapter<br>- [ ] Mô tả metric và kết quả nếu có | Draft PPE | [ ] |
| 22/08/2026 | Đệm cuối tuần | - [ ] Review draft báo cáo<br>- [ ] Ghi điểm yếu cần bổ sung<br>- [ ] Nghỉ/đệm tiến độ | Review draft 1 | [ ] |
| 23/08/2026 | Đệm cuối tuần | - [ ] Sắp xếp hình/bảng cần có<br>- [ ] Chuẩn bị slide outline<br>- [ ] Không thêm scope mới | Slide outline | [ ] |
| 24/08/2026 | Viết phần edge | - [ ] Mô tả edge simulation<br>- [ ] Mô tả compute/communication cost<br>- [ ] Mô tả hạn chế thiết bị | Draft edge | [ ] |
| 25/08/2026 | Viết phần method | - [ ] Mô tả method candidate<br>- [ ] Mô tả lý do chọn<br>- [ ] Mô tả kết quả hoặc kỳ vọng | Draft method | [ ] |
| 26/08/2026 | Kết luận kỹ thuật | - [ ] Viết kết luận cái đã làm được<br>- [ ] Viết cái chưa làm được<br>- [ ] Viết hướng tiếp theo | Draft kết luận | [ ] |
| 27/08/2026 | Làm slide | - [ ] Tạo slide bối cảnh và mục tiêu<br>- [ ] Tạo slide pipeline/demo<br>- [ ] Tạo slide kết quả/method | Draft slide | [ ] |
| 28/08/2026 | Review báo cáo | - [ ] Soát logic báo cáo<br>- [ ] Soát không nói train full model<br>- [ ] Soát bảng metric và milestone | Báo cáo gần cuối | [ ] |
| 29/08/2026 | Đệm cuối tuần | - [ ] Tập trình bày thử<br>- [ ] Ghi câu hỏi có thể bị hỏi<br>- [ ] Chuẩn bị câu trả lời | Q&A note | [ ] |
| 30/08/2026 | Đệm cuối tuần | - [ ] Chỉnh slide theo Q&A<br>- [ ] Chỉnh phần hạn chế/hướng tiếp theo<br>- [ ] Nghỉ/đệm cuối | Slide gần cuối | [ ] |
| 31/08/2026 | Hoàn thiện | - [ ] Hoàn thiện báo cáo cuối<br>- [ ] Hoàn thiện slide<br>- [ ] Tổng kết kế hoạch và next steps | Báo cáo + slide cuối | [ ] |

---

## 5. Tiêu chí hoàn thành

- [ ] Có repo skeleton rõ vai trò từng thư mục.
- [x] Có demo Flower/PyTorch chạy được hoặc có spec đủ chi tiết để triển khai ngay.
- [x] Có đủ 3 mode: `centralized`, `local-only`, `federated`.
- [ ] Có thiết kế hoặc demo face theo hướng freeze backbone, train embedding/head.
- [ ] Có thiết kế hoặc demo quần áo/PPE theo hướng freeze backbone, train head/adapter.
- [x] Có chia dữ liệu theo client và có kịch bản non-IID.
- [x] Có metric theo từng client/site.
- [x] Có đo hoặc thiết kế đo communication cost, update size, training time.
- [ ] Có nhận xét Flower giải quyết phần nào và phần nào phải tự nghiên cứu.
- [ ] Có ít nhất 1 hướng method nhẹ để tiếp tục đào sâu.
- [ ] Có báo cáo cuối và slide trình bày.

---

## 6. Ghi chú cập nhật

- 11/06/2026: Kế hoạch này được tạo theo hướng face + quần áo/PPE, dùng Flower trước, không train full model từ đầu.
- 12/06/2026: Thiết lập flow ghi chép nhật ký thử nghiệm tại docs/journal/ để tự động hóa việc lưu trữ thông tin đối chiếu baseline, tài nguyên và chi tiết client bởi AI Agents (Codex/Claude Code).
- 12/06/2026: Hoàn thành demo synthetic Flower + PyTorch đầu tiên với 3 mode `centralized`, `local-only`, `federated`; kết quả được lưu tại `outputs/EXP-001/` và `docs/journal/2026-06-12_EXP-001_synthetic-flower-pytorch-baseline.md`.
- Khi có dữ liệu nội bộ hoặc yêu cầu cụ thể hơn từ quản lý, cập nhật lại các ngày liên quan dataset, task và metric.
