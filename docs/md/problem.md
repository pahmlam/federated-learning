# Problem Statement: Federated Learning cho camera edge

File này dùng để ghi rõ **vấn đề cần nghiên cứu**.  
Chưa đưa giải pháp triển khai, chưa chốt method, chưa chốt framework mới.

---

## 1. Bối cảnh

Bài toán đang xét là áp dụng Federated Learning (FL) cho hệ thống camera/vision triển khai ở nhiều địa điểm khác nhau. Hai nhóm use case trước mắt là:

- **Face:** nhận diện, xác thực hoặc thích nghi embedding/head theo site.
- **Quần áo/PPE:** nhận diện đồng phục, mũ bảo hộ, áo phản quang, khẩu trang hoặc trạng thái tuân thủ bảo hộ.

Đặc điểm chung của bối cảnh camera:

- Dữ liệu ảnh/video sinh ra liên tục và có dung lượng lớn.
- Dữ liệu nằm phân tán theo camera, cụm camera, site, edge box, NVR hoặc server local.
- Dữ liệu có yếu tố nhạy cảm: khuôn mặt, người lao động, layout nhà máy, quy trình vận hành.
- Các site/camera có điều kiện khác nhau: góc nhìn, ánh sáng, background, chất lượng camera, quy định vận hành.
- Không nên mặc định dữ liệu thô có thể gom về server trung tâm.

---

## 2. Phạm vi kỹ thuật đã chốt

Trong giai đoạn hiện tại, phạm vi kỹ thuật không phải là huấn luyện một mô hình thị giác lớn từ đầu.

Mặc định nghiên cứu:

- Dùng backbone pretrained.
- Freeze hoặc gần freeze backbone.
- Chỉ huấn luyện phần nhẹ như `embedding`, `head`, `adapter` hoặc `LoRA`.
- Mỗi client/site chỉ sử dụng dữ liệu local.
- Server chỉ nhận model update hoặc phần update được chọn, không nhận ảnh/video thô.

Điều này làm vấn đề nghiên cứu hẹp hơn:

> Làm sao để huấn luyện và tổng hợp các phần nhẹ của model một cách hiệu quả trong bối cảnh dữ liệu camera phân tán, nhạy cảm, non-IID và có nhãn không hoàn hảo.

### 2.1. Trọng tâm nghiên cứu thực sự

Như vậy, nghiên cứu hiện tại không nằm ở việc dùng FL để train một model vision lớn từ đầu. Trọng tâm cần làm rõ là:

- Phần nào của model nên được train tại client.
- Phần nào nên giữ global giữa các site.
- Phần nào nên giữ local để thích nghi với từng camera/site.
- Update của `embedding`, `head`, `adapter` hoặc `LoRA` nên được gửi và kiểm soát như thế nào.
- Aggregation cần xử lý ra sao khi dữ liệu giữa các camera/site non-IID.
- Chất lượng sau FL có đủ tốt so với `local-only` và `centralized` không.
- Việc chỉ train phần nhẹ giảm được bao nhiêu compute và communication so với full fine-tuning.

---

## 3. Vấn đề cốt lõi

Vòng FL cơ bản như server gửi model, client train local, client gửi update, server aggregate là một khung đã có. Phần khó trong bối cảnh camera không nằm ở việc mô tả lại vòng lặp này.

Vấn đề cốt lõi là:

> Khi triển khai trên nhiều camera/site, dữ liệu local có khối lượng lớn, phân phối khác nhau, nhãn khó tạo và chất lượng nhãn không đồng đều. Nếu dữ liệu local và nhãn không đủ tốt, update từ client có thể nhiễu hoặc lệch, làm FL không cải thiện được mô hình chung.

Nói cách khác:

- FL cần dữ liệu local có giá trị.
- Dữ liệu camera thô quá nhiều để dùng trực tiếp.
- Nhãn đúng không tự nhiên có sẵn.
- Gắn nhãn thủ công toàn bộ không khả thi khi quy mô lên hàng trăm camera.
- Gắn nhãn tự động hoàn toàn có rủi ro tạo nhãn sai.
- Client có dữ liệu/nhãn kém vẫn có thể gửi update làm hại global model.

---

## 4. Các nhóm vấn đề cần làm rõ

### 4.1. Vấn đề dữ liệu

Camera tạo ra lượng video/frame rất lớn. Không thể giả định toàn bộ dữ liệu đều được lưu, xử lý và dùng để train.

Các câu hỏi cần trả lời trong quá trình nghiên cứu và triển khai:

- Dữ liệu nào thật sự có giá trị để học thêm?
- Có cần dùng toàn bộ frame hay chỉ một phần nhỏ?
- Làm sao tránh trùng lặp quá nhiều giữa các frame gần giống nhau?
- Dữ liệu mới từ camera/site có khác gì so với dữ liệu model đã từng thấy?
- Dữ liệu local có đủ đại diện cho lỗi thực tế tại site không?

### 4.2. Vấn đề nhãn

FL không tự giải quyết việc tạo nhãn. Với camera, nhãn có thể đến từ con người, model hiện tại, rule vận hành, feedback của người dùng hoặc một cơ chế bán tự động nào đó. Tuy nhiên, chất lượng nhãn là vấn đề chưa được đảm bảo.

Các câu hỏi cần trả lời trong quá trình nghiên cứu và triển khai:

- Ai tạo nhãn cho dữ liệu local?
- Nhãn nào cần con người xác nhận?
- Khi nào có thể tin nhãn tự động?
- Nhãn sai ảnh hưởng thế nào tới local update?
- Có thể đo chất lượng nhãn trước khi dùng để train không?
- Khi quy mô hàng trăm camera, khối lượng review của con người có vượt quá khả năng vận hành không?

### 4.3. Vấn đề non-IID giữa site/camera

Dữ liệu giữa các site/camera thường không cùng phân phối. Đây là một đặc điểm tự nhiên của hệ thống camera.

Nguồn non-IID có thể gồm:

- Góc đặt camera khác nhau.
- Ánh sáng ngày/đêm, trong nhà/ngoài trời, ngược sáng hoặc ánh sáng yếu.
- Background khác nhau giữa nhà máy, kho, văn phòng, cổng, công trường.
- Chất lượng camera, độ phân giải, FPS, nén video khác nhau.
- Phân phối lớp khác nhau giữa các site.
- Quy định vận hành khác nhau, đặc biệt với PPE hoặc vùng cấm.

Các câu hỏi cần trả lời trong quá trình nghiên cứu và triển khai:

- Mức độ non-IID nào làm FedAvg hoặc aggregation cơ bản kém hiệu quả?
- Global model có thể tốt trung bình nhưng kém ở một số site không?
- Khi nào cần giữ phần local riêng cho từng site?
- Có cần đánh giá từng site thay vì chỉ dùng trung bình toàn hệ thống không?

### 4.4. Vấn đề edge và tài nguyên

Trong triển khai thực tế, client không nhất thiết là camera đơn lẻ. Client có thể là edge box, NVR hoặc server local. Các thiết bị này có tài nguyên khác nhau.

Các câu hỏi cần trả lời trong quá trình nghiên cứu và triển khai:

- Client thực tế có đủ tài nguyên để train local không?
- Chi phí local training có ảnh hưởng tới inference đang chạy không?
- Kích thước update có phù hợp với băng thông không?
- Số round FL cần thiết có thực tế không?
- Việc chỉ train phần nhẹ có giảm đủ compute và communication không?

### 4.5. Vấn đề đánh giá

Nếu chỉ đo accuracy trung bình, có thể đánh giá sai hiệu quả của FL trong hệ camera.

Các câu hỏi cần trả lời trong quá trình nghiên cứu và triển khai:

- Metric nào phản ánh đúng lỗi vận hành?
- Có cần đo theo từng client/site không?
- Có cần đo false alarm, false negative, recall lớp vi phạm hoặc verification metric không?
- Có cần đo thời gian train, RAM/CPU/GPU, update size và communication cost không?
- Có cần so sánh với `local-only` và `centralized` không?

---

## 5. Vấn đề riêng của từng use case

### 5.1. Face

Face là nhóm dữ liệu rất nhạy cảm. Việc gom dữ liệu thô về server trung tâm có thể không phù hợp về privacy, quyền sử dụng dữ liệu hoặc chính sách vận hành.

Các vấn đề cần làm rõ:

- Nhãn identity hoặc pair same/different được tạo như thế nào?
- Dữ liệu face giữa các site có khác nhau về camera, ánh sáng, góc mặt, độ phân giải không?
- Có nguy cơ rò rỉ thông tin cá nhân qua model update không?
- Nếu chỉ train embedding/head, có đủ để thích nghi với site mới không?
- Đánh giá nên dùng identification, verification hay embedding quality?

### 5.2. Quần áo/PPE

Quần áo/PPE phụ thuộc mạnh vào quy định từng site. Một site có thể yêu cầu mũ và áo phản quang, site khác có thể yêu cầu đồng phục, khẩu trang hoặc găng tay.

Các vấn đề cần làm rõ:

- Label space giữa các site có thống nhất không?
- Quy định PPE khác nhau có làm task giữa client không đồng nhất không?
- Vật thể nhỏ như mũ, khẩu trang, găng tay có đủ rõ trong ảnh không?
- Case vi phạm có đủ nhiều để train không?
- Khi cảnh báo sai, feedback từ người vận hành được xử lý như thế nào?
- Nếu nhãn PPE sai hoặc thiếu, local update ảnh hưởng thế nào tới global model?

---

## 6. Bottleneck lớn nhất

Bottleneck lớn nhất có thể không phải là triển khai khung client-server của FL, mà là:

> Nếu sản phẩm nhắm đến việc tự động hoá quá trình thu dữ liệu, gắn nhãn, huấn luyện ngay tại biên thì làm sao tạo được dữ liệu local có giá trị và nhãn đủ tin cậy ở quy mô nhiều camera/site.

Đây là vấn đề quan trọng vì:

- Dữ liệu camera quá nhiều.
- Con người không thể xem và gắn nhãn tất cả.
- Nhãn tự động hoàn toàn có thể sai.
- Dữ liệu dễ bị trùng lặp hoặc toàn mẫu dễ.
- Client có dữ liệu/nhãn kém có thể gửi update kém chất lượng.
- Update nhiễu từ nhiều client có thể làm global model khó cải thiện.

Do đó, trước khi bàn method mới, cần hiểu rõ:

- Dữ liệu local được chọn như thế nào.
- Nhãn local được tạo và kiểm soát chất lượng ra sao.
- Mức độ nhiễu nhãn ảnh hưởng thế nào tới FL.
- Quy mô review của con người có khả thi không.
- Client nào đủ điều kiện gửi update.

---

## 7. Câu hỏi nghiên cứu ban đầu

Các câu hỏi cần được dùng để định hướng nghiên cứu tiếp theo:

1. Trong bài toán face/PPE trên camera edge, dữ liệu local cần đạt chất lượng tối thiểu nào để local update có ích?
2. Khi dữ liệu giữa các site non-IID, việc chỉ train `embedding/head/adapter` có đủ để cải thiện model không?
3. Nhãn sai hoặc nhãn thiếu ảnh hưởng thế nào tới local update và global aggregation?
4. Có thể đánh giá chất lượng client/update trước khi aggregate không?
5. Chi phí gắn nhãn của con người tăng thế nào khi số camera/site tăng?
6. Với cùng ngân sách nhãn, FL có cải thiện tốt hơn local-only không?
7. Global model có làm giảm chất lượng ở một số site đặc thù không?
8. Communication cost của phần update nhẹ có đủ nhỏ để phù hợp edge không?

---

## 8. Các hướng nghiên cứu chính trong áp dụng hệ thống FL:

### A. Tự động gắn nhãn (Auto-labeling) 
Việc thiếu hụt nhãn dữ liệu (label deficiency) tại các thiết bị biên đang là một vấn đề rất lớn của FL. 
*   **Rào cản thực tế:** Hiện tại, các hệ thống như FedVision vẫn phải dựa vào "Crowdsourced Image Annotation" (con người tự dùng chuột khoanh vùng và dán nhãn thủ công). Việc kỳ vọng người dùng cuối hoặc quản trị viên tại các site (camera, điện thoại) tự giác dán nhãn cho dữ liệu riêng tư của họ là cực kỳ khó khăn.

### B. Xử lý Dữ liệu phi đồng nhất (Non-IID Data)
*   **Vấn đề:** Dữ liệu thu thập từ các camera hoặc thiết bị khác nhau sẽ cực kỳ mất cân bằng (ví dụ: camera ở góc phố này thấy nhiều xe tải, camera ở góc khác chỉ thấy xe máy). Điều này làm cho dữ liệu có tính chất Không độc lập và không phân phối đồng nhất (Non-IID). Thuật toán FL cơ bản như FedAvg khi gặp dữ liệu Non-IID sẽ bị giảm độ chính xác nghiêm trọng, thậm chí không hội tụ.

### C. Giảm chi phí và Quá tải truyền thông (Communication Overhead)
*   **Vấn đề:** Môi trường mạng của các thiết bị biên (IoT, camera) thường yếu và không ổn định. Quá trình gửi và nhận hàng triệu tham số mô hình (model parameters) lặp đi lặp lại liên tục giữa máy chủ và client gây tốn băng thông cực lớn.

### D. Giới hạn tài nguyên phần cứng (Resource Constraints)
*   **Vấn đề:** Quá trình huấn luyện (training) tốn bộ nhớ và tính toán hơn rất nhiều so với quá trình suy luận (inference). .



