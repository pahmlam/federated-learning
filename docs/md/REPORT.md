# Federated Learning cho các bài toán Camera/Vision

## 1. Mục tiêu báo cáo

Báo cáo này tổng hợp thông tin nghiên cứu ban đầu về việc áp dụng **Federated Learning (FL)** cho các bài toán camera/vision triển khai ở nhiều địa điểm khác nhau.

Phạm vi hiện tại:

- Tổng quan về FL trong ngữ cảnh camera.
- Xác định vì sao dữ liệu camera nhiều địa điểm là bối cảnh đáng nghiên cứu ứng dụng FL.
- Liệt kê các nhóm use case vision cần khảo sát.
- Tổng hợp các nhận xét thực tế đã rút ra từ tài liệu nghiên cứu hiện có.

## 2. Tóm tắt về Federated Learning trong bài toán Camera AI

Federated Learning là cơ chế huấn luyện trong đó nhiều client cùng tham gia cải thiện mô hình mà không cần gửi dữ liệu gốc về server trung tâm.

Trong hệ camera, dữ liệu gốc thường là ảnh/video từ các camera hoặc cụm camera. Thay vì gom dữ liệu thô về trung tâm, hệ thống FL có thể vận hành theo luồng:

```text
Server trung tâm giữ global model
        ↓
Gửi model xuống các client
        ↓
Client huấn luyện/fine-tune bằng dữ liệu local
        ↓
Client gửi model update/trọng số/metric về server
        ↓
Server aggregate các update
        ↓
Tạo global model mới cho vòng tiếp theo
```

Trong triển khai camera thực tế, client không nhất thiết là chính camera. Client có thể là:

- Camera AI nếu thiết bị đủ năng lực tính toán.
- Edge box gần camera.
- NVR hoặc server local tại địa điểm triển khai.
- Một cụm camera trong cùng nhà máy, kho, công trường hoặc khu vực.
- Một server đại diện cho một site/khách hàng.

Điểm cần phân biệt:

- **Inference** vẫn có thể chạy như hệ thống camera AI thông thường.
- **Training/fine-tuning** mới là phần được tổ chức theo kiểu federated.
- Giai đoạn nghiên cứu có thể mô phỏng nhiều client trên một máy trước khi xét tới edge device thật.

## 3. Vì sao Camera nhiều địa điểm là bối cảnh phù hợp để nghiên cứu áp dụng FL

Dữ liệu camera ở nhiều địa điểm thường có tính **non-IID** rất rõ. Các client cùng làm một task nhưng phân phối dữ liệu local khác nhau.

Các nguồn biến thiên phổ biến:

- Góc đặt camera: nhìn ngang, nhìn từ trên cao, nhìn xa/gần.
- Ánh sáng: trong nhà, ngoài trời, ban đêm, ngược sáng, ánh sáng yếu.
- Background: nhà xưởng, kho, công trường, đường phố, bãi xe.
- Môi trường: bụi, hơi nước, mưa, nắng, khói, phản xạ ánh sáng.
- Chất lượng thiết bị: độ phân giải, FPS, nén hình, hồng ngoại.
- Mật độ đối tượng: đông người, ít người, nhiều occlusion.
- Quy định từng site: đồng phục, vùng cấm, quy trình an toàn, layout vận hành.

FL đáng nghiên cứu trong bối cảnh này vì:

- Dữ liệu phân tán tự nhiên theo camera/site.
- Ảnh/video có thể nhạy cảm, không tiện gom về trung tâm.
- Mỗi site có kiểu lỗi riêng, ví dụ false alarm hoặc miss detection do môi trường.
- Mỗi site có thể có ít dữ liệu, nhưng nhiều site cộng lại tạo nguồn tri thức lớn hơn.
- Video có dung lượng lớn, nên cần cân nhắc giữa truyền dữ liệu thô và truyền model update.

Các điểm trên chỉ cho thấy bối cảnh phù hợp để nghiên cứu FL, chưa đủ để kết luận FL luôn tốt hơn centralized hoặc local-only. Việc này cần được kiểm chứng bằng thực nghiệm.

## 4. Những vấn đề cần giải quyết khi làm việc với công nghệ FL

Khi áp dụng FL cho hệ camera/vision, vấn đề không chỉ là thay đổi cách huấn luyện từ centralized sang federated. Hệ thống cần xử lý đồng thời các vấn đề về dữ liệu, thiết bị, truyền thông, bảo mật và đánh giá.

| Vấn đề | Biểu hiện trong hệ camera/vision | Tác động | Hướng cần xem xét khi nghiên cứu |
| --- | --- | --- | --- |
| Data heterogeneity / Non-IID | Mỗi site có góc camera, ánh sáng, background, tỷ lệ lớp, chất lượng dữ liệu và số lượng mẫu khác nhau | Local update giữa các client có thể lệch hướng, làm global model hội tụ chậm hoặc giảm chất lượng trên một số site | Phân tích feature skew, label skew, quantity skew, quality skew; đánh giá riêng từng site thay vì chỉ dùng trung bình toàn cục |
| System heterogeneity | Camera AI, edge box, NVR hoặc server local có CPU/GPU/RAM và băng thông khác nhau | Thiết bị yếu có thể thành straggler; một số client chỉ đủ inference, không đủ training | Xác định client thực tế là camera, cụm camera, edge box hay server local; cân nhắc partial training hoặc train head/embedding |
| Client availability | Site mất kết nối, edge device quá tải do inference, client không hoàn thành local training đúng thời gian | Update bị thiếu hoặc thiên lệch về nhóm client ổn định; round đồng bộ có thể bị chậm | Nghiên cứu client selection, scheduling, cơ chế xử lý client dropout hoặc asynchronous FL nếu cần |
| Communication cost và scalability | Mỗi round cần gửi model xuống client và gửi update về server; model vision có thể lớn | Chi phí truyền thông tăng theo số client, số round và kích thước model | Đo kích thước update, số round, thời gian truyền; xem xét compression, quantization, partial update hoặc chỉ train một phần model |
| Privacy và security | FL không gửi ảnh/video thô, nhưng model update vẫn có thể chứa rủi ro rò rỉ thông tin | Không thể coi FL là bảo mật tuyệt đối; server trung tâm vẫn là điểm cần bảo vệ | Ghi nhận privacy là động lực dùng FL; secure aggregation, differential privacy và mã hóa kênh truyền là hướng nghiên cứu riêng nếu triển khai |
| Model và task heterogeneity | Các site có thể khác nhãn, khác quy định vận hành, khác pipeline giữa classification/detection/temporal behavior | FedAvg thông thường khó áp dụng nếu model hoặc label space không đồng nhất | Xác định task và label space chung; cân nhắc personalized FL hoặc chỉ chia sẻ một phần model nếu các site khác biệt lớn |
| Evaluation và baseline | Accuracy chung không phản ánh đủ lỗi vận hành, đặc biệt với bài toán cảnh báo hiếm positive | Có thể đánh giá sai hiệu quả FL nếu không so sánh với local-only/centralized và không đo lỗi theo site | So sánh local-only, federated, centralized nếu có thể; dùng recall, false alarm rate, false negative rate, performance theo site và communication cost |

## 5. Các nhóm Use Case Vision cần khảo sát

Các nhóm use case hiện cần khảo sát gồm:

| Nhóm use case | Ví dụ bài toán | Lý do cần khảo sát |
| --- | --- | --- |
| Người | phát hiện người, đếm người, xâm nhập vùng cấm,... | dữ liệu nhạy cảm, biến thiên mạnh theo camera/site |
| Phương tiện | phát hiện xe, phân loại xe, đếm xe, vi phạm làn/vùng,... | biến thiên theo đường, bãi xe, cổng, góc camera |
| Hành vi | ngã, đánh nhau, bất thường,... | phụ thuộc bối cảnh, cần dữ liệu video/temporal |
| Cháy/lửa/khói | phát hiện khói, lửa, cảnh báo cháy sớm | biến thiên môi trường và false alarm theo site |
| Quần áo/PPE | đồng phục, mũ bảo hộ, áo phản quang, khẩu trang | liên quan người, chịu ảnh hưởng ánh sáng/góc nhìn/occlusion |


## 6. Phân tích chi tiết use case

### 6.1. Use Case: Cháy/Lửa/Khói

#### 6.1.1. Bối cảnh bài toán

Phát hiện cháy/lửa/khói là nhóm bài toán cảnh báo an toàn, trong đó yêu cầu quan trọng không chỉ là nhận diện đúng sự kiện cháy mà còn phải giảm cảnh báo giả. Dữ liệu của bài toán này có tính biến thiên mạnh theo địa điểm:

- Môi trường trong nhà/ngoài trời khác nhau.
- Ánh sáng, phản xạ, hơi nước, bụi, sương, khói thật và khói giả có thể gây nhiễu.
- Camera hoặc cảm biến ở các site khác nhau có chất lượng và góc quan sát khác nhau.
- Lớp sự kiện cháy thật thường hiếm, trong khi dữ liệu normal hoặc false alarm có thể nhiều hơn.

#### 6.1.2. Cách FL được đặt vào bài toán

Trong hướng fire detection, FL được dùng để khai thác dữ liệu từ nhiều thiết bị hoặc cơ sở phân tán mà không cần tập trung dữ liệu thô về server. Mỗi cơ sở/thiết bị có thể giữ dữ liệu local, huấn luyện mô hình cục bộ, sau đó chỉ gửi tham số hoặc update về server để tổng hợp.

Với bài toán camera nhiều địa điểm, cách hiểu tương ứng là:

- Site/cụm camera/cảm biến = client
- Server trung tâm = nơi tổng hợp update
- Dữ liệu local = ảnh/video/cảm biến tại từng site

Điểm quan trọng là FL không trực tiếp giải quyết toàn bộ bài toán cảnh báo cháy. FL giải quyết phần **học cộng tác từ nhiều nguồn dữ liệu phân tán**. Các thành phần như model thị giác, cảm biến, cách fusion dữ liệu và metric cảnh báo vẫn phải được thiết kế riêng theo use case.

#### 6.1.3. Các vấn đề thực tế rút ra

| Vấn đề | Biểu hiện trong use case cháy/lửa/khói | Liên hệ với FL |
| --- | --- | --- |
| Dữ liệu phân tán | Dữ liệu đến từ nhiều nhà ở, tòa nhà, cơ sở, camera hoặc cảm biến khác nhau | FL cho phép huấn luyện trên dữ liệu local mà không gom dữ liệu thô |
| Quyền riêng tư | Ảnh/video giám sát trong không gian sống hoặc làm việc có tính nhạy cảm | Chỉ gửi update mô hình thay vì gửi ảnh/video |
| Non-IID | Khác loại camera, điều kiện ánh sáng, môi trường, hạ tầng và loại cảm biến giữa các cơ sở | FedAvg truyền thống có thể hội tụ chậm hoặc không ổn định |
| False alarm | Bụi, hơi nước, ánh sáng, phản xạ hoặc hoạt động sinh hoạt có thể giống khói/lửa | Cần dữ liệu đa dạng theo site để mô hình học các trường hợp gây nhiễu |
| Communication overhead | Model update hoặc gradient có thể lớn, đặc biệt khi nhiều client tham gia | Một số hướng như CMFL, federated dropout, gradient selection được dùng để giảm truyền thông |
| Heterogeneous hardware | Thiết bị ở mỗi cơ sở có khả năng tính toán khác nhau | Có thể cần gửi sub-model hoặc giảm phần model cần train |

#### 6.1.4. Vai trò của dữ liệu đa phương thức

Một điểm đáng chú ý trong hướng fire detection là bài toán không nhất thiết chỉ dùng ảnh RGB. Dữ liệu có thể được mở rộng sang nhiều nguồn tín hiệu:

- Camera nhiệt để quan sát biến đổi nhiệt.
- Cảm biến khí, khói hoặc hóa chất để nhận biết dấu hiệu môi trường.
- Mô hình học sâu như CNN/BiLSTM để trích xuất và kết hợp đặc trưng.

Ý nghĩa của multimodal data:

- Giảm phụ thuộc vào một nguồn tín hiệu đơn lẻ.
- Hỗ trợ phân biệt cháy thật với các nguyên nhân gây báo động giả như bụi hoặc hơi nước.
- Tăng độ tin cậy trong bối cảnh môi trường đô thị hoặc tòa nhà có nhiều nguồn nhiễu.

Điểm cần chú ý khi liên hệ với hệ camera hiện tại: nếu chỉ có RGB camera, cần tách rõ phần nào có thể áp dụng trực tiếp, phần nào phụ thuộc vào camera nhiệt hoặc cảm biến bổ sung.

#### 6.1.5. Liên hệ với bài toán camera nhiều địa điểm

Use case cháy/lửa/khói có nhiều điểm phù hợp để nghiên cứu FL:

- Dữ liệu có thể sinh ra từ nhiều site/camera/cảm biến phân tán.
- Điều kiện môi trường khác nhau tạo non-IID rõ.
- Dữ liệu ảnh/video giám sát có yếu tố privacy.
- False alarm là vấn đề thực tế, chịu ảnh hưởng mạnh bởi từng môi trường.
- Communication cost là vấn đề đáng chú ý nếu dùng model vision lớn hoặc nhiều client.

Các điểm cần làm rõ trước khi kết luận tính phù hợp:

- Hệ thống hiện có chỉ dùng RGB camera hay có thêm camera nhiệt/cảm biến?
- Có dữ liệu positive cháy/lửa/khói thật không?
- Có log false alarm theo site không?
- Client thực tế sẽ là camera, cụm camera, edge box hay server local?
- Metric ưu tiên là accuracy, recall, false alarm rate hay thời gian phát hiện?

### 6.2. Use Case: Người

#### 6.2.1. Bối cảnh bài toán

Nhóm người bao gồm các bài toán như phát hiện người, đếm người, phát hiện xâm nhập vùng cấm hoặc theo dõi mật độ người trong một khu vực. Đây là nhóm bài toán phổ biến trong camera giám sát vì người thường là đối tượng trung tâm của các hệ thống an ninh, an toàn và vận hành.

Dữ liệu của nhóm này có yếu tố nhạy cảm cao. Ảnh/video có thể chứa khuôn mặt, dáng người, hành vi cá nhân, bối cảnh sinh hoạt hoặc không gian làm việc. Vì vậy, việc không gom dữ liệu thô về trung tâm là một động lực rõ ràng để nghiên cứu FL.

#### 6.2.2. Cách FL được đặt vào bài toán

Trong bài toán người, client FL có thể là một site, một cụm camera trong cùng khu vực, edge box, NVR hoặc server local. Mỗi client giữ ảnh/video local để huấn luyện hoặc fine-tune model phát hiện người, đếm người hoặc nhận diện xâm nhập. Server trung tâm chỉ nhận update mô hình để tổng hợp thành global model.

Cách đặt bài toán có thể chia thành vài hướng:

- **Person detection:** local client train detector để nhận diện người trong khung hình.
- **People counting:** local client train model đếm người hoặc head hồi quy mật độ.
- **Intrusion detection:** phần model phát hiện người có thể học chung, còn vùng cấm/rule vận hành có thể được cấu hình riêng theo site.
- **Site adaptation:** global model học từ nhiều môi trường, sau đó mỗi site có thể fine-tune thêm để phù hợp với góc camera và layout riêng.

#### 6.2.3. Các vấn đề thực tế rút ra

| Vấn đề | Biểu hiện trong use case người | Liên hệ với FL |
| --- | --- | --- |
| Dữ liệu nhạy cảm | Video có thể chứa khuôn mặt, dáng người, nơi làm việc hoặc sinh hoạt | FL giúp giảm nhu cầu truyền dữ liệu thô về server trung tâm |
| Non-IID theo camera | Góc nhìn, độ cao camera, mật độ người, background và ánh sáng khác nhau giữa site | Local update giữa các client có thể rất khác nhau, đặc biệt giữa cảnh đông người và cảnh vắng |
| Occlusion | Người bị che bởi người khác, máy móc, kệ hàng hoặc vật cản | Cần dữ liệu đa dạng từ nhiều site để model học các tình huống che khuất |
| Site-specific rule | Vùng cấm, khu vực cần đếm, giờ hoạt động và quy định cảnh báo khác nhau | Global model có thể học phần nhận diện người, còn rule/vùng cấm cần giữ riêng theo site |
| Class imbalance | Một số site có rất nhiều frame không có người, một số site luôn đông người | Cần đánh giá theo site để tránh model chỉ tốt trên nhóm site chiếm nhiều dữ liệu |
| Edge workload | Camera/edge device thường phải chạy inference liên tục | Training local cần được đặt lịch hoặc giới hạn tài nguyên để không ảnh hưởng vận hành |

#### 6.2.4. Các nguồn biến thiên giữa site/camera

Các nguồn biến thiên đáng chú ý:

- Góc nhìn từ trên cao, nhìn ngang, nhìn xa hoặc nhìn gần.
- Camera trong nhà, ngoài trời, hành lang, sảnh, nhà xưởng hoặc khu vực công cộng.
- Ánh sáng ngày/đêm, ngược sáng, ánh sáng yếu, hồng ngoại hoặc camera bị nén mạnh.
- Mật độ người khác nhau: cảnh vắng, cảnh đông, hàng chờ, đám đông di chuyển.
- Occlusion do người che nhau, máy móc, kệ hàng hoặc vật cản.
- Đồng phục, màu quần áo, mũ, balo hoặc đồ bảo hộ khác nhau theo site.
- Vùng cấm và layout vận hành khác nhau giữa nhà máy, kho, văn phòng, công trường.

#### 6.2.5. Metric và baseline cần quan tâm

Metric phụ thuộc vào task cụ thể:

- Với **person detection**: mAP, precision, recall, false positive, miss detection.
- Với **people counting**: MAE/RMSE sai số đếm, sai số theo khung giờ cao điểm/thấp điểm.
- Với **intrusion detection**: false alarm rate, false negative rate, recall với sự kiện xâm nhập, thời gian phát hiện.
- Với mọi task: cần báo cáo performance theo từng site, không chỉ trung bình toàn hệ thống.

Baseline cần có nếu làm thực nghiệm:

- Local-only: mỗi site tự train/fine-tune model riêng.
- Federated: nhiều site cùng train qua FL.
- Centralized: chỉ dùng nếu có thể gom dữ liệu hoặc mô phỏng tập trung để so sánh trần hiệu năng.

#### 6.2.6. Điểm khó khi áp dụng FL

Các điểm khó chính:

- Nếu mục tiêu là xâm nhập vùng cấm, phần vùng cấm thường là cấu hình riêng của site, không phải tri thức chung để aggregate.
- Nếu mỗi site có camera quá khác nhau, global model có thể tốt trung bình nhưng không tốt cho site đặc thù.
- Dữ liệu người dễ nhạy cảm, nhưng model update vẫn có rủi ro rò rỉ thông tin nếu không có kênh truyền và cơ chế bảo vệ phù hợp.
- Các site có số lượng dữ liệu rất khác nhau có thể làm aggregation thiên về site lớn.
- Việc annotation người/vùng cấm/sự kiện xâm nhập cần thống nhất tiêu chuẩn để kết quả so sánh có ý nghĩa.

#### 6.2.7. Liên hệ với bài toán camera nhiều địa điểm

Use case người có nhiều điểm phù hợp để nghiên cứu FL:

- Dữ liệu phân tán tự nhiên theo site/camera.
- Dữ liệu có yếu tố privacy rõ.
- Non-IID mạnh do góc nhìn, mật độ người, background và rule vận hành.
- Có thể bắt đầu từ task đơn giản như person detection trước khi mở rộng sang counting hoặc intrusion.

Các điểm cần làm rõ trước khi kết luận tính phù hợp:

- Task chính là detection, counting hay intrusion?
- Dữ liệu hiện có theo site hay chỉ là dữ liệu trộn?
- Có annotation vùng cấm hoặc event xâm nhập không?
- Client thực tế là camera đơn, cụm camera, edge box hay server local?
- Metric ưu tiên là recall người, sai số đếm, false alarm xâm nhập hay latency phát hiện?

### 6.3. Use Case: Phương Tiện

#### 6.3.1. Bối cảnh bài toán

Nhóm phương tiện bao gồm phát hiện xe, phân loại xe, đếm xe, nhận diện luồng ra/vào, giám sát bãi xe, cổng nhà máy hoặc một số vi phạm vùng/làn.

Dữ liệu phương tiện thường ít nhạy cảm hơn dữ liệu người, nhưng vẫn có thể chứa biển số, tuyến đường, thói quen di chuyển, mật độ giao thông hoặc thông tin vận hành của khách hàng. Ngoài ra, dữ liệu giữa các địa điểm có domain shift rất mạnh vì mỗi camera nhìn một kiểu đường, bãi hoặc cổng khác nhau.

#### 6.3.2. Cách FL được đặt vào bài toán

Trong FL, mỗi tuyến đường, bãi xe, cổng ra vào hoặc cụm camera giao thông có thể là một client. Client huấn luyện/fine-tune local model theo bối cảnh riêng, còn server tổng hợp update để tạo model học được nhiều kiểu môi trường hơn.

Cách đặt bài toán có thể gồm:

- **Vehicle detection:** phát hiện phương tiện trong từng frame.
- **Vehicle classification:** phân loại xe máy, ô tô, xe tải, xe bus hoặc nhóm phương tiện theo nhu cầu vận hành.
- **Vehicle counting/flow monitoring:** đếm lượt xe, đo lưu lượng ra/vào, thống kê mật độ.
- **Violation/event detection:** phát hiện xe vào vùng cấm, sai làn, dừng đỗ sai vị trí hoặc đi ngược chiều nếu có rule tương ứng.

#### 6.3.3. Các vấn đề thực tế rút ra

| Vấn đề | Biểu hiện trong use case phương tiện | Liên hệ với FL |
| --- | --- | --- |
| Domain shift theo địa điểm | Đường phố, bãi xe, cổng nhà máy và kho vận có góc nhìn, background, mật độ xe rất khác nhau | FL có thể giúp model học từ nhiều bối cảnh mà không cần gom video về trung tâm |
| Thời tiết và thời điểm | Mưa, nắng gắt, ban đêm, đèn pha, bóng đổ hoặc camera hồng ngoại ảnh hưởng mạnh tới ảnh | Dữ liệu từ nhiều site có thể bổ sung các điều kiện hiếm cho nhau |
| Khác biệt loại phương tiện | Một số site nhiều xe máy, site khác nhiều xe tải/container hoặc xe con | Label distribution giữa client có thể lệch mạnh, gây khó cho aggregation |
| Privacy và dữ liệu nhạy cảm | Video có thể chứa biển số, tuyến đường, lịch sử ra/vào | FL giảm nhu cầu chia sẻ dữ liệu thô nhưng vẫn cần bảo vệ update |
| Object scale | Xe ở xa nhỏ, xe gần chiếm gần hết khung hình, xe bị che bởi xe khác | Model cần học nhiều scale và occlusion từ dữ liệu đa site |
| Quy định vận hành | Cách định nghĩa vi phạm hoặc loại xe cần thống kê khác nhau giữa đường, bãi và cổng | Cần thống nhất label/task trước khi huấn luyện FL |

#### 6.3.4. Các nguồn biến thiên giữa site/camera

Các nguồn biến thiên đáng chú ý:

- Camera đặt trên cột cao, cổng ra/vào, tầng hầm, bãi ngoài trời hoặc bên đường.
- Hướng xe đi ngang, đi thẳng vào camera, quay đầu hoặc dừng chờ.
- Mật độ giao thông thay đổi theo giờ cao điểm, ngày thường, cuối tuần hoặc ca sản xuất.
- Loại phương tiện phổ biến khác nhau theo khu vực: xe máy, ô tô, xe tải, container, xe nâng.
- Điều kiện thời tiết và ánh sáng: mưa, nắng, đèn pha, bóng cây, ngược sáng, ban đêm.
- Background khác nhau giữa đường đô thị, bãi xe, kho, trạm kiểm soát, cổng nhà máy.
- Chất lượng camera và nén video ảnh hưởng tới biển số, cạnh xe và vật thể nhỏ.

#### 6.3.5. Metric và baseline cần quan tâm

Metric phụ thuộc task:

- Với **vehicle detection**: mAP, precision, recall theo từng loại xe.
- Với **vehicle classification**: accuracy, macro-F1, confusion matrix giữa các loại xe dễ nhầm.
- Với **counting/flow**: MAE/RMSE sai số đếm, sai số theo khung giờ, sai số theo hướng di chuyển.
- Với **violation detection**: false alarm rate, false negative rate, recall với sự kiện vi phạm, thời gian phát hiện.
- Với mọi task: cần tách kết quả theo site, thời tiết, ngày/đêm và mật độ giao thông.

Baseline cần có nếu làm thực nghiệm:

- Local-only theo từng site để kiểm tra khả năng thích nghi cục bộ.
- Federated để kiểm tra lợi ích học cộng tác.
- Centralized nếu có thể mô phỏng gom dữ liệu để so sánh giới hạn hiệu năng.

#### 6.3.6. Điểm khó khi áp dụng FL

Các điểm khó chính:

- Label space có thể không thống nhất: site A phân loại xe máy/ô tô, site B cần thêm xe tải/container/xe nâng.
- Dữ liệu mỗi site có phân phối rất lệch, ví dụ một bãi xe gần như không có xe tải, trong khi cổng kho lại chủ yếu có xe tải.
- Nếu task liên quan rule vi phạm, rule này có thể phụ thuộc layout từng site và không thể aggregate trực tiếp như một label chung.
- Model detection cho phương tiện có thể lớn, làm communication cost tăng nếu gửi full model update nhiều round.
- Dữ liệu ban đêm, mưa lớn hoặc camera rung có thể ít nhưng lại quyết định chất lượng vận hành.

#### 6.3.7. Liên hệ với bài toán camera nhiều địa điểm

Use case phương tiện phù hợp để nghiên cứu FL ở góc độ domain shift giữa địa điểm:

- Dữ liệu phân tán theo tuyến đường, bãi xe, cổng, kho hoặc khu công nghiệp.
- Mỗi site có loại phương tiện, góc nhìn và mật độ riêng.
- Video có thể lớn và có thông tin nhạy cảm như biển số hoặc lịch sử di chuyển.
- FL có thể được thử nghiệm từ task detection/classification trước khi mở rộng sang counting hoặc vi phạm.

Các điểm cần làm rõ trước khi kết luận tính phù hợp:

- Task chính là detection, classification, counting hay violation?
- Label xe có thống nhất giữa các site không?
- Có cần xử lý biển số hoặc che biển số không?
- Dữ liệu có đủ ngày/đêm, mưa/nắng, giờ cao điểm/thấp điểm không?
- Client thực tế là camera, roadside unit, edge box, NVR hay server local?

### 6.4. Use Case: Hành Vi

#### 6.4.1. Bối cảnh bài toán

Nhóm hành vi bao gồm phát hiện ngã, đánh nhau, tụ tập bất thường, đi vào vùng nguy hiểm, thao tác sai quy trình hoặc các bất thường theo chuỗi thời gian. Đây là nhóm bài toán phức tạp hơn detection/classification ảnh đơn vì thường cần ngữ cảnh nhiều frame, tracking hoặc mô hình temporal.

Dữ liệu hành vi thường nhạy cảm hơn dữ liệu object detection thông thường vì nó mô tả hoạt động của con người trong một không gian cụ thể. Ngoài ra, các hành vi nguy hiểm hoặc bất thường thường hiếm, khó thu thập và khó gán nhãn nhất quán.

#### 6.4.2. Cách FL được đặt vào bài toán

Trong FL, mỗi site có thể giữ video local để huấn luyện phần nhận diện hành vi hoặc temporal classifier. Server chỉ tổng hợp update, còn dữ liệu video nhạy cảm không rời khỏi site.

Cách đặt bài toán có thể gồm:

- **Frame-based pipeline:** dùng detector/tracker để lấy thông tin người rồi phân loại trạng thái từ từng frame hoặc vài frame ngắn.
- **Clip-based classification:** phân loại hành vi trên một đoạn video ngắn.
- **Temporal modeling:** dùng chuỗi đặc trưng theo thời gian để nhận diện hành vi hoặc bất thường.
- **Site adaptation:** mỗi site có thể fine-tune thêm vì hành vi bình thường và bất thường phụ thuộc layout, quy trình và bối cảnh.

#### 6.4.3. Các vấn đề thực tế rút ra

| Vấn đề | Biểu hiện trong use case hành vi | Liên hệ với FL |
| --- | --- | --- |
| Dữ liệu video nhạy cảm | Video ghi lại hoạt động, thói quen, quy trình làm việc hoặc sự cố của con người | FL giảm nhu cầu gửi clip nhạy cảm về trung tâm |
| Positive hiếm | Ngã, đánh nhau, tai nạn hoặc thao tác sai thường ít xuất hiện | Cần chia dữ liệu và metric cẩn thận để không bị accuracy đánh lừa |
| Nhãn khó | Ranh giới bắt đầu/kết thúc hành vi, mức độ nguy hiểm và định nghĩa bất thường có thể không rõ | Các client cần label convention thống nhất trước khi aggregate |
| Phụ thuộc bối cảnh | Một hành động có thể bình thường ở site này nhưng bất thường ở site khác | Có thể cần personalized FL hoặc giữ rule riêng theo site |
| Dữ liệu temporal nặng | Video clip và model temporal tốn tài nguyên hơn ảnh đơn | Cần cân nhắc train head/embedding, sampling frame hoặc partial update |
| Occlusion và tracking lỗi | Người bị che, mất track hoặc hành động diễn ra nhanh | Lỗi ở pipeline trước có thể làm local update nhiễu |

#### 6.4.4. Các nguồn biến thiên giữa site/camera

Các nguồn biến thiên đáng chú ý:

- Layout khu vực: hành lang, nhà xưởng, kho, sảnh, sân ngoài trời, khu vực máy móc.
- Góc camera và vùng quan sát làm thay đổi dáng người, tốc độ chuyển động và mức độ che khuất.
- Quy trình vận hành từng site: hành vi bình thường ở site này có thể là bất thường ở site khác.
- Mật độ người và tương tác giữa nhiều người.
- Độ dài clip, FPS, motion blur, camera rung hoặc mất khung hình.
- Cách gán nhãn: frame-level, clip-level, event-level hoặc chỉ có timestamp cảnh báo.
- Tần suất event positive rất thấp so với video normal.

#### 6.4.5. Metric và baseline cần quan tâm

Metric cần ưu tiên theo hướng cảnh báo:

- Recall với event nguy hiểm hoặc bất thường.
- False alarm rate để đo mức độ gây nhiễu vận hành.
- False negative rate vì bỏ sót sự kiện nguy hiểm thường là lỗi nghiêm trọng.
- F1 hoặc macro-F1 nếu có nhiều lớp hành vi.
- Latency phát hiện, đặc biệt nếu cần cảnh báo real-time.
- Đánh giá theo event/clip/site, không chỉ theo frame.

Baseline cần có nếu làm thực nghiệm:

- Local-only theo site để biết mỗi site tự học được đến đâu.
- Federated để kiểm tra học cộng tác có cải thiện recall hoặc giảm false alarm không.
- Centralized nếu có thể mô phỏng gom dữ liệu, nhưng cần ghi rõ đây chỉ là baseline so sánh.

#### 6.4.6. Điểm khó khi áp dụng FL

Các điểm khó chính:

- Model video/temporal có thể nặng, không phù hợp với camera yếu nếu train trực tiếp trên thiết bị.
- Nhãn hành vi khó thống nhất giữa site, đặc biệt với các khái niệm như bất thường, tụ tập hoặc thao tác sai.
- Dữ liệu positive ít làm local training dễ overfit hoặc không đủ gradient hữu ích.
- Nếu mỗi site có hành vi bình thường khác nhau, một global model duy nhất có thể không phù hợp cho tất cả.
- Cần xác định rõ phần nào học bằng model, phần nào là rule theo site, ví dụ vùng nguy hiểm hoặc thời gian cho phép.

#### 6.4.7. Liên hệ với bài toán camera nhiều địa điểm

Use case hành vi có liên hệ mạnh với FL vì dữ liệu video thường nhạy cảm và phân tán:

- Mỗi site có quy trình, layout và hành vi bình thường riêng.
- Event nguy hiểm hiếm, nên nhiều site cộng lại có thể tạo nguồn học tốt hơn.
- Privacy là động lực rõ vì clip hành vi khó chia sẻ hơn ảnh object thông thường.
- Cần đánh giá theo site để biết global model có làm giảm chất lượng ở site đặc thù không.

Các điểm cần làm rõ trước khi kết luận tính phù hợp:

- Hành vi cụ thể cần nghiên cứu là gì?
- Dữ liệu có clip positive không, hay chỉ có log/cảnh báo?
- Annotation ở mức frame, clip hay event?
- Có yêu cầu real-time không?
- Model sẽ dùng ảnh đơn, tracking, skeleton/keypoint hay video sequence?
- Client có đủ tài nguyên train local model temporal không?

### 6.5. Use Case: Quần Áo/PPE

#### 6.5.1. Bối cảnh bài toán

Nhóm quần áo/PPE bao gồm phát hiện mũ bảo hộ, áo phản quang, khẩu trang, đồng phục, găng tay hoặc các yêu cầu an toàn lao động theo từng môi trường. Bài toán thường xuất hiện ở công trường, nhà máy, kho, khu vực sản xuất hoặc khu vực có quy định bảo hộ.

Đặc điểm quan trọng của nhóm này là quy định an toàn có thể thay đổi theo site. Một nhà máy yêu cầu mũ và áo phản quang, một kho có thể cần đồng phục, một khu vực khác cần khẩu trang hoặc găng tay. Vì vậy, bài toán không chỉ là phát hiện vật thể, mà còn là kiểm tra trạng thái tuân thủ trong một bối cảnh cụ thể.

#### 6.5.2. Cách FL được đặt vào bài toán

Trong FL, mỗi nhà máy, công trường, kho hoặc khu vực sản xuất có thể là một client. Client huấn luyện local model theo đồng phục, màu PPE, góc camera và quy định riêng của site, sau đó gửi update để server tổng hợp.

Cách đặt bài toán có thể gồm:

- **PPE object detection:** phát hiện mũ, áo phản quang, khẩu trang, găng tay hoặc các vật thể bảo hộ.
- **Person-PPE association:** xác định PPE có thuộc đúng người trong khung hình không.
- **Compliance classification:** phân loại trạng thái người lao động là đủ/thiếu PPE theo rule của site.
- **Site adaptation:** global model học đặc trưng chung của người/PPE, còn rule và ngưỡng cảnh báo có thể giữ riêng theo site.

#### 6.5.3. Các vấn đề thực tế rút ra

| Vấn đề | Biểu hiện trong use case quần áo/PPE | Liên hệ với FL |
| --- | --- | --- |
| Quy định khác nhau theo site | Loại PPE bắt buộc, màu đồng phục, vùng cần kiểm tra và mức cảnh báo khác nhau | Cần tách phần tri thức chung của model và rule riêng của site |
| Non-IID mạnh | Mỗi site có đồng phục, màu mũ, ánh sáng, background và camera khác nhau | FL có thể giúp model học đa dạng PPE mà không gom ảnh công nhân về trung tâm |
| Vật thể nhỏ | Mũ, khẩu trang, găng tay hoặc dây phản quang có thể rất nhỏ trong ảnh | Model dễ miss khi người ở xa hoặc camera đặt cao |
| Occlusion | PPE bị che bởi máy móc, hàng hóa, người khác hoặc tư thế làm việc | Cần dữ liệu nhiều bối cảnh để giảm false alarm/miss detection |
| Class imbalance | Case vi phạm thường ít hơn case tuân thủ | Accuracy dễ gây hiểu nhầm, cần metric tập trung vào lỗi vi phạm |
| Privacy và bảo mật vận hành | Ảnh công nhân có thể lộ người, quy trình sản xuất hoặc layout nhà máy | FL giảm chia sẻ ảnh/video thô nhưng vẫn cần bảo vệ update |

#### 6.5.4. Các nguồn biến thiên giữa site/camera

Các nguồn biến thiên đáng chú ý:

- Màu sắc và kiểu dáng đồng phục, mũ, áo phản quang, khẩu trang, găng tay.
- Camera nhìn từ trên cao, nhìn xiên, nhìn xa hoặc đặt trong khu vực thiếu sáng.
- Khoảng cách người tới camera làm PPE trở thành object nhỏ.
- Tư thế làm việc: cúi người, quay lưng, đội mũ lệch, áo bị che bởi vật dụng.
- Occlusion do máy móc, kệ hàng, xe nâng, hàng hóa hoặc người khác.
- Ánh sáng nhà xưởng, bụi, phản xạ kim loại, đèn công nghiệp hoặc camera bị rung.
- Quy định PPE khác nhau giữa khu vực sản xuất, kho, công trường và văn phòng kỹ thuật.

#### 6.5.5. Metric và baseline cần quan tâm

Metric cần ưu tiên theo hướng cảnh báo an toàn:

- Recall với các lỗi thiếu PPE quan trọng, ví dụ không đội mũ hoặc không mặc áo phản quang.
- Precision của cảnh báo vi phạm để tránh gây quá nhiều false alarm.
- False negative rate vì bỏ sót vi phạm an toàn là rủi ro lớn.
- mAP theo từng loại PPE nếu task là detection.
- F1 hoặc macro-F1 nếu task là classification trạng thái tuân thủ.
- Performance theo khoảng cách, góc camera, điều kiện ánh sáng và từng site.

Baseline cần có nếu làm thực nghiệm:

- Local-only để kiểm tra model riêng của từng site có đủ tốt không.
- Federated để kiểm tra học từ nhiều site có giúp nhận diện PPE đa dạng hơn không.
- Centralized nếu có thể mô phỏng gom dữ liệu để so sánh.

#### 6.5.6. Điểm khó khi áp dụng FL

Các điểm khó chính:

- Label space có thể không thống nhất vì mỗi site yêu cầu loại PPE khác nhau.
- Nếu chỉ train global model chung, model có thể học tốt object phổ biến nhưng kém với PPE đặc thù của một site.
- Cần xác định rõ output là bounding box PPE, trạng thái từng người hay cảnh báo vi phạm theo rule.
- Case vi phạm có thể ít, làm local training khó học đủ tín hiệu.
- Camera xa hoặc góc cao làm PPE rất nhỏ, khiến update từ các site camera khó có độ nhiễu cao hơn.

#### 6.5.7. Liên hệ với bài toán camera nhiều địa điểm

Use case quần áo/PPE phù hợp để nghiên cứu FL ở góc độ khác biệt quy định và môi trường giữa site:

- Dữ liệu phân tán theo nhà máy, kho, công trường hoặc khu sản xuất.
- Ảnh công nhân và quy trình sản xuất có yếu tố privacy.
- Non-IID đến từ đồng phục, màu PPE, background, ánh sáng và rule vận hành.
- Có thể bắt đầu từ detection PPE phổ biến trước khi mở rộng sang compliance theo rule từng site.

Các điểm cần làm rõ trước khi kết luận tính phù hợp:

- Các loại PPE cần nhận diện là gì?
- Label có thống nhất giữa site không?
- Task là detection từng vật thể, association người-PPE hay classification trạng thái an toàn?
- Có đủ case vi phạm không?
- Có cần rule riêng theo từng khu vực/site không?
- Client thực tế là camera, cụm camera, edge box hay server local?

## 7. Nhận xét kỹ thuật tạm thời

### 7.1. Non-IID là vấn đề trung tâm

Dữ liệu camera giữa các địa điểm khác nhau thường không cùng phân phối. Khác biệt có thể đến từ:

- Góc đặt, độ cao, khoảng cách và chất lượng camera.
- Ánh sáng, thời tiết, background, phản xạ, bụi, hơi nước hoặc môi trường nhà xưởng.
- Mật độ người/phương tiện, mức độ occlusion và thời điểm trong ngày.
- Quy định vận hành từng site như vùng cấm, loại PPE, đồng phục hoặc quy trình an toàn.
- Tần suất xuất hiện lớp positive thấp trong các bài toán cảnh báo, hành vi bất thường hoặc vi phạm an toàn.

Điều này có thể làm local update giữa các client lệch nhau, gây khó cho aggregation.

### 7.2. Communication Cost cần được đo

Communication overhead là điểm cần đo khi nghiên cứu FL cho camera:

- Video thô rất nặng.
- Model update cũng có thể lớn nếu model lớn.
- Nếu dùng backbone lớn, cần cân nhắc partial update, compression hoặc chỉ train một phần model.

### 7.3. Metric không nên chỉ dùng accuracy

Với các bài toán cảnh báo, an toàn hoặc sự kiện hiếm, accuracy có thể gây hiểu nhầm do dữ liệu positive ít. Cần quan tâm:

- Recall với lớp sự kiện/vi phạm quan trọng.
- Precision của cảnh báo.
- False alarm rate.
- False negative rate.
- mAP hoặc F1 nếu bài toán là detection/classification.
- Thời gian phát hiện nếu tài liệu hoặc thực nghiệm có xét yếu tố thời gian.

### 7.4. Client thực tế nên được định nghĩa theo site hoặc edge node

Trong bối cảnh camera, định nghĩa client cần phản ánh cấu trúc triển khai thực tế. Các lựa chọn hợp lý:

- Một site là một client.
- Một cụm camera cùng môi trường là một client.
- Một edge box/NVR/server local là một client.

Không nên mặc định camera đơn lẻ luôn là client nếu camera không đủ tài nguyên để train.
