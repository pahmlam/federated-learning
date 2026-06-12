
# I. Federated Learning 

## I.1. Centralized Learning

Mô hình **Centralized Learning** (Học tập trung) là phương pháp truyền thống trong việc xây dựng các mô hình học máy, nơi toàn bộ dữ liệu được thu thập và xử lý tại một thực thể trung tâm.

### Quy trình hoạt động của Centralized Learning

Quy trình này được thực hiện thông qua hai luồng chính:

1.  **Luồng dữ liệu tổng quát:**
    *   Nhiều khối dữ liệu nhỏ từ các nguồn khác nhau được gom lại thành một khối dữ liệu duy nhất gọi là **Global data**.
    *   Khối **Global data** này sau đó được sử dụng để huấn luyện một **Global Model**.

2.  **Mô hình Client-Server:**
    *   Hệ thống bao gồm nhiều người dùng (ví dụ: từ User 1 đến User 5) sử dụng các thiết bị đầu cuối khác nhau như laptop, smartphone và tablet.
    *   Mỗi thiết bị sẽ gửi trực tiếp dữ liệu cục bộ của mình (**data**) thông qua các kết nối mạng hướng về một máy chủ trung tâm (**Server model**), thường được biểu diễn dưới dạng điện toán đám mây.
    *   Quá trình này được gọi là cập nhật dữ liệu cục bộ (**Update Local Data**) lên máy chủ để phục vụ cho việc huấn luyện mô hình.

### Các vấn đề tồn tại (Problems)

Mặc dù phổ biến, mô hình **Centralized Learning** đối mặt với nhiều thách thức và hạn chế nghiêm trọng:

-   **Vi phạm quyền riêng tư (Privacy violation):** Dữ liệu cá nhân và nhạy cảm buộc phải rời khỏi thiết bị của người dùng để truyền về máy chủ, gây ra rủi ro lộ lọt thông tin.
-   **Tuân thủ quy định (Regulatory compliance):** Việc tập trung dữ liệu gây khó khăn trong việc đáp ứng các quy định pháp lý nghiêm ngặt về bảo vệ dữ liệu (như GDPR).
-   **Chi phí truyền thông (Communication overhead):** Việc truyền tải một lượng lớn dữ liệu thô từ hàng triệu thiết bị về máy chủ trung tâm gây tốn kém băng thông và tài nguyên mạng.
-   **Tấn công dữ liệu đang truyền (Data-in-transit attack):** Dữ liệu có nguy cơ bị đánh chặn hoặc tấn công trong quá trình di chuyển từ thiết bị người dùng đến máy chủ.
-   **Điểm lỗi duy nhất (Single point of failure):** Toàn bộ hệ thống phụ thuộc hoàn toàn vào máy chủ trung tâm. Nếu máy chủ này gặp sự cố hoặc bị tấn công, toàn bộ quá trình huấn luyện và dịch vụ sẽ bị đình trệ.

Những hạn chế này là động lực chính dẫn đến sự ra đời của **Federated Learning**, một giải pháp thay thế giúp huấn luyện mô hình mà không cần di chuyển dữ liệu gốc ra khỏi thiết bị.

## I.2. Federated Learning là gì?

Federated Learning (FL) là một phương pháp huấn luyện máy học phi tập trung (decentralized). Triết lý cốt lõi của phương pháp này được tóm gọn qua câu nói: 
> "Thay vì đưa dữ liệu đến model, FL đưa model đến dữ liệu."

### Đặc điểm

- **Huấn luyện tại chỗ:** Các **client** thực hiện huấn luyện **model** tại local ngay trên thiết bị của mình mà không chia sẻ dữ liệu thô ra bên ngoài.
- **Trao đổi tham số:** Chỉ có các bản cập nhật **model** hoặc các tham số được trao đổi với **server**.
- **Mục tiêu:** Học từ nguồn dữ liệu phân tán trong khi vẫn bảo vệ quyền riêng tư (**privacy**) cho người dùng.

### Quy trình hoạt động và cấu trúc hệ thống

Quy trình Federated Learning bao gồm sự phối hợp giữa các thiết bị người dùng và hệ thống máy chủ:

1.  **Thành phần tại Client (Local Data & Local Models):**
    - Hệ thống bao gồm nhiều người dùng (ví dụ: User 1 đến User 5) sử dụng các thiết bị khác nhau như laptop, smartphone, hoặc tablet.
    - Mỗi người dùng sở hữu cơ sở dữ liệu riêng (**Local Data**) và một **Local Model**.

2.  **Luồng xử lý tại Client:**
    - Dữ liệu từ thiết bị được đưa vào các **Local Models** để tiến hành huấn luyện ngay trên thiết bị của người dùng. Quá trình này đảm bảo dữ liệu cá nhân không bao giờ rời khỏi thiết bị.

3.  **Trao đổi tham số (Exchange Model Parameters):**
    - Sau khi huấn luyện cục bộ, các thiết bị sẽ gửi các tham số **model** đã học được (như **weights** hoặc **gradients**) lên **server** trên đám mây thay vì gửi dữ liệu thô.

4.  **Xử lý tại Server:**
    - Tại **server**, một khối chức năng thực hiện nhiệm vụ **Aggregating and updating global model** (Tổng hợp và cập nhật **model** toàn cục).
    - Kết quả của quá trình tổng hợp này là tạo ra một **Global model** hoàn chỉnh, mang tri thức từ tất cả các **client** tham gia.

### Ý nghĩa của Federated Learning

Khác với phương pháp học tập trung (Centralized Learning) truyền thống, Federated Learning cho phép tối ưu hóa việc học từ nguồn dữ liệu lớn và đa dạng mà vẫn đảm bảo tính bảo mật. Việc giữ dữ liệu tại thiết bị cá nhân và chỉ chia sẻ các bản cập nhật trọng số giúp bảo vệ thông tin nhạy cảm của người dùng một cách hiệu quả.


## I.3. Standard Workflow

Quy trình chuẩn của Federated Learning được thực hiện thông qua các vòng truyền thông (One communication round), còn được gọi là Global round hoặc Iteration.

### Các khái niệm cơ bản về mô hình
- **Global model**: Mô hình dùng chung được duy trì bởi server.
- **Local model**: Mô hình được cập nhật tại phía client trong một vòng lặp (round).
- **Epochs vs steps**: Các **epoch** và **step** diễn ra cục bộ (locally) bên trong một vòng truyền thông duy nhất.

### Quy trình chi tiết của một vòng truyền thông
Hoạt động của một vòng lặp trong Federated Learning diễn ra theo các bước tuần tự sau:

1.  **Khởi tạo (Initialize global model)**: Thực hiện khởi tạo **global model** tại vòng đầu tiên.
2.  **Phân phối mô hình**: Server nắm giữ **global model** $w^t$ tại thời điểm $t$ và gửi bản sao của mô hình này xuống các client được lựa chọn.
3.  **Huấn luyện cục bộ (Local training)**:
    - Các client được chọn (ví dụ: Client 1 và Client 2) sẽ nhận bản sao cục bộ của **global model** là $w_{i}^{t}$.
    - Quá trình huấn luyện diễn ra trên dữ liệu riêng tư (**Private local dataset**) của từng client.
    - Việc huấn luyện bao gồm nhiều **local epochs** (E) và nhiều **gradient steps**.
4.  **Tải lên bản cập nhật (Upload local updates)**: Sau khi hoàn tất huấn luyện cục bộ, các client gửi các bản cập nhật hoặc tham số mô hình mới $w_i^{t+1}$ lên server.
5.  **Tổng hợp (Aggregation)**: Server thực hiện tổng hợp các tham số nhận được từ Client 1 và Client 2 
6.  **Cập nhật Global Model**: Kết quả của quá trình tổng hợp sẽ tạo ra mô hình Global mới $w^{t+1}$ cho vòng tiếp theo.

### Trạng thái của Client
Trong mỗi vòng truyền thông, không phải tất cả các client đều tham gia vào quá trình huấn luyện:
- **Client đang hoạt động (Client 1 & 2)**: Thực hiện nhận mô hình, huấn luyện trên dữ liệu riêng và gửi bản cập nhật về server.
- **Client không tham gia (Client 3)**: Có thể ở trạng thái không được chọn trong vòng này (**Not selected this round**).

### Nguyên tắc cốt lõi
Điểm mấu chốt trong quy trình này là dữ liệu (**Private local dataset**) luôn được lưu giữ an toàn tại phía client. Chỉ có các tham số mô hình hoặc bản cập nhật (**weight**) mới được truyền tải qua mạng về phía server để thực hiện **Aggregation**.


## I.4. Federated Averaging (FedAvg)

### Mục tiêu: 

Nội dung này dựa trên nghiên cứu của *Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera y Arcas*: Communication-Efficient Learning of Deep Networks from Decentralized Data, được công bố trong *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, PMLR 54:1273-1282, 2017*.

#### Tổng quan 
Các thiết bị di động hiện đại có quyền truy cập vào lượng dữ liệu phong phú phù hợp để huấn luyện các **model**, từ đó cải thiện đáng kể trải nghiệm người dùng (ví dụ: **language model** giúp nhận dạng giọng nói và nhập văn bản tốt hơn, **image model** tự động chọn ảnh đẹp). Tuy nhiên, dữ liệu này thường nhạy cảm về quyền riêng tư hoặc có dung lượng quá lớn, gây khó khăn cho việc gửi về trung tâm dữ liệu để huấn luyện theo phương pháp thông thường.

Giải pháp được đề xuất là **Federated Learning**:
- Dữ liệu huấn luyện được để phân tán trên các thiết bị di động.
- Học một **model** dùng chung bằng cách tổng hợp các bản cập nhật được tính toán cục bộ.
- Phương pháp thực tế này dựa trên việc tính trung bình **model** lặp đi lặp lại.
- **Đánh giá thực nghiệm:** Thử nghiệm trên 5 kiến trúc **model** và 4 tập dữ liệu cho thấy phương pháp này mạnh mẽ đối với các phân phối dữ liệu không cân bằng và **non-IID** (đây là đặc điểm cốt lõi của bối cảnh dữ liệu phân tán).
- **Hiệu quả truyền thông:** Vì chi phí truyền thông là ràng buộc chính, thuật toán này giúp giảm số lượng vòng truyền thông cần thiết từ 10-100 lần so với **SGD** (stochastic **gradient** descent) đồng bộ.

### Cơ sở lý thuyết toán học

Thuật toán FedAvg hướng tới việc tối ưu hóa một **model** chung (**Global Model**) bằng cách tối thiểu hóa hàm **loss** trung bình có trọng số từ tất cả các **client** tham gia.

#### 1. Hàm mục tiêu toàn cục (Global Objective Function)
Server cần tối thiểu hóa hàm $F(\Theta)$, được xác định bằng tổng trọng số của các hàm mục tiêu cục bộ:
$$\min_{\Theta} F(\Theta) = \sum_{k=1}^{K} \frac{n_k}{N} F_k(\Theta)$$

#### 2. Hàm mục tiêu cục bộ (Local Objective Function)
Tại mỗi **client** $k$, hàm mục tiêu được tính dựa trên tập dữ liệu riêng của nó:
$$F_k(\Theta) = \frac{1}{n_k} \sum_{(x,y) \in \mathcal{D}_k} \mathcal{L}(\Theta; x, y)$$

#### 3. Các ký hiệu toán học
- $\theta$: **model**.
- $F(\theta)$: **Global objective** (mục tiêu toàn cục mà server cần tối thiểu hóa).
- $F_k(\theta)$: **Local objective** (mục tiêu cục bộ tại mỗi thiết bị).
- $K$: Tổng số lượng **client** tham gia vào hệ thống.
- $\mathcal{D}_k$: Tập dữ liệu cục bộ được lưu trữ trên **client** $k$.
- $n_k = |\mathcal{D}_k|$: Số lượng mẫu dữ liệu cục bộ trên **client** $k$.
- $N = \sum_{k=1}^{K} n_k$: Tổng số lượng mẫu dữ liệu trên tất cả các **client** cộng lại.

**Lưu ý:** Trọng số của mỗi **client** trong hàm mục tiêu toàn cục được xác định bởi tỷ lệ số lượng mẫu dữ liệu cục bộ ($n_k$) trên tổng số mẫu của toàn hệ thống ($N$). Điều này đảm bảo các **client** có nhiều dữ liệu hơn sẽ đóng góp đáng kể hơn vào quá trình cập nhật **model** chung.


## I.5. Server Procedure

Phần này mô tả thuật toán tổng quát để vận hành một hệ thống Federated Learning. Đây là khung sườn cơ bản của các thuật toán như FedAvg, thực hiện thông qua các vòng giao tiếp (communication rounds) lặp đi lặp lại giữa Server và các client. Quy trình bao gồm việc khởi tạo **model** global ban đầu, sau đó thực hiện lặp lại việc chọn nhóm client, huấn luyện local và tổng hợp kết quả.

### Thuật toán 

*Input*: $\mathcal{C}$, $T$, $\rho$ 

*Output*: $\theta^{(T)}$

Step 1 — Initialize: $\theta^{(0)} \leftarrow$ random or pretrained

for $t = 0$ to $T - 1$ do

    Step 2 — Select: identify available clients $\mathcal{A}_t$; sample $\mathcal{S}_t \subseteq \mathcal{A}_t$ 

    Step 3 — Broadcast: send model $\theta^{(t)}$ to every client $k \in \mathcal{S}_t$

    Step 4 — Local Train: each client $k \in \mathcal{S}_t$ updates $\theta_k^{(t+1)} \leftarrow \text{CLIENTUPDATE}(k, \theta^{(t)})$

    Step 5 — Aggregate: $\theta^{(t+1)} \leftarrow \text{AGGREGATE}(\{\theta_k^{(t+1)}, n_k\}_{k \in \mathcal{S}_t})$ 

end for

return $\theta^{(T)}$


### Các bước thực hiện trong quy trình

Quy trình vận hành tại Server được chia thành các giai đoạn cụ thể như sau:

1.  **Thiết lập đầu vào và đầu ra:**
    -   **Input:** Hệ thống nhận vào danh sách các client $\mathcal{C}$, tổng số vòng lặp $T$, và tỷ lệ tham gia $\rho$ của các client trong mỗi vòng.
    -   **Output:** Kết quả cuối cùng là **model** global $\theta^{(T)}$ sau khi đã trải qua $T$ vòng huấn luyện.

2.  **Bước 1 — Khởi tạo (Initialization):**
    -   Thiết lập giá trị ban đầu cho **weight** và **bias** của **model** ($\theta^{(0)}$).
    -   Giá trị này có thể được khởi tạo ngẫu nhiên hoặc sử dụng một **model** đã được huấn luyện trước (pretrained).

3.  **Vòng lặp giao tiếp (Communication Rounds):**
    Thực hiện lặp từ vòng $t = 0$ đến $T-1$. Trong mỗi vòng, Server thực hiện các bước:
    -   **Bước 2 — Chọn (Selection):** Server xác định danh sách các client đang ở trạng thái sẵn sàng $\mathcal{A}_t$. Từ đó, một tập con các client $\mathcal{S}_t$ được lấy mẫu để tham gia vào vòng huấn luyện hiện tại (chi tiết về cơ chế chọn sẽ được giải thích ở phần sau).
    -   **Bước 3 — Phát sóng (Broadcast):** Server gửi thông số **model** hiện tại $\theta^{(t)}$ tới tất cả các client $k$ thuộc tập hợp đã chọn $\mathcal{S}_t$.
    -   **Bước 4 — Huấn luyện tại chỗ (Local Train):** Mỗi client $k$ nhận **model** và tự thực hiện quá trình cập nhật cục bộ dựa trên dữ liệu riêng của mình thông qua hàm `CLIENTUPDATE`. Kết quả thu được là các thông số **model** mới $\theta_k^{(t+1)}$ (chi tiết về quá trình này được trình bày ở phần sau).
    -   **Bước 5 — Tổng hợp (Aggregate):** Server thu thập các bản cập nhật $\theta_k^{(t+1)}$ cùng với kích thước dữ liệu tương ứng $n_k$ từ các client. Sau đó, Server sử dụng hàm `AGGREGATE` để kết hợp các kết quả này lại, cập nhật thành **model** global $\theta^{(t+1)}$ cho vòng lặp tiếp theo (các phương pháp tổng hợp chi tiết ở phần sau).

4.  **Kết thúc:**
    -   Sau khi hoàn thành đủ $T$ vòng lặp, Server trả về **model** đã tối ưu $\theta^{(T)}$.

## I.6. Client Selection

Việc lựa chọn client là một thành phần quan trọng trong quy trình Federated Learning. Thay vì sử dụng toàn bộ các thiết bị có sẵn trong mỗi round huấn luyện (điều này có thể gây tốn kém tài nguyên hệ thống và băng thông mạng), hệ thống chỉ chọn một tỷ lệ nhất định các thiết bị tham gia.

### Tại sao việc lựa chọn Client lại quan trọng?

- **JOIN RATIO ($\lambda$)**: Đây là tham số xác định tỷ lệ client tham gia. Ví dụ, với $\lambda = 0.5$, có 50% số lượng client hiện có sẽ tham gia vào mỗi round huấn luyện.
- **Tác động đến Federated Learning**:
    - Ảnh hưởng đến tốc độ hội tụ (**convergence**) nhanh hơn hoặc chậm hơn của mô hình.
    - Quyết định độ chính xác (**accuracy**) cuối cùng của mô hình toàn cục (global model).
    - Cải thiện tính công bằng (**fairness**) giữa các client khác nhau trong hệ thống.
    - Tăng cường hiệu quả truyền thông (**communication efficiency**).

### Các chiến lược lựa chọn Client phổ biến 

1. **Random selection**: Lựa chọn ngẫu nhiên các client, phương pháp này đóng vai trò là baseline cho thuật toán FedAvg.
2. **System-aware selection**: Ưu tiên lựa chọn các client có tốc độ xử lý nhanh và độ tin cậy cao dựa trên các thông số kỹ thuật của hệ thống (như phần cứng, trạng thái kết nối).
3. **Data-aware selection**: Thực hiện lựa chọn dựa trên đặc điểm dữ liệu của client nhằm cải thiện tính đa dạng và tính đại diện cho tập dữ liệu tổng thể.
4. **Adaptive / importance-based selection**: Chiến lược lựa chọn thích nghi hoặc dựa trên tầm quan trọng, căn cứ vào hiệu quả của quá trình huấn luyện. Các tiêu chí đánh giá thường bao gồm: **gradient** norm, **loss**, hoặc độ trễ (**staleness**).

### Quy trình lựa chọn Client qua các Communication Round

Quy trình này mô tả cách thức các client được huy động linh hoạt qua từng giai đoạn huấn luyện:

1. **Global Pool of Available Clients**: Hệ thống quản lý một tập hợp các thiết bị sẵn có (ví dụ: một nhóm gồm 8 thiết bị từ C1 đến C8, bao gồm cả điện thoại thông minh và máy tính xách tay).
2. **Cơ chế lựa chọn**: Áp dụng tỷ lệ tham gia **JOIN RATIO ($\lambda$) = 0.5**, nghĩa là trong mỗi round, hệ thống sẽ chọn ngẫu nhiên 4 trong số 8 client (50%) để thực hiện huấn luyện.
3. **Sự thay đổi qua các Round**:
    - **ROUND t**: Các client C1, C3, C7, C8 được đánh dấu là **SELECTED**.
    - **ROUND t + 1**: Các client C1, C3, C4, C8 được đánh dấu là **SELECTED**.
    - **ROUND t + 2**: Các client C1, C2, C6, C7 được đánh dấu là **SELECTED**.

Sự thay đổi linh hoạt của tập hợp các client qua từng round giúp tối ưu hóa tài nguyên trong khi vẫn đảm bảo mô hình học hỏi được từ nhiều nguồn dữ liệu khác nhau.

## I.7. Server Aggregation

Cơ chế Server Aggregation là thành phần quan trọng nhất của thuật toán Federated Averaging (FedAvg). Thay vì thu thập dữ liệu người dùng, server thu thập các tham số **model** ($\theta$) từ các client được chọn, sau đó thực hiện tính trung bình có trọng số để tạo ra một global **model** mới cho vòng lặp tiếp theo.

Công thức tổng hợp tại Server:

$$\theta^{(t+1)} = \sum_{k \in \mathcal{S}_t} \alpha_k * \theta_k^t$$

Điều kiện ràng buộc:
$$\sum_{k \in \mathcal{S}_t} \alpha_k = 1$$

Trong đó:
- $\theta$: **model**.
- $\alpha$: **weight**/hệ số tổng hợp (aggregation **weight**/coefficient).
- $k$: tổng số client trong tập con (subset).
- $T$: vòng lặp/phiên bản toàn cục (global round / iteration).
- $\mathcal{S}_t$: các client được lựa chọn (selected clients).

### Các đặc điểm chính:
- Được coi là cốt lõi của thuật toán FedAvg.
- Được xem như một kỹ thuật ensemble.
- Phương pháp này đơn giản nhưng mang lại hiệu quả cao.
- **"Học" mà không cần dữ liệu:** Quá trình học trong Federated Learning thực chất là việc tổng hợp các tri thức đã được học cục bộ tại các client. Server không cần truyền dữ liệu thô về máy chủ trung tâm, giúp bảo mật dữ liệu gốc trong khi vẫn cải thiện được **model** chung.
- Các phương pháp Federated Learning dựa trên cơ chế aggregation thường tập trung vào việc thay đổi hoặc tối ưu hóa giá trị của **weight** $\alpha$.

## I.8. Equal Weight Aggregation

Trong phương pháp này, tất cả các client được gán trọng số bằng nhau bất kể kích thước mẫu (sample size) của tập dữ liệu mà họ sở hữu. Đây là phương pháp aggregation đơn giản nhất trong Federated Learning, trong đó server sẽ lấy trung bình cộng trực tiếp các tham số (**weight**) nhận được từ tất cả các client để cập nhật global model. Điểm quan trọng là phương pháp này coi đóng góp của mọi client là như nhau, không phân biệt client đó huấn luyện trên tập dữ liệu lớn hay nhỏ.

Công thức tính toán **weight** cho global model:
$$w_{global} = \frac{w_1 + w_2}{2}$$

Trong đó:
- $w_1$: **weight** của Client 1 model.
- $w_2$: **weight** của Client 2 model.
- $w_{global}$: **weight** của Aggregated global model.

### Quy trình Aggregation minh họa

Quy trình tổng hợp các client model thành global model được thực hiện thông qua việc tính trung bình cộng các phần tử tương ứng trong ma trận **weight**:

1.  **Client 1 model ($w_1$):** Một ma trận **weight** kích thước 3x3 với các giá trị:
    $$\begin{bmatrix} 0.7 & 0.4 & 0.3 \\ 0.4 & 0.5 & 0.3 \\ 0.2 & 0.8 & 1.3 \end{bmatrix}$$

2.  **Client 2 model ($w_2$):** Một ma trận **weight** kích thước 3x3 với các giá trị:
    $$\begin{bmatrix} 1.0 & 0.3 & 4.3 \\ 0.7 & 0.5 & 1.6 \\ 1.6 & 0.6 & 0.9 \end{bmatrix}$$

3.  **Cơ chế xử lý:** Hai luồng dữ liệu từ $w_1$ và $w_2$ được đưa vào quá trình Aggregation. Tại đây, giá trị của mỗi phần tử trong ma trận kết quả là trung bình cộng của các phần tử ở vị trí tương ứng từ hai ma trận client.
    *   *Ví dụ:* Phần tử đầu tiên được tính bằng $(0.7 + 1.0) / 2 = 0.85$.

4.  **Aggregated global model ($w_{global}$):** Ma trận kết quả cuối cùng sau khi tổng hợp:
    $$\begin{bmatrix} 0.85 & 0.35 & 2.3 \\ 0.55 & 0.5 & 0.95 \\ 0.9 & 0.7 & 1.1 \end{bmatrix}$$

## I.9. Federated Learning - Weighted Aggregation (Tổng hợp có trọng số)

Trong Federated Learning, phương pháp **Weighted Aggregation** được sử dụng để kết hợp các mô hình từ các **client** khác nhau thành một mô hình chung duy nhất. Thay vì sử dụng trung bình cộng đơn giản, mô hình toàn cục (**global model**) được tính toán dựa trên trung bình có trọng số của các mô hình địa phương (**local models**).

**Chuẩn hóa mẫu huấn luyện (Training Samples Normalized):**

Các trọng số được gán cho mỗi **client** dựa trên tỷ lệ số lượng mẫu huấn luyện (**training samples**) mà **client** đó sở hữu so với tổng số dữ liệu của toàn hệ thống. Tổng các trọng số này phải được chuẩn hóa để bằng 1. Trong ví dụ này:
- Trọng số của **Client 1**: $0.8$ (tương ứng 80% dữ liệu)
- Trọng số của **Client 2**: $0.2$ (tương ứng 20% dữ liệu)
- Tổng cộng: $0.8 + 0.2 = 1$

**Chi tiết các mô hình thành phần:**

Quy trình **Aggregation** bắt đầu với các ma trận tham số (**weight**) có kích thước $3 \times 3$ từ hai **client**:

- **Client 1 model ($w_1$):**
$$
\begin{bmatrix}
0.7 & 0.4 & 0.3 \\
0.4 & 0.5 & 0.3 \\
0.2 & 0.8 & 1.3
\end{bmatrix}
$$

- **Client 2 model ($w_2$):**
$$
\begin{bmatrix}
1.0 & 0.3 & 4.3 \\
0.7 & 0.5 & 1.6 \\
1.6 & 0.6 & 0.9
\end{bmatrix}
$$

**Công thức tính toán:**

Mô hình toàn cục được tạo ra bằng cách tổng hợp các ma trận **local** theo công thức:
$$w_{global} = 0.8w_1 + 0.2w_2$$

**Kết quả Aggregated global model ($w_{global}$):**

Sau khi thực hiện phép tính tổng hợp có trọng số, ta thu được ma trận kết quả cho mô hình toàn cục:
$$
\begin{bmatrix}
0.76 & 0.38 & 1.1 \\
0.46 & 0.5 & 0.56 \\
0.48 & 0.76 & 1.22
\end{bmatrix}
$$

**Nguyên lý hoạt động:**

- Các trọng số (**weights**) trong phép tổng hợp được điều chỉnh dựa trên số lượng mẫu huấn luyện tương đối của mỗi **client**.
- Các **client** đóng góp nhiều dữ liệu hơn sẽ có ảnh hưởng lớn hơn đến kết quả cuối cùng của mô hình toàn cục. 
- Trong ví dụ cụ thể này, vì **Client 1** chiếm 80% lượng dữ liệu nên các tham số của nó có tác động mạnh mẽ hơn đến $w_{global}$ so với **Client 2** (chỉ chiếm 20%). Điều này giúp mô hình toàn cục phản ánh chính xác hơn đặc điểm của tập dữ liệu lớn hơn trong hệ thống.

## I.10. Client Local Training Procedure

Quy trình này mô tả chi tiết các bước huấn luyện diễn ra tại mỗi client trong hệ thống Federated Learning. Điểm mấu chốt là dữ liệu thô không bao giờ rời khỏi thiết bị của người dùng, giúp đảm bảo tính bảo mật và quyền riêng tư.

### Các tham số và ký hiệu

Trong thuật toán, các ký hiệu toán học được định nghĩa như sau:

- $\theta^{(t)}$: **global model** tại vòng lặp thứ $t$.
- $E$: Số lượng **local epochs** (số lần lặp lại việc huấn luyện trên toàn bộ dữ liệu cục bộ).
- $B$: **batch size** cho quá trình huấn luyện cục bộ.
- $\mathcal{D}_k$: Tập dữ liệu cục bộ (**local dataset**) được lưu trữ tại client $k$.
- $\eta$: **local learning rate**.

### Thuật toán cập nhật tại Client (Client Update Procedure)

Mỗi client thực hiện hàm `CLIENTUPDATE` để cập nhật mô hình dựa trên dữ liệu riêng của mình. Thay vì huấn luyện từ đầu, client bắt đầu từ **global model** hiện tại do server cung cấp.

```python
1: procedure CLIENTUPDATE(k, theta^(t), E, B, eta)
2:   Nhận global model theta^(t) từ server
3:   theta_k^(t,0) <- theta^(t)  # bắt đầu từ global model, không phải từ đầu
4:   Shuffle local dataset D_k
5:   for e = 1 to E do
6:     for mỗi mini-batch B thuộc D_k với kích thước B do
7:       l <- 1/|B| * sum_{(x,y) thuộc B} L(theta_k^(t,e-1); x, y)
8:       theta_k^(t,e) <- theta_k^(t,e-1) - eta * grad(l)
9:     end for
10:  end for
11:  theta_k^(t+1) <- theta_k^(t,E)
12:  Gửi theta_k^(t+1) tới server  # chỉ gửi weights - không gửi gradient, không gửi dữ liệu thô
13:  return theta_k^(t+1)
14: end procedure
```

### Chi tiết các bước thực hiện

1.  **Khởi tạo**: Client $k$ nhận **global model** $\theta^{(t)}$ từ server. Trọng số cục bộ ban đầu $\theta_k^{(t,0)}$ được thiết lập bằng chính trọng số của **global model**.
2.  **Chuẩn bị dữ liệu**: Tập dữ liệu cục bộ $\mathcal{D}_k$ được xáo trộn (**shuffle**) để đảm bảo tính ngẫu nhiên trong quá trình huấn luyện.
3.  **Huấn luyện cục bộ (Local Training)**: Client thực hiện $E$ **epoch** huấn luyện. Trong mỗi **epoch**, dữ liệu được chia thành các **mini-batch** có kích thước $B$.
    -   **Tính toán loss**: Với mỗi **mini-batch**, client tính toán giá trị **loss** trung bình $\ell$ dựa trên hàm **loss** $\mathcal{L}$:
        $$\ell \leftarrow \frac{1}{|\mathcal{B}|} \sum_{(x,y) \in \mathcal{B}} \mathcal{L}(\theta_k^{(t,e-1)}; x, y)$$
    -   **Cập nhật trọng số**: Client cập nhật **weight** cục bộ bằng phương pháp Stochastic Gradient Descent (SGD):
        $$\theta_k^{(t,e)} \leftarrow \theta_k^{(t,e-1)} - \eta \nabla \ell$$
4.  **Gửi kết quả**: Sau khi hoàn thành tất cả các **epoch**, client xác định trọng số mới $\theta_k^{(t+1)}$ và gửi chúng trở lại server.
    -   **Lưu ý quan trọng**: Client chỉ gửi các **weight** đã cập nhật. Tuyệt đối không gửi **gradient** hay dữ liệu thô (**raw data**) lên server, điều này giúp tối ưu băng thông và bảo vệ quyền riêng tư của người dùng.

# II. DATA PARTITION

## II.1. Federated Learning
Trong kiến trúc **Federated Learning**, việc phân chia và quản lý dữ liệu tuân thủ các nguyên tắc cốt lõi nhằm đảm bảo tính bảo mật và hiệu quả của hệ thống.

Mỗi client chỉ có:
`Private Training Data`

Server có:
`Shared / Public Test Set`

Tức là sau khi aggregate, server có thể đánh giá global model trên một bộ test chung/public. Mục tiêu là tạo một global model chung tốt trên test set chung.

> “Send trained model” luôn là client → server, còn “Receive aggregated model” là server → client.

## II.2. Personalized Federated Learning

Trong kiến trúc **Personalized Federated Learning**, việc phân chia dữ liệu và quy trình tương tác giữa các thành phần được thiết kế để tối ưu hóa tính cá nhân hóa và bảo mật thông tin.

Mỗi client có:

`Private Training and Testing Dat`

Server chỉ làm:

`Global aggregation, not evaluation`

Tức là server không có test set chung và không đánh giá global model tập trung. Sau khi nhận aggregated model, từng client có thể personalize/fine-tune/evaluate trên test data riêng của mình.

Mục tiêu là mỗi client có model phù hợp với dữ liệu riêng, không nhất thiết một global model tốt cho tất cả.

> “Send trained model” luôn là client → server, còn “Receive aggregated model” là server → client.

## II.3. Decentralized Federated Learning (DFL)

**Decentralized Federated Learning (DFL)** là một mô hình machine learning trong đó nhiều thiết bị (**clients**) cộng tác huấn luyện một **model** mà không cần một **server** điều phối trung tâm.

### Kiến trúc của Decentralized Federated Learning (DFL)

Mô hình DFL được xây dựng dựa trên cấu trúc mạng lưới (**mesh network**) hoàn chỉnh với các đặc điểm sau:

- **Thành phần hệ thống**: Bao gồm nhiều **clients** (thiết bị đầu cuối) đóng vai trò ngang hàng trong mạng lưới.
- **Dữ liệu cục bộ (Private Training and Testing Data)**: Mỗi **client** sở hữu tập dữ liệu huấn luyện và kiểm thử riêng tư. Dữ liệu này được lưu trữ phân tán và hoàn toàn không rời khỏi thiết bị, đảm bảo tính bảo mật.
- **Cơ chế giao tiếp (Model Update Exchange)**: 
    - Việc trao đổi thông tin diễn ra trực tiếp giữa các thiết bị theo mô hình ngang hàng (**Peer-to-Peer**).
    - Các **clients** kết nối trực tiếp với nhau để trao đổi các bản cập nhật **model** thông qua các luồng giao tiếp hai chiều.
- **Đặc điểm cấu trúc**: Điểm khác biệt cốt lõi của DFL so với Federated Learning truyền thống là việc loại bỏ hoàn toàn **server** trung tâm.

### Cơ chế hoạt động và Ưu điểm

Trong mô hình này, các **clients** không chỉ thực hiện huấn luyện cục bộ trên dữ liệu riêng tư của mình mà còn trực tiếp chia sẻ kết quả huấn luyện với các **clients** khác trong mạng lưới. Quá trình này giúp:

1. **Tối ưu hóa model chung**: Các thiết bị cùng nhau cộng tác để cải thiện chất lượng **model** mà không cần sự điều phối từ bên thứ ba.
2. **Tăng tính phi tập trung**: Loại bỏ sự phụ thuộc vào một thực thể quản lý duy nhất.
3. **Giảm thiểu rủi ro**: Tránh được lỗi từ điểm duy nhất (**single point of failure**) thường xảy ra khi **server** trung tâm gặp sự cố trong các mô hình tập trung.

## II.4. Chiến lược phân chia dữ liệu

Trong Federated Learning, việc phân chia tập dữ liệu đóng vai trò quan trọng trong việc xác định cách thức huấn luyện và đánh giá mô hình. Dưới đây là ba chiến lược phân chia dữ liệu chính:

### 1. Centralized FL Split
Chiến lược này tập trung vào việc tối ưu hóa một **model** global duy nhất và đánh giá nó dựa trên một tiêu chuẩn tập trung.
- **Cơ chế (Pre-Split):** Tập dữ liệu kiểm tra (TEST DATA) được giữ cố định tại server trung tâm. Trong khi đó, dữ liệu huấn luyện được phân chia thành các phần (TRAIN DATA SPLIT) và phân phối xuống các thiết bị (Client 1, Client 2, Client 3).
- **Đánh giá:** Hiệu suất của **model** được đánh giá trực tiếp dựa trên tập test global (toàn cục) tại server.

### 2. Personalized FL Split
Chiến lược này phù hợp khi mục tiêu là tùy chỉnh **model** cho từng người dùng cụ thể, cho phép mỗi **client** có tập test riêng từ một nguồn dữ liệu chung.
- **Cơ chế (Merge then split):** Quy trình bắt đầu từ một tập dữ liệu đầy đủ (Full Dataset) được hợp nhất thành Merged Dataset. Sau đó, dữ liệu từ khối chung này được phân phối đến các **client**.
- **Phân chia tại Client:** Tại mỗi **client**, dữ liệu nhận được sẽ được chia thành hai phần ngay tại thiết bị: **TRAIN SPLIT** và **TEST SPLIT**. 
- **Ứng dụng:** Đây là phương pháp lý tưởng cho các kỹ thuật Personalized FL (Federated Learning cá nhân hóa).

### 3. Separate Client Data
Đây là kịch bản phản ánh thực tế nhất của Federated Learning, nơi dữ liệu phát sinh tự nhiên tại các thiết bị đầu cuối và không bao giờ rời khỏi thiết bị.
- **Cơ chế (Native Separation):** Dữ liệu của mỗi **client** là duy nhất (unique) ngay từ đầu. Dữ liệu này tồn tại độc lập tại các địa điểm khác nhau (Location 1, 2, 3) và trên các loại thiết bị khác nhau như máy tính bàn, máy tính bảng, điện thoại hoặc smartphone.
- **Phân chia cục bộ:** Mỗi thiết bị tự thực hiện việc chia **LOCAL TRAIN SPLIT** và **LOCAL TEST SPLIT** trên chính nguồn dữ liệu thực tế cục bộ của mình.
- **Đánh giá:** Việc đánh giá được thực hiện trực tiếp trên dữ liệu **client** đa dạng (ví dụ: tập dữ liệu Weather - 5k), phục vụ việc huấn luyện và kiểm thử trên dữ liệu thực tế tại chỗ.

## II.5. Statistical Heterogeneity - Distribution Skew

Trong Federated Learning, tính không đồng nhất về mặt thống kê (**Statistical Heterogeneity**), cụ thể là hiện tượng lệch phân phối (**Distribution Skew**), là một đặc điểm quan trọng cần xem xét khi phân chia dữ liệu.

### Phương pháp Practical Dirichlet Partition ($DIR = 0.1$)

Phương pháp này được sử dụng để mô phỏng sự phân bổ dữ liệu của mỗi **client** trên các tập dữ liệu MNIST, CIFAR-10, và CIFAR-100 trong thiết lập **heterogeneous** (không đồng nhất) mặc định. Trong các biểu đồ phân tán minh họa:
- Trục X đại diện cho **Client IDs** (từ 0 đến 19).
- Trục Y đại diện cho **Class IDs** (các lớp nhãn).
- Kích thước của vòng tròn đại diện cho số lượng mẫu (**samples**).

Phân tích sự phân bổ trên các tập dữ liệu cụ thể với 20 **clients**:

1.  **MNIST ($DIR = 0.1$):**
    - Các vòng tròn đỏ phân bố rải rác.
    - Kích thước vòng tròn khác nhau cho thấy số lượng mẫu của mỗi **class** tại mỗi **client** là không đồng đều.
    - Một số **client** chỉ sở hữu rất ít **class**, trong khi những **client** khác có nhiều lớp hơn nhưng với số lượng mẫu khác nhau.

2.  **CIFAR-10 ($DIR = 0.1$):**
    - Mật độ các điểm đỏ dày đặc hơn so với MNIST.
    - Mỗi cột (tương ứng với một **client**) cho thấy sự hiện diện của nhiều **class**, nhưng kích thước vòng tròn biến thiên mạnh mẽ. Điều này thể hiện tính không đồng nhất (**heterogeneity**) cao giữa các **clients**.

3.  **CIFAR-100 ($DIR = 0.1$):**
    - Các điểm đỏ tạo thành một dải đường chéo rõ rệt từ góc dưới bên trái lên góc trên bên phải của biểu đồ.
    - Đây là minh chứng rõ nét cho hiện tượng lệch phân phối nhãn (**label skew**): mỗi **client** chỉ sở hữu một tập hợp con các **class** liên tiếp (ví dụ: **Client** 0 giữ các lớp từ 0-10, trong khi **Client** 19 giữ các lớp từ 80-100).

**Tổng kết về tính không đồng nhất**

Trong thực tế, dữ liệu tại các **client** không bao giờ giống nhau (**Non-IID**). Mỗi **client** có thể sở hữu số lượng mẫu khác nhau và các lớp dữ liệu khác nhau. Đặc biệt như trong ví dụ CIFAR-100, sự lệch phân phối thể hiện cực kỳ rõ ràng khi mỗi **client** chỉ tập trung vào một nhóm nhỏ các lớp nhất định. Điều này gây khó khăn lớn cho việc huấn luyện một mô hình global có khả năng tổng quát hóa tốt trên tất cả các lớp dữ liệu.


### Tham số tập trung Dirichlet $\beta$: 
- Đây là tham số then chốt kiểm soát mức độ không đồng nhất (heterogeneity) của dữ liệu.
- **Mối quan hệ nghịch biến**: Khi giá trị $\beta$ trong $DIR(\beta)$ tăng lên, mức độ không đồng nhất giữa các client sẽ giảm xuống. Ngược lại, $\beta$ càng nhỏ thì sự phân bổ dữ liệu giữa các client càng khác biệt (Distribution Skew), gây khó khăn lớn cho quá trình huấn luyện.
- **Minh họa trên tập dữ liệu Tiny-ImageNet**: Sự phân bổ dữ liệu của mỗi client được thể hiện qua ba thiết lập không đồng nhất khác nhau, trong đó kích thước của các vòng tròn đại diện cho số lượng mẫu (samples) của một lớp cụ thể tại một client cụ thể.

#### Phân tích các thiết lập phân phối dữ liệu

Dưới đây là sự thay đổi của phân phối dữ liệu trên Tiny-ImageNet (với trục hoành là Client IDs từ 0-19 và trục tung là Class IDs từ 0-180) dựa trên các giá trị $\beta$ khác nhau:

1.  **Tiny-ImageNet — $DIR(0.01)$ (Mức độ không đồng nhất cực cao):**
    - Các vòng tròn xuất hiện rất thưa thớt trên biểu đồ.
    - Mỗi Client ID chỉ sở hữu một vài Class IDs nhất định.
    - Dữ liệu bị lệch rất nặng, minh họa cho tình trạng một client chỉ có dữ liệu của một số rất ít nhãn lớp.

2.  **Tiny-ImageNet — $DIR(0.1)$ (Mức độ không đồng nhất trung bình):**
    - Số lượng các vòng tròn xuất hiện nhiều hơn trên mỗi cột của client so với thiết lập trước.
    - Tuy nhiên, vẫn còn nhiều khoảng trống trắng trên biểu đồ, cho thấy nhiều client vẫn thiếu hụt đáng kể các lớp dữ liệu.

3.  **Tiny-ImageNet — $DIR(0.5)$ (Mức độ không đồng nhất thấp):**
    - Các vòng tròn xuất hiện dày đặc và bao phủ hầu hết các Class IDs cho mỗi Client ID.
    - Kích thước các vòng tròn có xu hướng đồng đều hơn giữa các lớp và các client.
    - Dữ liệu bắt đầu phân bố đều hơn, tiến gần hơn đến trạng thái đồng nhất (IID - Independent and Identically Distributed).

**Tóm tắt cấu trúc biểu đồ minh họa:**
- **Trục hoành (X-axis):** Client IDs (từ 0 đến 19).
- **Trục tung (Y-axis):** Class IDs (từ 0 đến 180).
- **Kích thước vòng tròn:** Tỉ lệ thuận với số lượng mẫu của một lớp (class) cụ thể tại một client cụ thể.

### Pathological partitioning
- Trong phương pháp **pathological partitioning**, mỗi **client** chỉ được gán một số lượng nhỏ các lớp (classes).
- Kỹ thuật này tạo ra một thiết lập **non-IID** cực đoan, thường được sử dụng để kiểm tra tính mạnh mẽ (**robustness**) của mô hình dưới tình trạng **label skew** nghiêm trọng.
- Mục tiêu chính của phương pháp này là mô phỏng kịch bản trong Federated Learning để đánh giá khả năng hội tụ và tính hiệu quả của các mô hình học máy khi dữ liệu phân tán không đại diện cho toàn bộ quần thể.

**Minh họa phân bổ dữ liệu trong thiết lập pathological heterogeneous**

Sự phân bổ dữ liệu của mỗi **client** trên các tập dữ liệu MNIST, CIFAR-10 và CIFAR-100 được minh họa qua các biểu đồ phân tán (scatter plots). Trong đó, kích thước của các vòng tròn đại diện cho số lượng mẫu dữ liệu (number of samples).

Cấu trúc của các biểu đồ:
- **Trục hoành (X-axis):** Biểu thị **Client** IDs (từ 0 đến 19).
- **Trục tung (Y-axis):** Biểu thị Class IDs (Mã định danh lớp).

Phân tích cụ thể trên từng tập dữ liệu:
1. **MNIST - Pathological:** Mỗi **client** chỉ nắm giữ 2 classes. Các vòng tròn tạo thành cấu trúc hình bậc thang, cho thấy từng nhóm **client** liên tiếp sở hữu các cặp nhãn khác nhau (ví dụ: các **client** từ 0-3 chỉ giữ dữ liệu của class 0 và class 1).
2. **CIFAR-10 - Pathological:** Tương tự như MNIST, mỗi **client** chỉ có dữ liệu của 2 classes. Điều này dẫn đến sự thiếu hụt dữ liệu trầm trọng về các lớp khác trên mỗi thiết bị, tạo ra kịch bản **label skew** cực hạn.
3. **CIFAR-100 - Pathological:** Mỗi **client** được gán 10 classes. Do tập dữ liệu CIFAR-100 có nhiều lớp hơn, các cụm vòng tròn hiển thị dày đặc hơn nhưng vẫn duy trì cấu trúc phân mảnh theo từng nhóm **client** nhất định.

#### Non-IID

```text
Non-Independent and Identically Distributed
```

Trong Federated Learning, hiểu đơn giản là:

**Dữ liệu ở các client không giống phân phối nhau.**

Ngược lại, **IID** nghĩa là dữ liệu ở các client được chia khá đều và giống nhau.

Ví dụ bài toán phân loại số viết tay MNIST có 10 class: 0–9.

##### IID

Mỗi client đều có dữ liệu tương đối giống nhau:

```text
Client 1: số 0,1,2,3,4,5,6,7,8,9
Client 2: số 0,1,2,3,4,5,6,7,8,9
Client 3: số 0,1,2,3,4,5,6,7,8,9
```

Tỉ lệ các class cũng gần giống nhau.

##### Non-IID

Dữ liệu mỗi client bị lệch:

```text
Client 1: chủ yếu số 0, 1
Client 2: chủ yếu số 2, 3
Client 3: chủ yếu số 7, 8, 9
```

Hoặc:

```text
Client 1: rất nhiều data
Client 2: rất ít data
Client 3: data bị nhiễu nhiều hơn
```

Trong thực tế, non-IID rất phổ biến.

Ví dụ với bàn phím điện thoại:

```text
Người A hay gõ tiếng Việt
Người B hay gõ tiếng Anh
Người C hay dùng từ chuyên ngành y tế
Người D hay viết tắt, dùng emoji
```

Tất cả đều dùng cùng một mô hình keyboard, nhưng dữ liệu của từng người rất khác nhau.

Vấn đề của non-IID trong Federated Learning là:

```text
Local model của mỗi client học theo dữ liệu riêng của nó
        ↓
Các update gửi về server rất khác nhau
        ↓
Server aggregate khó hơn
        ↓
Global model có thể hội tụ chậm hoặc accuracy thấp
```

Nói ngắn gọn:

**Non-IID = dữ liệu giữa các client không cùng phân phối, không đại diện giống nhau cho toàn bộ bài toán.**

Trong FL, đây là một trong những lý do chính khiến personalized federated learning cần thiết.

### Extended Dirichlet
Để tái hiện sự không đồng nhất này, phương pháp **Extended Dirichlet** được sử dụng với tham số $Dir=100$. Phương pháp này cho phép phân mảnh dữ liệu sao cho mỗi client chỉ sở hữu một số lượng lớp nhãn giới hạn, thay vì có đầy đủ các lớp như trong phân phối IID (Independent and Identically Distributed).

#### Thực nghiệm trên các tập dữ liệu
Việc phân chia dữ liệu được minh họa thông qua 20 clients (Client IDs từ 0 đến 19) trên ba tập dữ liệu phổ biến:

- **MNIST — Extended Dirichlet (num_class=2)**:
    - Trục tung hiển thị các Class IDs từ 0 đến 9.
    - Mỗi client trong số 20 clients chỉ sở hữu dữ liệu của **2 lớp nhãn** nhất định (được biểu diễn bằng các chấm đỏ trên biểu đồ).
- **CIFAR-10 — Extended Dirichlet (num_class=2)**:
    - Tương tự như MNIST, mỗi client chỉ nắm giữ dữ liệu của **2 trong số 10 lớp nhãn** của tập CIFAR-10.
- **CIFAR-100 — Extended Dirichlet (num_class=10)**:
    - Trục tung hiển thị các Class IDs từ 0 đến 99.
    - Mỗi client chỉ sở hữu dữ liệu của **10 lớp nhãn** khác nhau. Mặc dù mật độ các chấm đỏ cho thấy sự phân bổ nhãn đa dạng hơn so với MNIST hay CIFAR-10, nhưng nó vẫn bị giới hạn nghiêm ngặt ở mức 10 lớp cho mỗi client.

#### Ý nghĩa trong Federated Learning
Việc phân chia dữ liệu theo cách này phản ánh thực tế khách quan trong các hệ thống Federated Learning:
- Dữ liệu tại mỗi thiết bị người dùng (client) thường không đại diện cho toàn bộ phân phối dữ liệu toàn cục.
- Hiện tượng **Label Skew** này tạo ra thách thức lớn cho quá trình huấn luyện, làm cho việc hội tụ của mô hình trở nên khó khăn hơn do sự khác biệt lớn giữa các local **gradient** và **weight** được cập nhật từ mỗi client.


# III. EVALUATION

Việc đánh giá một hệ thống Federated Learning (FL) đòi hỏi một khung tham chiếu toàn diện. Hiệu quả của một thuật toán FL trong thực tế không chỉ dừng lại ở độ chính xác của mô hình mà còn phải được xem xét dựa trên sự cân bằng giữa bốn trụ cột chính: Hiệu suất mô hình, Tính công bằng, Tính toán và Truyền thông.

#### 1. Model Performance (Hiệu suất mô hình)
Trụ cột này tập trung vào chất lượng và khả năng học tập của mô hình trong môi trường phân tán:
- **Generalization (Khả năng tổng quát hóa):** Khả năng mô hình hoạt động tốt trên các dữ liệu mới, chưa từng xuất hiện trong quá trình huấn luyện.
- **Personalization (Cá nhân hóa):** Khả năng điều chỉnh mô hình để phù hợp với đặc thù dữ liệu riêng biệt của từng client.
- **Convergence (Sự hội tụ):** Tốc độ và khả năng mô hình đạt đến trạng thái tối ưu ổn định.

#### 2. Fairness (Tính công bằng)
Đánh giá sự phân bổ hiệu quả giữa các client tham gia vào hệ thống, đảm bảo không có thiết bị nào bị yếu thế:
- **Variance of per-client accuracy (Phương sai của độ chính xác trên mỗi client):** Đo lường mức độ chênh lệch về hiệu suất giữa các client khác nhau. Phương sai thấp cho thấy hệ thống hoạt động ổn định trên toàn bộ mạng lưới.
- **Worst-client accuracy (Độ chính xác của client tệ nhất):** Đánh giá hiệu suất của thiết bị có kết quả thấp nhất để đảm bảo tính công bằng tối thiểu cho mọi thành viên.

#### 3. Computation (Tính toán)
Đo lường các nguồn lực tài nguyên tính toán cần thiết để vận hành thuật toán:
- **FLOPs per round (Số lượng FLOPs trên mỗi round):** Tổng số lượng phép tính dấu phẩy động cần thực hiện trong một round huấn luyện.
- **Wall-clock time (Thời gian thực thi thực tế):** Thời gian thực tế trôi qua để hoàn thành các tác vụ tính toán.

#### 4. Communication (Truyền thông)
Đo lường hiệu quả của việc trao đổi dữ liệu giữa server và các client, vốn là một nút thắt cổ chai quan trọng trong FL:
- **Total communication cost (Tổng chi phí truyền thông):** Tổng lượng dữ liệu cần trao đổi trong toàn bộ quá trình huấn luyện.
- **Upload / download volume per client (Lưu lượng upload / download trên mỗi client):** Lượng dữ liệu mà mỗi thiết bị cụ thể phải tải lên và tải xuống, ảnh hưởng trực tiếp đến băng thông và chi phí của người dùng.

Bốn yếu tố này tạo thành một hệ thống tương tác qua lại. Việc tối ưu hóa một thuật toán FL đòi hỏi phải cân nhắc đồng thời cả chất lượng mô hình, tính công bằng giữa các thiết bị, cũng như tối ưu hóa tài nguyên về mặt tính toán và băng thông truyền tải.

## III.1. Model Performance
### Best-centralized

Trong hệ thống Federated Learning, việc đánh giá hiệu năng mô hình được thực hiện theo phương pháp **Best-centralized** thông qua quy trình vòng huấn luyện (**Round**).

- **Thành phần tham gia:**
    - **Client Layer:** Bao gồm 4 **Client** khác nhau (được minh họa bằng các màu tím, xanh dương, xanh lá và cam).
    - **Server:** Thành phần trung tâm đóng vai trò thu thập dữ liệu và đánh giá mô hình.

- **Quá trình huấn luyện và đánh giá:**
    1. **Huấn luyện local:** Trong mỗi **Round** (từ 1 đến 8), mỗi **Client** thực hiện quá trình huấn luyện local trên sơ đồ mạng nơ-ron riêng (bao gồm các **layer** và các kết nối nội bộ).
    2. **Tổng hợp tại Server:** Sau mỗi **Round**, các kết quả từ **Client** được gửi về **Server**. Tại đây, **Server** thực hiện ghi nhận và báo cáo số đo hiệu năng toàn cục.
    3. **Chỉ số hiệu năng toàn cục (Global Performance Metric):** Giá trị $Acc_{global}$ được tính toán cụ thể sau mỗi **Round**.

- **Xác định hiệu năng cuối cùng (Final Performance):**
    - Thay vì chỉ sử dụng kết quả của vòng huấn luyện cuối cùng, hệ thống sẽ theo dõi toàn bộ chuỗi giá trị $Acc_{global}$ từ **Round** 1 đến **Round** 8.
    - Các giá trị này được đưa vào một khối xử lý **MAX** để tìm ra kết quả tốt nhất.
    - **Hiệu năng cuối cùng** được xác định bằng giá trị lớn nhất đạt được trong suốt quá trình hội tụ:
    $$MAX(Acc_{global}^{(1)}, Acc_{global}^{(2)}, ..., Acc_{global}^{(8)})$$

Phương pháp này đảm bảo ghi lại được trạng thái tối ưu nhất mà mô hình có thể đạt được trong toàn bộ quá trình huấn luyện, giúp phản ánh chính xác khả năng của mô hình thay vì rủi ro lấy phải kết quả ở vòng cuối cùng nếu quá trình hội tụ không ổn định.

### Best-personalized/p2p

Quy trình đánh giá hiệu suất cuối cùng cho mô hình Personalized Federated Learning (hoặc p2p) không chỉ dựa vào kết quả ở vòng cuối cùng mà dựa trên một quá trình tính toán tổng thể qua các giai đoạn huấn luyện.

**Cấu trúc và quy trình đánh giá:**

- **Round (Vòng):** Quá trình thực nghiệm bao gồm 8 vòng giao tiếp, được đánh số từ 1 đến 8.
- **Thành phần tham gia:** Có 4 **client** tham gia vào quá trình Federated Learning (được đại diện bởi 4 màu sắc: Tím, Xanh dương, Xanh lá, Cam). Tại mỗi **round**, mỗi **client** sẽ thực hiện đánh giá model và ghi lại kết quả (biểu diễn bằng các icon tài liệu và biểu đồ tròn).
- **Mean (Trung bình):** Sau mỗi **round**, hệ thống sẽ tính toán giá trị hiệu suất trung bình của tất cả các **client** tham gia trong vòng đó.
- **Best (Tốt nhất):** Hệ thống sẽ so sánh các giá trị Mean của cả 8 vòng để tìm ra giá trị trung bình cao nhất.
- **Final Performance (Hiệu suất cuối cùng):** Kết quả cuối cùng được xác định chính là giá trị Mean tốt nhất thu được trong suốt quá trình 8 vòng thử nghiệm.

**Logic xác định kết quả:**

Thay vì chỉ lấy kết quả tại thời điểm kết thúc (Round 8), phương pháp này theo dõi hiệu suất trung bình của toàn bộ hệ thống qua từng bước. Bằng cách chọn ra giá trị "Best Mean" từ tập hợp các giá trị trung bình của 8 vòng, hệ thống có thể ghi lại được đỉnh cao hiệu suất (peak performance) mà mô hình đạt được trong quá trình hội tụ. Điều này giúp phản ánh chính xác hơn khả năng tối ưu của các mô hình cá nhân hóa hoặc mạng ngang hàng (p2p).

### Vòng cuối cùng (Last)

- **Global Performance Metric:** Chỉ số hiệu suất toàn cục dùng để đo lường chất lượng của mô hình sau khi hợp nhất.
- **$Acc_{global}$:** Độ chính xác toàn cục, được tính toán cụ thể cho mỗi vòng sau khi các mô hình từ các **Clients** được tổng hợp.
- **LAST ROUND:** Vòng cuối cùng của quá trình huấn luyện (vòng 8).
- **Final Performance:** Hiệu suất cuối cùng của hệ thống, được xác định dựa trên kết quả tại vòng cuối cùng.

Việc đánh giá liên tục qua từng vòng giúp theo dõi sự tiến triển của mô hình trước khi đi đến kết quả sau cùng tại vòng giao tiếp cuối cùng.

### Mean of K-Last

Trong pp này hiệu suất cuối cùng không chỉ dựa trên một vòng huấn luyện đơn lẻ mà được tính toán dựa trên giá trị trung bình của các vòng cuối cùng để đảm bảo tính ổn định.

**Ý nghĩa của phương pháp:**

Việc sử dụng giá trị trung bình của $K=4$ vòng cuối cùng thay vì chỉ dựa vào vòng huấn luyện duy nhất cuối cùng mang lại nhiều lợi ích:
- **Giảm nhiễu:** Loại bỏ các biến động ngẫu nhiên hoặc nhiễu có thể xảy ra tại một thời điểm huấn luyện nhất định.
- **Độ tin cậy cao:** Cung cấp một con số đánh giá ổn định và đáng tin cậy hơn về khả năng hội tụ của mô hình.
- **Đánh giá chính xác:** Phản ánh đúng độ chính xác bền vững của mô hình toàn cục trong môi trường Federated Learning sau khi đã trải qua đủ số lượng vòng lặp cần thiết.

### Target Performance / Convergence

Đây là phương pháp đánh giá **model performance theo target performance / convergence** trong Federated Learning.

Ý chính: **không chỉ báo accuracy cuối cùng**, mà xem mô hình cần **bao nhiêu vòng giao tiếp** để đạt accuracy mục tiêu.

Ví dụ trong hình:

```text
Target Accuracy = 85%
```

Mỗi cột là một **communication round**:

```text
Round 1 → Round 2 → ... → Round 8
```

Ở mỗi round:

```text
Server gửi global model hiện tại cho các client
        ↓
Mỗi client train local trên private data
        ↓
Client gửi trained model / update lên server
        ↓
Server aggregate thành global model mới
        ↓
Đánh giá global model → Acc_global
```

Sau mỗi round sẽ có một giá trị:

```text
Acc_global round 1
Acc_global round 2
Acc_global round 3
...
```

Mục tiêu là kiểm tra:

```text
Khi nào Acc_global đạt 85%?
```

Nếu đến **round 8** mới đạt 85%, thì báo cáo sẽ là:

```text
Final Performance = 85%
Number of communication rounds = 8
```

Ngoài số round, phương pháp này còn có thể report thêm:

```text
Total training time
Total communication time
Total data volume truyền qua mạng
```

Nói ngắn gọn:

**Phương pháp này đánh giá tốc độ hội tụ của FL.**
Không chỉ hỏi “model đạt accuracy bao nhiêu?”, mà hỏi thêm:

```text
Để đạt accuracy mục tiêu 85%, hệ thống cần bao nhiêu round,
mất bao lâu, và tốn bao nhiêu dữ liệu truyền thông?
```

Ví dụ so sánh:

```text
Method A đạt 85% sau 5 rounds
Method B đạt 85% sau 8 rounds
```

Thì nếu accuracy cuối đều là 85%, **Method A tốt hơn về convergence/communication efficiency** vì cần ít vòng giao tiếp hơn.

## III.2. Example

Phần này trình bày các kết quả thực nghiệm so sánh hiệu năng của phương pháp **FedALA** với 13 thuật toán **Federated Learning** khác nhau. Các thử nghiệm được thực hiện trên nhiều bộ dữ liệu đa dạng, bao gồm xử lý ảnh (MNIST, CIFAR-10, CIFAR-100, TINY ImageNet) và xử lý ngôn ngữ tự nhiên (AG News).

### Ví dụ so sánh hiệu năng

Dưới đây là bảng thống kê độ chính xác kiểm tra (%) trong hai thiết lập dữ liệu không đồng nhất: (**pathological heterogeneous setting**) và (**practical heterogeneous setting**).

**Bảng 2: Độ chính xác kiểm tra (%) trong thiết lập không đồng nhất bệnh lý và thiết lập không đồng nhất thực tế.**

| Settings | Pathological heterogeneous setting | | | Practical heterogeneous setting | | | | |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Methods** | **MNIST** | **Cifar10** | **Cifar100** | **Cifar10** | **Cifar100** | **TINY** | **TINY*** | **AG News** |
| FedAvg | $97.93\pm0.05$ | $55.09\pm0.83$ | $25.98\pm0.13$ | $59.16\pm0.47$ | $31.89\pm0.47$ | $19.46\pm0.20$ | $19.45\pm0.13$ | $79.57\pm0.17$ |
| FedProx | $98.01\pm0.09$ | $55.06\pm0.75$ | $25.94\pm0.16$ | $59.21\pm0.40$ | $31.99\pm0.41$ | $19.37\pm0.22$ | $19.27\pm0.23$ | $79.35\pm0.23$ |
| FedAvg-C | $99.79\pm0.00$ | $92.13\pm0.03$ | $66.17\pm0.03$ | $90.34\pm0.01$ | $51.80\pm0.02$ | $30.67\pm0.08$ | $36.94\pm0.10$ | $95.89\pm0.25$ |
| FedProx-C | $99.80\pm0.04$ | $92.12\pm0.03$ | $66.07\pm0.08$ | $90.33\pm0.01$ | $51.84\pm0.07$ | $30.77\pm0.13$ | $38.78\pm0.52$ | $96.10\pm0.22$ |
| Per-FedAvg | $99.63\pm0.02$ | $89.63\pm0.23$ | $56.80\pm0.26$ | $87.74\pm0.19$ | $44.28\pm0.33$ | $25.07\pm0.07$ | $21.81\pm0.54$ | $93.27\pm0.25$ |
| FedRep | $99.77\pm0.03$ | $91.93\pm0.14$ | $67.56\pm0.31$ | $90.40\pm0.24$ | $52.39\pm0.35$ | $37.27\pm0.20$ | $39.95\pm0.61$ | $96.28\pm0.14$ |
| pFedMe | $99.75\pm0.02$ | $90.11\pm0.10$ | $58.20\pm0.14$ | $88.09\pm0.32$ | $47.34\pm0.46$ | $26.93\pm0.19$ | $33.44\pm0.33$ | $91.41\pm0.22$ |
| Ditto | $99.81\pm0.00$ | $92.39\pm0.06$ | $67.23\pm0.07$ | $90.59\pm0.01$ | $52.87\pm0.64$ | $32.15\pm0.04$ | $35.92\pm0.43$ | $95.45\pm0.17$ |
| FedAMP | $99.76\pm0.02$ | $90.79\pm0.16$ | $64.34\pm0.37$ | $88.70\pm0.18$ | $47.69\pm0.49$ | $27.99\pm0.11$ | $29.11\pm0.15$ | $94.18\pm0.09$ |
| FedPHP | $99.73\pm0.00$ | $90.01\pm0.00$ | $63.09\pm0.04$ | $88.92\pm0.02$ | $50.52\pm0.16$ | $35.69\pm3.26$ | $29.90\pm0.51$ | $94.38\pm0.12$ |
| FedFomo | $99.83\pm0.00$ | $91.85\pm0.02$ | $62.49\pm0.22$ | $88.06\pm0.02$ | $45.39\pm0.45$ | $26.33\pm0.22$ | $26.84\pm0.11$ | $95.84\pm0.15$ |
| APPLE | $99.75\pm0.01$ | $90.97\pm0.05$ | $65.80\pm0.08$ | $89.37\pm0.11$ | $53.22\pm0.20$ | $35.04\pm0.47$ | $39.93\pm0.52$ | $95.63\pm0.21$ |
| PartialFed | $99.86\pm0.01$ | $89.60\pm0.13$ | $61.39\pm0.12$ | $87.38\pm0.08$ | $48.81\pm0.20$ | $35.26\pm0.18$ | $37.50\pm0.16$ | $85.20\pm0.16$ |
| **FedALA** | $\mathbf{99.88\pm0.01}$ | $\mathbf{92.44\pm0.02}$ | $\mathbf{67.83\pm0.06}$ | $\mathbf{90.67\pm0.03}$ | $\mathbf{55.92\pm0.03}$ | $\mathbf{40.54\pm0.02}$ | $\mathbf{41.94\pm0.05}$ | $\mathbf{96.52\pm0.08}$ |

# IV. CHALLENGES

## IV.1. Cross-device vs Cross-silo

Trong Federated Learning, các thách thức được phân loại dựa trên hai kịch bản chính: **Cross-silo** và **Cross-device**. Sự khác biệt giữa hai mô hình này nằm ở quy mô, tính ổn định và bản chất của các client tham gia.

### 1. Cross-silo
Mô hình này tập trung vào sự hợp tác ở cấp độ tổ chức (Institution level collaboration).

- **Số lượng client:** Ít, thường chỉ từ 10 đến 100 client (ví dụ: các bệnh viện, ngân hàng hoặc các tổ chức lớn).
- **Dữ liệu:** Mỗi client sở hữu tập dữ liệu cục bộ lớn và có cấu trúc.
- **Hạ tầng:** Sử dụng cơ sở hạ tầng mạng và phần cứng đáng tin cậy, mạnh mẽ.
- **Tính ổn định:** Quá trình giao tiếp và sự tham gia của các client ổn định hơn, ít khi xảy ra tình trạng ngắt kết nối đột ngột.

### 2. Cross-device
Mô hình này tập trung vào việc huấn luyện trên các thiết bị cá nhân ở quy mô lớn (Large Scale device collaboration).

- **Số lượng client:** Cực kỳ lớn, có thể lên đến hàng triệu hoặc hàng tỷ thiết bị (ví dụ: điện thoại thông minh, thiết bị IoT).
- **Sự biến động về tính khả dụng của client (Client availability variability):** Các thiết bị có thể tham gia hoặc rời khỏi quá trình huấn luyện bất cứ lúc nào do tình trạng pin, kết nối mạng hoặc người dùng sử dụng thiết bị.
- **Sự biến động về hiệu suất hệ thống (System performance variability):** Có sự khác biệt lớn về năng lực tính toán, bộ nhớ và tốc độ kết nối giữa các thiết bị khác nhau.
- **Tính không đồng nhất của dữ liệu (Data heterogeneity):** Dữ liệu trên mỗi thiết bị được tạo ra bởi người dùng cá nhân, dẫn đến sự khác biệt lớn về phân phối dữ liệu (Non-IID).
- **Các vấn đề về lòng tin:** Đối mặt với rủi ro từ các client đối kháng (**adversarial clients**) có thể cố gắng làm sai lệch mô hình hoặc thực hiện các cuộc tấn công bảo mật.

### So sánh tóm tắt giữa Cross-silo và Cross-device

| Đặc điểm | Cross-silo | Cross-device |
| :--- | :--- | :--- |
| **Đối tượng tham gia** | Tổ chức (Bệnh viện, Ngân hàng) | Thiết bị cá nhân (Smartphone, IoT) |
| **Số lượng client** | Ít (10 - 100) | Rất lớn (Hàng triệu - Hàng tỷ) |
| **Dữ liệu cục bộ** | Tập dữ liệu lớn | Dữ liệu phân tán, không đồng nhất |
| **Độ tin cậy hệ thống** | Hạ tầng ổn định, đáng tin cậy | Hiệu suất và tính khả dụng biến động cao |
| **Kết nối** | Giao tiếp ổn định | Kết nối dày đặc nhưng không ổn định |

## IV.2. Các thách thức chính

Các thách thức chính trong Federated Learning bao gồm năm khía cạnh cốt lõi, thường được trình bày dưới dạng một sơ đồ vòng tròn để thể hiện sự tương quan và mối liên hệ mật thiết giữa các thành phần trong việc triển khai hệ thống:

- **Statistical Heterogeneity** (Tính không đồng nhất về thống kê): Thách thức này xuất phát từ sự khác biệt về phân phối dữ liệu giữa các thiết bị tham gia. Do dữ liệu được thu thập cục bộ dựa trên hành vi người dùng, dữ liệu trên mỗi client thường không đồng nhất về mặt thống kê.
- **System Heterogeneity** (Tính không đồng nhất về hệ thống): Đề cập đến sự khác biệt về năng lực phần cứng (như CPU, RAM, bộ nhớ) và khả năng kết nối mạng (băng thông, độ ổn định) giữa các thiết bị khác nhau trong mạng lưới.
- **Client Availability** (Sự sẵn có của Client): Việc các thiết bị tham gia không ổn định là một thách thức lớn. Các client có thể ngoại tuyến, ngắt kết nối hoặc không sẵn sàng tham gia vào quá trình huấn luyện do các yếu tố khách quan.
- **Scalability** (Khả năng mở rộng): Hệ thống Federated Learning phải đối mặt với bài toán quản lý và điều phối hiệu quả khi số lượng thiết bị tham gia tăng lên quy mô cực lớn.
- **Privacy & Security** (Quyền riêng tư và Bảo mật): Đảm bảo an toàn dữ liệu và bảo vệ quyền riêng tư của người dùng là yêu cầu tiên quyết, tránh các nguy cơ tấn công hoặc rò rỉ thông tin trong quá trình trao đổi giữa client và server.

Năm yếu tố này tạo thành khung tham chiếu cơ bản cho các cuộc thảo luận chi tiết về những khó khăn thực tế mà các hệ thống Federated Learning phải giải quyết.


### System Heterogeneity (Tính không đồng nhất của hệ thống)

Trong Federated Learning (FL), một trong những thách thức lớn nhất là **System Heterogeneity** (Tính không đồng nhất của hệ thống). Thách thức này phát sinh từ sự đa dạng và khác biệt đáng kể giữa các thiết bị tham gia vào quá trình huấn luyện.

Tính không đồng nhất của hệ thống được thể hiện qua các khía cạnh chính sau:

- **Hardware Heterogeneity (Tính không đồng nhất về phần cứng):** Có sự chênh lệch lớn về sức mạnh tính toán giữa các thiết bị di động khác nhau. 
    - Các thiết bị cấu hình cao như laptop có thể sở hữu **CPU** và **GPU** mạnh mẽ, cho phép xử lý các tác vụ huấn luyện nhanh chóng.
    - Ngược lại, các thiết bị di động hoặc máy tính bảng đời cũ có tài nguyên hạn chế về **CPU**, **GPU** và **RAM**, dẫn đến tốc độ xử lý chậm hơn.
- **Network Volatility (Sự biến động của mạng):** Tốc độ kết nối không ổn định và khác nhau giữa các thiết bị.
    - Tốc độ upload và download thay đổi tùy thuộc vào loại kết nối mà thiết bị đang sử dụng, chẳng hạn như 3G, 4G, 5G hoặc Wi-Fi.
    - Sự khác biệt về băng thông (bandwidth) giữa các thiết bị khách (clients) gây ra sự không đồng đều trong việc gửi và nhận tham số mô hình từ Server.
- **The Straggler Problem (Vấn đề thiết bị chậm chạp):** 
    - Trong mô hình tương tác giữa Server trung tâm và các thiết bị khách, các vòng huấn luyện toàn cục thường bị nghẽn (**bottlenecked**) bởi thiết bị tham gia có tốc độ chậm nhất.
    - Ngay cả khi có nhiều thiết bị phản hồi nhanh, Server vẫn phải chờ đợi các thiết bị chậm hơn ("**Stragglers**") hoàn thành công việc của chúng trước khi có thể tổng hợp kết quả và chuyển sang vòng huấn luyện tiếp theo.
- **Synchronization Overhead (Chi phí đồng bộ hóa):** 
    - Thời gian chờ đợi các "**Stragglers**" tạo ra một khoản chi phí về mặt thời gian và tài nguyên, làm giảm hiệu quả tổng thể của hệ thống Federated Learning.
    - Việc quản lý sự đồng bộ giữa một nhóm thiết bị có tốc độ phản hồi không đồng nhất (một số nhanh, một số chậm) là một bài toán phức tạp trong việc tối ưu hóa hiệu suất huấn luyện.

### Client Availability

Thách thức về sự sẵn có của các **client** là một vấn đề trọng tâm trong Federated Learning, đặc biệt khi các thiết bị tham gia thường là thiết bị cá nhân như điện thoại di động, máy tính bảng hoặc laptop.

#### Các yếu tố ảnh hưởng đến sự sẵn có

- **Dynamic Connectivity (Kết nối động)**: Các **client** thường xuyên tham gia hoặc rời khỏi mạng một cách không thể dự đoán trước, không có lịch trình cố định.
- **Unreliable Conditions (Điều kiện không đáng tin cậy)**: Quá trình huấn luyện có thể bị gián đoạn bởi nhiều yếu tố khách quan:
    - **Devices out of battery**: Thiết bị hết pin trong quá trình xử lý.
    - **Devices crashes**: Lỗi hệ thống hoặc xung đột phần mềm trên thiết bị.
    - **Connection loss / Out of Wi-Fi range**: Thiết bị di chuyển ra khỏi phạm vi phủ sóng Wi-Fi hoặc mất tín hiệu mạng.
- **Asynchronous Participation (Tham gia không đồng bộ)**: Không phải tất cả các **client** được hệ thống lựa chọn đều có khả năng hoàn thành nhiệm vụ huấn luyện cục bộ (local training) của họ để gửi kết quả về **server**.

#### Mô tả hệ thống và các tình huống lỗi

Trong mô hình Federated Learning, một **server** trung tâm kết nối với nhiều thiết bị **client** thông qua các đường truyền mạng không ổn định. Các kịch bản lỗi phổ biến bao gồm:

1. **Devices go offline mid-training**: Thiết bị ngoại tuyến ngay khi đang thực hiện các bước tính toán huấn luyện.
2. **System crashes**: Biểu tượng lỗi hệ thống (màn hình lỗi) xuất hiện trên thiết bị của người dùng.
3. **Battery depletion**: Thiết bị cạn kiệt năng lượng, buộc phải dừng mọi hoạt động xử lý.
4. **Signal warning**: Mất kết nối hoàn toàn với cột phát sóng hoặc tín hiệu mạng chập chờn.

#### Hệ quả 

Sự thiếu ổn định của các thiết bị dẫn đến tỷ lệ **client dropout** cao. Điều này gây ra những tác động tiêu cực trực tiếp đến mô hình chung:

- **Biased updates**: Các bản cập nhật gửi về **server** có thể bị thiên kiến (sai lệch) do chỉ thu thập được dữ liệu từ một nhóm các **client** có kết nối ổn định nhất, thay vì đại diện cho toàn bộ tập người dùng.
- **Slower convergence**: Quá trình tổng hợp dữ liệu tại **server** không đầy đủ, khiến mô hình toàn cục đạt đến trạng thái hội tụ (**convergence**) chậm hơn so với dự kiến.

### Khắc phục System Heterogeneity - Client Availability

Trong Federated Learning, sự không đồng nhất về hệ thống (System Heterogeneity) và tính sẵn có của các thiết bị (Client Availability) là những thách thức lớn. Để giải quyết vấn đề này, hai chiến lược chính được đề xuất:

#### 1. Asynchronous Federated Learning (Federated Learning bất đồng bộ)

Cơ chế này tập trung vào việc tối ưu hóa thời gian huấn luyện bằng cách thay đổi cách thức tương tác giữa server và client:

- **Nguyên lý**: Server trung tâm sẽ cập nhật **model** toàn cục ngay lập tức khi nhận được dữ liệu cập nhật từ bất kỳ client nào, thay vì phải chờ đợi tất cả các client trong một vòng huấn luyện hoàn tất.
- **So sánh cơ chế Synchronous (Đồng bộ) và Asynchronous (Bất đồng bộ)**:
    - **Synchronous**: Quá trình huấn luyện được chia theo các **round** nghiêm ngặt. Các client hoàn thành sớm (ví dụ: Client  C) sẽ rơi vào trạng thái chờ (**Idle**) để đợi các client chậm hơn (stragglers - ví dụ: Client A, B, D) hoàn thành việc tính toán (**Computing update**) và truyền tải (**Transmitting update**). Điều này gây lãng phí tài nguyên và kéo dài thời gian hội tụ.
    - **Asynchronous**: Các client không cần chờ đợi nhau. Ngay sau khi kết thúc việc truyền tải dữ liệu lên server, client có thể bắt đầu ngay một chu kỳ tính toán mới. Cơ chế này giúp loại bỏ tình trạng trì trệ do các thiết bị yếu gây ra, từ đó tăng tốc độ hội tụ tổng thể của hệ thống.

#### 2. (System-aware) Client Selection (Lựa chọn Client dựa trên hệ thống)

Thay vì lựa chọn ngẫu nhiên các thiết bị tham gia, hệ thống thực hiện lựa chọn một cách chủ động và thông minh:

- **Nguyên lý**: Chủ động lựa chọn các thiết bị tham gia vào một vòng huấn luyện dựa trên khả năng hệ thống cục bộ của chúng trong thời gian thực.
- **Cơ chế hoạt động**:
    - Server trung tâm kết nối với đa dạng các thiết bị như Laptop, Desktop PC, Smartphone, Tablet.
    - Hệ thống sẽ đánh giá các thông số tài nguyên thực tế của từng thiết bị bao gồm:
        - Hiệu suất **CPU**.
        - Dung lượng pin.
        - Băng thông mạng.
    - Chỉ những thiết bị có đủ khả năng đáp ứng yêu cầu tại thời điểm đó mới được chọn vào nhóm tham gia huấn luyện (được đánh dấu bằng khung lựa chọn). Việc này đảm bảo quá trình huấn luyện diễn ra ổn định, hiệu quả và tránh tình trạng ngắt quãng do thiết bị hết pin hoặc mất kết nối giữa chừng.

### Scalability

Thách thức lớn nhất được đề cập là **Scalability** (Khả năng mở rộng). Tổng chi phí truyền thông và tính toán trong hệ thống được biểu diễn qua công thức:

$$T * \Sigma * |\mathcal{S}| * 2$$

Trong đó:
- $T$: Số vòng huấn luyện (training rounds).
- $\Sigma$: Kích thước mô hình (model size).
- $|\mathcal{S}|$: Số lượng client tham gia.
- Hệ số $2$ đại diện cho quá trình truyền tải hai chiều (tải lên và tải xuống) giữa client và server.

Để giải quyết vấn đề này, ba kỹ thuật chính được áp dụng:

#### 1. Quantization
**Quantization** là quá trình ánh xạ các giá trị đầu vào từ một tập hợp lớn, liên tục hoặc có độ chính xác cao sang một tập hợp nhỏ hơn, rời rạc hoặc có độ chính xác thấp hơn.
- **Cơ chế**: Ví dụ, dữ liệu ở định dạng **FP32** (biểu diễn bằng đồ thị phân phối chuẩn hình chuông) được ánh xạ xuống các ô vuông rời rạc nhãn **INT4**.

#### 2. Partial Training
**Partial Training** cho phép mỗi client huấn luyện một sub-model nhỏ hơn được trích xuất từ global server model lớn ban đầu.
- **Cấu trúc**: Giả sử Global Server Model là một mạng neural đầy đủ với 5 node ở hidden layer được ký hiệu là $\{a, b, c, d, e\}$.
- **Quá trình phân tách qua các Round**:
    - **Round $j$**: Mô hình được tách thành các sub-models khác nhau tùy theo năng lực thiết bị. Ví dụ: **Small-capacity Client Model** (chứa các node $\{b, c, d\}$) và **Large-capacity Client Model** (chứa các node $\{a, c, d, e\}$).
    - **Round $j+1$**: Tiếp tục thay đổi cấu hình trích xuất, ví dụ một bên chứa $\{a, d, e\}$ và một bên chứa $\{a, b, c, e\}$.
- **Lợi ích**: Phương pháp này cho phép các thiết bị có cấu hình phần cứng khác nhau (từ yếu đến mạnh) đều có thể tham gia huấn luyện các phần của mô hình tổng thể.

#### 3. LoRA (Low-Rank Adaptation)
**LoRA** là một kỹ thuật fine-tuning hiệu quả về tham số (parameter-efficient), giúp thích ứng các mô hình machine learning tiền huấn luyện (pre-trained) lớn bằng cách đóng băng các **weight** gốc và tiêm các ma trận low-rank nhỏ, có thể huấn luyện được vào các **transformer layer**.

**Kiến trúc của LoRA:**
- Luồng dữ liệu bắt đầu từ input $x$ có kích thước $d$.
- Luồng dữ liệu được chia làm hai nhánh song song:
    - **Nhánh trái**: Đi qua khối **Pretrained Weights** $W \in \mathbb{R}^{d \times d}$. Khối này được giữ cố định (frozen), không cập nhật trong quá trình huấn luyện.
    - **Nhánh phải**: Đi qua hai ma trận low-rank chồng lên nhau để tạo ra sự thay đổi trọng số $\Delta W$. Ma trận phía dưới là $A$ được khởi tạo theo phân phối chuẩn $A = \mathcal{N}(0, \sigma^2)$, và ma trận phía trên là $B$ được khởi tạo bằng $0$. Kích thước cổ chai (bottleneck) của hai ma trận này được ký hiệu là $r$.
- **Kết quả**: Đầu ra $h$ là tổng của hai nhánh ($h = Wx + BAx$). Kỹ thuật này giúp giảm đáng kể tài nguyên cần thiết vì chỉ cần huấn luyện một lượng nhỏ tham số bổ sung thay vì toàn bộ mô hình khổng lồ.

### Privacy & Security

Trong kiến trúc hệ thống, **Single Point of Failure (SPOF)** được định nghĩa là một phần của hệ thống mà nếu nó gặp lỗi, toàn bộ hệ thống sẽ ngừng hoạt động. Trong mô hình Federated Learning tập trung, Central Server đóng vai trò là một **SPOF**, gây ra nhiều rủi ro nghiêm trọng:

- **Service Interruption (Gián đoạn dịch vụ):** Nếu server trung tâm bị sập do lỗi phần cứng hoặc mất điện, toàn bộ quá trình huấn luyện toàn cục sẽ bị dừng lại ngay lập tức.
- **Communication Bottleneck (Nghẽn cổ chai truyền thông):** Khi số lượng **client** tăng lên hàng triệu, băng thông của server trở thành một yếu tố hạn chế, gây khó khăn cho việc truyền tải dữ liệu từ tất cả các nguồn.
- **Resource Exhaustion (Cạn kiệt tài nguyên):** Tải tính toán cao trong giai đoạn **aggregation** (tổng hợp mô hình) có thể dẫn đến sự mất ổn định của hệ thống.
- **Network Partitioning (Chia cắt mạng):** Nếu kết nối tới server bị mất, các **client** không thể đóng góp vào mô hình chung, ngay cả khi họ đang sở hữu dữ liệu cục bộ chất lượng cao.

#### Sơ đồ luồng kết nối giữa **client** và server minh họa rõ nét điểm yếu này:

1. **Application clients (người dùng cuối):** Các icon người dùng cho thấy họ không thể kết nối hoặc bị ảnh hưởng trực tiếp khi hệ thống gặp lỗi.
2. **Internet router:** Đóng vai trò trung gian, nếu router gặp sự cố, luồng dữ liệu cũng sẽ bị chặn đứng.
3. **Load balancer:** Đây là thiết bị nhận dữ liệu từ router để phân phối cho các server. Trong kiến trúc này, **Load balancer** được xác định là **Single point of failure (SPOF)**. Khi thiết bị này gặp sự cố (được đánh dấu bằng dấu 'X' đỏ), toàn bộ luồng trao đổi dữ liệu giữa **client** và server bị ngắt quãng hoàn toàn.
4. **Application servers:** Mặc dù các cụm server phía sau có thể vẫn đang hoạt động bình thường (hiển thị trạng thái xanh), nhưng chúng sẽ bị cô lập hoàn toàn và không thể tiếp nhận đóng góp từ phía người dùng do sự cố tại điểm trung gian.

Sự phụ thuộc vào kiến trúc tập trung khiến Central Server trở thành "điểm yếu duy nhất". Nếu thành phần trung tâm này hoặc các thiết bị hạ tầng quan trọng như **Load balancer** gặp sự cố, toàn bộ hệ thống Federated Learning sẽ bị tê liệt, bất kể các thành phần khác có hoạt động tốt hay không.

#### Topology (Server-Client)

Kiến trúc Client-Server là một mô hình mạng truyền thống, hoạt động bằng cách cho phép nhiều **client** giao tiếp với một **server** trung tâm. Trong cấu trúc này, **server** đóng vai trò là thực thể cung cấp các dịch vụ khác nhau cho các nút mạng thành viên.

Các ví dụ phổ biến về dịch vụ sử dụng mô hình này bao gồm:
- **Web services**: Sử dụng máy chủ Web trung tâm (central Web server).
- **FTP services**: Sử dụng máy chủ FTP trung tâm (central FTP server).
- **DHCP services**: Sử dụng máy chủ DHCP trung tâm (central DHCP server).

##### Sơ đồ cấu trúc liên kết (Topology)
Mô hình Client-Server được tổ chức theo dạng hình sao với các đặc điểm kết nối sau:
- Một **server** đóng vai trò là nút gốc đặt tại trung tâm của hệ thống.
- Các thiết bị đầu cuối bao gồm **Client 1**, **Client 2**, và **Client 3** được bố trí xung quanh.
- Các mũi tên hai chiều thiết lập kết nối trực tiếp giữa **server** và từng **client**, cho phép luồng giao tiếp song phương diễn ra giữa thực thể trung tâm và các nút mạng thành viên.

##### Ưu điểm và Nhược điểm

| Đặc điểm | Chi tiết |
| :--- | :--- |
| **Ưu điểm** | - **Tăng hiệu suất**: Do **server** trung tâm thường được thiết kế với cấu hình mạnh mẽ để hoạt động ở tốc độ cao.<br>- **Điểm quản trị duy nhất**: Mọi tài nguyên và dịch vụ đều được lưu trữ tập trung tại **server**, giúp việc quản lý ứng dụng trở nên thuận tiện hơn. |
| **Nhược điểm** | - **Chi phí đầu tư cao**: Đòi hỏi nguồn kinh phí lớn để thiết lập hạ tầng ở phía **server side**.<br>- **Chi phí bảo trì cao**: Việc duy trì và vận hành hệ thống máy chủ tiêu tốn nhiều nguồn lực.<br>- **Độ phức tạp**: Hệ thống trở nên cực kỳ phức tạp khi mở rộng trong các môi trường có quy mô lớn. |

Trong bối cảnh nghiên cứu về Federated Learning, việc xem xét mô hình Client-Server truyền thống là rất quan trọng. Đây là nền tảng để so sánh các thách thức về quyền riêng tư và bảo mật dữ liệu so với các cấu trúc liên kết mạng khác, đồng thời đánh giá khả năng mở rộng của hệ thống khi số lượng các nút mạng gia tăng.

#### Topology trong Decentralized Federated Learning

Trong hệ thống Decentralized Federated Learning (DFL), do không có server trung tâm để điều phối, cách thức các client kết nối với nhau đóng vai trò quyết định đến việc truyền tải thông tin, tính bảo mật và hiệu quả của quá trình hội tụ **model**.

- Mỗi nút (**node**) trong mạng đại diện cho một client.
- Các cấu trúc liên kết (**topologies**) khác nhau xác định cách các client trao đổi các bản cập nhật **model**.
- Mỗi loại cấu trúc liên kết sẽ có những ưu và nhược điểm riêng về băng thông, độ trễ và khả năng chống chịu lỗi.

##### Các loại cấu trúc liên kết tiêu biểu

Dưới đây là các dạng cấu trúc mạng phổ biến được sử dụng để kết nối các client (ký hiệu từ C1, C2, ...):

1. **Fully Connected (Complete Graph)**:
   - Đây là mô hình đồ thị đầy đủ.
   - Mỗi client đều có kết nối trực tiếp với tất cả các client còn lại trong mạng (ví dụ: mạng gồm 6 client từ C1-C6).

2. **K-Connected (3-Regular Example)**:
   - Mỗi client được kết nối với đúng $K$ client khác.
   - Ví dụ: Cấu trúc 3-Regular gồm 8 client (C1-C8), trong đó mỗi client có liên kết với đúng 3 client khác, tạo thành một mạng lưới có tính đối xứng cao.

3. **Mesh Topology**:
   - Cấu trúc lưới không đồng nhất.
   - Các client (ví dụ: C1-C8) kết nối với nhau tạo thành các hình tam giác và tứ giác đan xen.

4. **Ring Topology**:
   - Cấu trúc vòng khép kín.
   - Các client (ví dụ: C1-C8) kết nối nối tiếp nhau; mỗi client chỉ thiết lập kết nối với hai client lân cận trực tiếp.

5. **Grid / Torus Topology**:
   - Cấu trúc lưới tọa độ hoặc Torus.
   - Các client được sắp xếp theo dạng lưới (ví dụ: lưới 3x3 gồm 9 client C1-C9). Mỗi client kết nối với các client hàng xóm theo chiều ngang và chiều dọc.

6. **Random Graph (Erdos-Renyi)**:
   - Đồ thị ngẫu nhiên.
   - Các đường kết nối được thiết lập ngẫu nhiên giữa các cặp nút trong mạng (ví dụ: mạng gồm 10 client từ C1-C10).

### Data/Statistical Heterogeneity

**Data/Statistical Heterogeneity** (Tính không đồng nhất về dữ liệu/thống kê) là một trong những thách thức cốt lõi trong Federated Learning. Khái niệm này chỉ việc phân phối dữ liệu của các client tham gia vào hệ thống không giống nhau.

#### Phân biệt IID và Non-IID

Trong các bài toán học máy truyền thống, dữ liệu thường được giả định là IID, nhưng thực tế trong Federated Learning, dữ liệu thường mang tính chất Non-IID.

1. **IID (Independent and Identically Distributed - Độc lập và Phân phối giống hệt nhau)**
   - **Independent (Độc lập):** Việc biết một mẫu dữ liệu không cung cấp thông tin về mẫu tiếp theo; không tồn tại sự tương quan hay quan hệ nhân quả giữa các điểm dữ liệu.
   - **Identically (Phân phối giống hệt nhau):** Mọi mẫu dữ liệu đều được rút ra từ cùng một phân phối. Trên tất cả các client, các yếu tố sau là đồng nhất:
     - Không gian nhãn (label space).
     - Tỷ lệ các lớp (class ratios).
     - Phạm vi đặc trưng (feature range).
   - **Kịch bản IID lý tưởng:** Trong mô hình lý tưởng, các client (ví dụ: các điện thoại thông minh) đều có dữ liệu hoàn toàn giống hệt nhau về cấu trúc và số lượng (ví dụ: mỗi client đều có 3 mẫu Class A, 3 mẫu Class B và 3 mẫu Class C). Khi đó, các mũi tên cập nhật từ client sẽ hội tụ một cách thuận lợi về **Global Model** ở trung tâm.

2. **Non-IID (Không độc lập và không phân phối giống hệt nhau)**
   - **Not Independent:** Các mẫu dữ liệu có sự tương quan nhất định trong nội bộ của một client.
   - **Not Identically:** Các client có các phân phối dữ liệu khác nhau. Điều này dẫn đến sự khác biệt về không gian nhãn, tỷ lệ các lớp và phạm vi đặc trưng giữa các client.

#### Các loại lệch dữ liệu (Data Skew) trong thực tế

Sự không đồng nhất về dữ liệu (Data Heterogeneity) thường được thể hiện qua ba kịch bản lệch dữ liệu chính:

- **Quantity Skew (Lệch về số lượng):** Các thiết bị khác nhau đóng góp lượng dữ liệu không đồng đều. Ví dụ: một Smartphone có thể có rất nhiều dữ liệu, trong khi một IoT sensor node hoặc Laptop lại có lượng dữ liệu ít hơn đáng kể.
- **Label Distribution Skew (Lệch về phân phối nhãn):** Tỷ lệ các lớp giữa các client không đồng nhất. Một client có thể chỉ sở hữu dữ liệu của một nhãn duy nhất (ví dụ: chỉ có nhãn màu đỏ, trong khi client khác chỉ có nhãn màu xanh hoặc vàng). Điều này khiến biểu đồ phân chia mục tiêu chung (**Global Target**) bị lệch nghiêm trọng.
- **Feature Distribution Skew (Lệch về phân phối đặc trưng):** Ngay cả khi các client cùng huấn luyện một mục tiêu (ví dụ: bộ phát hiện mèo - "Cat detector"), đặc điểm dữ liệu vẫn có thể khác biệt. Mỗi client có thể sở hữu hình ảnh mèo với các đặc điểm, góc chụp, ánh sáng hoặc bối cảnh khác nhau.

#### Tác động đến hệ thống

Sự không đồng nhất về dữ liệu gây ra những khó khăn và xung đột lớn trong quá trình tổng hợp mô hình. Biểu hiện cụ thể là sự xuất hiện của các xung đột khi kết hợp các cập nhật từ client về **Global Model**. So với các phương pháp học máy tập trung truyền thống, tính chất Non-IID làm cho quá trình huấn luyện trở nên phức tạp hơn, đòi hỏi các thuật toán tối ưu hóa mạnh mẽ hơn và khiến mô hình khó đạt được sự hội tụ.

### Statistical Heterogeneity (Tính không đồng nhất về thống kê)

Trong Federated Learning, một trong những thách thức lớn nhất là tính không đồng nhất về thống kê giữa các client. Điều này dẫn đến các **Skew Patterns** — sự sai lệch hệ thống trong phân phối dữ liệu giữa các client phi tập trung, vi phạm giả định **IID** (Independent and Identically Distributed - Độc lập và phân phối đồng nhất) thường thấy trong học máy truyền thống.

Việc hiểu rõ các mô hình skew này là cực kỳ quan trọng để thiết kế các thuật toán Federated Learning có khả năng hội tụ và đạt độ chính xác cao trên dữ liệu thực tế. Có năm mô hình skew chính trong tính không đồng nhất của dữ liệu:

- **Distribution skew**: 
    - Đây là trường hợp phổ biến khi các client có tỷ lệ nhãn khác nhau. 
    - Ví dụ: Client $i$ và Client $j$ cùng sở hữu các chữ số viết tay nhưng tần suất xuất hiện của mỗi con số là khác nhau. 
    - Trong thực nghiệm, sự phân bổ này thường được minh họa qua phân phối Dirichlet với các giá trị $\beta$ khác nhau (như $\beta = 0.3, \beta = 0.5, \beta = 1$) để thể hiện các mức độ lệch dữ liệu.

- **Label Skew**: 
    - Xảy ra khi phân phối của các nhãn $P(y)$ thay đổi giữa các client. 
    - Trong mô hình này, mỗi client có thể chỉ sở hữu một tập con các nhãn nhất định thay vì toàn bộ danh mục nhãn của hệ thống. 
    - Ví dụ: Client $i$ chỉ có các nhãn {1, 3, 9}, Client $j$ có {2, 3, 7}, và Client $k$ có {6, 8, 0}.

- **Feature Skew**: 
    - Xảy ra khi phân phối biên (marginal distribution) $P(x)$ thay đổi giữa các client, ngay cả khi phân phối có điều kiện $P(y|x)$ vẫn giữ nguyên. 
    - Ví dụ: Khi xét cùng một nhãn là "Dog", nhưng hình ảnh (feature) tại mỗi client lại khác nhau hoàn toàn do khác biệt về giống chó, bối cảnh môi trường hoặc điều kiện ánh sáng giữa Client $i, j, k$.

- **Quantity Skew**: 
    - Sự mất cân bằng về khối lượng dữ liệu giữa các client. 
    - Biểu đồ số lượng mẫu (number of samples) sẽ cho thấy sự chênh lệch lớn: một số client (như Client $i$) có lượng dữ liệu rất lớn, trong khi các client khác (như Client $j$) lại có rất ít dữ liệu.

- **Quality Skew**: 
    - Sự khác biệt về chất lượng dữ liệu giữa các thiết bị. 
    - Các dạng nhiễu phổ biến bao gồm:
        - **Label noise skew**: Các nhãn dữ liệu bị gán sai một cách hệ thống tại một số client.
        - **Sample noise skew**: Bản thân dữ liệu đầu vào bị nhiễu (ví dụ: hình ảnh bị nhiễu hạt, mờ, chất lượng thấp).

Dữ liệu thực tế từ các thiết bị người dùng cuối hiếm khi tuân theo giả định IID. Do đó, việc giải quyết 5 dạng sai lệch trên là chìa khóa để đảm bảo mô hình Federated Learning hoạt động hiệu quả trên môi trường phi tập trung.

### Data Heterogeneity

Thách thức chính trong Federated Learning liên quan đến **Data Heterogeneity** (Tính không đồng nhất của dữ liệu), bao gồm hai hiện tượng quan trọng: **Client drift** và **Concept drift**.

#### 1. Client drift
**Client drift** xảy ra khi các **local model** phân kỳ khỏi **global optimum** vì mỗi client thực hiện tối ưu hóa dựa trên phân phối dữ liệu riêng biệt của nó. 
- Các **gradient** từ các client khác nhau trỏ theo các hướng xung đột nhau.
- Các hướng xung đột này kéo **global model** ra xa khỏi mục tiêu chung thực sự (**true global objective**), gây khó khăn cho việc hội tụ.

**So sánh cơ chế cập nhật giữa dữ liệu IID và Non-IID:**
- **Trường hợp IID (Independent and Identically Distributed):** Các **local update** (mũi tên xanh dương) từ trạng thái hiện tại $w$ đến các trạng thái tiếp theo $w_1^{t+1}$ và $w_2^{t+1}$ đều có xu hướng hướng về phía **global model optimal** ($w^*$).
- **Trường hợp Non-IID:** Các **local update** lại hướng về các **local model optimal** riêng biệt ($w_1^*$ và $w_2^*$) của từng client. Do đó, **global update** tổng hợp (mũi tên xanh lá) không thể tiếp cận hiệu quả điểm tối ưu chung của toàn hệ thống ($w^*$).

#### 2. Concept drift
**Concept drift** xảy ra khi phân phối dữ liệu cơ sở thay đổi theo thời gian. Khi thế giới thực tiến hóa, mối quan hệ giữa đầu vào và nhãn $P(y|x)$ bị thay đổi, khiến các mô hình (patterns) đã học được trước đó trở nên lỗi thời hoặc không còn chính xác.

**Minh họa sự thay đổi của Concept Drift:**
- **Tại thời điểm $t$:** Dữ liệu (biểu diễn bằng hình tròn và tam giác xanh) được phân chia rõ ràng bởi một đường ranh giới quyết định (decision boundary) màu xanh.
- **Tại thời điểm $t+1$:** Xuất hiện sự thay đổi $P_t(y|X)$ (**drift**) trong khi $P_t(X)$ vẫn giữ nguyên. Dữ liệu mới (màu đỏ) có phân phối khác, dẫn đến đường ranh giới quyết định ban đầu không còn phù hợp và phải dịch chuyển sang vị trí mới (đường màu đỏ).

Cả **Client drift** và **Concept drift** đều là những rào cản lớn, làm chệch hướng mô hình và gây khó khăn trong việc duy trì độ chính xác ổn định cho **global model**.


### Model Heterogeneity

**Model heterogeneity** là một mô hình (paradigm) trong đó các client tham gia sử dụng các kiến trúc local model đa dạng, được tùy chỉnh theo các hạn chế phần cứng (như CPU, RAM, bộ nhớ) và yêu cầu quyền riêng tư cụ thể của họ, thay vì sử dụng một cấu trúc global đồng nhất. 

Thách thức này đòi hỏi việc học hỏi kiến thức từ những người khác mà không cần chia sẻ dữ liệu riêng tư hoặc tiết lộ thông tin chi tiết về cấu trúc local model.

#### Quy trình Federated Learning với sự không đồng nhất về mô hình

Sự không đồng nhất về mô hình trong hệ thống Federated Learning được thể hiện qua các giai đoạn sau:

1.  **Local Learning (Local update):**
    -   **Các Client:** Ví dụ là các bệnh viện khác nhau đóng vai trò là các đơn vị tham gia huấn luyện.
    -   **Heterogeneous Dataset:** Mỗi client sở hữu một tập dữ liệu riêng biệt với kích thước khác nhau.
    -   **Heterogeneous model:** Mỗi client sử dụng một kiến trúc mạng thần kinh khác nhau tùy theo năng lực thiết bị:
 
2.  **Quá trình truyền tải thông tin:**
    -   Một **Server** trung tâm đóng vai trò điều phối.
    -   Các luồng thông tin được truyền tải hai chiều giữa các client và **Server** để thực hiện quá trình tổng hợp tri thức.

3.  **Global Learning (Global update):**
    -   **Server** thực hiện tổng hợp thông tin từ tất cả các loại kiến trúc model khác nhau từ phía các client.
    -   Mục tiêu cuối cùng là tối ưu hóa kết quả phân loại cho các lớp dữ liệu (ví dụ: **Class 0**, **Class 1**, và **Class 2**).

Tóm lại, hệ thống Federated Learning cần phải có khả năng tổng hợp tri thức từ các cấu trúc mạng khác nhau mà vẫn đảm bảo tính riêng tư của dữ liệu và không yêu cầu các client phải công khai chi tiết cấu trúc model nội bộ của mình.

### Knowledge-Distillation-based Federated Learning

**Knowledge-Distillation-based Federated Learning** là một lớp các thuật toán Federated Learning tích hợp một loại **loss** gọi là **knowledge distillation loss** vào một thời điểm nào đó trong quy trình huấn luyện (pipeline), có thể được thực hiện ở phía client, phía server, hoặc cả hai.

Mục đích chính của phương pháp này là để truyền đạt tri thức thông qua đầu ra của mô hình (**soft targets**, **logits**) hoặc các biểu diễn trung gian (**intermediate representations**), thay vì chỉ dựa hoàn toàn vào việc trung bình hóa tham số (**parameter averaging**) trực tiếp. Điều này giúp giải quyết thách thức về sự không đồng nhất của mô hình (Model Heterogeneity), cho phép các mô hình có cấu trúc khác nhau vẫn có thể trao đổi tri thức.

Nói dễ hiểu: trong FL bình thường, server hay **average weights/parameters** của các client. Nhưng nếu mỗi client dùng model khác kiến trúc, ví dụ client A dùng CNN nhỏ, client B dùng ResNet, client C dùng MLP, thì **không thể average trực tiếp parameters** vì shape khác nhau. Vì vậy người ta dùng **knowledge distillation**: truyền “kiến thức” qua **output của model** thay vì truyền trực tiếp weights.

##### Ví dụ
- Trong hình:

  - $D_k$ = dữ liệu riêng của client k 
  - $x_i$ = input sample
  - $y_i$ = nhãn thật của sample đó

Client lấy cùng một input $x_i$ đưa vào hai model:

- Model trên: $W_t$
- Model dưới: $W^k_{t+1}$

Có thể hiểu đơn giản:

- $W_t$ = model global / teacher model ở round hiện tại
- $W^k_{t+1}$ = model local của client k đang được train


Cả hai model cho ra output, rồi đi qua **softmax** để biến output thành phân phối xác suất.

Ví dụ với bài toán phân loại ảnh:

```text
Teacher model W_t dự đoán:
cat = 0.7, dog = 0.2, fox = 0.1

Student/local model W^k_{t+1} dự đoán:
cat = 0.5, dog = 0.4, fox = 0.1
```

Sau đó có hai loại loss:

1. **CE loss**

`CE loss` là loss học từ **nhãn thật** $y_i$.

Ví dụ nhãn thật là:

$y_i$ = cat

Thì model local phải học để dự đoán `cat` đúng hơn.

Đây là cách train supervised bình thường:

Prediction của local model vs label thật $y_i$

2. **KD loss**

`KD loss` là loss học từ **output mềm của teacher model**.

Tức là local model không chỉ học “ảnh này là cat”, mà còn học cách teacher nhìn bài toán:

```text
cat 0.7
dog 0.2
fox 0.1
```

Thông tin này giàu hơn nhãn cứng `cat`, vì nó cho biết các class khác giống/khác nhau thế nào.

KD loss ép output của local model gần với output của teacher model:

Output của $W^k_{t+1}$ ≈ Output của $W_t$

### Local loss cuối cùng

Hai loss được cộng lại:

```text
local loss = CE loss + KD loss
```

Hoặc thường viết đầy đủ hơn:

```text
local loss = CE loss + λ * KD loss
```

Trong đó `λ` điều chỉnh mức độ tin vào knowledge distillation.

Ý nghĩa toàn bộ sơ đồ là:

```text
Client train model local bằng dữ liệu riêng
Nhưng không chỉ học từ nhãn thật
Mà còn học từ output/knowledge của global hoặc teacher model
```

Điểm quan trọng nhất:

**KD-based FL không cần các client có cùng kiến trúc model.**
Miễn là các model có thể cho output trên cùng task, ví dụ cùng số class, thì vẫn có thể truyền knowledge qua logits/softmax output.

So với FedAvg thường:

| Cách        | Truyền cái gì?                | Cần model giống kiến trúc không? |
| ----------- | ----------------------------- | -------------------------------- |
| FedAvg      | Parameters / weights          | Có                               |
| KD-based FL | Output / logits / soft labels | Không nhất thiết                 |

### Agnostic Personalized Federated Learning

**Agnostic Personalized Federated Learning** là một kịch bản mà bất kỳ thiết bị tham gia cục bộ nào đến từ các **domain** đa dạng, với các sơ đồ gán nhãn (**labeling schemes**) cá nhân hóa riêng của nó, đều có thể phối hợp học tập để mang lại lợi ích cho nhau. Tuy nhiên, mô hình này đối mặt với hai thách thức lớn về sự không đồng nhất:

#### 1. Label Heterogeneous (Dị biệt nhãn)

Thách thức này phát sinh khi các client sử dụng cùng một bộ chỉ số nhãn nhưng ý nghĩa thực tế của chúng lại khác nhau:

- Mỗi client sử dụng một hoán vị nhãn (**permutation of labels**) tùy ý và riêng tư.
- Ngay cả các **layer** **feature-extraction** được chia sẻ cũng bị phân kỳ. Nguyên nhân là do các **gradient** từ các **classifier** không tương thích sẽ **backpropagate** các tín hiệu xung đột với nhau.

- **Sơ đồ minh họa:** Một Server trung tâm kết nối hai chiều với nhiều Client (Client 1, Client 2, ..., Client K). Mặc dù các client đều có bảng nhãn từ 1 đến 10, nhưng ý nghĩa của chúng bị hoán vị

- **Hệ quả:** Cùng một chỉ số nhãn nhưng đại diện cho các đối tượng khác nhau ở mỗi client, gây ra xung đột nghiêm trọng khi huấn luyện các **layer** chung.

#### 2. Domain Heterogeneous (Dị biệt domain)

Thách thức này xảy ra khi các client hoạt động trên các lĩnh vực hoàn toàn khác nhau, dẫn đến sự khác biệt sâu sắc về cấu trúc dữ liệu:

- Các client huấn luyện trên các tập dữ liệu hoàn toàn rời rạc, không có sự chồng chéo về thực thể (**instance**) hoặc lớp (**class**).
- Số lượng **class** mục tiêu có thể khác nhau giữa các client.
- Quá trình huấn luyện cũng gặp phải tình trạng **backpropagate** các tín hiệu xung đột.
- **Sơ đồ minh họa:** Server trung tâm kết nối với các Client thuộc các lĩnh vực không có sự giao thoa

- **Hệ quả:** Các client không có sự giao thoa về dữ liệu hay số lượng lớp mục tiêu, khiến việc xây dựng một global model thống nhất trở nên cực kỳ khó khăn.

Trong các hệ thống thực tế, Federated Learning cần phải giải quyết được sự không đồng nhất không chỉ ở mức phân phối dữ liệu mà còn ở mức cấu trúc nhãn và lĩnh vực dữ liệu để các client có thể hỗ trợ lẫn nhau một cách hiệu quả.

### Agnostic Personalized Federated Learning

**LoRA-based Federated Learning** là một nhóm các phương pháp trong đó các tham số mô hình của mỗi client được phân rã thành một cấu trúc low-rank. Cấu trúc này thường được phân tích thành hai hoặc nhiều thành phần có số chiều thấp hơn, nhằm cho phép thực hiện aggregation có chọn lọc trong hệ thống federated learning.

#### Cơ chế LoRA-based Federated Learning và Similarity Matching

Quy trình này tập trung vào việc xử lý các thành phần low-rank thay vì toàn bộ mô hình, bao gồm các bước và thành phần chính sau:

1.  **Cấu trúc tại Client**:
    - Hệ thống bao gồm một danh sách các client từ Client 1, Client 2 đến Client K.
    - Mỗi client sở hữu một mạng nơ-ron và thực hiện khối chức năng phân rã tham số.

2.  **Phân rã tham số (Parameter Factorization)**:
    - Tại mỗi client $k$, các tham số được phân rã thành các thành phần như $u_{fk}$, $v_{fk}$, và $\mu_{fk}$.
    - Một thành phần cụ thể là vector $v_{fk}^{L-1}$ (ví dụ: $v_{f1}^{L-1}, v_{f2}^{L-1}, \dots, v_{fK}^{L-1}$) sẽ được trích xuất từ các client để gửi đi.

3.  **Không gian tham số đã phân tích nhân tử (Factorized Parameter Space)**:
    - Các vector trích xuất từ các client được ánh xạ vào một không gian tọa độ chung.

4.  **Khớp nối độ tương đồng (Similarity Matching)**:
    - Trong không gian này, các điểm đại diện cho mỗi client được so sánh với nhau để xác định mức độ tương đồng.
    - Việc aggregation được thực hiện dựa trên mức độ tương đồng giữa các thành phần low-rank của các client. Các trọng số aggregation (ví dụ: $0.1$, $0.25$, và $0.65$) được áp dụng để ưu tiên kết hợp thông tin từ các client có đặc điểm tương đồng.

#### Ý nghĩa và Hiệu quả

Phương pháp này là một kỹ thuật tiên tiến để giải quyết thách thức về tính cá nhân hóa (Personalized FL) với các ưu điểm:

- **Tối ưu hóa truyền tải**: Thay vì truyền tải và thực hiện aggregation toàn bộ **weight** của mô hình, phương pháp này chỉ tập trung vào các thành phần low-rank có số chiều thấp.
- **Aggregation có chọn lọc (Selective Aggregation)**: Cho phép server ưu tiên kết hợp thông tin từ các client tương đồng, giúp mô hình toàn cục hoặc mô hình cá nhân hóa đạt hiệu quả cao hơn.
- **Tiết kiệm tài nguyên**: Giảm thiểu chi phí tính toán và băng thông truyền tải dữ liệu giữa client và server.


# V. APPLICATIONS

## Google Keyboard (GBoard)

Google Keyboard (GBoard) là một ứng dụng thực tế tiêu biểu minh chứng cho việc áp dụng Federated Learning nhằm cân bằng giữa tính năng thông minh và quyền riêng tư của người dùng.

**Các tính năng cốt lõi (Core Features)**

GBoard cung cấp trải nghiệm nhập liệu tối ưu thông qua các tính năng:
- **Next-word prediction** (Dự đoán từ tiếp theo) và **Auto-correction** (Tự động sửa lỗi): Hệ thống hiển thị các từ gợi ý (ví dụ: "so much", "too", "and") ngay phía trên hàng phím chữ để tăng tốc độ soạn thảo.
- **Voice** (Giọng nói): Giao diện nhập liệu bằng giọng nói cho phép người dùng chuyển lời nói thành văn bản với biểu tượng micro và thông báo "Speak now".
- **Swipe** (Vuốt): Hỗ trợ thao tác vuốt (glide typing) qua các chữ cái trên bàn phím để tạo thành từ một cách liên tục.
- **Smart Suggestion** (Gợi ý thông minh): Đưa ra các gợi ý nội dung đa phương tiện, chẳng hạn như hình ảnh (GIF) "Typing dog", dựa trên nội dung hội thoại của người dùng.

**Quyền riêng tư theo thiết kế (Privacy by Design)**

Hệ thống được xây dựng dựa trên các nguyên tắc bảo mật nghiêm ngặt:
- **Local Learning**: Mọi thao tác phím nhấn và dữ liệu nhập liệu được xử lý trực tiếp trên thiết bị của người dùng để huấn luyện mô hình cục bộ.
- **Data Sovereignty**: Đảm bảo quyền chủ quyền dữ liệu, trong đó dữ liệu cá nhân của người dùng không bao giờ rời khỏi thiết bị.
- **Differential Privacy (DP)**: Áp dụng các kỹ thuật toán học để cung cấp các đảm bảo về quyền riêng tư, ngăn chặn việc rò rỉ thông tin cá nhân từ các bản cập nhật mô hình.

## Apple - Siri & iOS

Apple áp dụng Federated Learning vào Siri và hệ điều hành iOS nhằm giải quyết bài toán quyền riêng tư dữ liệu người dùng, đảm bảo các tính năng thông minh hoạt động hiệu quả mà không cần thu thập dữ liệu thô.

**Quyền riêng tư kỹ thuật (Technical Privacy):**
- **On-device Training:** Các model thực hiện học trực tiếp từ các tương tác và âm thanh cục bộ của người dùng ngay trên thiết bị.
- **Zero Leakage:** Đảm bảo âm thanh thô và văn bản không bao giờ rời khỏi thiết bị, tránh nguy cơ rò rẻ dữ liệu cá nhân.
- **Differential Privacy:** Nhiễu toán học được thêm vào các bản cập nhật cá nhân để bảo vệ thông tin trước khi gửi về máy chủ, đảm bảo không thể truy ngược lại danh tính người dùng.

**Ưu điểm (Advantages):**
- **Latency:** Phản hồi được thực hiện tức thì thông qua quá trình inference cục bộ trên thiết bị, không phụ thuộc vào tốc độ mạng.
- **Brand Value:** Chiến lược "Privacy by Design" (Quyền riêng tư ngay từ khâu thiết kế) là một yếu tố cốt lõi tạo nên sự khác biệt cho thương hiệu Apple.
- **Regulatory:** Khả năng tuân thủ các quy định pháp lý khắt khe được tích hợp sẵn, chẳng hạn như GDPR hoặc các yêu cầu về dữ liệu lưu trú tại địa phương (Local data residency).

**Quy trình Federated Speaker Recognition (Siri):**
Để cải thiện khả năng nhận diện giọng nói của Siri mà vẫn bảo mật, Apple sử dụng quy trình Federated Learning theo các bước sau:

1.  **Local Training (On-Device):** Các thiết bị iPhone thực hiện trích xuất các đặc trưng giọng nói (voice features). Lưu ý quan trọng: Không có âm thanh thô (raw audio) nào được lưu trữ trong quá trình này.
2.  **Encrypted Updates:** Các bản cập nhật model (model gradients) được mã hóa an toàn và gửi từ thiết bị cá nhân đến máy chủ (Server).
3.  **Secure Aggregation:** Tại Server, các bản cập nhật từ hàng triệu thiết bị khác nhau được tổng hợp lại (biểu tượng $\Sigma$) để cải thiện model chung mà không cần xem nội dung chi tiết của từng bản cập nhật đơn lẻ.
4.  **Distributed Global Model:** Model toàn cục đã được cập nhật (Updated Global Model) sau đó được phân phối ngược lại cho các thiết bị để hoàn tất chu trình, giúp Siri nhận diện giọng nói chính xác hơn trên mọi thiết bị của người dùng.

## Owkin - Healthcare/Cancer (Y tế/Ung thư)

**Lâm sàng (Nature Medicine, 2023):**
- **Dự đoán hóa trị TNBC:** Thực hiện nghiên cứu đa trung tâm tại 4 bệnh viện với quy mô hơn 650 bệnh nhân. Kết quả nghiên cứu này đã được đăng trên tạp chí **Nature Medicine** (Quyển 29, Số 8, tháng 8/2023).
- **Lần đầu tiên:** Triển khai mô hình **Federated Learning** cho hình ảnh giải phẫu bệnh (histopathology imaging) đa bệnh viện, đánh dấu một bước tiến quan trọng trong việc ứng dụng AI vào chẩn đoán hình ảnh y khoa quy mô lớn.

**Dược phẩm (Dự án MELLODDY):**
- **Liên minh:** Sự hợp tác của 10 tập đoàn dược phẩm lớn (bao gồm Sanofi, GSK, và các đơn vị khác).
- **Khám phá thuốc (Drug Discovery):** Cho phép các công ty thực hiện huấn luyện cộng tác để tìm kiếm các loại thuốc mới mà không làm lộ các thư viện cấu trúc hóa chất độc quyền của mỗi bên.

**Quyền riêng tư & Pháp lý:**
- **Tuân thủ:** Hệ thống đảm bảo tuân thủ hoàn toàn các quy định khắt khe về bảo mật dữ liệu y tế như GDPR và HIPAA.
- **Chủ quyền dữ liệu (Data Sovereignty):** Dữ liệu thô được giữ an toàn tuyệt đối sau tường lửa của bệnh viện. Trong quá trình huấn luyện, chỉ có các **weight** và **parameters** của mô hình được chia sẻ và tổng hợp, không có dữ liệu bệnh nhân nào bị chuyển ra ngoài.

**Tác động (Impact):**
- **Khả năng tổng quát hóa (Generalization):** Việc kết nối nhiều tổ chức giúp mô hình tiếp cận được các tập dữ liệu về những trường hợp bệnh hiếm gặp, từ đó cải thiện độ chính xác và khả năng ứng dụng thực tế.
- **Hiệu quả:** Đẩy nhanh tiến độ Nghiên cứu và Phát triển (R&D) trong khi vẫn duy trì các tiêu chuẩn đạo đức và bảo mật nghiêm ngặt nhất.

**Sơ đồ mối quan hệ:**
Các hoạt động trong lĩnh vực **Lâm sàng** (Clinical), **Dược phẩm** (Pharma) cùng với việc đảm bảo **Quyền riêng tư & Pháp lý** (Privacy & Legal) là những tiền đề trực tiếp dẫn đến các kết quả tích cực về mặt **Tác động** (Impact) cho ngành y tế và nghiên cứu khoa học. Sự kết nối giữa các tổ chức như **Owkin** và liên minh **MELLODDY** minh chứng cho sức mạnh của việc chia sẻ tri thức mà không cần chia sẻ dữ liệu thô.

## Google Cloud & Swift – Financial Fraud Detection

Đây là một dự án trọng điểm được triển khai vào **Tháng 12/2024** nhằm giải quyết các thách thức trong ngành tài chính.

- **Mục tiêu:** Phát hiện các hành vi gian lận thanh toán trong các giao dịch xuyên biên giới.
- **Quy mô:** Triển khai hệ thống **Federated Learning** đa khu vực pháp lý với sự tham gia của 12 ngân hàng toàn cầu.
- **Cơ chế hoạt động:** Các bên tham gia chia sẻ **Fraud Labels** (nhãn gian lận) để cùng huấn luyện mô hình mà không làm tiết lộ dữ liệu giao dịch thô của khách hàng.

### Công nghệ và Bảo mật

Dự án kết hợp các công nghệ tiên tiến để đảm bảo tính an toàn cho dữ liệu:
- **Secure Sandbox:** Cung cấp một môi trường được kiểm soát chặt chẽ để các ngân hàng có thể cộng tác với nhau.
- **Confidential Computing:** Sử dụng khả năng cách ly dựa trên phần cứng của Google Cloud để bảo vệ các bản cập nhật trong quá trình **Federated Learning**.

### Rào cản Pháp lý và Cạnh tranh

Việc ứng dụng **Federated Learning** giúp vượt qua hai rào cản lớn nhất trong ngành ngân hàng:
- **Bảo mật Ngân hàng (Banking Secrecy):** Đảm bảo tuân thủ các quy định nghiêm ngặt về lưu trú dữ liệu tài chính tại từng quốc gia.
- **Co-opetition (Hợp tác đối đầu):** Cho phép các ngân hàng đối thủ hợp tác với nhau để chống lại tội phạm tài chính mà không lo ngại việc rò rỉ thông tin khách hàng cho đối phương.

### Quy trình Federated Learning trong dự án Google Cloud và Swift

Hệ thống được thiết kế để áp dụng cho nhiều kịch bản khác nhau với các thực thể tham gia bao gồm:
- **Data in Bank 1:** Dữ liệu được lưu trữ trên Cloud.
- **Data in Bank 2:** Dữ liệu lưu trữ tại chỗ (On-Premise).
- **Data in Bank 3:** Dữ liệu lưu trữ trên bất kỳ nhà cung cấp Cloud nào khác.
- **Federated Learning Orchestrator:** Bộ điều phối trung tâm nằm trên nền tảng Cloud.

**Quy trình hoạt động (Workflow):**

1. **Gửi model đến dữ liệu (Send model to data):** Bộ điều phối (Orchestrator) gửi mô hình đến các điểm lưu trữ dữ liệu của từng ngân hàng.
2. **Huấn luyện tại chỗ và xuất weight (Train Model. Export Weights):** Mỗi ngân hàng thực hiện huấn luyện mô hình trực tiếp trên dữ liệu tại chỗ của mình. Sau đó, họ chỉ gửi các thông số **weight** đã được cập nhật về lại bộ điều phối trung tâm, không gửi dữ liệu thô.
3. **Tổng hợp weight vào global model (Aggregate weights into global model):** Tại bộ điều phối Orchestrator, các **weight** nhận được từ tất cả các ngân hàng thành viên sẽ được tổng hợp lại để cập nhật và tạo ra một mô hình chung (global model) hoàn thiện hơn.

Ứng dụng này chứng minh rằng **Federated Learning** là giải pháp hiệu quả để cân bằng giữa nhu cầu nâng cao độ chính xác của mô hình thông qua hợp tác và yêu cầu bảo mật dữ liệu khắt khe của ngành tài chính.

## IBM & Handelsbanken - Anti-Money Laundering

**Bối cảnh chiến lược (GAIA 2025)**

- **Mục tiêu:** Chống lại hoạt động rửa tiền trên quy mô toàn cầu, ước tính lên đến $3.1T$ tiền bất hợp pháp.

**Khung kỹ thuật (Technical Framework)**

- **Pattern Sharing (Chia sẻ mẫu):** Các ngân hàng thực hiện cộng tác dựa trên các dấu hiệu gian lận (fraud signatures) mà không làm lộ PII (Personally Identifiable Information - Thông tin nhận dạng cá nhân).
- **Synthetic Data (Dữ liệu tổng hợp):** Tạo ra các bộ dữ liệu tổng hợp có độ trung thực cao nhằm mục đích xác thực **model** giữa các ngân hàng khác nhau.

**Cơ sở hạ tầng tin cậy (Trust Infrastructure)**

- Hệ thống được lưu trữ trung lập bởi **AI Sweden**, đóng vai trò trung gian để kết nối các ngân hàng vốn đang cạnh tranh trực tiếp với nhau trên thị trường.

**Sơ đồ quy trình AML dựa trên Federated Learning (FL-Powered AML)**

Quy trình này cho phép phát hiện gian lận cộng tác mà không cần chia sẻ dữ liệu trực tiếp giữa các bên:

1.  **Các thực thể đầu cuối:**
    - **Bank 1 (Handelsbanken)** và **Bank 2** (ví dụ: các ngân hàng khu vực).
2.  **Quy trình nội bộ tại mỗi ngân hàng (Bank's Local Process):**
    - Sử dụng **Local Sensitive Data** (Dữ liệu nhạy cảm cục bộ) để thực hiện **Local AML Model Training** (Huấn luyện **model** AML cục bộ). Dữ liệu nhạy cảm này không bao giờ rời khỏi hạ tầng của ngân hàng.
3.  **Trung tâm điều phối:**
    - Một **Trusted Aggregator** (Bộ tổng hợp tin cậy) do AI Sweden quản lý đóng vai trò điều phối chung.
4.  **Luồng truyền tin và tổng hợp:**
    - Các ngân hàng gửi **Secure Update / Updated Model** (Bản cập nhật bảo mật hoặc **model** đã được cập nhật cục bộ) lên **Trusted Aggregator**.
    - **Trusted Aggregator** thực hiện tổng hợp các thông tin này để tạo ra **Global Model** (Model toàn cầu).
    - **Global Model** sau đó được gửi ngược lại cho các ngân hàng để áp dụng vào hệ thống của họ.

**Tác động và Ý nghĩa thực tiễn**

Việc áp dụng Federated Learning trong ngành tài chính mang lại những lợi ích then chốt:
- **Enhanced Accuracy:** Tăng cường độ chính xác của hệ thống phát hiện gian lận.
- **Cross-Bank Detection:** Cho phép phát hiện các hành vi gian lận xuyên ngân hàng, điều mà các hệ thống đơn lẻ khó thực hiện được.
- **Vượt qua rào cản pháp lý và cạnh tranh:** Cho phép các ngân hàng đối thủ cùng nhau cải thiện khả năng bảo mật thông qua việc chia sẻ các "mẫu" (patterns) và bản cập nhật **model** thay vì chia sẻ dữ liệu khách hàng. Điều này giúp tuân thủ các quy định về quyền riêng tư (PII) và giữ vững lợi thế cạnh tranh trong khi vẫn đạt được mục tiêu chung là ngăn chặn dòng tiền bất hợp pháp quy mô lớn.

## NVIDIA & NHS - Cancer Detection (Phát hiện ung thư)

Ứng dụng này minh họa một trường hợp thực tế của Federated Learning (FL) trong lĩnh vực y tế thông qua sự hợp tác giữa NVIDIA và NHS (Dịch vụ Y tế Quốc gia Anh) nhằm mục đích phát hiện ung thư. Giải pháp này cho phép các bệnh viện cùng nhau huấn luyện một mô hình AI mạnh mẽ mà không cần chia sẻ dữ liệu bệnh nhân nhạy cảm ra khỏi cơ sở.

### Kiến trúc hệ thống (Mô hình Hub-and-Spoke)

Hệ thống được tổ chức theo cấu trúc trung tâm và các nhánh (nút), đảm bảo tính riêng tư và hiệu quả kinh tế.

**1. Các nút mạng của các Hospital Trust (Hospital Trust Nodes):**
Hệ thống kết nối các đơn vị bệnh viện cụ thể bao gồm:
- **Oxford Hospital Trust**
- **Birmingham Hospital Trust**
- **Bedfordshire Hospital Trust**
- **Portsmouth Hospital Trust**

**Đặc điểm kỹ thuật tại mỗi nút (Node):**
- **Dữ liệu nhạy cảm cục bộ (Local Sensitive Data):** Hồ sơ y tế của bệnh nhân được lưu trữ và xử lý tại chỗ, đảm bảo tuân thủ đầy đủ quy định GDPR.
- **Phần cứng chi phí thấp (Low-cost hardware):** Mỗi địa điểm sử dụng các thiết bị phần cứng giá rẻ như Raspberry Pi với chi phí chỉ từ £45-85, giúp giải pháp dễ dàng tiếp cận.
- **Huấn luyện mô hình cục bộ (Local model training):** Quá trình huấn luyện mạng thần kinh diễn ra trực tiếp trên dữ liệu tại mỗi bệnh viện.

**2. Thành phần trung tâm (Central Aggregator):**
- **Máy chủ điều phối (Coordinator Server):** Đóng vai trò quản lý và điều phối toàn bộ quy trình Federated Learning.
- **Cập nhật mô hình toàn cục (Global Model Update):** Thực hiện việc tổng hợp các tham số từ các nút để cải thiện mô hình chung.
- **Công nghệ hỗ trợ:** Sử dụng bộ công cụ **NVIDIA FLARE SDK**.
- **SECURE AGGREGATION (Tổng hợp bảo mật):** Áp dụng các phương thức tổng hợp an toàn để bảo vệ thông tin từ các bản cập nhật cục bộ.

**3. Luồng dữ liệu và thông tin:**
- **Trung tâm gửi đến các Nút:** Chuyển giao mô hình toàn cục ban đầu (**INITIAL GLOBAL MODEL**) hoặc các phiên bản mô hình toàn cục đã được cập nhật (**UPDATED GLOBAL MODEL**).
- **Các Nút gửi về Trung tâm:** Chỉ gửi các bản cập nhật mô hình đã được mã hóa (**ENCRYPTED MODEL UPDATES**), tuyệt đối không gửi dữ liệu thô.
- **Trao đổi thông tin:** Quá trình này tạo ra các thông tin chuyên sâu (**INSIGHTS**) hai chiều giữa trung tâm và các bệnh viện thành viên.

### Kết quả và Tác động

Việc triển khai Federated Learning mang lại những lợi ích chiến lược quan trọng:

- **Tuân thủ pháp lý (Legal Compliance):** Đáp ứng yêu cầu pháp lý của NHS về việc dữ liệu y tế không được phép tập trung hóa tại một kho lưu trữ duy nhất.
- **Hiệu suất (Performance):** Federated Learning giúp cải thiện hiệu suất mô hình thêm **27.6%** so với việc chỉ huấn luyện mô hình riêng lẻ tại một bệnh viện duy nhất.
- **Chủ quyền dữ liệu (Data Sovereignty):** Quyền riêng tư của bệnh nhân được bảo vệ tối đa do dữ liệu không rời khỏi biên giới của bệnh viện.
- **Khả năng mở rộng và Chi phí (Scalability & Cost):** Việc sử dụng Raspberry Pi chứng minh rằng Federated Learning có thể triển khai trên quy mô lớn với chi phí cực kỳ thấp.
- **Sự công bằng (Equity):** Tạo ra sự bình đẳng khi tất cả các bệnh viện tham gia đều được hưởng lợi từ tri thức tập thể của toàn hệ thống.

# VI. REPOSITORY

## Flower

Flower là một framework phổ biến và mạnh mẽ được thiết kế để triển khai Federated Learning. Framework này được xây dựng với các đặc điểm nổi bật sau:

- **Customizable** (Có thể tùy chỉnh)
- **Extendable** (Có thể mở rộng)
- **Understandable** (Dễ hiểu)
- **Framework-agnostic** (Không phụ thuộc framework): Flower hỗ trợ hầu hết các thư viện Machine Learning và xử lý dữ liệu hiện nay, bao gồm: PyTorch, TensorFlow, Hugging Face Transformers, PyTorch Lightning, scikit-learn, JAX, TFLite, MONAI, fastai, MLX, XGBoost, CatBoost, LeRobot, Pandas và NumPy.

### Kiến trúc của Framework Flower

Kiến trúc của Flower được thiết kế theo dạng phân tầng, cho phép quản lý đồng thời cả thiết bị vật lý thực tế và các client ảo, giúp tối ưu hóa việc nghiên cứu và triển khai thực tế:

1.  **Tầng điều khiển trung tâm:**
    - **Strategy**: Định nghĩa các chiến lược tổng hợp và điều phối.
    - **Global Model**: Lưu trữ mô hình chung của toàn hệ thống.
2.  **Tầng cấu hình:**
    - Bao gồm hai khối **Configure train/eval** đóng vai trò trung gian điều phối việc huấn luyện và đánh giá giữa mô hình trung tâm và các client.
3.  **Tầng quản lý kết nối:**
    - **Client Manager**: Quản lý danh sách các client tham gia.
    - **RPC Server**: Thực hiện giao tiếp giữa server và các client.
4.  **Tầng Proxy (Đại diện):**
    - **Edge Client Proxy**: Đại diện cho các thiết bị ngoại vi thực tế.
    - **Virtual Client Proxy**: Đại diện cho các thực thể ảo trong môi trường mô phỏng.
5.  **Tầng thực thi (Client):**
    - **Edge Client**: Chứa **RPC Client**, **Training Pipeline** và dữ liệu (**Data**) cục bộ.
    - **Virtual Client (inactive/active)**: Chứa **Training Pipeline** và dữ liệu (**Data**) được mô phỏng.

**Luồng dữ liệu:**
Quá trình vận hành bắt đầu bằng việc gửi cấu hình từ **Strategy** xuống các Proxy, sau đó chuyển đến các Client. Tại đây, các Client thực hiện **Training Pipeline** trên dữ liệu nội bộ của mình. Cuối cùng, các kết quả cập nhật được gửi ngược lại để tổng hợp vào **Global Model**.

**Liên kết tham khảo:**
- GitHub: [https://github.com/flwrlabs/flower](https://github.com/flwrlabs/flower)
- Paper: [https://arxiv.org/abs/2007.14390](https://arxiv.org/abs/2007.14390)

## PFLLib

Trang web dự án: [https://github.com/TsingZ0/PFLlib](https://github.com/TsingZ0/PFLlib)

PFLLib là một thư viện mã nguồn mở mạnh mẽ dành cho Federated Learning (FL), đặc biệt tập trung vào các phương pháp cá nhân hóa (**Personalized FL** - pFL). Thư viện này cung cấp một tập hợp phong phú các thuật toán hiện đại (SOTA), giúp đơn giản hóa việc thử nghiệm và so sánh các phương pháp FL khác nhau trong nghiên cứu và ứng dụng.

### Cấu trúc thư mục và chức năng
Mã nguồn của PFLLib được thiết kế theo dạng module, cho phép người dùng dễ dàng mở rộng và triển khai các thuật toán mới bằng cách kế thừa và chỉnh sửa từ các lớp cơ sở (base class).

- **root**: Thư mục gốc của dự án.
    - **dataset**: Chứa các công cụ xử lý dữ liệu.
        - **utils**: Các tiện ích bổ trợ.
        - **generate_MNIST.py**: Chứa các kịch bản tích hợp sẵn (Built-in scenarios) để tạo tập dữ liệu cục bộ cho client dựa trên cấu hình cụ thể.
    - **system**: Chứa lõi hệ thống FL.
        - **flcore**:
            - **clients**:
                - **clientavg.py**: Triển khai client của thuật toán FedAvg. Đây là file mẫu để chỉnh sửa khi tạo thuật toán mới.
                - **clientbase.py**: Tạo lớp Client cơ sở.
            - **servers**:
                - **serveravg.py**: Triển khai server của thuật toán FedAvg. Đây là file mẫu để chỉnh sửa khi tạo thuật toán mới.
                - **serverbase.py**: Tạo lớp Server cơ sở.
    - **main.py**: Điểm bắt đầu của chương trình, dùng để cấu hình các siêu tham số (hyperparameters).

### Quy trình hoạt động (Workflow)
Hệ thống hoạt động dựa trên sự tương tác giữa Server trung tâm và nhiều Client (từ Client 1 đến Client N). Quá trình truyền tải mô hình hoặc **gradient** diễn ra liên tục trong mỗi round huấn luyện:

1. **Tại mỗi Client:** Thực hiện một chu trình khép kín bao gồm:
    - `train`: Huấn luyện cục bộ trên thiết bị với dữ liệu riêng.
    - `evaluate`: Đánh giá hiệu suất mô hình.
    - `send`: Gửi bản cập nhật mô hình lên server.
    - `receive`: Nhận mô hình toàn cục từ server để bắt đầu round tiếp theo.
2. **Tại Server:** Đóng vai trò điều phối trung tâm:
    - `receive`: Tiếp nhận các bản cập nhật từ tất cả các client tham gia.
    - `aggregate`: Thực hiện thuật toán tổng hợp các bản cập nhật.
    - `update`: Cập nhật mô hình toàn cục.
    - `send`: Gửi lại mô hình đã cập nhật cho các client.

### Phân loại thuật toán FL trong PFLLib
PFLLib hỗ trợ tổng cộng 37 thuật toán, được chia thành hai nhóm chính:

| Nhóm (Category) | Thuật toán (Algorithms) |
| :--- | :--- |
| **8 Thuật toán tFL (Traditional FL)** | |
| tFL cơ bản | FedAvg |
| tFL dựa trên hiệu chỉnh cập nhật | SCAFFOLD |
| tFL dựa trên **regularization** | FedProx và FedDyn |
| tFL dựa trên chia tách mô hình | MOON và FedLC |
| tFL dựa trên chưng cất tri thức | FedGen và FedNTD |
| **29 Thuật toán pFL (Personalized FL)** | |
| pFL dựa trên siêu học (Meta-learning) | Per-FedAvg |
| pFL dựa trên **regularization** | pFedMe và Ditto |
| pFL dựa trên tổng hợp cá nhân hóa | APFL, FedFomo, FedAMP, FedPHP, APPLE, và FedALA |
| pFL dựa trên chia tách mô hình | FedPer, LG-FedAvg, FedRep, FedRoD, FedBABU, FedGC, FedCP, GPFL, FedGH, DBE, FedCAC, và PFL-DA |
| pFL dựa trên chưng cất tri thức | FedDistill, FML, FedKD, FedProto, FedPCL, và FedPAC |
| pFL khác | FedMTL và FedBN |

## FLBench

**FLBench** là một công cụ benchmark toàn diện dành cho Federated Learning (FL), được thiết kế để hỗ trợ nhiều kịch bản và loại dữ liệu khác nhau trong quá trình đánh giá các thuật toán.

- **Link GitHub:** [https://github.com/KarhouTam/FL-bench](https://github.com/KarhouTam/FL-bench)

### Kiến trúc hệ thống FLBench

Sơ đồ kiến trúc của FLBench được cấu thành từ 4 khối chức năng chính, phối hợp với nhau để tạo ra một quy trình đánh giá hoàn chỉnh:

1. **Automated Deployment Tool (Công cụ triển khai tự động):**
   - Hỗ trợ triển khai trên nhiều môi trường khác nhau: **Mobile** (Di động), **Distributed** (Phân tán), và **Standalone** (Độc lập).
   - Khối này kết nối và tương tác với các thành phần khác của hệ thống thông qua các **APIs**.

2. **Scenario (Kịch bản):**
   - Chứa các **Evaluation Metrics** (Chỉ số đánh giá) để đo lường hiệu quả của mô hình.
   - **Fixed Scenario (Kịch bản cố định):** Bao gồm các kịch bản ứng dụng cổ điển như **Medicine** (Y tế), **Finance** (Tài chính), và **AIoT**. Các kịch bản này được thiết lập sẵn với các mức độ đánh giá: **Best** (Tốt nhất), **Medium** (Trung bình), và **Worst** (Tệ nhất).
   - **Customized Scenarios (Kịch bản tùy chỉnh):** Cho phép người dùng linh hoạt cấu hình các kịch bản riêng biệt từ 1 đến $n$.
   - Dữ liệu trong mỗi kịch bản được phân tách rõ ràng thành: **Training Set**, **Validation Set**, và **Test Set**.

3. **Scenario Configuration (Cấu hình kịch bản):**
   - Cung cấp **API for Configuration** để thiết lập các tham số hệ thống.
   - **Basic configurations (Cấu hình cơ bản):** Tập trung vào các yếu tố cốt lõi của FL bao gồm:
     - **Communication** (Giao tiếp).
     - **Scenario Transformation** (Chuyển đổi kịch bản).
     - **Data Distribution Heterogeneity** (Sự không đồng nhất trong phân phối dữ liệu).
     - **Privacy-preserving** (Bảo mật quyền riêng tư).
     - **Cooperation Strategy** (Chiến lược hợp tác).
   - Hỗ trợ **Natural Distribution of Data** (Phân phối dữ liệu tự nhiên).

4. **Input Data (Dữ liệu đầu vào):**
   - **Structured (Có cấu trúc):** Điển hình là mảng Y tế (Medicine) với các loại dữ liệu như Scale, CT, Gene... được lưu trữ dưới dạng Image, Table, Binary, hoặc Text.
   - **Semi-structured (Bán cấu trúc):** Điển hình là mảng Tài chính (Finance) với các dạng dữ liệu như Time Series, Table, và Text.
   - **Un-structured (Phi cấu trúc):** Điển hình là mảng AIoT với các định dạng đa dạng như Graph, Table, Video, Image, Audio, và Text.

### Quy trình hoạt động (Flow)

Hệ thống vận hành theo một luồng dữ liệu và điều khiển thống nhất:
1. Dữ liệu đầu vào (**Input Data**) từ các nguồn khác nhau được đưa vào khối **Scenario**.
2. Khối **Scenario Configuration** cung cấp các thiết lập và tham số cấu hình cụ thể cho kịch bản tương ứng.
3. Cuối cùng, **Automated Deployment Tool** sử dụng các **APIs** để thực thi và triển khai các kịch bản đã cấu hình lên các môi trường thực tế (Mobile, Distributed hoặc Standalone).

FLBench đóng vai trò như một framework mã nguồn mở mạnh mẽ, giúp đơn giản hóa việc thử nghiệm và so sánh các chiến lược Federated Learning trên nhiều loại dữ liệu từ có cấu trúc đến phi cấu trúc.

## FedProC

Trang web dự án: https://github.com/nclabteam/FedProC

FedProC là một framework toàn diện dành cho Federated Learning, hỗ trợ quy trình từ quản lý dữ liệu, huấn luyện mô hình đến đánh giá hệ thống. Hệ thống bao gồm các thành phần chính sau:

- **Datasets (Tập dữ liệu):**
    - **Different Domain (Đa dạng lĩnh vực):** Cung cấp dữ liệu từ nhiều lĩnh vực khác nhau như Finance (Tài chính), Weather (Thời tiết), và Traffic (Giao thông).
    - **Different Granularity (Đa dạng độ phân giải thời gian):** Hỗ trợ các mức độ chi tiết thời gian khác nhau bao gồm Daily (Hàng ngày), Hourly (Hàng giờ), và Minutely (Hàng phút).

- **Data Processing (Xử lý dữ liệu):**
    - **Splitting/Partitioning (Phân chia dữ liệu):** Thực hiện phân chia dữ liệu theo các Clients và phân định các tập Train/Test.
    - **Normalization (Chuẩn hóa):** Hỗ trợ các kỹ thuật chuẩn hóa dữ liệu như Robust, Standard, và Min Max.

- **Models (Mô hình):** Thư viện mô hình phong phú với nhiều kiến trúc mạng khác nhau:
    - **MLP**: DLinear, TSMixer, UMixer, CrossLinear.
    - **RNN**: LSTM, GRU, SegRNN, RWKV4TS.
    - **CNN**: MLCNN, SCINet, ModernTCN, xPatch.
    - **Others**: DSSRNN, TimeKAN, PAttn, CALF.

- **Strategies (Chiến lược):**
    - **tFL/pFL** (Mô hình Client-Server): FedAvg, FedProx, Elastic, FedAWA.
    - **dFL** (Decentralized - Mạng lưới ngang hàng): FullConnected, Ring, KConnected.
    - **nFL** (Non-federated - Cục bộ hoặc tập trung): Local, Centralized.

- **Logging (Ghi nhật ký):** Hệ thống ghi lại toàn bộ nhật ký quá trình hoạt động và thực thi của framework.

- **Evaluation (Đánh giá):** Đánh giá hệ thống dựa trên các tiêu chí đa dạng:
    - **Robustness** (Tính bền vững)
    - **Efficiency** (Hiệu suất)
    - **Stability** (Tính ổn định)
    - **Communication** (Truyền thông)

### Luồng hoạt động của Framework FedProC

Quy trình hoạt động của hệ thống được thực hiện theo các bước tuần tự từ trái sang phải:

1. **Datasets**: Dữ liệu đầu vào được thu thập từ các lĩnh vực (Tài chính, Thời tiết, Giao thông) với các độ phân giải thời gian tương ứng.
2. **Data Processing**: Dữ liệu từ nguồn được đưa vào xử lý, thực hiện phân chia cho các nhóm Clients, chia tập huấn luyện/kiểm thử và áp dụng các biểu đồ chuẩn hóa (Robust, Standard, Min Max).
3. **Models & Strategies**: Dữ liệu sau xử lý được nạp vào các kiến trúc mô hình và áp dụng các chiến lược Federated Learning (như cấu trúc tập trung Client-Server, mạng lưới ngang hàng phi tập trung, hoặc các phương thức không liên kết).
4. **Logging**: Các trang tài liệu ghi nhật ký sẽ lưu trữ lại toàn bộ diễn biến của quá trình.
5. **Evaluation**: Khối cuối cùng tiếp nhận kết quả để thực hiện đánh giá tổng thể về tính bền vững của mạng lưới, hiệu suất vận hành, độ ổn định của hệ thống và khả năng truyền thông giữa các thành phần. Luồng dữ liệu có tính chất quay vòng từ Logging về Evaluation để đảm bảo tính liên tục trong việc theo dõi và đánh giá.