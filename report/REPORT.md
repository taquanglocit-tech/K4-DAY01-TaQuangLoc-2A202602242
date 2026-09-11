# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
 ('468', 'cab', 1, 0.510915, 'ImageNet-1K')
- Record này mô tả toàn ảnh như thế nào?
Record này mô tả toàn ảnh có nhân vật chính là xe taxi (cab)
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
Con người sẽ định nghĩa class list mà checkpoint có thể đoán
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
ID sẽ là mã định danh cho từng ảnh, tên lớp giúp con người nhìn vào và hiểu được ý nghĩa, taxonomy chính là mã để xác định nguồn gốc hay thuộc bộ ảnh nào, cùng 1 tên lớp ở những bộ ảnh khác nhau rất có thể sẽ mang ý nghĩa khác nhau, cần phải phân biệt
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
Nếu ảnh có nhiều chủ thể guideline cần chọn vật có vị trí trung tâm và lớn nhất trong các vật thể
- Vì sao model score không phải ground truth?
Vì model score là điểm dự đoán của mô hình, nó được dùng để đánh giá thay vì xem nó như nhãn để chỉnh sửa model weight

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`): ('bus', '0.912558', [ 93.17, 187.95, 223.01, 320.91], 129.84, 132.96)
- Diễn giải vị trí box bằng lời: box này nằm ở tọa độ góc trên bên trái là (93.17, 187.95) và góc dưới bên phải là (223.01, 320.91), chiều rộng là 129.84 và chiều cao 132.96
- So sánh số prediction ở hai threshold: các score_threshold bằng nhau
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
So với phân lớp, độ bao phủ của một class giảm xuống và số lượng reviewer cần xem tăng lên
- Đề xuất một quy tắc box chặt: Quy tắc box chặt là box phải chạm vào pixel ngoài cùng của đường biên vật thể, không được cắt phạm vào trong hay để dư ở ngoài
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
guideline sẽ chọn việc chỉ vẽ box cho phần lộ ra hay tự dự đoán phần bị mất để vẽ box, còn escalation sẽ quyết định vật thể đó lộ ra khoản bao nhiêu % trở lên thì cần gán nhãn
## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
- Polygon bổ sung chi tiết gì so với box?
polygon là đường bao countour thực tế sẽ bám sát hình dạng của vật thể thay vì chỉ cho biết chiều dài, rộng của nó trong ảnh
- `instance_id` dùng để làm gì và không phải loại ID nào?
instance_id là mã định danh riêng cho từng vật thể trong ảnh
- Đề xuất một quy tắc biên mask: đường biên không được dư hoặc cắt vào trong vật thể, ở đoạn cong cần tăng lượng điểm đánh dấu hơn so với đoạn gấp khúc vuông để đường cong vật mượt mà vào ôm sát vật thể
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
guideline quy định độ mờ bao nhiêu thì bỏ qua, vật thể nào ở trước thì giữ nguyên đường biên và cắt ngắn đường biên của vật bị che, 
escalation quyết định xem vùng mờ đó có cần giữ lại không và khi tiếp xúc, các đối tượng đó có được đánh nhãn riêng không hay gắn thành một nhãn

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | class_name và class_id | ảnh có nhiều chủ thể khác class nhưng chỉ dự đoán cho 1 chủ thể | Gắn nhãn dữ liệu | Xem kết quả gắn nhãn của Annotator |
| Phát hiện vật thể | Tọa độ 2 điểm chéo của box hoặc tọa độ 1 điểm kèm chiều cao chiều rộng box | vật bị mờ, che khuất ở giữa có thể bị đoán thành nhiều vật thể khác nhau| Vẽ box cho các vật thể | Xem box đã khít chưa, có bị cắt hay lỏng không |
| Instance segmentation | Tọa độ các điểm đường biên poligon | vật bị mờ hoặc che ở giữa có thể bị nhầm thành nhiều đối tượng khác nhau | Vẽ polixy ôm sát đường biên | kiểm tra đường biên đã sát chưa, có lỏng hay bị cắt không |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: luôn ghi nguồn giữ liệu, sao lưu 
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: quản lý dự án

## 6. Danh sách bằng chứng

- [X] `classification_predictions.json`
- [X] `detection_predictions.json`
- [X] `segmentation_predictions.json`
- [X] `IMAGE_ATTRIBUTION.md`
- [X] `visuals/classification_top5.png`
- [X] `visuals/detection_predictions.png`
- [X] `visuals/segmentation_prediction.png`
- [X] Ô validation cuối notebook báo `PASS`.
- [X] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
