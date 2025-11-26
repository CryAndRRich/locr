# Text Detection: Tìm Kiếm Con Chữ Trong "Rừng" Pixel

Đây là nơi lưu trữ các tìm hiểu về **Text Detection** và mã nguồn thực hành về bài toán này

## Text Detection là gì?

Nói một cách đơn giản, **Text Detection** là nhiệm vụ xác định **vị trí** của văn bản trong một bức ảnh

Nếu coi cả hệ thống OCR (Optical Character Recognition) là một người đọc sách, thì:
* **Text Detection** là đôi mắt: Tìm xem chữ nằm ở đâu (đóng khung nó lại)
* **Text Recognition** là bộ não: Đọc xem trong cái khung đó viết chữ gì

**Mục tiêu:** Đầu ra của bài toán này thường là các tọa độ bao quanh văn bản (bounding boxes hoặc polygons)

## Tại sao nó lại khó?

Khác với việc quét tài liệu (document scanning) nơi giấy trắng mực đen thẳng hàng, việc phát hiện chữ ngoài đời thực là một "cơn ác mộng" vì những lý do sau:

### 1. Đa dạng hình thái (Diversity)
Chữ viết không chỉ có một kiểu. Chúng đa dạng về màu sắc, kích thước, phông chữ (font) và ngôn ngữ
* **Ví dụ:** Chữ trên biển quảng cáo neon khác hẳn chữ viết tay trên bảng menu

### 2. Nền phức tạp (Scene Complexity)
Trong ảnh tự nhiên, có rất nhiều vật thể nhìn "giống chữ" nhưng không phải là chữ (ví dụ: hàng rào, gạch men, lá cây). Mô hình rất dễ bị nhầm lẫn (gọi là nhiễu nền)

### 3. Biến dạng (Distortion & Orientation)
Đây là thách thức lớn nhất.
* **Đa hướng (Multi-oriented):** Chữ có thể nằm ngang, dọc, hoặc xoay nghiêng bất kỳ
* **Cong (Curved Text):** Chữ trên ly Starbucks, logo tròn, hoặc biển hiệu uốn lượn
* **Phối cảnh (Perspective):** Khi chụp nghiêng, chữ bị méo theo góc nhìn 3D

### 4. Chất lượng ảnh thấp (Degraded Image Quality)
Ảnh chụp thực tế thường bị mờ do chuyển động (motion blur), thiếu sáng, độ phân giải thấp hoặc bị nhiễu

### 5. Che khuất (Occlusion)
Đôi khi chữ bị cây cối, người đi đường hoặc các vật thể khác che mất một phần, nhưng con người vẫn đọc được, và máy tính cũng phải làm được điều đó

## Các hướng tiếp cận chính

Hiện nay, Deep Learning thống trị lĩnh vực này với 2 hướng đi chủ đạo:

1.  **Regression-based (Dựa trên hồi quy):** Coi văn bản là các vật thể (object) và dự đoán tọa độ hộp bao trực tiếp (giống như thuật toán YOLO hay SSD trong object detection)
2.  **Segmentation-based (Dựa trên phân vùng):** Phân loại từng điểm ảnh (pixel) xem nó thuộc về "văn bản" hay "nền", sau đó nhóm chúng lại thành các khối chữ. Phương pháp này thường tốt hơn cho chữ cong hoặc hình dạng kỳ dị

## Mô hình Text Detection

Thư mục này chứa các mã nguồn mô hình kèm notebook thực hành để giải quyết bài toán Text Detection:
* [EAST](EAST/EAST.md) - Efficient and Accurate Scene Text Detector

*(Danh sách sẽ được cập nhật liên tục trong quá trình học tập)*