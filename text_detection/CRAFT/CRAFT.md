CRAFT/ 
├── config/ # Quản lý cấu hình 
│ └── config.yaml # File YAML chứa toàn bộ tham số 
├── data/ # Pipeline xử lý dữ liệu 
│ ├── init.py 
│ ├── dataset.py # Custom Dataset Class 
│ ├── gaussian.py # Sinh Heatmap
│ └── augmentations.py # Augmentation (RandomCrop, Rotate) 
├── models/ # Định nghĩa mạng nơ-ron 
│ ├── init.py 
│ ├── craft.py # Class CRAFT và vgg16_bn
│ ├── refinenet.py # LinkRefiner
│ └── weights/ 
├── utils/
  ├── init.py 
│ ├── loss.py # Hàm loss OHEM 
│ ├── metrics.py # Tính IoU, F1-score 
│ ├── post_processing.py # getDetBoxes
│ └── logger.py # Ghi log huấn luyện 
├── train.py # Script huấn luyện chính 
├── eval.py # Script đánh giá