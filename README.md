# Garbage Classification (PyTorch)

Dự án phân loại rác thải sử dụng các mô hình Deep Learning phổ biến như ResNet, Vision Transformer (ViT) và CLIP.

## 📂 Cấu trúc thư mục

```text
Garbage_Classification/
│
├── Data/                    # Thư mục chứa dữ liệu ảnh (chia theo các class)
├── src/
│   ├── data/                # Dataset & Dataloader (TrashDataset)
│   ├── models/              # Định nghĩa mô hình (CNN, ViT, CLIP)
│   └── utils/               # Các hàm phụ trợ (Metrics, Seed, Plot)
├── train.py                 # Script huấn luyện chính
├── requirements.txt         # Các thư viện phụ thuộc
└── README.md                # Tài liệu hướng dẫn
```

**Các tham số chính:**
- `--data_dir`: Đường dẫn tới thư mục dataset.
- `--model_type`: Chọn mô hình (`cnn`, `vit`, `clip`).
- `--epochs`: Số vòng lặp huấn luyện.
- `--batch_size`: Kích thước lô dữ liệu (Mặc định: 32).
- `--lr`: Learning rate (Mặc định: 0.001).
- `--num_splits`: Số lượng Monte Carlo splits.
- `--single-split`: Thêm cờ này nếu chỉ muốn chạy thử 1 split duy nhất cho nhanh.
- `--save_dir`: Thư mục lưu biểu đồ và weights (`checkpoints/` mặc định).

**Ví dụ chạy mô hình CLIP với 1 split (Kaggle/Colab):**
```bash
python train.py --data_dir /kaggle/input/garbage-classification/garbage_classification --model_type clip --single-split
```
