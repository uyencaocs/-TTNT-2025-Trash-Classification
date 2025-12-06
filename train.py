# Giả sử trong train.py
from src.data_splitter import get_monte_carlo_splits

DATA_DIR = "D:\TTNT-2025\garbage_classification" # Đường dẫn tới thư mục data của bạn
# Lấy bộ generator
splits = get_monte_carlo_splits(DATA_DIR, num_splits=5)

# Chạy vòng lặp 5 lần thí nghiệm
for split_idx, (train_loader, val_loader, test_loader) in enumerate(splits):
    print(f"Đang training Split {split_idx + 1}...")
    
    