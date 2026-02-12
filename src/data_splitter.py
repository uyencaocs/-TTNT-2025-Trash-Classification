# File: src/data_splitter.py

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

from src.dataset import TrashDataset, load_data_paths 

def get_monte_carlo_splits(root_dir, num_splits=5, batch_size=32, img_size=224):
    """
    Tạo ra generator trả về 5 bộ DataLoaders khác nhau.
    Tỷ lệ chia: Train (70%), Val (15%), Test (15%)
    """
    # 1. Load toàn bộ dữ liệu thô
    all_paths, all_labels, class_names = load_data_paths(root_dir)
    
    # Define transforms (Chuẩn hóa dữ liệu cho Model)
    
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    # Val/Test chỉ cần Resize và chuyển thành Tensor
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # 2. Vòng lặp Monte Carlo (5 lần)
    for i in range(num_splits):
        print(f"\n--- Monte Carlo Split #{i+1} ---")
        
        # Bước 1: Tách Test ra trước (15%)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            all_paths, all_labels, test_size=0.15, random_state=None, stratify=all_labels
        )
        
        # Bước 2: Tách Train và Val từ phần còn lại 
        # (Val chiếm khoảng 15% tổng, tương đương ~17.6% phần còn lại)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.176, random_state=None, stratify=y_train_val
        )
        
        print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

      
        train_ds = TrashDataset(X_train, y_train, transform=train_transform)
        val_ds = TrashDataset(X_val, y_val, transform=eval_transform)
        test_ds = TrashDataset(X_test, y_test, transform=eval_transform)

        # 4. Tạo DataLoader
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        yield train_loader, val_loader, test_loader