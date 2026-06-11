import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, accuracy_score

# Import modules từ src/
from src.utils.seed import seed_everything
from src.utils.metrics import plot_cm, plot_loss_curve
from src.models.factory import build_model
from src.data.dataset import get_monte_carlo_splits

def run_experiment(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nSTARTING EXPERIMENT: {args.model_type.upper()}")
    print(f"--> Thiết bị: {DEVICE}")
    print(f"--> Data Dir: {args.data_dir}")
    
    # 1. Thiết lập Seed cho reproducibility
    seed_everything(args.seed)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 2. Khởi tạo Data Loaders
    splits = get_monte_carlo_splits(
        args.data_dir, 
        num_splits=args.num_splits, 
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    for split_idx, (train_loader, val_loader, test_loader, class_names) in enumerate(splits):
        print(f"\n{'='*20} SPLIT {split_idx + 1}/{args.num_splits} {'='*20}")
        
        start_time = time.time()
        
        # 3. Khởi tạo Model thông qua Factory
        model = build_model(args.model_type, num_classes=len(class_names), device=DEVICE)
        model = model.to(DEVICE)
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        
        train_losses = []
        val_losses = []
        
        best_val_loss = float('inf')
        best_model_path = os.path.join(args.save_dir, f'best_model_{args.model_type}_split{split_idx+1}.pth')
        
        # 4. Vòng lặp Huấn luyện (Training Loop)
        train_acc_final = 0
        
        for epoch in range(args.epochs): 
            # --- TRAIN ---
            model.train()
            running_train_loss = 0.0
            correct_train = 0
            total_train = 0
            
            train_pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{args.epochs} [Train]", leave=False)
            for inputs, labels in train_pbar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_train_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (preds == labels).sum().item()
            
            epoch_train_loss = running_train_loss / len(train_loader.dataset)
            train_losses.append(epoch_train_loss)
            train_acc_final = 100 * correct_train / total_train

            # --- VALIDATE ---
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item() * inputs.size(0)
            
            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            val_losses.append(epoch_val_loss)

            print(f"   -> Epoch {epoch+1:02d}: Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
            
            # --- SAVE BEST MODEL ---
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"      [*] Best model saved at Val Loss {best_val_loss:.4f}")

        end_time = time.time()
        train_duration = end_time - start_time
        
        # Vẽ biểu đồ Loss
        print("\n📈 Đang vẽ biểu đồ Loss...")
        plot_loss_curve(train_losses, val_losses, save_path=os.path.join(args.save_dir, f'loss_{args.model_type}_split{split_idx+1}.png'))
        
        # 5. Đánh giá trên tập Test
        print(f"Loading best model from {best_model_path} cho quá trình Test...")
        model.load_state_dict(torch.load(best_model_path))
        
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_acc = accuracy_score(all_labels, all_preds) * 100
        overfitting_score = train_acc_final - test_acc
        
        print(f"\n📊 KẾT QUẢ ĐÁNH GIÁ (SPLIT {split_idx+1}):")
        print("-" * 30)
        print(f"⏱️ Thời gian train: {train_duration:.2f} giây")
        print(f"📉 Best Val Loss   : {best_val_loss:.4f}")
        print(f"🎯 Train Accuracy  : {train_acc_final:.2f}%")
        print(f"🎯 Test Accuracy   : {test_acc:.2f}%")
        print(f"⚠️ Độ Overfitting  : {overfitting_score:.2f}%")
        print("-" * 30)
        
        print("Chi tiết từng lớp (Test Set):")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        
        print("Vẽ Confusion Matrix...")
        plot_cm(all_labels, all_preds, class_names, save_path=os.path.join(args.save_dir, f'cm_{args.model_type}_split{split_idx+1}.png'))
        
        if args.single_split:
            print("Đã hoàn thành 1 split theo yêu cầu (--single-split). Dừng.")
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình phân loại rác thải")
    parser.add_argument('--data_dir', type=str, default='./Data', help='Đường dẫn tới thư mục chứa dữ liệu')
    parser.add_argument('--model_type', type=str, default='vit', choices=['cnn', 'vit', 'clip'], help='Loại mô hình (cnn, vit, clip)')
    parser.add_argument('--epochs', type=int, default=5, help='Số lượng epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_splits', type=int, default=5, help='Số lượng Monte Carlo splits')
    parser.add_argument('--num_workers', type=int, default=2, help='Số workers cho DataLoader')
    parser.add_argument('--seed', type=int, default=42, help='Random seed để Reproducibility')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Thư mục lưu weights và plots')
    parser.add_argument('--single-split', action='store_true', help='Chỉ chạy 1 split thay vì tất cả (để test code nhanh)')
    
    args = parser.parse_args()
    
    run_experiment(args)