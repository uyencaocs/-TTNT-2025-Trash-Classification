import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA_DIR = "/kaggle/input/garbage-classification/garbage_classification" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_cm(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Dự đoán (Predicted)')
    plt.ylabel('Thực tế (True)')
    plt.title('Confusion Matrix')
    plt.show()


def plot_loss_curve(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue', marker='o')
    plt.plot(val_losses, label='Validation Loss', color='orange', marker='x')
    plt.title('Biểu đồ Loss qua các Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- 3. CHƯƠNG TRÌNH CHÍNH ---
def run_experiment_full_report(model_type='cnn'):
    print(f"\nSTARTING EXPERIMENT: {model_type.upper()}")
    print(f"--> Thiết bị: {DEVICE}")
    
    os.makedirs("models", exist_ok=True)
    
    splits = get_monte_carlo_splits(DATA_DIR, num_splits=5, batch_size=32)
    
    _, _, class_names = load_data_paths(DATA_DIR)

    for split_idx, (train_loader, val_loader, test_loader) in enumerate(splits):
        print(f"\n{'='*20} SPLIT {split_idx + 1} {'='*20}")
        
        start_time = time.time()
      
        if model_type == 'cnn':
            model = CNN_ResNet(num_classes=12)
        elif model_type == 'vit':
            model = ViT_Model(num_classes=12)
        elif model_type == 'clip':
            model = CLIP_Model(num_classes=12, backbone_name="ViT-B/16")
            
        model = model.to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
     
        train_losses = []
        val_losses = []
        
        # Training Loop
        train_acc_final = 0
        
        EPOCHS = 5 
        
        for epoch in range(EPOCHS): 
            # --- TRAIN ---
            model.train()
            running_train_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for inputs, labels in tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}", leave=False):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Cộng dồn loss
                running_train_loss += loss.item() * inputs.size(0)
                
                # Tính Accuracy train
                _, preds = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (preds == labels).sum().item()
            
            # Tính trung bình Train Loss của Epoch này
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
            
            # Tính trung bình Val Loss của Epoch này
            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            val_losses.append(epoch_val_loss)

            print(f"   -> Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        end_time = time.time()
        train_duration = end_time - start_time
        
        # --- VẼ BIỂU ĐỒ LOSS (MỚI) ---
        print("\nĐang vẽ biểu đồ Loss...")
        plot_loss_curve(train_losses, val_losses)
        
        # Evaluation (Test set)
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
        
        # Tính toán chỉ số
        test_acc = accuracy_score(all_labels, all_preds) * 100
        overfitting_score = train_acc_final - test_acc
        
        print(f"\nKẾT QUẢ ĐÁNH GIÁ (SPLIT {split_idx+1}):")
        print("-" * 30)
        print(f"⏱Thời gian train: {train_duration:.2f} giây")
        print(f"Final Train Loss: {train_losses[-1]:.4f}")
        print(f"Final Val Loss  : {val_losses[-1]:.4f}")
        print(f"Train Accuracy  : {train_acc_final:.2f}%")
        print(f"Test Accuracy   : {test_acc:.2f}%")
        print(f"Độ Overfitting  : {overfitting_score:.2f}%")
        print("-" * 30)
        
        print("Chi tiết từng lớp:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        
        print("Vẽ Confusion Matrix...")
        plot_cm(all_labels, all_preds, class_names)
        
        torch.save(model.state_dict(), f'model_{model_type}_full_metrics.pth')


#run_experiment_full_report('vit')
#run_experiment_full_report('cnn')
#run_experiment_full_report('clip')