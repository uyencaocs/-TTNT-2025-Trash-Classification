from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from PIL import Image
import os
import glob

class TrashDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def load_data_paths(root_dir):
    all_image_paths = []
    all_labels = []
 
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    print(f"--> Tìm thấy {len(classes)} loại: {classes}")

    for cls_name in classes:
        cls_folder = os.path.join(root_dir, cls_name)
       
        files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            files.extend(glob.glob(os.path.join(cls_folder, ext)))
            files.extend(glob.glob(os.path.join(cls_folder, ext.upper()))) # Tìm cả đuôi viết hoa

        for f_path in files:
            all_image_paths.append(f_path)
            all_labels.append(class_to_idx[cls_name])
            
    return all_image_paths, all_labels, classes

def get_monte_carlo_splits(root_dir, num_splits=5, batch_size=32):
    all_paths, all_labels, class_names = load_data_paths(root_dir)
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for i in range(num_splits):
       
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            all_paths, all_labels, test_size=0.15, stratify=all_labels, random_state=i*42
        )
       
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.176, stratify=y_train_val, random_state=i*42
        )
        
        train_ds = TrashDataset(X_train, y_train, transform=train_transform)
        val_ds = TrashDataset(X_val, y_val, transform=eval_transform)
        test_ds = TrashDataset(X_test, y_test, transform=eval_transform)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

        yield train_loader, val_loader, test_loader