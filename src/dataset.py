import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import glob



class TrashDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list): Danh sách đường dẫn tới từng file ảnh.
            labels (list): Danh sách nhãn tương ứng (dạng số nguyên 0, 1, 2...).
            transform (callable, optional): Các hàm xử lý ảnh (Resize, Normalize...).
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # Trả về tổng số lượng ảnh trong tập này
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Lấy đường dẫn ảnh và nhãn tại vị trí idx
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 2. Mở ảnh bằng thư viện PIL (Python Imaging Library)
        # convert('RGB') để đảm bảo ảnh có 3 kênh màu
        image = Image.open(img_path).convert("RGB")

        # 3. Áp dụng transform (nếu có) để chuyển thành Tensor
        if self.transform:
            image = self.transform(image)

        return image, label


def load_data_paths(root_dir):
    
    all_image_paths = []
    all_labels = []
    classes = sorted(os.listdir(root_dir))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    print(f"Tìm thấy {len(classes)} lớp: {classes}")

    for cls_name in classes:
        cls_folder = os.path.join(root_dir, cls_name)
        if not os.path.isdir(cls_folder):
            continue
       
        files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            files.extend(glob.glob(os.path.join(cls_folder, ext)))
            
        for f_path in files:
            all_image_paths.append(f_path)
            all_labels.append(class_to_idx[cls_name])
            
    return all_image_paths, all_labels, classes