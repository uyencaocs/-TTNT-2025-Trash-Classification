import torch
import torch.nn as nn
import timm

# Thử import CLIP, nếu chưa cài thì bỏ qua (để tránh lỗi khi chạy CNN/ViT thường)
try:
    import clip
except ImportError:
    clip = None

# ==========================================
# 1. MODEL CNN (RESNET)
# ==========================================
class CNN_ResNet(nn.Module):
    def __init__(self, num_classes=12):
        super(CNN_ResNet, self).__init__()
        print("--> Đang khởi tạo CNN (ResNet50)...")
        # pretrained=True: Sử dụng Transfer Learning từ ImageNet
        self.model = timm.create_model('resnet50', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

# ==========================================
# 2. MODEL VISION TRANSFORMER (ViT)
# ==========================================
class ViT_Model(nn.Module):
    def __init__(self, num_classes=12):
        super(ViT_Model, self).__init__()
        print("--> Đang khởi tạo Vision Transformer (ViT-Base)...")
        # vit_base_patch16_224: Model chuẩn, cắt ảnh thành ô 16x16
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

# ==========================================
# 3. MODEL CLIP (VISION-LANGUAGE)
# ==========================================
class CLIP_Model(nn.Module):
    def __init__(self, num_classes=12, backbone_name="ViT-B/16"):
        super(CLIP_Model, self).__init__()
        
        # Kiểm tra xem đã cài thư viện CLIP chưa
        if clip is None:
            raise ImportError("Chưa cài thư viện CLIP. Hãy chạy: pip install git+https://github.com/openai/CLIP.git")

        print(f"--> Đang khởi tạo CLIP ({backbone_name})...")
        
        # 1. Tải backbone (phần nhìn ảnh của CLIP)
        # device='cpu' để load ban đầu, sau đó train.py sẽ chuyển sang cuda sau
        self.model, _ = clip.load(backbone_name, device='cpu', jit=False)
        
        # 2. Đóng băng (Freeze) phần nhìn ảnh để giữ kiến thức gốc
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 3. Thêm lớp phân loại (Classification Head)
        # ViT-B/16 có đầu ra là vector 512 chiều, RN50 là 1024
        embed_dim = 512 if backbone_name == "ViT-B/16" else 1024
        
        self.head = nn.Linear(embed_dim, num_classes)
            
        # Chuyển model về float32
        self.model = self.model.float()

    def forward(self, x):
        # Cho ảnh đi qua CLIP để lấy đặc trưng (features)
        features = self.model.encode_image(x)
        # Cho đặc trưng đi qua lớp phân loại
        return self.head(features)