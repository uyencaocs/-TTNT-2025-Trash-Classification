import torch
import torch.nn as nn
import timm
import clip 
# --- 1. CNN ResNet ---
class CNN_ResNet(nn.Module):
    def __init__(self, num_classes=12):
        super(CNN_ResNet, self).__init__()
        self.model = timm.create_model('resnet50', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

# --- 2. ViT ---
class ViT_Model(nn.Module):
    def __init__(self, num_classes=12):
        super(ViT_Model, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

# --- 3. CLIP ---
class CLIP_Model(nn.Module):
    def __init__(self, num_classes=12, backbone_name="ViT-B/16"):
        super(CLIP_Model, self).__init__()
        self.model, _ = clip.load(backbone_name, device='cpu', jit=False)

        for param in self.model.parameters():
            param.requires_grad = False
            
        embed_dim = 512 if backbone_name == "ViT-B/16" else 1024
        self.head = nn.Linear(embed_dim, num_classes)
        self.model = self.model.float()

    def forward(self, x):
        features = self.model.encode_image(x)
        return self.head(features)