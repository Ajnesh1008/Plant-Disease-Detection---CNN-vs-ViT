# ViT model wrapper
import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16

class ViTModel(nn.Module):
    def __init__(self, num_classes=15):
        super(ViTModel, self).__init__()
        self.vit = vit_b_16(pretrained=True)
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)
