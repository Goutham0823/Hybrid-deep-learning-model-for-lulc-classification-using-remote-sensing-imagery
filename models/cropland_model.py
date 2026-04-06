import torch
import torch.nn as nn
from models.fusion_attention_model import FusionAttentionNet

class CroplandSuitabilityModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = FusionAttentionNet(num_classes=10)

        # Replace classifier with binary head
        self.backbone.classifier = nn.Sequential(
            nn.Linear(512 + 320, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.backbone(x)
