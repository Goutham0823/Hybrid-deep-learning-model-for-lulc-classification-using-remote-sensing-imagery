import torch
import torch.nn as nn

from models.ibnr65 import IBNR65
from models.densenet64 import DenseNet64
from models.self_attention import SelfAttention

class FusionAttentionNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.ibnr = IBNR65(num_classes=num_classes)
        self.densenet = DenseNet64(num_classes=num_classes)

        # Remove classifiers
        self.ibnr.classifier = nn.Identity()
        self.densenet.fc = nn.Identity()

        fused_dim = 512 + 320  # confirmed earlier

        self.attention = SelfAttention(fused_dim)

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        f1 = self.ibnr.extract_features(x)
        f2 = self.densenet.extract_features(x)

        fused = torch.cat([f1, f2], dim=1)
        attended = self.attention(fused)

        return self.classifier(attended)
