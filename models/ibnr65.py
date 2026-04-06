import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=6):
        super().__init__()
        hidden_dim = in_ch * expand_ratio
        self.use_residual = (stride == 1 and in_ch == out_ch)

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim,
                bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            nn.Conv2d(hidden_dim, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class IBNR65(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layers = nn.Sequential(
            InvertedResidualBlock(32, 64, stride=2),
            InvertedResidualBlock(64, 64),
            InvertedResidualBlock(64, 128, stride=2),
            InvertedResidualBlock(128, 128),
            InvertedResidualBlock(128, 256, stride=2),
            InvertedResidualBlock(256, 256),
            InvertedResidualBlock(256, 512, stride=2),
            InvertedResidualBlock(512, 512),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

    # 🔹 FEATURE EXTRACTION (FOR FUSION)
    def extract_features(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)

    # 🔹 NORMAL FORWARD (FOR SINGLE-MODEL TRAINING)
    def forward(self, x):
        x = self.extract_features(x)
        return self.classifier(x)
