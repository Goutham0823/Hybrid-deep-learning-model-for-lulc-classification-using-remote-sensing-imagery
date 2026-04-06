import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        out = self.layer(x)
        return torch.cat([x, out], 1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(num_layers):
            layers.append(DenseLayer(channels, growth_rate))
            channels += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x):
        return self.block(x)


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.trans = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        return self.trans(x)


class DenseNet64(nn.Module):
    def __init__(self, num_classes=10, growth_rate=16):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.block1 = DenseBlock(4, 64, growth_rate)
        self.trans1 = TransitionLayer(self.block1.out_channels, 128)

        self.block2 = DenseBlock(4, 128, growth_rate)
        self.trans2 = TransitionLayer(self.block2.out_channels, 256)

        self.block3 = DenseBlock(4, 256, growth_rate)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.block3.out_channels, num_classes)

    # 🔹 FEATURE EXTRACTION (FOR FUSION)
    def extract_features(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.trans1(x)
        x = self.block2(x)
        x = self.trans2(x)
        x = self.block3(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)

    # 🔹 NORMAL FORWARD (FOR SINGLE-MODEL TRAINING)
    def forward(self, x):
        x = self.extract_features(x)
        return self.fc(x)
