import torch
from models.fusion_model import FusionNet

model = FusionNet(num_classes=10)
x = torch.randn(2, 3, 64, 64)

y = model(x)
print("Fusion output shape:", y.shape)
