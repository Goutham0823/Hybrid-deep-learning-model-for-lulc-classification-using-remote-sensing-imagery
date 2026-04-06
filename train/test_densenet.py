import torch
from models.densenet64 import DenseNet64

model = DenseNet64(num_classes=10)
x = torch.randn(1, 3, 64, 64)

y = model(x)
print("Output shape:", y.shape)
