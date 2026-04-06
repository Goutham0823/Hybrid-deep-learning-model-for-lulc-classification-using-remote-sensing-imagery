import torch
from models.ibnr65 import IBNR65

model = IBNR65(num_classes=10)
x = torch.randn(1, 3, 64, 64)

y = model(x)
print("Output shape:", y.shape)
