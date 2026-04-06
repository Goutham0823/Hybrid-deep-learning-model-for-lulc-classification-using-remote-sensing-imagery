import torch
from models.fusion_attention_model import FusionAttentionNet

model = FusionAttentionNet(num_classes=10)
x = torch.randn(2, 3, 64, 64)

y = model(x)
print("Output shape:", y.shape)
