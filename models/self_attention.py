import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Linear(in_dim, in_dim // 8)
        self.key   = nn.Linear(in_dim, in_dim // 8)
        self.value = nn.Linear(in_dim, in_dim)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        x: [B, F]
        """
        Q = self.query(x)          # [B, F/8]
        K = self.key(x)            # [B, F/8]
        V = self.value(x)          # [B, F]

        attention = torch.softmax(Q @ K.T, dim=-1)  # [B, B]
        out = attention @ V                         # [B, F]

        return self.gamma * out + x
