import torch
import torch.nn as nn
from .moe_layer import MoELayer

class SimpleMoEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, moe_hidden_dim=None, k=1):
        super().__init__()
        if moe_hidden_dim is None:
            moe_hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.moe = MoELayer(hidden_dim, moe_hidden_dim, hidden_dim, num_experts, k)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch, input_dim]
        h = self.input_proj(x)
        h = self.moe(h)
        out = self.output_proj(h)
        return out

# 示例用法：
# model = SimpleMoEModel(input_dim=32, hidden_dim=64, output_dim=10, num_experts=4, k=2)
# x = torch.randn(8, 32)
# y = model(x)
