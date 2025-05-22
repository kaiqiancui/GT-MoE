import torch
import torch.nn as nn

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.ffn(x)

# 支持多个专家的容器
class Experts(nn.Module):
    def __init__(self, num_experts, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)
        ])

    def forward(self, x, expert_indices):
        # x: [batch, input_dim], expert_indices: [batch]
        # 每个样本分配给一个专家
        out = torch.zeros(x.size(0), self.experts[0].ffn[-1].out_features, device=x.device)
        for i, expert in enumerate(self.experts):
            mask = (expert_indices == i)
            if mask.any():
                out[mask] = expert(x[mask])
        return out
