import torch
import torch.nn as nn
from .experts import Experts
from .router import AuctionRouter

class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, k=1):
        super().__init__()
        self.experts = Experts(num_experts, input_dim, hidden_dim, output_dim)
        self.router = AuctionRouter(input_dim, num_experts, k)
        self.k = k

    def forward(self, x):
        # x: [batch, input_dim]
        topk_indices, topk_scores = self.router(x)  # [batch, k], [batch, k]
        batch_size = x.size(0)
        output_dim = self.experts.experts[0].ffn[-1].out_features
        out = torch.zeros(batch_size, output_dim, device=x.device)
        # 对每个token，聚合k个专家的输出
        for i in range(self.k):
            indices = topk_indices[:, i]  # [batch]
            scores = topk_scores[:, i].unsqueeze(-1)  # [batch, 1]
            expert_out = self.experts(x, indices)  # [batch, output_dim]
            out += expert_out * scores  # 加权聚合
        return out
