import torch
import torch.nn as nn
import torch.nn.functional as F

class AuctionRouter(nn.Module):
    def __init__(self, input_dim, num_experts, k=1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k  # top-k
        self.router = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # x: [batch, input_dim]
        logits = self.router(x)  # [batch, num_experts]
        scores = F.softmax(logits, dim=-1)  # 作为拍卖分数
        # 选出top-k个专家
        topk_scores, topk_indices = torch.topk(scores, self.k, dim=-1)  # [batch, k]
        # 返回每个样本分配的专家索引和分数
        return topk_indices, topk_scores
