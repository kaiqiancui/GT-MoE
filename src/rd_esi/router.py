import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Any, Optional


class RDESIRouter(nn.Module):
    """
    基于声誉的动态专家选择与自适应激励 (RD-ESI) 路由器。

    该路由器实现了 RD-ESI 机制，该机制结合了：
    1. 基础门控分数 (g_i(x)) - 可通过梯度下降进行训练。
    2. 动态声誉分数 (R_i(t)) - 通过启发式规则更新。
    3. 负载感知 (L_i(t)) - 通过启发式规则更新。
    4. 探索奖励 (Exploration Bonus) - 启发式部分。
    
    该路由器通过两种方式工作：
    - 一个基于启发式规则的快速更新系统（在 @torch.no_grad() 中），用于调整声誉和负载。
    - 一个可微分的、标准的负载均衡辅助损失，用于通过梯度下降来训练基础门控网络，
      确保模型在端到端训练中学会如何均衡地分配任务。

    最终选择分数的计算公式为:
    SelectionScore_i(x,t) = g_i(x) + β * R_i(t) - γ * L_i(t) + ExplorationBonus_i(x,t)
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        beta: float = 0.1,
        gamma: float = 0.1,
        alpha: float = 0.9,
        use_exploration_bonus: bool = True,
        exploration_c: float = 0.1,
        use_reputation_decay: bool = True,
        decay_rate: float = 0.99,
        load_ema_alpha: float = 0.9,
    ):
        """
        初始化 RD-ESI 路由器。

        参数:
            hidden_size (int): 输入特征的维度。
            num_experts (int): 路由的专家数量。
            top_k (int): 为每个令牌（token）选择的专家数量。
            beta (float): 声誉分数的权重。
            gamma (float): 负载惩罚的权重。
            alpha (float): 声誉 EMA 更新的平滑因子。
            use_exploration_bonus (bool): 是否使用探索奖励。
            exploration_c (float): UCB 探索奖励的常数。
            use_reputation_decay (bool): 是否使用声誉衰减。
            decay_rate (float): 声誉衰减的速率。
            load_ema_alpha (float): 负载 EMA 更新的平滑因子。
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.use_exploration_bonus = use_exploration_bonus
        self.exploration_c = exploration_c
        self.use_reputation_decay = use_reputation_decay
        self.decay_rate = decay_rate
        self.load_ema_alpha = load_ema_alpha
        
        # 基础门控投影器 (g_i(x)) - 这是模型中唯一可训练的部分
        self.gate_projector = nn.Linear(hidden_size, num_experts, bias=False)
        
        # 注册缓冲区（buffers），用于在训练步骤之间保持持久状态
        # 声誉分数 R_i(t)
        self.register_buffer("reputation_scores", torch.zeros(num_experts))
        # 负载跟踪 L_i(t)
        self.register_buffer("expert_loads", torch.zeros(num_experts))
        # 专家选择计数 N_i(t)，用于探索奖励
        self.register_buffer("expert_counts", torch.zeros(num_experts))
        # 已做出的总路由决策数 (N)
        self.register_buffer("total_routing_decisions", torch.tensor(0))
        
    def forward(
        self, 
        x: torch.Tensor,
        current_batch_assignments: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        RD-ESI 路由器的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, sequence_length, hidden_size]。
            current_batch_assignments (Optional[torch.Tensor]): 可选张量，指示批处理中当前的专家分配情况，用于负载计算。
            
        返回:
            一个元组，包含:
            - routing_weights (torch.Tensor): 形状为 [batch_size, sequence_length, top_k] 的张量，包含所选专家的归一化权重。
            - expert_indices (torch.Tensor): 形状为 [batch_size, sequence_length, top_k] 的张量，包含所选专家的索引。
            - aux_outputs (Dict[str, Any]): 包含辅助输出的字典，其中包括用于训练和分析的
              **可微分负载均衡损失 (`loss`)**。
        """
        batch_size, sequence_length, hidden_size = x.shape
        
        # 为路由重塑输入
        x_reshaped = x.view(-1, hidden_size)  # [batch_size * sequence_length, hidden_size]
        
        # 1. 计算基础门控分数 g_i(x)
        base_logits = self.gate_projector(x_reshaped)  # [batch_size * sequence_length, num_experts]
        
        # 2. 更新专家负载 L_i(t) (启发式部分)
        if current_batch_assignments is not None:
            current_loads = current_batch_assignments
        else:
            current_loads = self.expert_loads
            
        # 使用 EMA 更新负载
        updated_loads = self.load_ema_alpha * current_loads + (1 - self.load_ema_alpha) * self.expert_loads
        
        # 3. 计算探索奖励 (启发式部分)
        exploration_bonus = torch.zeros_like(base_logits)
        if self.use_exploration_bonus:
            epsilon = 1e-10
            exploration_term = self.exploration_c * torch.sqrt(
                torch.log(self.total_routing_decisions + 1.0) / (self.expert_counts.unsqueeze(0) + epsilon)
            )
            exploration_bonus = exploration_term.expand_as(base_logits)
        
        # 4. 计算最终选择分数（结合可微分和不可微分部分）
        selection_scores = (
            base_logits +  # g_i(x) - 可微分
            self.beta * self.reputation_scores.unsqueeze(0) -  # β * R_i(t) - 不可微分
            self.gamma * updated_loads.unsqueeze(0) +  # γ * L_i(t) - 不可微分
            exploration_bonus  # ExplorationBonus_i(x,t) - 不可微分
        )
        
        # 5. 选择 top-k 专家
        top_k_scores, top_k_indices = torch.topk(selection_scores, self.top_k, dim=-1)
        
        # 6. 计算路由权重 (在所选专家的分数上进行 softmax)
        routing_weights = F.softmax(top_k_scores, dim=-1)
        
        # 将输出重塑为原始维度
        routing_weights = routing_weights.view(batch_size, sequence_length, self.top_k)
        expert_indices = top_k_indices.view(batch_size, sequence_length, self.top_k)
        
        # --- 可微分的负载均衡损失计算 ---
        # 这个部分确保梯度可以流回 gate_projector，教它如何进行负载均衡。

        # 1. 计算所有专家的被选择概率（用于损失计算）
        # 注意：这里我们对所有专家的 `selection_scores` 进行 softmax，而不仅仅是 top-k 的。
        router_probs = F.softmax(selection_scores, dim=-1, dtype=torch.float32)

        # 2. 计算每个专家被选中的 token 比例
        # 首先，我们需要一个独热编码（one-hot）来表示哪些专家被选中
        flat_expert_indices = expert_indices.view(-1, self.top_k)
        expert_gate = F.one_hot(flat_expert_indices, num_classes=self.num_experts).sum(dim=1)
        tokens_per_expert = torch.mean(expert_gate.float(), dim=0)
        
        # 3. 计算每个专家的平均路由概率
        router_prob_per_expert = torch.mean(router_probs, dim=0)

        # 4. 计算最终的负载均衡损失
        # loss = (对所有专家求和) (tokens_per_expert * router_prob_per_expert) * num_experts
        aux_loss = (tokens_per_expert * router_prob_per_expert).sum() * self.num_experts

        # 准备辅助输出，现在包含了可微分的损失
        aux_outputs = {
            "router_logits": base_logits,         # 原始 logits，用于调试和分析
            "selection_scores": selection_scores,
            "updated_loads": updated_loads,
            "expert_indices": expert_indices,
            "loss": aux_loss                      # 新增的、可微分的损失项
        }
        
        return routing_weights, expert_indices, aux_outputs
        
    @torch.no_grad()
    def update_states(
        self,
        expert_indices: torch.Tensor,
        current_performances: torch.Tensor,
        batch_size: int = None,
        sequence_length: int = None,
    ) -> None:
        """
        根据路由决策和专家表现更新路由器状态（启发式部分）。
        
        注意：此方法在 @torch.no_grad() 下运行，不会影响梯度计算。

        参数:
            expert_indices (torch.Tensor): 形状为 [batch_size, sequence_length, top_k] 的张量，包含所选专家的索引。
            current_performances (torch.Tensor): 形状为 [batch_size, sequence_length, top_k] 的张量，包含每个所选专家的性能指标。
            batch_size (int): 批大小（可选，如未提供将自动推断）。
            sequence_length (int): 序列长度（可选，如未提供将自动推断）。
        """
        # 为计数压平索引
        flat_indices = expert_indices.view(-1)
        
        # 为探索奖励更新专家计数
        new_counts = torch.bincount(flat_indices, minlength=self.num_experts).float()
        self.expert_counts.add_(new_counts)
        
        # 如果未提供，则推断 batch_size 和 sequence_length
        if batch_size is None or sequence_length is None:
            batch_size, sequence_length = expert_indices.shape[0], expert_indices.shape[1]
            
        # 更新总路由决策数
        self.total_routing_decisions.add_(batch_size * sequence_length * self.top_k)
        
        # --- 安全且向量化的实现，用于计算每个专家的平均性能 ---
        
        # 为便于处理，压平输入张量
        flat_indices = expert_indices.view(-1)
        flat_performances = current_performances.view(-1)

        # 在正确的设备上创建临时张量以聚合批处理结果
        batch_perf_sum = torch.zeros_like(self.reputation_scores)
        batch_counts = torch.zeros_like(self.reputation_scores) # 使用 float 类型进行累加

        # 使用 scatter_add 高效地对每个专家的性能和计数求和
        batch_perf_sum.scatter_add_(0, flat_indices, flat_performances)
        batch_counts.scatter_add_(0, flat_indices, torch.ones_like(flat_performances))
        
        # 创建一个掩码，标记在此批处理中实际使用过的专家，以避免除以零
        used_experts_mask = batch_counts > 0
        
        # 初始化 current_performance_i 为零；对于未使用的专家，它将保持为零
        current_performance_i = torch.zeros_like(self.reputation_scores)
        
        # 仅为被使用过的专家计算平均性能
        current_performance_i[used_experts_mask] = batch_perf_sum[used_experts_mask] / batch_counts[used_experts_mask]
        
        # 使用安全的 current_performance_i 张量更新声誉
        if self.use_reputation_decay:
            self.reputation_scores *= self.decay_rate
        self.reputation_scores = self.alpha * current_performance_i + (1 - self.alpha) * self.reputation_scores

        # 更新专家负载
        if self.load_ema_alpha < 1.0:
            self.expert_loads = self.load_ema_alpha * batch_counts + (1 - self.load_ema_alpha) * self.expert_loads
        else:
            self.expert_loads = batch_counts