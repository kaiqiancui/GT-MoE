import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional


class TopKGatingRouter(nn.Module):
    """
    标准的 Top-K 门控路由器，用于 MoE 模型。
    
    该路由器实现标准的 Top-K 门控机制，其中令牌（token）被路由到具有最高路由器分数的 top-k 个专家。
    
    属性:
        hidden_size (int): 输入特征的维度。
        num_experts (int): 用于路由的专家数量。
        top_k (int): 为每个令牌选择的专家数量。
        use_aux_loss (bool): 是否使用辅助的负载均衡损失。
        aux_loss_weight (float): 辅助负载均衡损失的权重。
        expert_counts (torch.Tensor): 用于跟踪路由到每个专家的令牌数量的缓冲区。
                                     为了与 Trainer 兼容，它也将充当 'expert_loads' 的角色。
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        use_aux_loss: bool = True,
        aux_loss_weight: float = 0.01,
    ):
        """
        初始化 TopKGatingRouter。
        
        参数:
            hidden_size (int): 输入特征的维度。
            num_experts (int): 用于路由的专家数量。
            top_k (int): 为每个令牌选择的专家数量。
            use_aux_loss (bool): 是否使用辅助的负载均衡损失。
            aux_loss_weight (float): 辅助负载均衡损失的权重。
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_aux_loss = use_aux_loss
        self.aux_loss_weight = aux_loss_weight
        
        # 用于计算路由概率的门控投影器
        self.gate_projector = nn.Linear(hidden_size, num_experts, bias=False)
        
        # 注册用于跟踪专家计数的缓冲区
        # 为了与 Trainer 的 _get_expert_loads 兼容，它也将作为 'expert_loads' 被访问，
        # 因为它代表了专家的负载/利用率。
        self.register_buffer("expert_counts", torch.zeros(num_experts, dtype=torch.float32))
        # 将 expert_loads 设置为 expert_counts 的别名，以兼容 Trainer
        self.expert_loads = self.expert_counts

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        TopKGatingRouter 的前向传播。
        
        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, sequence_length, hidden_size]。
            
        返回:
            元组，包含:
            - routing_weights (torch.Tensor): 形状为 [batch_size, sequence_length, top_k] 的张量，
                                             包含所选专家的归一化权重。
            - expert_indices (torch.Tensor): 形状为 [batch_size, sequence_length, top_k] 的张量，
                                            包含所选专家的索引。
            - aux_outputs (Dict[str, Any]): 包含辅助输出的字典，其中包括负载均衡损失。
        """
        batch_size, sequence_length, _ = x.shape
        
        # 重塑输入以进行路由
        x_reshaped = x.view(-1, self.hidden_size)  # [batch_size * sequence_length, hidden_size]
        
        # 计算门控 logits
        gate_logits = self.gate_projector(x_reshaped)  # [batch_size * sequence_length, num_experts]
        
        # 计算路由概率
        routing_probs = F.softmax(gate_logits, dim=-1)  # [batch_size * sequence_length, num_experts]
        
        # 选择 top-k 专家
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        
        # 归一化 top-k 专家的概率
        # 添加一个小的 epsilon 以防止在 top_k_probs_sum 为零时除以零
        top_k_probs_sum = top_k_probs.sum(dim=-1, keepdim=True) + 1e-6 
        top_k_probs_normalized = top_k_probs / top_k_probs_sum
        
        # 重塑输出以匹配输入维度
        routing_weights = top_k_probs_normalized.view(batch_size, sequence_length, self.top_k)
        expert_indices = top_k_indices.view(batch_size, sequence_length, self.top_k)
        
        # 如果启用，则计算辅助的负载均衡损失
        aux_loss_value = torch.tensor(0.0, device=x.device, dtype=x.dtype) #确保 aux_loss_value 在任何情况下都被定义为张量
        if self.use_aux_loss and self.training:
            # expert_mask: [batch_size * sequence_length, num_experts]
            # scatter_期望索引张量是long类型的
            # 注意：top_k_indices 需要被 reshape 以匹配 expert_mask 的前导维度
            expert_mask = torch.zeros(
                batch_size * sequence_length, self.num_experts, device=x.device, dtype=torch.float32
            ).scatter_(1, top_k_indices.view(-1, self.top_k), 1.0) 
            
            # router_prob_per_expert: [num_experts]
            router_prob_per_expert = routing_probs.mean(dim=0)
            # router_assignment_per_expert: [num_experts]
            router_assignment_per_expert = expert_mask.mean(dim=0)
            
            # 计算负载均衡损失
            # 我们希望 router_prob_per_expert 和 router_assignment_per_expert 是均匀的，
            # 即每个专家都是 1/num_experts
            aux_loss_value = (
                # 鼓励均匀的概率
                torch.sum(router_prob_per_expert * router_prob_per_expert) * self.num_experts +
                # 鼓励均匀的分配
                torch.sum(router_assignment_per_expert * router_assignment_per_expert) * self.num_experts
            )
        
        # 更新专家计数以进行监控
        # 确保直接更新缓冲区，而不是其副本
        if self.training:
            # 使用 expert_indices 的展平版本进行 bincount，以避免图问题（如果不需要梯度）
            # 并确保 expert_counts 被正确更新。
            flat_expert_indices = expert_indices.view(-1) # 将 expert_indices 展平
            # 确保 expert_counts 与 flat_expert_indices 在同一设备上
            current_expert_counts = torch.bincount(flat_expert_indices, minlength=self.num_experts).to(self.expert_counts.device, dtype=self.expert_counts.dtype)
            self.expert_counts.add_(current_expert_counts) # 原地加法
        
        # 准备辅助输出
        aux_outputs = {
            "gate_logits": gate_logits,
            "routing_probs": routing_probs,
            "aux_loss": aux_loss_value, # 这是实际的辅助损失张量
        }
        
        return routing_weights, expert_indices, aux_outputs


class TopKGatingMoELayer(nn.Module):
    """
    具有标准 Top-K 门控的 MoE 层。
    
    该层将 TopKGatingRouter 与一组专家网络集成在一起。
    
    属性:
        hidden_size (int): 输入和输出特征的维度。
        num_experts (int): 层中的专家数量。
        top_k (int): 为每个令牌选择的专家数量。
        router (TopKGatingRouter): 用于专家选择的路由器。
        experts (nn.ModuleList): 专家网络列表。
        expert_dropout_rate (float): 专家输出的 dropout 概率。
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int = 2,
        use_aux_loss: bool = True,      # 这个参数传递给内部的 TopKGatingRouter
        aux_loss_weight: float = 0.01,  # 这个参数也传递给内部的 TopKGatingRouter
        expert_dropout: float = 0.0,
    ):
        """
        初始化 TopKGatingMoELayer。
        
        参数:
            hidden_size (int): 输入和输出特征的维度。
            intermediate_size (int): 每个专家的中间特征维度。
            num_experts (int): 层中的专家数量。
            top_k (int): 为每个令牌选择的专家数量。
            use_aux_loss (bool): 是否使用辅助的负载均衡损失（传递给路由器）。
            aux_loss_weight (float): 辅助负载均衡损失的权重（传递给路由器）。
            expert_dropout (float): 专家输出的 dropout 概率。
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_dropout_rate = expert_dropout # 重命名以避免与 self.dropout 模块冲突
        
        # 初始化路由器
        self.router = TopKGatingRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            use_aux_loss=use_aux_loss,
            aux_loss_weight=aux_loss_weight,
        )
        
        # 初始化专家
        self.experts = nn.ModuleList([
            self._create_expert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])
        
        # 专家输出的 dropout
        self.dropout = nn.Dropout(self.expert_dropout_rate)
        
    def _create_expert(self, hidden_size: int, intermediate_size: int) -> nn.Module:
        """
        创建一个专家网络。
        
        参数:
            hidden_size (int): 输入和输出特征的维度。
            intermediate_size (int): 中间特征的维度。
            
        返回:
            nn.Module: 专家网络模块。
        """
        # 专家通常是一个简单的前馈网络
        return nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, bias=False),
            nn.GELU(), # 如果其他部分使用SiLU或SwiGLU，可以考虑保持一致性
            nn.Linear(intermediate_size, hidden_size, bias=False),
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        TopKGatingMoELayer 的前向传播。
        
        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, sequence_length, hidden_size]。
            
        返回:
            元组，包含:
            - output_tensor (torch.Tensor): 形状为 [batch_size, sequence_length, hidden_size] 的输出张量。
            - aux_outputs (Dict[str, Any]): 包含辅助输出的字典，其中包括负载均衡损失。
        """
        batch_size, sequence_length, _ = x.shape
        
        # 获取路由器输出
        # TopKGatingRouter 返回的 router_aux 包含 "aux_loss"
        routing_weights, expert_indices, router_aux = self.router(x)
        
        # 重塑输入以便专家处理
        x_reshaped = x.view(-1, self.hidden_size)  # [batch_size * sequence_length, hidden_size]
        
        # 初始化输出张量
        final_output = torch.zeros_like(x_reshaped)
        
        # 处理每个专家
        # 这是一个常见的MoE分发实现方式，但可能较慢。
        # 对于少量专家（例如您配置中的8个），这可能是可接受的。
        for i in range(self.num_experts): # 遍历专家列表中的每个专家
            expert = self.experts[i]
            # expert_mask: [batch_size, sequence_length, top_k]
            # token_indices: 被路由到专家 'i' 的令牌在展平后 x_reshaped 中的索引
            # top_k_slot_indices: 对于该令牌，其 top_k 选择中哪一个对应于专家 'i'
            # .view(-1, self.top_k) 将 expert_indices 展平为 [N*L, top_k]
            token_indices, top_k_slot_indices = torch.where(expert_indices.view(-1, self.top_k) == i)
            
            if token_indices.numel() > 0: # 如果有令牌被路由到这个专家
                expert_inputs = x_reshaped[token_indices] # 获取这些令牌的输入
                expert_outputs_for_expert_i = expert(expert_inputs) # 通过专家网络处理
                
                # 获取这些特定令牌-专家分配的权重
                # .view(-1, self.top_k) 将 routing_weights 展平为 [N*L, top_k]
                # .unsqueeze(1) 将权重形状变为 [num_selected_tokens, 1] 以便广播
                weights_for_expert_i = routing_weights.view(-1, self.top_k)[token_indices, top_k_slot_indices].unsqueeze(1)
                
                # 使用 index_add_ 将加权后的专家输出累加到 final_output
                # 这确保了梯度能够正确传播
                final_output.index_add_(0, token_indices, expert_outputs_for_expert_i * weights_for_expert_i)
        
        # 应用 dropout
        final_output = self.dropout(final_output)
        
        # 将输出重塑回原始形状
        final_output = final_output.view(batch_size, sequence_length, self.hidden_size)
        
        # 准备辅助输出给 Trainer
        # CustomMoETransformer 期望在这个字典中直接找到 'aux_loss' 键
        aux_outputs = {
            "router_aux": router_aux, # 包含路由器本身的原始 aux_loss
            "expert_indices": expert_indices,
            "routing_weights": routing_weights,
            # 直接从 router_aux 中提取 "aux_loss"，如果不存在则默认为0
            "aux_loss": router_aux.get("aux_loss", torch.tensor(0.0, device=x.device, dtype=x.dtype))
        }
        
        return final_output, aux_outputs