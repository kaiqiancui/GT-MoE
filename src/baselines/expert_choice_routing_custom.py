import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional


class ExpertChoiceRouter(nn.Module):
    """
    用于 MoE 模型的 Expert Choice 路由器。
    
    与令牌选择专家（Top-K）的路由方式不同，在 Expert Choice 路由中，是专家选择令牌。
    每个专家选择与其最相关的前k个令牌。
    这种机制旨在确保完美的负载均衡，因为每个专家处理完全相同数量的令牌（由其容量决定）。
    
    属性:
        hidden_size (int): 输入特征的维度。
        num_experts (int): 用于路由的专家数量。
        capacity_factor (float): 决定每个专家处理多少令牌的因子。
        gate_projector (nn.Linear): 用于计算令牌-专家亲和度的线性层。
        expert_counts (torch.Tensor): 跟踪每个专家在训练期间被分配到的令牌总数。
                                     也通过 self.expert_loads 别名暴露，以兼容 Trainer。
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        capacity_factor: float = 1.0, # 每个专家容量 = capacity_factor * (num_tokens / num_experts)
    ):
        """
        初始化 ExpertChoiceRouter。
        
        参数:
            hidden_size (int): 输入特征的维度。
            num_experts (int): 用于路由的专家数量。
            capacity_factor (float): 决定每个专家处理多少令牌的因子。
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        
        # 用于计算令牌-专家亲和度的门控投影器
        self.gate_projector = nn.Linear(hidden_size, num_experts, bias=False)

        # 注册缓冲区以跟踪每个专家被分配的令牌数（主要用于训练期间的指标监控）
        self.register_buffer("expert_counts", torch.zeros(num_experts, dtype=torch.float32))
        # 创建别名以兼容 Trainer 可能期望的 'expert_loads' 属性
        self.expert_loads = self.expert_counts
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        ExpertChoiceRouter 的前向传播。
        
        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, sequence_length, hidden_size]。
            
        返回:
            元组，包含:
            - routing_weights (torch.Tensor): 形状为 [batch_size, sequence_length] 的张量，
                                             包含每个令牌被其选定专家处理的权重（或分数）。
            - expert_indices (torch.Tensor): 形状为 [batch_size, sequence_length] 的张量，
                                            包含每个令牌被分配到的专家索引。
            - aux_outputs (Dict[str, Any]): 包含辅助输出的字典（通常不包含aux_loss，因为负载均衡是内建的）。
        """
        batch_size, sequence_length, _ = x.shape # 使用 _ 接收 hidden_size
        
        # 重塑输入以进行路由
        x_reshaped = x.view(-1, self.hidden_size)  # [N*L, D] 其中 N=batch_size, L=sequence_length, D=hidden_size
        num_tokens = x_reshaped.shape[0]
        
        # 计算令牌-专家亲和度 (logits)
        # 形状: [num_tokens, num_experts]
        router_logits = self.gate_projector(x_reshaped)
        
        # 计算路由概率 (通常在 expert choice 中，专家基于 logits 直接选择，而不是基于 token 的 softmax 概率)
        # 但这里的 router_probs (token视角) 主要用于处理未分配的token，或作为辅助信息
        # 形状: [num_tokens, num_experts]
        router_probs_for_unassigned = F.softmax(router_logits, dim=-1) # 用于后续处理未分配的令牌
        
        # 计算专家容量：每个专家将处理的令牌数量
        # tokens_per_expert 是专家选择的 "top-k" 中的 k
        tokens_per_expert = int(num_tokens * self.capacity_factor / self.num_experts)
        tokens_per_expert = max(1, tokens_per_expert)  # 确保每个专家至少处理1个令牌
        tokens_per_expert = min(tokens_per_expert, num_tokens) # 不能超过总令牌数

        # 获取专家对令牌的亲和度分数 (转置router_logits)
        # 形状: [num_experts, num_tokens]
        # 使用原始 logits 进行 top-k 选择，因为 softmax 可能会压缩分数差异
        expert_token_affinities = router_logits.t() 
        
        # 每个专家选择其亲和度最高的 top-k (即 tokens_per_expert) 个令牌
        # expert_token_scores_selected: [num_experts, tokens_per_expert] 选定令牌的分数
        # expert_token_indices_selected: [num_experts, tokens_per_expert] 选定令牌的原始索引 (在 num_tokens 维度上)
        expert_token_scores_selected, expert_token_indices_selected = torch.topk(
            expert_token_affinities, k=tokens_per_expert, dim=1
        )
        
        # 创建一个从令牌到其被分配的专家的映射
        # 初始化为 -1，表示令牌尚未被任何专家选中
        # token_to_expert_assignment: [num_tokens] 记录每个令牌最终被分配给哪个专家
        # token_score_from_assigned_expert: [num_tokens] 记录每个令牌从其分配的专家那里获得的分数
        token_to_expert_assignment = torch.full((num_tokens,), -1, dtype=torch.long, device=x.device)
        token_score_from_assigned_expert = torch.zeros((num_tokens,), dtype=x.dtype, device=x.device) # 使用 x.dtype

        # 解决冲突：一个令牌可能被多个专家选为 top-k。这里采用的策略是，令牌被分配给对其具有最高亲和度分数的专家。
        # 将 expert_token_indices_selected 展平，并创建对应的专家索引和分数张量
        # flat_selected_token_indices: [num_experts * tokens_per_expert]
        flat_selected_token_indices = expert_token_indices_selected.reshape(-1)
        # flat_expert_indices_for_selection: [num_experts * tokens_per_expert], 代表是哪个专家做出的选择
        flat_expert_indices_for_selection = torch.arange(self.num_experts, device=x.device).unsqueeze(1).expand(-1, tokens_per_expert).reshape(-1)
        # flat_expert_token_scores_selected: [num_experts * tokens_per_expert]
        flat_expert_token_scores_selected = expert_token_scores_selected.reshape(-1)

        # 使用 scatter_max 来高效地为每个令牌找到选择它的、亲和度最高的专家
        # 先用一个足够小的负数填充，确保任何有效分数都比它大
        # index_fill_value = torch.finfo(token_score_from_assigned_expert.dtype).min
        # token_score_from_assigned_expert.fill_(index_fill_value)
        # scatter_max_ 会更新 token_score_from_assigned_expert 和 token_to_expert_assignment
        # 注意: scatter_max_对于index参数有特定要求, 这里简化处理逻辑
        # 以下是原代码的循环逻辑，更易于理解：
        for expert_idx in range(self.num_experts):
            current_expert_selected_tokens = expert_token_indices_selected[expert_idx]
            current_expert_selected_scores = expert_token_scores_selected[expert_idx]
            
            for token_idx, score in zip(current_expert_selected_tokens, current_expert_selected_scores):
                if token_to_expert_assignment[token_idx] == -1 or score > token_score_from_assigned_expert[token_idx]:
                    token_to_expert_assignment[token_idx] = expert_idx
                    token_score_from_assigned_expert[token_idx] = score
        
        # 处理可能未被任何专家选中的令牌 (如果 capacity_factor * num_tokens < num_tokens)
        # 这种情况通常在 capacity_factor < 1.0 时发生，或者由于 topk 选择的限制
        unassigned_tokens_mask = (token_to_expert_assignment == -1)
        if unassigned_tokens_mask.any():
            unassigned_token_indices = unassigned_tokens_mask.nonzero().squeeze(-1)
            if unassigned_token_indices.numel() > 0:
                # 对于未分配的令牌，根据它们对所有专家的原始亲和度概率，将它们分配给概率最高的专家
                unassigned_probs = router_probs_for_unassigned[unassigned_token_indices] # 使用softmax概率
                assigned_experts_for_unassigned = torch.argmax(unassigned_probs, dim=1)
                token_to_expert_assignment[unassigned_token_indices] = assigned_experts_for_unassigned
                
                # 更新这些令牌的分数 (可以使用原始logits或概率作为分数)
                # 这里我们使用它们被分配到的专家的原始router_logits值作为分数，以保持一致性
                # 或者也可以用 router_probs_for_unassigned
                for i, token_idx in enumerate(unassigned_token_indices):
                    assigned_expert = assigned_experts_for_unassigned[i]
                    # 使用原始logits可能更好，因为它未经softmax的尺度变换
                    token_score_from_assigned_expert[token_idx] = router_logits[token_idx, assigned_expert]


        # 更新 expert_counts (用于监控和 Trainer 中的指标)
        if self.training or True: # 假设在评估时也可能需要这些计数
            # token_to_expert_assignment 中可能仍有 -1 (如果 capacity_factor 非常小且没有处理未分配令牌的逻辑)
            # 我们只对实际分配了的令牌进行计数
            valid_assignments = token_to_expert_assignment[token_to_expert_assignment != -1]
            if valid_assignments.numel() > 0:
                current_batch_expert_counts = torch.bincount(valid_assignments, minlength=self.num_experts).to(self.expert_counts.device, dtype=self.expert_counts.dtype)
                # Trainer 通常期望 router.expert_counts 是一个在整个训练过程中累积的计数
                # 如果是这样，应该使用 self.expert_counts.add_()
                # 如果 Trainer 每次都获取当前批次的分配情况，则直接赋值或返回 current_batch_expert_counts
                # 根据之前的 TopKGatingRouter 修改，我们假设 Trainer 期望累积计数
                self.expert_counts.add_(current_batch_expert_counts)


        # 将 token_to_expert_assignment 重塑为批次维度
        # expert_indices_output: [batch_size, sequence_length] 每个令牌对应的专家索引
        expert_indices_output = token_to_expert_assignment.view(batch_size, sequence_length)
        
        # routing_weights_output: [batch_size, sequence_length] 每个令牌对应的路由权重/分数
        # 这个权重将用于 ExpertChoiceMoELayer 中对专家输出进行加权
        routing_weights_output = token_score_from_assigned_expert.view(batch_size, sequence_length)
        # 注意: Expert Choice 的 "权重" 通常是二进制的 (要么选要么不选)，
        # 但这里我们返回了分数，MoELayer 可以选择如何使用它 (例如，直接乘以专家输出，或者先通过softmax处理)
        # 实际的 Switch Transformers 和一些实现中，这个权重就是 router_probs[token_indices, expert_indices_for_those_tokens]
        # 这里我们用的是 token_score_from_assigned_expert，它源于 topk 选择的 expert_token_affinities 或 router_logits
        # 为了让 MoE 层能像 Top-K 那样使用 softmax 归一化的权重，可以考虑在这里对 token_score_from_assigned_expert 做处理
        # 但通常 ExpertChoice 的输出直接就是专家处理的结果，权重体现在选择本身。
        # 这里返回的 routing_weights_output 将由 ExpertChoiceMoELayer 使用。
        # 对于 Expert Choice，通常令牌只被一个专家处理，所以权重更像是一个指示器或原始分数。
        # 在 ExpertChoiceMoELayer 中，这个 routing_weight 可能会被用来乘以专家输出。
        
        # 准备辅助输出
        aux_outputs = {
            "router_logits": router_logits,
            # router_probs_for_unassigned 仅用于未分配令牌的处理，可能不需要传出
            # "router_probs": router_probs_for_unassigned, 
        }
        
        # ExpertChoice 通常没有辅助损失，因为负载均衡是通过其设计实现的
        # 但为了与 Trainer 和其他 MoE 层的接口保持一致，可以显式地添加一个零值的 aux_loss
        aux_outputs["aux_loss"] = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        return routing_weights_output, expert_indices_output, aux_outputs


class ExpertChoiceMoELayer(nn.Module):
    """
    采用 Expert Choice 路由的 MoE 层。
    
    该层将 ExpertChoiceRouter 与一组专家网络集成。
    
    属性:
        hidden_size (int): 输入和输出特征的维度。
        num_experts (int): 层中的专家数量。
        capacity_factor (float): 决定每个专家处理多少令牌的因子。
        router (ExpertChoiceRouter): 用于专家选择的路由器。
        experts (nn.ModuleList): 专家网络列表。
        expert_dropout_rate (float): 专家输出的 dropout 概率。
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        expert_dropout: float = 0.0, # 参数名与 self.dropout 模块一致，在内部重命名
    ):
        """
        初始化 ExpertChoiceMoELayer。
        
        参数:
            hidden_size (int): 输入和输出特征的维度。
            intermediate_size (int): 每个专家的中间特征维度。
            num_experts (int): 层中的专家数量。
            capacity_factor (float): 决定每个专家处理多少令牌的因子。
            expert_dropout (float): 专家输出的 dropout 概率。
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.expert_dropout_rate = expert_dropout # 将输入参数保存到新变量名
        
        # 初始化路由器
        self.router = ExpertChoiceRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
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
        return nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, bias=False),
            nn.GELU(), # 与 TopKGatingMoELayer 中的专家激活函数保持一致
            nn.Linear(intermediate_size, hidden_size, bias=False),
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        ExpertChoiceMoELayer 的前向传播。
        
        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, sequence_length, hidden_size]。
            
        返回:
            元组，包含:
            - output_tensor (torch.Tensor): 形状为 [batch_size, sequence_length, hidden_size] 的输出张量。
            - aux_outputs (Dict[str, Any]): 包含辅助输出的字典。
        """
        batch_size, sequence_length, _ = x.shape # 使用 _ 接收 hidden_size
        
        # 获取路由器输出
        # routing_weights_from_router: [batch_size, sequence_length], 每个token的分数/权重
        # expert_indices_from_router: [batch_size, sequence_length], 每个token被分配的专家索引
        # router_aux: 包含 "aux_loss" (为0) 和其他路由信息的字典
        routing_weights_from_router, expert_indices_from_router, router_aux = self.router(x)
        
        # 重塑输入以进行专家处理
        x_reshaped = x.view(-1, self.hidden_size)  # [N*L, D]
        
        # 初始化输出张量
        final_output = torch.zeros_like(x_reshaped)
        
        # expert_indices_from_router 是展平的，形状为 [N*L]
        flat_expert_indices = expert_indices_from_router.view(-1)
        # routing_weights_from_router 也是展平的，形状为 [N*L]
        flat_routing_weights = routing_weights_from_router.view(-1)

        for expert_idx in range(self.num_experts):
            expert_module = self.experts[expert_idx]
            
            # 找到被分配给当前专家的令牌的索引
            # token_indices_for_current_expert 是一个一维张量，包含所有被路由到 expert_idx 的令牌的索引（在 x_reshaped 中）
            token_indices_for_current_expert = (flat_expert_indices == expert_idx).nonzero(as_tuple=False).squeeze(-1)
            
            if token_indices_for_current_expert.numel() > 0:
                # 获取这些令牌的输入
                expert_inputs = x_reshaped[token_indices_for_current_expert]
                
                # 通过专家网络处理
                expert_processing_output = expert_module(expert_inputs)
                
                # 获取这些令牌的路由权重/分数，并调整形状以进行乘法
                # weights_for_these_tokens: [num_tokens_for_expert, 1]
                weights_for_these_tokens = flat_routing_weights[token_indices_for_current_expert].unsqueeze(-1)
                
                # 将加权后的专家输出加到最终输出中
                # 注意：在许多Expert Choice实现中，权重通常是1（因为专家选择了这些令牌），
                # 或者这里的权重直接就是 router logits 的值，然后后续可能没有额外的softmax。
                # 如果 routing_weights_from_router 是原始的亲和度分数，那么这样乘是合理的。
                final_output.index_add_(0, token_indices_for_current_expert, expert_processing_output * weights_for_these_tokens)
        
        # 应用 dropout
        final_output = self.dropout(final_output)
        
        # 将输出重塑回原始形状
        final_output = final_output.view(batch_size, sequence_length, self.hidden_size)
        
        # 准备辅助输出
        # 确保 "aux_loss" 键存在，即使 ExpertChoice 通常不需要辅助损失
        aux_outputs = {
            "router_aux": router_aux, # 包含路由器自身的辅助输出 (例如 "aux_loss": 0)
            "expert_indices": expert_indices_from_router, # [batch_size, sequence_length]
            "routing_weights": routing_weights_from_router, # [batch_size, sequence_length]
            "aux_loss": router_aux.get("aux_loss", torch.tensor(0.0, device=x.device, dtype=x.dtype))
        }
        
        return final_output, aux_outputs