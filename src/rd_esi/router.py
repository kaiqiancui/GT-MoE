import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Any, Optional


class RDESIRouter(nn.Module):
    """
    Reputation-based Dynamic Expert Selection with Adaptive Incentives (RD-ESI) Router.
    
    This router implements the RD-ESI mechanism which combines:
    1. Base gating scores (g_i(x))
    2. Dynamic reputation scores (R_i(t))
    3. Load awareness (L_i(t))
    4. Exploration bonuses
    
    The final selection score is calculated as:
    SelectionScore_i(x,t) = g_i(x) + β * R_i(t) - γ * L_i(t) + ExplorationBonus_i(x,t)
    
    Attributes:
        hidden_size (int): Dimension of input features
        num_experts (int): Number of experts to route between
        top_k (int): Number of experts to select for each token
        beta (float): Weight for reputation score
        gamma (float): Weight for load penalty
        alpha (float): Smoothing factor for reputation EMA updates
        use_exploration_bonus (bool): Whether to use exploration bonus
        exploration_c (float): Constant for UCB exploration bonus
        use_reputation_decay (bool): Whether to use reputation decay
        decay_rate (float): Rate at which reputation decays
        load_ema_alpha (float): Smoothing factor for load EMA updates
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
        Initialize the RD-ESI Router.
        
        Args:
            hidden_size: Dimension of input features
            num_experts: Number of experts to route between
            top_k: Number of experts to select for each token
            beta: Weight for reputation score
            gamma: Weight for load penalty
            alpha: Smoothing factor for reputation EMA updates
            use_exploration_bonus: Whether to use exploration bonus
            exploration_c: Constant for UCB exploration bonus
            use_reputation_decay: Whether to use reputation decay
            decay_rate: Rate at which reputation decays
            load_ema_alpha: Smoothing factor for load EMA updates
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
        
        # Base gating projector (g_i(x))
        self.gate_projector = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Register buffers for persistent state across training steps
        # Reputation scores R_i(t)
        self.register_buffer("reputation_scores", torch.zeros(num_experts))
        # Load tracking L_i(t)
        self.register_buffer("expert_loads", torch.zeros(num_experts))
        # Expert selection counts N_i(t) for exploration bonus
        self.register_buffer("expert_counts", torch.zeros(num_experts))
        # Total routing decisions made (N)
        self.register_buffer("total_routing_decisions", torch.tensor(0))
        
    def forward(
        self, 
        x: torch.Tensor,
        current_batch_assignments: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass for the RD-ESI router.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, hidden_size]
            current_batch_assignments: Optional tensor indicating current expert assignments
                in the batch, used for load calculation
                
        Returns:
            Tuple containing:
            - routing_weights: Tensor of shape [batch_size, sequence_length, top_k]
              containing normalized weights for selected experts
            - expert_indices: Tensor of shape [batch_size, sequence_length, top_k]
              containing indices of selected experts
            - aux_outputs: Dictionary with auxiliary outputs for state updates
        """
        batch_size, sequence_length, hidden_size = x.shape
        
        # Reshape input for routing
        x_reshaped = x.view(-1, hidden_size)  # [batch_size * sequence_length, hidden_size]
        
        # 1. Calculate base gating scores g_i(x)
        base_logits = self.gate_projector(x_reshaped)  # [batch_size * sequence_length, num_experts]
        
        # 2. Update expert loads L_i(t)
        if current_batch_assignments is not None:
            # If we have current batch assignments, use them to update loads
            # This would typically be a tensor counting how many tokens are assigned to each expert
            current_loads = current_batch_assignments
        else:
            # Otherwise, use a simple approximation based on previous routing decisions
            current_loads = self.expert_loads
            
        # Update loads using EMA
        updated_loads = self.load_ema_alpha * current_loads + (1 - self.load_ema_alpha) * self.expert_loads
        
        # 3. Calculate exploration bonus (if enabled)
        exploration_bonus = torch.zeros_like(base_logits)
        if self.use_exploration_bonus:
            # Add small epsilon to prevent division by zero
            epsilon = 1e-10
            # UCB-like exploration bonus: C * sqrt(log(N) / N_i)
            exploration_term = self.exploration_c * torch.sqrt(
                torch.log(self.total_routing_decisions + 1.0) / (self.expert_counts.unsqueeze(0) + epsilon)
            )
            exploration_bonus = exploration_term.expand_as(base_logits)
        
        # 4. Calculate final selection scores
        # SelectionScore_i(x,t) = g_i(x) + β * R_i(t) - γ * L_i(t) + ExplorationBonus_i(x,t)
        selection_scores = (
            base_logits +  # g_i(x)
            self.beta * self.reputation_scores.unsqueeze(0) -  # β * R_i(t)
            self.gamma * updated_loads.unsqueeze(0) +  # γ * L_i(t)
            exploration_bonus  # ExplorationBonus_i(x,t)
        )
        
        # 5. Select top-k experts
        top_k_scores, top_k_indices = torch.topk(selection_scores, self.top_k, dim=-1)
        
        # 6. Compute routing weights (softmax over selected experts' scores)
        routing_weights = F.softmax(top_k_scores, dim=-1)
        
        # Reshape outputs to match input dimensions
        routing_weights = routing_weights.view(batch_size, sequence_length, self.top_k)
        expert_indices = top_k_indices.view(batch_size, sequence_length, self.top_k)
        
        # Prepare auxiliary outputs for state updates
        aux_outputs = {
            "base_logits": base_logits,
            "selection_scores": selection_scores,
            "updated_loads": updated_loads,
            "expert_indices": expert_indices,
        }
        
        return routing_weights, expert_indices, aux_outputs
    
    def update_states(
        self,
        expert_indices: torch.Tensor,
        current_performances: torch.Tensor,
        batch_size: int,
        sequence_length: int,
    ) -> None:
        """
        Update router states based on routing decisions and expert performances.
        
        Args:
            expert_indices: Tensor of shape [batch_size, sequence_length, top_k]
              containing indices of selected experts
            current_performances: Tensor of shape [batch_size, sequence_length, top_k]
              containing performance metrics for each selected expert
            batch_size: Batch size
            sequence_length: Sequence length
        """
        # Flatten indices for counting
        flat_indices = expert_indices.view(-1)
        
        # Update expert counts for exploration bonus
        new_counts = torch.bincount(flat_indices, minlength=self.num_experts).float()
        self.expert_counts.add_(new_counts)
        
        # Update total routing decisions
        self.total_routing_decisions.add_(batch_size * sequence_length * self.top_k)
        
        # Aggregate current performance metrics for each expert
        # Initialize performance accumulator
        performance_sum = torch.zeros_like(self.reputation_scores)
        performance_count = torch.zeros_like(self.reputation_scores)
        
        # Reshape for easier processing
        expert_indices_flat = expert_indices.view(-1, self.top_k)
        current_performances_flat = current_performances.view(-1, self.top_k)
        
        # For each token and selected expert, accumulate performance
        for i in range(expert_indices_flat.size(0)):
            for k in range(self.top_k):
                expert_idx = expert_indices_flat[i, k].item()
                performance = current_performances_flat[i, k].item()
                performance_sum[expert_idx] += performance
                performance_count[expert_idx] += 1
        
        # Compute average performance for each expert
        # Avoid division by zero
        mask = performance_count > 0
        avg_performance = torch.zeros_like(performance_sum)
        avg_performance[mask] = performance_sum[mask] / performance_count[mask]
        
        # Update reputation scores using EMA
        # R_i(t) = α * current_performance_i + (1-α) * R_i(t-1)
        self.reputation_scores = self.alpha * avg_performance + (1 - self.alpha) * self.reputation_scores
        
        # Apply reputation decay if enabled
        if self.use_reputation_decay:
            # Decay reputation scores of experts that weren't selected in this batch
            unused_mask = performance_count == 0
            self.reputation_scores[unused_mask] *= self.decay_rate
        
        # Update expert loads based on current batch
        # This is a simple approximation - in practice, more sophisticated load tracking might be used
        self.expert_loads = self.load_ema_alpha * new_counts + (1 - self.load_ema_alpha) * self.expert_loads
