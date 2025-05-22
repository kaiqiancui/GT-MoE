import torch
import torch.nn as nn
from typing import Optional


class ReputationScorer:
    """
    Handles the calculation and updating of dynamic reputation scores R_i(t) for experts.
    
    The reputation score is updated using an Exponential Moving Average (EMA):
    R_i(t) = α * current_performance_i + (1-α) * R_i(t-1)
    
    Attributes:
        num_experts (int): Number of experts in the model
        alpha (float): EMA smoothing factor controlling how much weight to give to recent performance
        use_decay (bool): Whether to apply reputation decay for unused experts
        decay_rate (float): Rate at which reputation decays for unused experts
    """
    
    def __init__(
        self,
        num_experts: int,
        alpha: float = 0.9,
        use_decay: bool = True,
        decay_rate: float = 0.99,
    ):
        """
        Initialize the ReputationScorer.
        
        Args:
            num_experts: Number of experts to track reputation for
            alpha: EMA smoothing factor (0 <= alpha <= 1)
            use_decay: Whether to apply reputation decay for unused experts
            decay_rate: Rate at which reputation decays for unused experts
        """
        self.num_experts = num_experts
        self.alpha = alpha
        self.use_decay = use_decay
        self.decay_rate = decay_rate
        
        # Initialize reputation scores to zeros
        self.reputation_scores = torch.zeros(num_experts)
        
    def calculate_performance_from_activations(
        self,
        expert_outputs: torch.Tensor,
        expert_indices: torch.Tensor,
        top_k: int
    ) -> torch.Tensor:
        """
        Calculate current_performance_i based on expert activation norms.
        
        This implements the "based on activation norm" approach mentioned in the algorithm
        description, where the L2 norm of expert outputs is used as a proxy for expert
        confidence/performance.
        
        Args:
            expert_outputs: Tensor of shape [batch_size, sequence_length, top_k, hidden_size]
                containing the outputs from each expert before gating
            expert_indices: Tensor of shape [batch_size, sequence_length, top_k]
                containing indices of selected experts
            top_k: Number of experts selected for each token
                
        Returns:
            Tensor of shape [batch_size, sequence_length, top_k] containing
            performance metrics for each selected expert
        """
        # Calculate L2 norm of expert outputs as a performance metric
        # This serves as a proxy for expert "confidence" or "awareness"
        performance_metrics = torch.norm(expert_outputs, p=2, dim=-1)
        
        return performance_metrics
    
    def update_reputation_scores(
        self,
        expert_indices: torch.Tensor,
        current_performances: torch.Tensor,
    ) -> None:
        """
        Update reputation scores based on current performances.
        
        Args:
            expert_indices: Tensor of shape [batch_size, sequence_length, top_k]
                containing indices of selected experts
            current_performances: Tensor of shape [batch_size, sequence_length, top_k]
                containing performance metrics for each selected expert
        """
        # Flatten tensors for easier processing
        batch_size, seq_len, top_k = expert_indices.shape
        expert_indices_flat = expert_indices.view(-1, top_k)
        current_performances_flat = current_performances.view(-1, top_k)
        
        # Initialize accumulators for performance metrics
        performance_sum = torch.zeros(self.num_experts, device=expert_indices.device)
        performance_count = torch.zeros(self.num_experts, device=expert_indices.device)
        
        # Aggregate performance metrics for each expert
        for i in range(expert_indices_flat.size(0)):
            for k in range(top_k):
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
        if self.use_decay:
            # Decay reputation scores of experts that weren't selected in this batch
            unused_mask = performance_count == 0
            self.reputation_scores[unused_mask] *= self.decay_rate
            
    def get_reputation_scores(self) -> torch.Tensor:
        """
        Get the current reputation scores for all experts.
        
        Returns:
            Tensor of shape [num_experts] containing reputation scores
        """
        return self.reputation_scores
