import torch
import torch.nn as nn
import math
from typing import Optional


class MitigationStrategies:
    """
    Implements strategies to mitigate the Matthew effect in MoE routing.
    
    The Matthew effect is the phenomenon where high-reputation experts continue to be selected,
    gaining more opportunities to learn and improve, while low-reputation experts are rarely
    selected and have limited opportunities to improve.
    
    This class implements two main mitigation strategies:
    1. Exploration Bonus: Adds a bonus to expert selection scores to encourage exploration
    2. Reputation Decay: Gradually reduces the reputation of unused experts
    
    Attributes:
        num_experts (int): Number of experts in the model
        use_exploration_bonus (bool): Whether to use exploration bonus
        exploration_c (float): Constant for UCB exploration bonus
        use_epsilon_greedy (bool): Whether to use epsilon-greedy exploration
        epsilon (float): Probability of random expert selection in epsilon-greedy
        use_noise_injection (bool): Whether to add noise to selection scores
        noise_std (float): Standard deviation of noise for noise injection
        use_reputation_decay (bool): Whether to apply reputation decay
        decay_rate (float): Rate at which reputation decays
        decay_threshold (int): Number of steps after which to apply decay
    """
    
    def __init__(
        self,
        num_experts: int,
        use_exploration_bonus: bool = True,
        exploration_c: float = 0.1,
        use_epsilon_greedy: bool = False,
        epsilon: float = 0.05,
        use_noise_injection: bool = False,
        noise_std: float = 0.01,
        use_reputation_decay: bool = True,
        decay_rate: float = 0.99,
        decay_threshold: int = 1,
    ):
        """
        Initialize the MitigationStrategies.
        
        Args:
            num_experts: Number of experts in the model
            use_exploration_bonus: Whether to use UCB-like exploration bonus
            exploration_c: Constant for UCB exploration bonus
            use_epsilon_greedy: Whether to use epsilon-greedy exploration
            epsilon: Probability of random expert selection in epsilon-greedy
            use_noise_injection: Whether to add noise to selection scores
            noise_std: Standard deviation of noise for noise injection
            use_reputation_decay: Whether to apply reputation decay
            decay_rate: Rate at which reputation decays
            decay_threshold: Number of steps after which to apply decay
        """
        self.num_experts = num_experts
        self.use_exploration_bonus = use_exploration_bonus
        self.exploration_c = exploration_c
        self.use_epsilon_greedy = use_epsilon_greedy
        self.epsilon = epsilon
        self.use_noise_injection = use_noise_injection
        self.noise_std = noise_std
        self.use_reputation_decay = use_reputation_decay
        self.decay_rate = decay_rate
        self.decay_threshold = decay_threshold
        
        # Initialize expert counts for exploration bonus
        self.expert_counts = torch.zeros(num_experts)
        self.total_routing_decisions = 0
        
    def calculate_exploration_bonus(
        self,
        selection_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate exploration bonus for expert selection.
        
        Args:
            selection_scores: Tensor of shape [batch_size * sequence_length, num_experts]
                containing the current selection scores
                
        Returns:
            Tensor of the same shape as selection_scores with exploration bonus applied
        """
        batch_size_seq_len = selection_scores.size(0)
        modified_scores = selection_scores.clone()
        
        if self.use_exploration_bonus:
            # Add small epsilon to prevent division by zero
            epsilon = 1e-10
            
            # UCB-like exploration bonus: C * sqrt(log(N) / N_i)
            exploration_term = self.exploration_c * torch.sqrt(
                torch.log(self.total_routing_decisions + 1.0) / 
                (self.expert_counts.unsqueeze(0) + epsilon)
            )
            
            # Add exploration bonus to selection scores
            modified_scores += exploration_term.expand_as(selection_scores)
            
        if self.use_epsilon_greedy and self.training:
            # Epsilon-greedy exploration
            # With probability epsilon, replace scores with random values
            random_mask = torch.rand(batch_size_seq_len, 1, device=selection_scores.device) < self.epsilon
            if random_mask.any():
                random_scores = torch.rand_like(selection_scores)
                modified_scores = torch.where(random_mask, random_scores, modified_scores)
                
        if self.use_noise_injection:
            # Add Gaussian noise to selection scores
            noise = torch.randn_like(selection_scores) * self.noise_std
            modified_scores += noise
            
        return modified_scores
    
    def apply_reputation_decay(
        self,
        reputation_scores: torch.Tensor,
        expert_usage: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply reputation decay to unused experts.
        
        Args:
            reputation_scores: Tensor of shape [num_experts] containing current reputation scores
            expert_usage: Tensor of shape [num_experts] indicating which experts were used
                (1 for used, 0 for unused)
                
        Returns:
            Updated reputation scores after applying decay
        """
        if not self.use_reputation_decay:
            return reputation_scores
            
        # Create a mask for unused experts
        unused_mask = expert_usage == 0
        
        # Apply decay to unused experts
        decayed_scores = reputation_scores.clone()
        decayed_scores[unused_mask] *= self.decay_rate
        
        return decayed_scores
    
    def update_counts(
        self,
        expert_indices: torch.Tensor,
        batch_size: int,
        sequence_length: int,
    ) -> None:
        """
        Update expert selection counts for exploration bonus calculation.
        
        Args:
            expert_indices: Tensor of shape [batch_size, sequence_length, top_k]
                containing indices of selected experts
            batch_size: Batch size
            sequence_length: Sequence length
        """
        # Flatten indices for counting
        flat_indices = expert_indices.view(-1)
        
        # Update expert counts
        new_counts = torch.bincount(flat_indices, minlength=self.num_experts).float()
        self.expert_counts += new_counts
        
        # Update total routing decisions
        top_k = expert_indices.size(-1)
        self.total_routing_decisions += batch_size * sequence_length * top_k
