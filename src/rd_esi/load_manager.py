import torch
import torch.nn as nn
from typing import Optional


class LoadManager:
    """
    Manages and tracks the load L_i(t) for each expert in the MoE model.
    
    The load can be calculated in different ways:
    1. Queue Length: Number of tokens assigned to each expert in the current batch
    2. Recent Usage Frequency: EMA of expert selection counts over time
    3. Capacity Utilization: Ratio of assigned tokens to expert capacity
    
    Attributes:
        num_experts (int): Number of experts in the model
        load_ema_alpha (float): Smoothing factor for EMA load calculation
        use_capacity_factor (bool): Whether to use capacity-based load calculation
        capacity_factor (float): Capacity factor for each expert
    """
    
    def __init__(
        self,
        num_experts: int,
        load_ema_alpha: float = 0.9,
        use_capacity_factor: bool = False,
        capacity_factor: float = 1.0,
    ):
        """
        Initialize the LoadManager.
        
        Args:
            num_experts: Number of experts to track load for
            load_ema_alpha: EMA smoothing factor (0 <= load_ema_alpha <= 1)
            use_capacity_factor: Whether to use capacity-based load calculation
            capacity_factor: Capacity factor for each expert
        """
        self.num_experts = num_experts
        self.load_ema_alpha = load_ema_alpha
        self.use_capacity_factor = use_capacity_factor
        self.capacity_factor = capacity_factor
        
        # Initialize expert loads to zeros
        self.expert_loads = torch.zeros(num_experts)
        
    def calculate_batch_loads(
        self,
        expert_indices: torch.Tensor,
        batch_size: int,
        sequence_length: int,
    ) -> torch.Tensor:
        """
        Calculate the current batch load for each expert based on token assignments.
        
        Args:
            expert_indices: Tensor of shape [batch_size, sequence_length, top_k]
                containing indices of selected experts
            batch_size: Batch size
            sequence_length: Sequence length
            
        Returns:
            Tensor of shape [num_experts] containing the current batch load for each expert
        """
        # Flatten expert indices
        flat_indices = expert_indices.view(-1)
        
        # Count occurrences of each expert in the current batch
        batch_loads = torch.bincount(flat_indices, minlength=self.num_experts).float()
        
        if self.use_capacity_factor:
            # Calculate capacity for each expert
            # Capacity = (batch_size * sequence_length * top_k) / num_experts * capacity_factor
            top_k = expert_indices.size(-1)
            total_tokens = batch_size * sequence_length * top_k
            capacity = (total_tokens / self.num_experts) * self.capacity_factor
            
            # Normalize load by capacity
            batch_loads = batch_loads / capacity
            
        return batch_loads
        
    def update_loads(
        self,
        batch_loads: torch.Tensor,
    ) -> None:
        """
        Update expert loads using EMA.
        
        Args:
            batch_loads: Tensor of shape [num_experts] containing the current batch load for each expert
        """
        # Update loads using EMA
        # L_i(t) = load_ema_alpha * batch_loads + (1 - load_ema_alpha) * L_i(t-1)
        self.expert_loads = self.load_ema_alpha * batch_loads + (1 - self.load_ema_alpha) * self.expert_loads
            
    def get_expert_loads(self) -> torch.Tensor:
        """
        Get the current load for all experts.
        
        Returns:
            Tensor of shape [num_experts] containing expert loads
        """
        return self.expert_loads
