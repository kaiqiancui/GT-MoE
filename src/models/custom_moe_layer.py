import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rd_esi.router import RDESIRouter


class Expert(nn.Module):
    """
    Expert module for MoE layers.
    
    This is a simple feed-forward network with SwiGLU activation.
    
    Attributes:
        w1 (nn.Linear): First linear layer
        w2 (nn.Linear): Second linear layer
        w3 (nn.Linear): Third linear layer for gating
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        """
        Initialize the Expert module.
        
        Args:
            hidden_size: Dimension of input and output features
            intermediate_size: Dimension of intermediate features
        """
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert module.
        
        Args:
            x: Input tensor of shape [batch_size, hidden_size]
            
        Returns:
            Output tensor of shape [batch_size, hidden_size]
        """
        # SwiGLU activation: (x * W1) * SiLU(x * W3)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CustomMoELayer(nn.Module):
    """
    Custom Mixture of Experts layer with RD-ESI routing.
    
    This layer integrates the RD-ESI router with a set of expert networks.
    
    Attributes:
        hidden_size (int): Dimension of input and output features
        num_experts (int): Number of experts in the layer
        top_k (int): Number of experts to select for each token
        router (RDESIRouter): RD-ESI router for expert selection
        experts (nn.ModuleList): List of expert networks
        expert_dropout (float): Dropout probability for expert outputs
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int = 2,
        router_config: Optional[Dict[str, Any]] = None,
        expert_dropout: float = 0.0,
    ):
        """
        Initialize the CustomMoELayer.
        
        Args:
            hidden_size: Dimension of input and output features
            intermediate_size: Dimension of intermediate features for each expert
            num_experts: Number of experts in the layer
            top_k: Number of experts to select for each token
            router_config: Configuration for the RD-ESI router
            expert_dropout: Dropout probability for expert outputs
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_dropout = expert_dropout
        
        # Initialize RD-ESI router
        router_kwargs = router_config or {}
        self.router = RDESIRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            **router_kwargs
        )
        
        # Initialize experts
        self.experts = nn.ModuleList([
            Expert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])
        
        # Dropout for expert outputs
        self.dropout = nn.Dropout(expert_dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass for the CustomMoELayer.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, hidden_size]
            
        Returns:
            Tuple containing:
            - Output tensor of shape [batch_size, sequence_length, hidden_size]
            - Dictionary with auxiliary outputs for metrics and state updates
        """
        batch_size, sequence_length, hidden_size = x.shape
        
        # Get router outputs
        routing_weights, expert_indices, router_aux = self.router(x)
        
        # Reshape input for expert processing
        x_reshaped = x.view(-1, hidden_size)  # [batch_size * sequence_length, hidden_size]
        
        # Initialize output tensor
        final_output = torch.zeros_like(x_reshaped)
        
        # Initialize tensor to store expert outputs for performance calculation
        expert_outputs = torch.zeros(
            batch_size * sequence_length, 
            self.top_k, 
            hidden_size, 
            device=x.device
        )
        
        # Process each expert
        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            # For each token, find which of its top_k slots (if any) selected this expert
            expert_mask = (expert_indices == i)  # [batch_size, sequence_length, top_k]
            expert_mask_flat = expert_mask.view(-1, self.top_k)  # [batch_size * sequence_length, top_k]
            
            # If no tokens routed to this expert, skip
            if not expert_mask_flat.any():
                continue
                
            # For each token-expert pair, get the corresponding weight and token index
            token_indices, top_k_indices = torch.where(expert_mask_flat)
            
            # Get inputs for this expert
            expert_inputs = x_reshaped[token_indices]
            
            # Process inputs with this expert
            expert_output = expert(expert_inputs)
            
            # Store expert outputs for performance calculation
            for j, (token_idx, top_k_idx) in enumerate(zip(token_indices, top_k_indices)):
                expert_outputs[token_idx, top_k_idx] = expert_output[j]
            
            # Get weights for this expert
            weights = routing_weights.view(-1, self.top_k)[token_indices, top_k_indices].unsqueeze(-1)
            
            # Add weighted expert output to final output
            final_output.index_add_(
                0, 
                token_indices, 
                expert_output * weights
            )
        
        # Apply dropout
        final_output = self.dropout(final_output)
        
        # Reshape output back to original shape
        final_output = final_output.view(batch_size, sequence_length, hidden_size)
        
        # Calculate expert performance metrics based on output norms
        # This implements the "based on activation norm" approach for current_performance_i
        performance_metrics = torch.norm(expert_outputs, p=2, dim=-1)
        performance_metrics = performance_metrics.view(batch_size, sequence_length, self.top_k)
        
        # Prepare auxiliary outputs
        aux_outputs = {
            "router_aux": router_aux,
            "expert_indices": expert_indices,
            "routing_weights": routing_weights,
            "performance_metrics": performance_metrics,
        }
        
        return final_output, aux_outputs
    
    def update_router_state(
        self,
        expert_indices: torch.Tensor,
        performance_metrics: torch.Tensor,
        batch_size: int,
        sequence_length: int,
    ) -> None:
        """
        Update router state based on routing decisions and expert performances.
        
        Args:
            expert_indices: Tensor of shape [batch_size, sequence_length, top_k]
                containing indices of selected experts
            performance_metrics: Tensor of shape [batch_size, sequence_length, top_k]
                containing performance metrics for each selected expert
            batch_size: Batch size
            sequence_length: Sequence length
        """
        self.router.update_states(
            expert_indices=expert_indices,
            current_performances=performance_metrics,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
