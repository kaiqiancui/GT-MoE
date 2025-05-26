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
        
        # ---------- VECTORIZED EXPERT COMPUTATION ----------
        # Step 1: Reshape input and routing tensors
        # Flatten the input tensor from [batch_size, sequence_length, hidden_size] to [batch_size*sequence_length, hidden_size]
        flat_x = x.reshape(-1, hidden_size)  # [batch_size * sequence_length, hidden_size]
        flat_routing_weights = routing_weights.reshape(-1, self.top_k)  # [batch_size * sequence_length, top_k]
        flat_expert_indices = expert_indices.reshape(-1, self.top_k)  # [batch_size * sequence_length, top_k]
        
        # Step 2: Create a token-to-expert mapping
        # Create a tensor of token indices
        token_indices = torch.arange(batch_size * sequence_length, device=x.device).unsqueeze(-1).expand(-1, self.top_k)
        
        # Step 3: Create a flattened representation for efficient processing
        # Stack token indices, expert indices, and top-k indices
        # This creates a tensor of shape [batch_size*sequence_length*top_k, 3]
        # Each row contains (token_idx, expert_idx, top_k_idx)
        token_expert_pairs = torch.stack([
            token_indices.reshape(-1),                   # Token index
            flat_expert_indices.reshape(-1),            # Expert index
            torch.arange(self.top_k, device=x.device).repeat(batch_size * sequence_length)  # Top-k index
        ], dim=1)
        
        # Step 4: Sort by expert index for batch processing
        # This groups all tokens going to the same expert together
        sorted_pairs, sort_indices = torch.sort(token_expert_pairs[:, 1])
        token_expert_pairs = token_expert_pairs[sort_indices]
        
        # Step 5: Initialize output tensors
        # Create a tensor to store expert outputs for each token-expert pair
        expert_outputs = torch.zeros(batch_size * sequence_length * self.top_k, hidden_size, device=x.device)
        
        # Step 6: Process each expert in a vectorized way
        # Get the unique expert indices and their counts
        unique_expert_indices, expert_counts = torch.unique_consecutive(token_expert_pairs[:, 1], return_counts=True)
        
        # Track the current position in the sorted array
        pos = 0
        for expert_idx, count in zip(unique_expert_indices, expert_counts):
            if count == 0:
                continue
                
            # Get the token indices for this expert
            expert_token_indices = token_expert_pairs[pos:pos+count, 0]
            
            # Get the inputs for this expert
            expert_inputs = flat_x[expert_token_indices]
            
            # Process inputs with this expert
            expert_idx = expert_idx.item()  # Convert to Python int for indexing
            expert_output = self.experts[expert_idx](expert_inputs)
            
            # Store the outputs
            expert_outputs[sort_indices[pos:pos+count]] = expert_output.to(expert_outputs.dtype)
            
            # Move to the next expert
            pos += count
        
        # Step 7: Reshape expert outputs and apply routing weights
        expert_outputs = expert_outputs.reshape(batch_size * sequence_length, self.top_k, hidden_size)
        flat_routing_weights = flat_routing_weights.unsqueeze(-1)  # [batch_size*sequence_length, top_k, 1]
        
        # Apply routing weights to expert outputs
        weighted_expert_outputs = expert_outputs * flat_routing_weights
        
        # Step 8: Sum over the top_k dimension to get the final output
        final_output = weighted_expert_outputs.sum(dim=1)  # [batch_size*sequence_length, hidden_size]
        
        # Apply dropout
        final_output = self.dropout(final_output)
        
        # Reshape output back to original shape
        final_output = final_output.reshape(batch_size, sequence_length, hidden_size)
        
        # Calculate expert performance metrics based on output norms
        performance_metrics = torch.norm(expert_outputs, p=2, dim=-1)
        performance_metrics = performance_metrics.reshape(batch_size, sequence_length, self.top_k)
        
        # Prepare auxiliary outputs
        aux_outputs = {
            "router_aux": router_aux,
            "expert_indices": expert_indices,
            "routing_weights": routing_weights,
            "performance_metrics": performance_metrics,
            # FIX: Extract and pass the auxiliary loss from the router
            "aux_loss": router_aux.get("loss", torch.tensor(0.0, device=x.device))
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
