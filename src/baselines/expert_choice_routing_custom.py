import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional


class ExpertChoiceRouter(nn.Module):
    """
    Expert Choice Router for MoE models.
    
    Unlike token choice (Top-K) routing where tokens select experts, in expert choice routing,
    experts select tokens. Each expert selects the top-k tokens that are most relevant to it.
    This ensures perfect load balancing as each expert processes exactly the same number of tokens.
    
    Attributes:
        hidden_size (int): Dimension of input features
        num_experts (int): Number of experts to route between
        capacity_factor (float): Factor to determine how many tokens each expert processes
        gate_projector (nn.Linear): Linear layer for computing token-expert affinities
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        capacity_factor: float = 1.0,
    ):
        """
        Initialize the ExpertChoiceRouter.
        
        Args:
            hidden_size: Dimension of input features
            num_experts: Number of experts to route between
            capacity_factor: Factor to determine how many tokens each expert processes
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        
        # Gate projector for computing token-expert affinities
        self.gate_projector = nn.Linear(hidden_size, num_experts, bias=False)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass for the ExpertChoiceRouter.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, hidden_size]
            
        Returns:
            Tuple containing:
            - routing_weights: Tensor containing weights for token-expert pairs
            - expert_indices: Tensor containing expert indices for each token
            - token_indices: Tensor containing token indices for each expert
            - aux_outputs: Dictionary with auxiliary outputs
        """
        batch_size, sequence_length, hidden_size = x.shape
        
        # Reshape input for routing
        x_reshaped = x.view(-1, hidden_size)  # [batch_size * sequence_length, hidden_size]
        num_tokens = x_reshaped.shape[0]
        
        # Compute token-expert affinities
        # Shape: [num_tokens, num_experts]
        router_logits = self.gate_projector(x_reshaped)
        
        # Compute routing probabilities
        # Shape: [num_tokens, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Compute expert capacity
        # This is the number of tokens each expert will process
        tokens_per_expert = int(num_tokens * self.capacity_factor / self.num_experts)
        tokens_per_expert = max(1, tokens_per_expert)  # Ensure at least 1 token per expert
        
        # Transpose to get expert-token affinities
        # Shape: [num_experts, num_tokens]
        expert_token_affinities = router_probs.t()
        
        # Each expert selects its top-k tokens
        # Shape: [num_experts, tokens_per_expert]
        expert_token_scores, expert_token_indices = torch.topk(
            expert_token_affinities, k=tokens_per_expert, dim=1
        )
        
        # Create a mapping from tokens to experts
        # Initialize with -1 to indicate tokens not selected by any expert
        token_to_expert = torch.full((num_tokens,), -1, dtype=torch.long, device=x.device)
        expert_to_token_scores = torch.zeros((num_tokens,), dtype=torch.float, device=x.device)
        
        # For each expert, assign tokens
        for expert_idx in range(self.num_experts):
            # Get indices of tokens selected by this expert
            selected_token_indices = expert_token_indices[expert_idx]
            selected_token_scores = expert_token_scores[expert_idx]
            
            # Assign expert to these tokens
            # If a token is already assigned to another expert with higher score, keep that assignment
            for i, (token_idx, score) in enumerate(zip(selected_token_indices, selected_token_scores)):
                if token_to_expert[token_idx] == -1 or score > expert_to_token_scores[token_idx]:
                    token_to_expert[token_idx] = expert_idx
                    expert_to_token_scores[token_idx] = score
        
        # Handle tokens not assigned to any expert (if any)
        unassigned_tokens = (token_to_expert == -1).nonzero().squeeze(-1)
        if unassigned_tokens.numel() > 0:
            # Assign unassigned tokens to experts with highest affinity
            unassigned_probs = router_probs[unassigned_tokens]
            assigned_experts = torch.argmax(unassigned_probs, dim=1)
            token_to_expert[unassigned_tokens] = assigned_experts
            for i, token_idx in enumerate(unassigned_tokens):
                expert_to_token_scores[token_idx] = unassigned_probs[i, assigned_experts[i]]
        
        # Reshape token_to_expert to match input dimensions
        expert_indices = token_to_expert.view(batch_size, sequence_length)
        
        # Create routing weights based on expert-token scores
        routing_weights = expert_to_token_scores.view(batch_size, sequence_length)
        
        # Prepare auxiliary outputs
        aux_outputs = {
            "router_logits": router_logits,
            "router_probs": router_probs,
        }
        
        return routing_weights, expert_indices, aux_outputs


class ExpertChoiceMoELayer(nn.Module):
    """
    MoE layer with Expert Choice routing.
    
    This layer integrates the ExpertChoiceRouter with a set of expert networks.
    
    Attributes:
        hidden_size (int): Dimension of input and output features
        num_experts (int): Number of experts in the layer
        capacity_factor (float): Factor to determine how many tokens each expert processes
        router (ExpertChoiceRouter): Router for expert selection
        experts (nn.ModuleList): List of expert networks
        expert_dropout (float): Dropout probability for expert outputs
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        expert_dropout: float = 0.0,
    ):
        """
        Initialize the ExpertChoiceMoELayer.
        
        Args:
            hidden_size: Dimension of input and output features
            intermediate_size: Dimension of intermediate features for each expert
            num_experts: Number of experts in the layer
            capacity_factor: Factor to determine how many tokens each expert processes
            expert_dropout: Dropout probability for expert outputs
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.expert_dropout = expert_dropout
        
        # Initialize router
        self.router = ExpertChoiceRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
        )
        
        # Initialize experts
        self.experts = nn.ModuleList([
            self._create_expert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])
        
        # Dropout for expert outputs
        self.dropout = nn.Dropout(expert_dropout)
        
    def _create_expert(self, hidden_size: int, intermediate_size: int) -> nn.Module:
        """
        Create an expert network.
        
        Args:
            hidden_size: Dimension of input and output features
            intermediate_size: Dimension of intermediate features
            
        Returns:
            Expert network module
        """
        return nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, bias=False),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size, bias=False),
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass for the ExpertChoiceMoELayer.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, hidden_size]
            
        Returns:
            Tuple containing:
            - Output tensor of shape [batch_size, sequence_length, hidden_size]
            - Dictionary with auxiliary outputs
        """
        batch_size, sequence_length, hidden_size = x.shape
        
        # Get router outputs
        routing_weights, expert_indices, router_aux = self.router(x)
        
        # Reshape input for expert processing
        x_reshaped = x.view(-1, hidden_size)  # [batch_size * sequence_length, hidden_size]
        
        # Initialize output tensor
        final_output = torch.zeros_like(x_reshaped)
        
        # Process each expert
        for expert_idx, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            token_mask = (expert_indices.view(-1) == expert_idx)
            
            # If no tokens routed to this expert, skip
            if not token_mask.any():
                continue
            
            # Get indices of tokens assigned to this expert
            token_indices = token_mask.nonzero().squeeze(-1)
            
            # Get inputs for this expert
            expert_inputs = x_reshaped[token_indices]
            
            # Process inputs with this expert
            expert_output = expert(expert_inputs)
            
            # Get weights for this expert
            weights = routing_weights.view(-1)[token_indices].unsqueeze(-1)
            
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
        
        # Prepare auxiliary outputs
        aux_outputs = {
            "router_aux": router_aux,
            "expert_indices": expert_indices,
            "routing_weights": routing_weights,
        }
        
        return final_output, aux_outputs
