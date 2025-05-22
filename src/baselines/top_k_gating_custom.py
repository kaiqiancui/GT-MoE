import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional


class TopKGatingRouter(nn.Module):
    """
    Standard Top-K Gating Router for MoE models.
    
    This router implements the standard Top-K gating mechanism where tokens are routed
    to the top-k experts with the highest router scores.
    
    Attributes:
        hidden_size (int): Dimension of input features
        num_experts (int): Number of experts to route between
        top_k (int): Number of experts to select for each token
        use_aux_loss (bool): Whether to use auxiliary load balancing loss
        aux_loss_weight (float): Weight for auxiliary load balancing loss
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
        Initialize the TopKGatingRouter.
        
        Args:
            hidden_size: Dimension of input features
            num_experts: Number of experts to route between
            top_k: Number of experts to select for each token
            use_aux_loss: Whether to use auxiliary load balancing loss
            aux_loss_weight: Weight for auxiliary load balancing loss
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_aux_loss = use_aux_loss
        self.aux_loss_weight = aux_loss_weight
        
        # Gate projector for computing routing probabilities
        self.gate_projector = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Register buffer for tracking expert counts
        self.register_buffer("expert_counts", torch.zeros(num_experts))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass for the TopKGatingRouter.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, hidden_size]
            
        Returns:
            Tuple containing:
            - routing_weights: Tensor of shape [batch_size, sequence_length, top_k]
              containing normalized weights for selected experts
            - expert_indices: Tensor of shape [batch_size, sequence_length, top_k]
              containing indices of selected experts
            - aux_outputs: Dictionary with auxiliary outputs including load balancing loss
        """
        batch_size, sequence_length, hidden_size = x.shape
        
        # Reshape input for routing
        x_reshaped = x.view(-1, hidden_size)  # [batch_size * sequence_length, hidden_size]
        
        # Compute gate logits
        gate_logits = self.gate_projector(x_reshaped)  # [batch_size * sequence_length, num_experts]
        
        # Compute routing probabilities
        routing_probs = F.softmax(gate_logits, dim=-1)  # [batch_size * sequence_length, num_experts]
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        
        # Normalize the probabilities of the top-k experts
        top_k_probs_sum = top_k_probs.sum(dim=-1, keepdim=True)
        top_k_probs_normalized = top_k_probs / top_k_probs_sum
        
        # Reshape outputs to match input dimensions
        routing_weights = top_k_probs_normalized.view(batch_size, sequence_length, self.top_k)
        expert_indices = top_k_indices.view(batch_size, sequence_length, self.top_k)
        
        # Compute auxiliary load balancing loss if enabled
        aux_loss = 0.0
        if self.use_aux_loss and self.training:
            # Compute fraction of tokens routed to each expert
            # First, create a one-hot encoding of expert assignments
            expert_mask = torch.zeros(
                batch_size * sequence_length, self.num_experts, device=x.device
            )
            expert_mask.scatter_(1, top_k_indices, 1.0)
            
            # Compute fraction of tokens routed to each expert
            # Shape: [num_experts]
            router_prob_per_expert = routing_probs.mean(dim=0)
            # Compute fraction of tokens assigned to each expert
            # Shape: [num_experts]
            router_assignment_per_expert = expert_mask.mean(dim=0)
            
            # Compute load balancing loss
            # We want router_prob_per_expert and router_assignment_per_expert to be uniform
            # i.e., 1/num_experts for each expert
            aux_loss = self.aux_loss_weight * (
                # Encourage uniform probability
                torch.sum(router_prob_per_expert * router_prob_per_expert) * self.num_experts
                # Encourage uniform assignment
                + torch.sum(router_assignment_per_expert * router_assignment_per_expert) * self.num_experts
            )
        
        # Update expert counts for monitoring
        if self.training:
            # Count occurrences of each expert
            for k in range(self.top_k):
                expert_idx = expert_indices[:, :, k].view(-1)
                for idx in expert_idx:
                    self.expert_counts[idx] += 1
        
        # Prepare auxiliary outputs
        aux_outputs = {
            "gate_logits": gate_logits,
            "routing_probs": routing_probs,
            "aux_loss": aux_loss,
        }
        
        return routing_weights, expert_indices, aux_outputs


class TopKGatingMoELayer(nn.Module):
    """
    MoE layer with standard Top-K gating.
    
    This layer integrates the TopKGatingRouter with a set of expert networks.
    
    Attributes:
        hidden_size (int): Dimension of input and output features
        num_experts (int): Number of experts in the layer
        top_k (int): Number of experts to select for each token
        router (TopKGatingRouter): Router for expert selection
        experts (nn.ModuleList): List of expert networks
        expert_dropout (float): Dropout probability for expert outputs
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int = 2,
        use_aux_loss: bool = True,
        aux_loss_weight: float = 0.01,
        expert_dropout: float = 0.0,
    ):
        """
        Initialize the TopKGatingMoELayer.
        
        Args:
            hidden_size: Dimension of input and output features
            intermediate_size: Dimension of intermediate features for each expert
            num_experts: Number of experts in the layer
            top_k: Number of experts to select for each token
            use_aux_loss: Whether to use auxiliary load balancing loss
            aux_loss_weight: Weight for auxiliary load balancing loss
            expert_dropout: Dropout probability for expert outputs
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_dropout = expert_dropout
        
        # Initialize router
        self.router = TopKGatingRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            use_aux_loss=use_aux_loss,
            aux_loss_weight=aux_loss_weight,
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
        Forward pass for the TopKGatingMoELayer.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, hidden_size]
            
        Returns:
            Tuple containing:
            - Output tensor of shape [batch_size, sequence_length, hidden_size]
            - Dictionary with auxiliary outputs including load balancing loss
        """
        batch_size, sequence_length, hidden_size = x.shape
        
        # Get router outputs
        routing_weights, expert_indices, router_aux = self.router(x)
        
        # Reshape input for expert processing
        x_reshaped = x.view(-1, hidden_size)  # [batch_size * sequence_length, hidden_size]
        
        # Initialize output tensor
        final_output = torch.zeros_like(x_reshaped)
        
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
        
        # Prepare auxiliary outputs
        aux_outputs = {
            "router_aux": router_aux,
            "expert_indices": expert_indices,
            "routing_weights": routing_weights,
        }
        
        return final_output, aux_outputs
