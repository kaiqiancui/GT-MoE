import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Any, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.custom_moe_layer import CustomMoELayer
from baselines.top_k_gating_custom import TopKGatingMoELayer
from baselines.expert_choice_routing_custom import ExpertChoiceMoELayer


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Attributes:
        dim (int): Dimension to normalize
        eps (float): Epsilon for numerical stability
        weight (nn.Parameter): Learnable scale parameter
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize RMSNorm.
        
        Args:
            dim: Dimension to normalize
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        x_normalized = x / rms
        return self.weight * x_normalized


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (rotary embeddings).
    
    Args:
        dim: Dimension of the embedding
        max_seq_len: Maximum sequence length
        theta: Base value for the frequency
        
    Returns:
        Precomputed frequency tensor
    """
    # Ensure dim is even
    assert dim % 2 == 0, "Dimension must be even for rotary embeddings"
    
    # Create position indices
    pos = torch.arange(max_seq_len).float()
    
    # Create frequency indices
    freqs = torch.arange(0, dim, 2).float() / dim
    
    # Compute frequencies
    inv_freq = 1.0 / (theta ** freqs)
    
    # Compute complex exponentials
    freqs_cis = torch.einsum("i,j->ij", pos, inv_freq)
    freqs_cis = torch.cat([freqs_cis.cos(), freqs_cis.sin()], dim=-1)
    
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to the input tensor.
    
    Args:
        x: Input tensor of shape [batch_size, seq_len, n_heads, head_dim]
        freqs_cis: Precomputed frequency tensor
        position_ids: Position indices
        
    Returns:
        Tensor with rotary embeddings applied
    """
    batch_size, seq_len, n_heads, head_dim = x.shape
    
    # Reshape for easier manipulation
    x = x.view(batch_size, seq_len, n_heads, head_dim // 2, 2)
    
    # Get the appropriate frequencies for each position
    freqs = freqs_cis[position_ids].view(batch_size, seq_len, 1, head_dim // 2, 2)
    
    # Apply complex multiplication
    # (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
    x_real, x_imag = x[..., 0], x[..., 1]
    freqs_real, freqs_imag = freqs[..., 0], freqs[..., 1]
    
    result_real = x_real * freqs_real - x_imag * freqs_imag
    result_imag = x_real * freqs_imag + x_imag * freqs_real
    
    # Combine real and imaginary parts
    result = torch.stack([result_real, result_imag], dim=-1)
    
    # Reshape back to original shape
    result = result.view(batch_size, seq_len, n_heads, head_dim)
    
    return result


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    
    Attributes:
        hidden_size (int): Size of hidden dimension
        num_heads (int): Number of attention heads
        head_dim (int): Dimension of each attention head
        wq (nn.Linear): Query projection
        wk (nn.Linear): Key projection
        wv (nn.Linear): Value projection
        wo (nn.Linear): Output projection
        attn_dropout (nn.Dropout): Dropout for attention weights
        resid_dropout (nn.Dropout): Dropout for residual connection
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        """
        Initialize MultiHeadAttention.
        
        Args:
            hidden_size: Size of hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        # Projection matrices
        self.wq = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wk = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wv = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wo = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for MultiHeadAttention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            freqs_cis: Precomputed frequency tensor for rotary embeddings
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # Default position ids if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Linear projections
        q = self.wq(x)  # [batch_size, seq_len, hidden_size]
        k = self.wk(x)  # [batch_size, seq_len, hidden_size]
        v = self.wv(x)  # [batch_size, seq_len, hidden_size]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply rotary embeddings to queries and keys
        q = apply_rotary_emb(q, freqs_cis, position_ids)
        k = apply_rotary_emb(k, freqs_cis, position_ids)
        
        # Transpose for batched matrix multiplication
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        v = v.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Compute attention scores
        # (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, head_dim, seq_len)
        # -> (batch_size, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project back to hidden_size
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.wo(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output


class FeedForward(nn.Module):
    """
    Feed-forward network (FFN) module.
    
    Attributes:
        w1 (nn.Linear): First linear layer
        w2 (nn.Linear): Second linear layer
        w3 (nn.Linear): Third linear layer for gating
        dropout (nn.Dropout): Dropout layer
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0,
    ):
        """
        Initialize FeedForward.
        
        Args:
            hidden_size: Size of hidden dimension
            intermediate_size: Size of intermediate dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for FeedForward.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # SwiGLU activation
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """
    Transformer block with attention and feed-forward/MoE layers.
    
    Attributes:
        attn (MultiHeadAttention): Multi-head attention module
        ffn (nn.Module): Feed-forward network or MoE layer
        attn_norm (RMSNorm): Layer normalization for attention
        ffn_norm (RMSNorm): Layer normalization for feed-forward/MoE
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        dropout: float = 0.0,
        layer_id: int = 0,
        use_moe: bool = False,
        moe_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize TransformerBlock.
        
        Args:
            hidden_size: Size of hidden dimension
            intermediate_size: Size of intermediate dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            layer_id: Layer identifier
            use_moe: Whether to use MoE for the feed-forward layer
            moe_config: Configuration for MoE layer
        """
        super().__init__()
        self.layer_id = layer_id
        self.use_moe = use_moe
        
        # Attention module
        self.attn = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Feed-forward or MoE module
        if use_moe:
            assert moe_config is not None, "moe_config must be provided when use_moe=True"
            self.ffn = CustomMoELayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                **moe_config,
            )
        else:
            self.ffn = FeedForward(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                dropout=dropout,
            )
        
        # Layer normalization
        self.attn_norm = RMSNorm(hidden_size)
        self.ffn_norm = RMSNorm(hidden_size)
        
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Forward pass for TransformerBlock.
        
        Args:
            x: Input tensor
            freqs_cis: Precomputed frequency tensor for rotary embeddings
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            
        Returns:
            Tuple containing:
            - Output tensor
            - Optional dictionary with auxiliary outputs (for MoE layers)
        """
        # Attention with residual connection
        h = x + self.attn(self.attn_norm(x), freqs_cis, attention_mask, position_ids)
        
        # Feed-forward/MoE with residual connection
        if self.use_moe:
            moe_output, moe_aux = self.ffn(self.ffn_norm(h))
            return h + moe_output, moe_aux
        else:
            return h + self.ffn(self.ffn_norm(h)), None


class CustomMoETransformer(nn.Module):
    """
    Transformer model with MoE layers.
    
    This model implements a transformer with MoE layers at specified positions.
    
    Attributes:
        vocab_size (int): Size of the vocabulary
        hidden_size (int): Dimension of hidden layers
        num_layers (int): Number of transformer layers
        num_heads (int): Number of attention heads
        intermediate_size (int): Dimension of intermediate feed-forward layers
        max_seq_len (int): Maximum sequence length
        dropout (float): Dropout probability
        moe_layers (List[int]): List of layer indices where MoE should be used
        moe_config (Dict[str, Any]): Configuration for MoE layers
        routing_type (str): Type of routing mechanism to use
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
    ):
        """
        Initialize the CustomMoETransformer.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__()
        
        # Extract configuration
        self.vocab_size = config.get("vocab_size", 50257)  # Default to GPT-2 vocab size
        self.hidden_size = config.get("hidden_size", 768)
        self.num_layers = config.get("num_layers", 12)
        self.num_heads = config.get("num_heads", 12)
        self.intermediate_size = config.get("intermediate_size", 3072)
        self.max_seq_len = config.get("max_seq_len", 1024)
        self.dropout = config.get("dropout", 0.1)
        
        # MoE configuration
        self.moe_layers = config.get("moe_layers", [])
        self.moe_config = config.get("moe_config", {})
        
        # Routing type (rd_esi, top_k, expert_choice)
        self.routing_type = self.moe_config.get("routing_type", "rd_esi")
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(self.vocab_size, self.hidden_size)
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(self.max_seq_len, self.hidden_size)
        
        # Layer normalization
        self.ln_f = nn.LayerNorm(self.hidden_size)
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            self._create_transformer_layer(layer_idx)
            for layer_idx in range(self.num_layers)
        ])
        
        # Output head
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """
        Initialize weights for the model.
        
        Args:
            module: Module to initialize
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def _create_transformer_layer(self, layer_idx: int) -> nn.Module:
        """
        Create a transformer layer, which may be a standard layer or an MoE layer.
        
        Args:
            layer_idx: Index of the layer
            
        Returns:
            Transformer layer module
        """
        # Create attention layer
        attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        
        # Create feed-forward layer (which may be MoE)
        if layer_idx in self.moe_layers:
            # Use MoE for feed-forward based on routing type
            if self.routing_type == "rd_esi":
                ffn = CustomMoELayer(
                    hidden_size=self.hidden_size,
                    intermediate_size=self.intermediate_size,
                    **self.moe_config,
                )
            elif self.routing_type == "top_k":
                # Extract top_k specific parameters
                top_k = self.moe_config.get("top_k", 2)
                num_experts = self.moe_config.get("num_experts", 16)
                router_config = self.moe_config.get("router_config", {})
                expert_dropout = self.moe_config.get("expert_dropout", 0.0)
                
                ffn = TopKGatingMoELayer(
                    hidden_size=self.hidden_size,
                    intermediate_size=self.intermediate_size,
                    num_experts=num_experts,
                    top_k=top_k,
                    use_aux_loss=router_config.get("use_aux_loss", True),
                    aux_loss_weight=router_config.get("aux_loss_weight", 0.01),
                    expert_dropout=expert_dropout,
                )
            elif self.routing_type == "expert_choice":
                # Extract expert choice specific parameters
                num_experts = self.moe_config.get("num_experts", 16)
                router_config = self.moe_config.get("router_config", {})
                expert_dropout = self.moe_config.get("expert_dropout", 0.0)
                
                ffn = ExpertChoiceMoELayer(
                    hidden_size=self.hidden_size,
                    intermediate_size=self.intermediate_size,
                    num_experts=num_experts,
                    capacity_factor=router_config.get("capacity_factor", 1.0),
                    expert_dropout=expert_dropout,
                )
            else:
                raise ValueError(f"Unknown routing type: {self.routing_type}")
                
            is_moe_layer = True
        else:
            # Use standard feed-forward
            ffn = nn.Sequential(
                nn.Linear(self.hidden_size, self.intermediate_size),
                nn.GELU(),
                nn.Linear(self.intermediate_size, self.hidden_size),
                nn.Dropout(self.dropout),
            )
            is_moe_layer = False
        
        # Create layer normalization
        ln1 = nn.LayerNorm(self.hidden_size)
        ln2 = nn.LayerNorm(self.hidden_size)
        
        # Return as dictionary
        return {
            "attention": attention,
            "ffn": ffn,
            "ln1": ln1,
            "ln2": ln2,
            "is_moe_layer": is_moe_layer,
        }
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the CustomMoETransformer.
        
        Args:
            input_ids: Tensor of token ids, shape [batch_size, sequence_length]
            attention_mask: Attention mask, shape [batch_size, sequence_length]
            labels: Optional tensor of target token ids for language modeling
            
        Returns:
            Dictionary with model outputs including loss if labels are provided
        """
        batch_size, sequence_length = input_ids.shape
        
        # Create position ids
        position_ids = torch.arange(0, sequence_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Get embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        
        # Combine embeddings
        h = token_embeds + position_embeds
        h = self.dropout_layer(h)
        
        # Create attention mask for transformer
        if attention_mask is None:
            # Default to causal mask
            attention_mask = torch.ones_like(input_ids)
        
        # Convert to attention mask expected by nn.MultiheadAttention
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=h.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(sequence_length, sequence_length, device=h.device), diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float("-inf"))
        
        # Initialize auxiliary outputs
        aux_outputs = {}
        
        # Process transformer layers
        for i, layer in enumerate(self.layers):
            # Layer normalization and attention
            h_ln1 = layer["ln1"](h)
            attn_output, _ = layer["attention"](
                query=h_ln1,
                key=h_ln1,
                value=h_ln1,
                attn_mask=causal_mask,
                key_padding_mask=~attention_mask.bool() if attention_mask is not None else None,
            )
            h = h + attn_output
            
            # Layer normalization and feed-forward (which may be MoE)
            h_ln2 = layer["ln2"](h)
            if layer["is_moe_layer"]:
                # MoE feed-forward
                ffn_output, ffn_aux = layer["ffn"](h_ln2)
                # Store auxiliary outputs
                aux_outputs[f"layer_{i}_moe"] = ffn_aux
            else:
                # Standard feed-forward
                ffn_output = layer["ffn"](h_ln2)
            
            h = h + ffn_output
        
        # Final layer normalization
        h = self.ln_f(h)
        
        # Language modeling head
        logits = self.lm_head(h)
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
            )
            
            # Add auxiliary losses if any
            aux_loss = 0.0
            for layer_aux in aux_outputs.values():
                if "router_aux" in layer_aux and "aux_loss" in layer_aux["router_aux"]:
                    aux_loss += layer_aux["router_aux"]["aux_loss"]
            
            if aux_loss > 0.0:
                loss += aux_loss
        
        # Prepare outputs
        outputs = {
            "logits": logits,
            "last_hidden_state": h,
        }
        
        if loss is not None:
            outputs["loss"] = loss
            
        if aux_outputs:
            outputs["aux_outputs"] = aux_outputs
            
        return outputs

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Initial token ids
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep for top-k sampling
            top_p: Cumulative probability for nucleus sampling
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to sample or use greedy decoding
            pad_token_id: ID of the padding token
            eos_token_id: ID of the end-of-sequence token
            
        Returns:
            Generated token ids
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize sequence with input_ids
        generated_ids = input_ids.clone()
        
        # Set attention mask
        attention_mask = torch.ones_like(generated_ids)
        
        # Generate tokens
        for _ in range(max_length - input_ids.shape[1]):
            # Get position ids
            position_ids = torch.arange(generated_ids.shape[1], device=device).unsqueeze(0).expand(batch_size, -1)
            
            # Forward pass
            outputs = self.forward(
                input_ids=generated_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
            )
            
            # Get logits for the next token
            next_token_logits = outputs["logits"][:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(generated_ids[i].tolist()):
                        next_token_logits[i, token_id] /= repetition_penalty
            
            # Apply top-k sampling
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float("Inf")
            
            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float("Inf")
            
            # Sample or greedy decoding
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append next token to generated ids
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
            
            # Check if all sequences have reached EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return generated_ids
