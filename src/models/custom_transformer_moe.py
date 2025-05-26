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
            # The attention mask is already in the correct format: [batch_size, 1, seq_len, seq_len]
            # Just ensure it has the right dtype
            attn_scores = attn_scores + attention_mask.to(dtype=attn_scores.dtype)
        
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
            routing_type = moe_config.get("routing_type", "rd_esi") # Default if not specified
            
            num_experts_val = moe_config.get("num_experts")
            expert_dropout_val = moe_config.get("expert_dropout", 0.0)
            router_specific_params = moe_config.get("router_config", {})

            if routing_type == "rd_esi":
                self.ffn = CustomMoELayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_experts=num_experts_val,
                    top_k=moe_config.get("top_k", 2),
                    router_config=router_specific_params, # Pass RD-ESI specific router params
                    expert_dropout=expert_dropout_val,
                )
            elif routing_type == "top_k":
                self.ffn = TopKGatingMoELayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_experts=num_experts_val,
                    top_k=moe_config.get("top_k", 2),
                    use_aux_loss=router_specific_params.get("use_aux_loss", True),
                    aux_loss_weight=float(router_specific_params.get("aux_loss_weight", 0.01)),
                    expert_dropout=expert_dropout_val,
                )
            elif routing_type == "expert_choice":
                self.ffn = ExpertChoiceMoELayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_experts=num_experts_val,
                    capacity_factor=float(router_specific_params.get("capacity_factor", 1.0)),
                    expert_dropout=expert_dropout_val,
                )
                
                
            else:
                raise ValueError(f"Unsupported routing_type: {routing_type}")
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
        
        # Layer normalization
        # FIX: Replaced nn.LayerNorm with RMSNorm for architecture consistency
        self.ln_f = RMSNorm(self.hidden_size)
        
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
        
        # Weight tying: 将token嵌入层的权重与输出层的权重绑定
        # 这是语言模型中的一种常见做法，可以减少参数数量并提高性能
        self.lm_head.weight = self.token_embeddings.weight
        
    def _init_weights(self, module):
        """
        Initialize weights for the model.
        
        Args:
            module: Module to initialize
        """
        if isinstance(module, nn.Linear):
                # FIX: Using standard initialization with std=0.02 (GPT-2 style) for more robust training
            std = 0.02
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
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
        # Check if this layer should use MoE
        use_moe = layer_idx in self.moe_layers
        
        # Create a TransformerBlock with or without MoE
        return TransformerBlock(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            layer_id=layer_idx,
            use_moe=use_moe,
            moe_config=self.moe_config if use_moe else None,
        )
        
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
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Create position ids
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        # FIX: Removed absolute position embeddings to resolve conflict with RoPE
        hidden_states = self.token_embeddings(input_ids)
        hidden_states = self.dropout_layer(hidden_states)
        
        # Create attention mask if needed
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=device)
            
        # Create causal mask for autoregressive language modeling
        # This ensures each token can only attend to previous tokens
        seq_length = hidden_states.size(1)
        causal_mask = torch.tril(torch.ones(seq_length, seq_length, device=device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_length, seq_length]
        
        # Combine with attention mask (1 = attend, 0 = mask)
        # First convert attention_mask from [batch_size, seq_length] to [batch_size, 1, 1, seq_length]
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # Combine masks: both must be 1 to attend
        combined_mask = causal_mask * attention_mask
        # Convert to format expected by attention mechanism (0 = attend, large negative = mask)
        attention_mask = (1.0 - combined_mask) * -10000.0
        
        # Precompute freqs_cis for rotary embeddings
        freqs_cis = precompute_freqs_cis(self.hidden_size // self.num_heads, self.max_seq_len).to(device)
        
        # Initialize auxiliary outputs
        aux_outputs = {}
        
        # Process through transformer layers
        for i, layer in enumerate(self.layers):
            # Apply transformer layer
            hidden_states, layer_aux = layer(
                hidden_states,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            
            # Store auxiliary outputs for this layer if any
            if layer_aux is not None:
                aux_outputs[f"layer_{i}"] = layer_aux
        
        # Apply final layer normalization
        hidden_states = self.ln_f(hidden_states)
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Calculate cross entropy loss
            # 使用ignore_index=-100，确保标签为-100的位置（填充标记）不参与损失计算
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
            
            # FIX: Add MoE auxiliary loss to the total loss
            if aux_outputs:
                # Extract and accumulate auxiliary losses from all MoE layers
                aux_loss = 0.0
                for layer_name, layer_aux in aux_outputs.items():
                    if 'aux_loss' in layer_aux:
                        aux_loss += layer_aux['aux_loss']
                
                # Apply weight to auxiliary loss and add to total loss
                aux_loss_alpha = self.moe_config.get('aux_loss_alpha', 0.01)
                loss = loss + aux_loss_alpha * aux_loss
        
        # Prepare outputs
        outputs = {
            "logits": logits,
            "last_hidden_state": hidden_states,
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
