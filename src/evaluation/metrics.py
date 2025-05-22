import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import math
import scipy.stats


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from loss.
    
    Args:
        loss: Cross-entropy loss
        
    Returns:
        Perplexity
    """
    return math.exp(loss)


def calculate_expert_load_metrics(expert_counts: torch.Tensor) -> Dict[str, float]:
    """
    Calculate metrics related to expert load distribution.
    
    Args:
        expert_counts: Tensor of shape [num_experts] containing the number of tokens
            assigned to each expert
            
    Returns:
        Dictionary containing metrics:
        - variance: Variance of token counts across experts
        - cv: Coefficient of Variation (standard deviation / mean)
        - entropy: Entropy of the expert assignment distribution
        - gini: Gini coefficient of the distribution
    """
    # Convert to numpy for easier calculation
    counts = expert_counts.cpu().numpy()
    
    # Calculate variance
    variance = np.var(counts)
    
    # Calculate coefficient of variation (CV)
    mean = np.mean(counts)
    std = np.std(counts)
    cv = std / mean if mean > 0 else 0.0
    
    # Calculate entropy
    # Normalize counts to get a probability distribution
    total = np.sum(counts)
    if total > 0:
        probs = counts / total
        # Add small epsilon to avoid log(0)
        probs = np.maximum(probs, 1e-10)
        entropy = -np.sum(probs * np.log(probs))
    else:
        entropy = 0.0
    
    # Calculate Gini coefficient
    # Sort the array
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    if n > 0 and np.sum(sorted_counts) > 0:
        # Calculate cumulative sum
        cumsum = np.cumsum(sorted_counts)
        # Calculate Gini coefficient
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0.0
    else:
        gini = 0.0
    
    return {
        "variance": float(variance),
        "cv": float(cv),
        "entropy": float(entropy),
        "gini": float(gini),
    }


def calculate_reputation_metrics(reputation_scores: torch.Tensor) -> Dict[str, float]:
    """
    Calculate metrics related to expert reputation scores.
    
    Args:
        reputation_scores: Tensor of shape [num_experts] containing reputation scores
            
    Returns:
        Dictionary containing metrics:
        - mean: Mean reputation score
        - std: Standard deviation of reputation scores
        - min: Minimum reputation score
        - max: Maximum reputation score
        - median: Median reputation score
        - skewness: Skewness of reputation score distribution
        - kurtosis: Kurtosis of reputation score distribution
    """
    # Convert to numpy for easier calculation
    scores = reputation_scores.cpu().numpy()
    
    # Calculate basic statistics
    mean = np.mean(scores)
    std = np.std(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    median = np.median(scores)
    
    # Calculate skewness and kurtosis
    skewness = scipy.stats.skew(scores) if len(scores) > 2 else 0.0
    kurtosis = scipy.stats.kurtosis(scores) if len(scores) > 3 else 0.0
    
    return {
        "mean": float(mean),
        "std": float(std),
        "min": float(min_score),
        "max": float(max_score),
        "median": float(median),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
    }


def calculate_expert_specialization(
    expert_indices: torch.Tensor,
    token_ids: torch.Tensor,
    num_experts: int,
    vocab_size: int,
) -> Dict[str, float]:
    """
    Calculate metrics related to expert specialization by analyzing token-expert assignments.
    
    Args:
        expert_indices: Tensor of shape [batch_size, sequence_length, top_k]
            containing indices of selected experts
        token_ids: Tensor of shape [batch_size, sequence_length]
            containing token IDs
        num_experts: Number of experts
        vocab_size: Size of vocabulary
            
    Returns:
        Dictionary containing metrics:
        - token_expert_mi: Mutual information between tokens and experts
        - expert_token_entropy: Conditional entropy of tokens given experts
    """
    # Flatten tensors
    batch_size, seq_len = token_ids.shape
    top_k = expert_indices.shape[-1]
    
    # Create a mapping of (token_id, expert_id) pairs
    flat_tokens = token_ids.view(-1).cpu().numpy()
    flat_experts = expert_indices.view(-1).cpu().numpy()
    
    # Count occurrences of each token
    token_counts = np.zeros(vocab_size)
    for token_id in flat_tokens:
        if 0 <= token_id < vocab_size:
            token_counts[token_id] += 1
    
    # Count occurrences of each expert
    expert_counts = np.zeros(num_experts)
    for expert_id in flat_experts:
        if 0 <= expert_id < num_experts:
            expert_counts[expert_id] += 1
    
    # Count occurrences of (token_id, expert_id) pairs
    token_expert_counts = np.zeros((vocab_size, num_experts))
    for i in range(len(flat_tokens)):
        token_id = flat_tokens[i]
        expert_id = flat_experts[i]
        if 0 <= token_id < vocab_size and 0 <= expert_id < num_experts:
            token_expert_counts[token_id, expert_id] += 1
    
    # Calculate joint and marginal probabilities
    total_pairs = np.sum(token_expert_counts)
    if total_pairs > 0:
        # Joint probability P(token, expert)
        p_token_expert = token_expert_counts / total_pairs
        
        # Marginal probabilities P(token) and P(expert)
        p_token = token_counts / total_pairs
        p_expert = expert_counts / total_pairs
        
        # Calculate mutual information
        mi = 0.0
        for t in range(vocab_size):
            for e in range(num_experts):
                if p_token_expert[t, e] > 0 and p_token[t] > 0 and p_expert[e] > 0:
                    mi += p_token_expert[t, e] * np.log(p_token_expert[t, e] / (p_token[t] * p_expert[e]))
        
        # Calculate conditional entropy H(token|expert)
        h_token_given_expert = 0.0
        for e in range(num_experts):
            if p_expert[e] > 0:
                for t in range(vocab_size):
                    if p_token_expert[t, e] > 0:
                        h_token_given_expert -= p_token_expert[t, e] * np.log(p_token_expert[t, e] / p_expert[e])
    else:
        mi = 0.0
        h_token_given_expert = 0.0
    
    return {
        "token_expert_mi": float(mi),
        "expert_token_entropy": float(h_token_given_expert),
    }


def calculate_performance_correlation(
    expert_indices: torch.Tensor,
    performance_metrics: torch.Tensor,
    reputation_scores: torch.Tensor,
) -> Dict[str, float]:
    """
    Calculate correlation between expert performance metrics and reputation scores.
    
    Args:
        expert_indices: Tensor of shape [batch_size, sequence_length, top_k]
            containing indices of selected experts
        performance_metrics: Tensor of shape [batch_size, sequence_length, top_k]
            containing performance metrics for each selected expert
        reputation_scores: Tensor of shape [num_experts] containing reputation scores
            
    Returns:
        Dictionary containing metrics:
        - performance_reputation_correlation: Correlation between performance and reputation
    """
    # Flatten tensors
    flat_indices = expert_indices.view(-1)
    flat_performance = performance_metrics.view(-1)
    
    # Create arrays for correlation calculation
    expert_ids = []
    perf_values = []
    rep_values = []
    
    # Collect data points
    for i in range(len(flat_indices)):
        expert_id = flat_indices[i].item()
        perf = flat_performance[i].item()
        rep = reputation_scores[expert_id].item()
        
        expert_ids.append(expert_id)
        perf_values.append(perf)
        rep_values.append(rep)
    
    # Calculate correlation
    if len(perf_values) > 1:
        correlation, p_value = scipy.stats.pearsonr(perf_values, rep_values)
    else:
        correlation = 0.0
        p_value = 1.0
    
    return {
        "performance_reputation_correlation": float(correlation),
        "correlation_p_value": float(p_value),
    }
