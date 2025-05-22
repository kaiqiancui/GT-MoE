import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import os


def calculate_load_metrics(expert_counts: torch.Tensor) -> Dict[str, float]:
    """
    Calculate load distribution metrics for MoE experts.
    
    Args:
        expert_counts: Tensor of shape [num_experts] containing the number of tokens
            assigned to each expert
            
    Returns:
        Dictionary containing load distribution metrics:
        - variance: Variance of token counts across experts
        - cv: Coefficient of Variation (standard deviation / mean)
        - entropy: Entropy of the expert assignment distribution
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
    
    return {
        "variance": float(variance),
        "cv": float(cv),
        "entropy": float(entropy),
    }


def plot_expert_metrics(
    metrics: Dict[str, List[float]],
    save_path: str,
    title: str = "Expert Metrics Over Time",
) -> None:
    """
    Plot expert metrics over time.
    
    Args:
        metrics: Dictionary mapping metric names to lists of values over time
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    for metric_name, values in metrics.items():
        plt.plot(values, label=metric_name)
    
    plt.title(title)
    plt.xlabel("Training Steps")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()


def plot_reputation_distribution(
    reputation_scores: torch.Tensor,
    save_path: str,
    step: int = 0,
) -> None:
    """
    Plot the distribution of expert reputation scores.
    
    Args:
        reputation_scores: Tensor of shape [num_experts] containing reputation scores
        save_path: Path to save the plot
        step: Current training step
    """
    plt.figure(figsize=(12, 6))
    
    # Convert to numpy
    scores = reputation_scores.cpu().numpy()
    
    # Create histogram
    plt.hist(scores, bins=30, alpha=0.7)
    
    plt.title(f"Expert Reputation Distribution (Step {step})")
    plt.xlabel("Reputation Score")
    plt.ylabel("Number of Experts")
    plt.grid(True)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()


def plot_expert_usage(
    expert_counts: torch.Tensor,
    save_path: str,
    step: int = 0,
    top_n: int = 20,
) -> None:
    """
    Plot the usage distribution of experts.
    
    Args:
        expert_counts: Tensor of shape [num_experts] containing the number of times
            each expert was selected
        save_path: Path to save the plot
        step: Current training step
        top_n: Number of top experts to highlight
    """
    plt.figure(figsize=(14, 8))
    
    # Convert to numpy
    counts = expert_counts.cpu().numpy()
    
    # Sort experts by usage
    sorted_indices = np.argsort(counts)[::-1]
    sorted_counts = counts[sorted_indices]
    
    # Plot all experts
    plt.bar(range(len(counts)), sorted_counts, alpha=0.7)
    
    # Highlight top-n experts
    if top_n > 0 and top_n < len(counts):
        plt.bar(range(top_n), sorted_counts[:top_n], alpha=0.7, color='red')
    
    plt.title(f"Expert Usage Distribution (Step {step})")
    plt.xlabel("Expert Rank")
    plt.ylabel("Selection Count")
    plt.grid(True, axis='y')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()


def save_metrics_to_json(
    metrics: Dict[str, List[float]],
    save_path: str,
) -> None:
    """
    Save metrics to a JSON file.
    
    Args:
        metrics: Dictionary mapping metric names to lists of values
        save_path: Path to save the JSON file
    """
    import json
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert any numpy or torch values to Python native types
    serializable_metrics = {}
    for key, values in metrics.items():
        serializable_metrics[key] = [float(v) for v in values]
    
    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
