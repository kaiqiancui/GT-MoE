import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Any, Optional, Union
import os
import time
import logging
import numpy as np
from tqdm import tqdm
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.custom_transformer_moe import CustomMoETransformer
from evaluation.metrics import (
    calculate_perplexity,
    calculate_expert_load_metrics,
    calculate_reputation_metrics,
    calculate_expert_specialization,
    calculate_performance_correlation
)
from rd_esi.utils import (
    plot_expert_metrics,
    plot_reputation_distribution,
    plot_expert_usage,
    save_metrics_to_json
)


class Evaluator:
    """
    Evaluator for the CustomMoETransformer model.
    
    This evaluator handles model evaluation and metrics calculation.
    
    Attributes:
        model (CustomMoETransformer): The model to evaluate
        dataloader (DataLoader): DataLoader for evaluation data
        device (torch.device): Device to evaluate on
        config (Dict[str, Any]): Evaluation configuration
        logger (logging.Logger): Logger for evaluation information
    """
    
    def __init__(
        self,
        model: CustomMoETransformer,
        dataloader: DataLoader,
        config: Dict[str, Any] = None,
    ):
        """
        Initialize the Evaluator.
        
        Args:
            model: The model to evaluate
            dataloader: DataLoader for evaluation data
            config: Evaluation configuration
        """
        self.model = model
        self.dataloader = dataloader
        self.config = config or {}
        
        # Set device
        self.device = torch.device(self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        
        # Create output directory
        self.output_dir = self.config.get("output_dir", "results/evaluation")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
        
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model.
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        # Initialize metrics
        all_expert_indices = []
        all_performance_metrics = []
        all_token_ids = []
        
        # Collect expert states
        expert_loads = self._get_expert_loads()
        expert_counts = self._get_expert_counts()
        reputation_scores = self._get_reputation_scores()
        
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs["loss"]
                
                # Update total loss
                total_loss += loss.item() * batch["input_ids"].size(0)
                total_tokens += batch["input_ids"].size(0)
                
                # Collect MoE metrics
                for layer_outputs in outputs.get("moe_aux_outputs", []):
                    all_expert_indices.append(layer_outputs["expert_indices"].detach().cpu())
                    all_performance_metrics.append(layer_outputs["performance_metrics"].detach().cpu())
                
                # Collect token IDs
                all_token_ids.append(batch["input_ids"].detach().cpu())
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens
        perplexity = calculate_perplexity(avg_loss)
        
        # Initialize metrics dictionary
        metrics = {
            "loss": avg_loss,
            "perplexity": perplexity,
        }
        
        # Calculate MoE-specific metrics if available
        if expert_loads is not None and expert_loads.numel() > 0:
            # Expert load metrics
            load_metrics = calculate_expert_load_metrics(expert_loads)
            metrics.update({f"expert_load_{k}": v for k, v in load_metrics.items()})
            
            # Plot expert load distribution
            plot_expert_usage(
                expert_counts,
                os.path.join(self.output_dir, "plots", "expert_usage.png"),
            )
        
        if reputation_scores is not None and reputation_scores.numel() > 0:
            # Reputation metrics
            reputation_metrics = calculate_reputation_metrics(reputation_scores)
            metrics.update({f"reputation_{k}": v for k, v in reputation_metrics.items()})
            
            # Plot reputation distribution
            plot_reputation_distribution(
                reputation_scores,
                os.path.join(self.output_dir, "plots", "reputation_distribution.png"),
            )
        
        # Calculate specialization metrics if we have collected data
        if all_expert_indices and all_token_ids:
            # Concatenate collected data
            expert_indices = torch.cat(all_expert_indices, dim=0)
            token_ids = torch.cat(all_token_ids, dim=0)
            
            # Calculate specialization metrics
            specialization_metrics = calculate_expert_specialization(
                expert_indices=expert_indices,
                token_ids=token_ids,
                num_experts=self.model.moe_config.get("num_experts", 16),
                vocab_size=self.model.vocab_size,
            )
            metrics.update({f"specialization_{k}": v for k, v in specialization_metrics.items()})
        
        # Calculate performance correlation if we have all necessary data
        if all_expert_indices and all_performance_metrics and reputation_scores is not None:
            # Concatenate collected data
            expert_indices = torch.cat(all_expert_indices, dim=0)
            performance_metrics = torch.cat(all_performance_metrics, dim=0)
            
            # Calculate correlation metrics
            correlation_metrics = calculate_performance_correlation(
                expert_indices=expert_indices,
                performance_metrics=performance_metrics,
                reputation_scores=reputation_scores,
            )
            metrics.update({f"correlation_{k}": v for k, v in correlation_metrics.items()})
        
        # Log metrics
        self.logger.info(f"Evaluation - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
        for name, value in metrics.items():
            if name not in ["loss", "perplexity"]:
                self.logger.info(f"  {name}: {value:.4f}")
        
        # Save metrics to file
        metrics_file = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def _get_expert_loads(self) -> Optional[torch.Tensor]:
        """
        Get the current expert loads from all MoE layers.
        
        Returns:
            Tensor containing expert loads or None if not available
        """
        expert_loads = []
        
        # Collect expert loads from all MoE layers
        for layer_id in self.model.moe_layers:
            layer = self.model.layers[layer_id]
            if hasattr(layer.ffn, "router"):
                expert_loads.append(layer.ffn.router.expert_loads)
        
        # Concatenate expert loads
        if expert_loads:
            return torch.cat(expert_loads)
        else:
            return None
    
    def _get_expert_counts(self) -> Optional[torch.Tensor]:
        """
        Get the current expert selection counts from all MoE layers.
        
        Returns:
            Tensor containing expert counts or None if not available
        """
        expert_counts = []
        
        # Collect expert counts from all MoE layers
        for layer_id in self.model.moe_layers:
            layer = self.model.layers[layer_id]
            if hasattr(layer.ffn, "router"):
                expert_counts.append(layer.ffn.router.expert_counts)
        
        # Concatenate expert counts
        if expert_counts:
            return torch.cat(expert_counts)
        else:
            return None
    
    def _get_reputation_scores(self) -> Optional[torch.Tensor]:
        """
        Get the current reputation scores from all MoE layers.
        
        Returns:
            Tensor containing reputation scores or None if not available
        """
        reputation_scores = []
        
        # Collect reputation scores from all MoE layers
        for layer_id in self.model.moe_layers:
            layer = self.model.layers[layer_id]
            if hasattr(layer.ffn, "router"):
                reputation_scores.append(layer.ffn.router.reputation_scores)
        
        # Concatenate reputation scores
        if reputation_scores:
            return torch.cat(reputation_scores)
        else:
            return None
