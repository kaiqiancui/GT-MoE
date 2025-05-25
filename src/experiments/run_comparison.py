#!/usr/bin/env python
"""
Run comparison experiments between different routing mechanisms.

This script automates the process of training and evaluating models with
different routing mechanisms (RD-ESI, Top-K, Expert Choice) and compares
their performance.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.custom_transformer_moe import CustomMoETransformer
from data_utils.c4_loader import create_c4_dataloaders
from data_utils.tokenizer_utils import load_tokenizer
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from training.training_utils import set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run comparison experiments between routing mechanisms")
    
    parser.add_argument(
        "--config_dir",
        type=str,
        default="configs",
        help="Directory containing configuration files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/comparison",
        help="Output directory for results",
    )
    parser.add_argument(
        "--mechanisms",
        type=str,
        nargs="+",
        default=["rd_esi", "top_k", "expert_choice"],
        help="Routing mechanisms to compare",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5000,
        help="Number of training steps",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=500,
        help="Evaluation interval in steps",
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def update_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Update configuration with command line arguments.
    
    Args:
        config: Original configuration
        args: Command line arguments
        
    Returns:
        Updated configuration
    """
    # Update device
    config["trainer"]["device"] = args.device
    config["evaluator"]["device"] = args.device
    
    # Update training steps
    config["trainer"]["max_steps"] = args.steps
    
    # Update evaluation interval
    config["trainer"]["eval_interval"] = args.eval_interval
    
    return config


def create_model(config: Dict[str, Any]) -> CustomMoETransformer:
    """
    Create a model based on the configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Initialized model
    """
    model_config = config.get("model", {})
    
    # Create model
    model = CustomMoETransformer(model_config)
    
    return model


def run_experiment(mechanism: str, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run an experiment for a specific routing mechanism.
    
    Args:
        mechanism: Routing mechanism name
        args: Command line arguments
        
    Returns:
        Dictionary with experiment results
    """
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config_path = os.path.join(args.config_dir, f"{mechanism}_custom_model_config.yaml")
    config = load_config(config_path)
    
    # Update configuration
    config = update_config(config, args)
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, mechanism)
    os.makedirs(output_dir, exist_ok=True)
    
    # Update output directories in config
    config["trainer"]["output_dir"] = os.path.join(output_dir, "train")
    config["evaluator"]["output_dir"] = os.path.join(output_dir, "eval")
    
    # Load tokenizer
    tokenizer_config = config.get("tokenizer", {})
    tokenizer = load_tokenizer(**tokenizer_config)
    
    # Create dataloaders
    data_config = config.get("data", {})
    train_dataloader, val_dataloader, test_dataloader = create_c4_dataloaders(
        tokenizer=tokenizer,
        **data_config,
    )
    
    # Create model
    model = create_model(config)
    
    # Create trainer
    trainer_config = config["trainer"]
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        config=trainer_config,
    )
    
    # Train model
    logging.info(f"Starting training for {mechanism}...")
    metrics = trainer.train()
    
    # Save final model
    final_checkpoint_path = os.path.join(output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_checkpoint_path)
    logging.info(f"Final model saved to {final_checkpoint_path}")
    
    # Create evaluator
    evaluator_config = config["evaluator"]
    evaluator = Evaluator(
        model=model,
        dataloader=test_dataloader,
        config=evaluator_config,
    )
    
    # Evaluate model
    logging.info(f"Starting evaluation for {mechanism}...")
    eval_metrics = evaluator.evaluate()
    
    # Combine metrics
    results = {
        "train_metrics": metrics,
        "eval_metrics": eval_metrics,
    }
    
    # Save results
    results_path = os.path.join(output_dir, "results.yaml")
    with open(results_path, "w") as f:
        yaml.dump(results, f)
    
    return results


def plot_comparison(results: Dict[str, Dict[str, Any]], output_dir: str):
    """
    Plot comparison of results from different routing mechanisms.
    
    Args:
        results: Dictionary with results from different mechanisms
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    for mechanism, mechanism_results in results.items():
        train_metrics = mechanism_results["train_metrics"]
        steps = list(range(len(train_metrics["loss"])))
        plt.plot(steps, train_metrics["loss"], label=mechanism)
    
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "training_loss_comparison.png"))
    plt.close()
    
    # Plot validation perplexity
    plt.figure(figsize=(10, 6))
    for mechanism, mechanism_results in results.items():
        train_metrics = mechanism_results["train_metrics"]
        eval_steps = train_metrics["eval_steps"]
        val_perplexity = train_metrics["val_perplexity"]
        plt.plot(eval_steps, val_perplexity, label=mechanism)
    
    plt.xlabel("Training Steps")
    plt.ylabel("Validation Perplexity")
    plt.title("Validation Perplexity Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "validation_perplexity_comparison.png"))
    plt.close()
    
    # Plot expert utilization
    plt.figure(figsize=(10, 6))
    bar_width = 0.25
    index = np.arange(3)  # Three metrics: expert utilization, load balancing, specialization
    
    for i, (mechanism, mechanism_results) in enumerate(results.items()):
        eval_metrics = mechanism_results["eval_metrics"]
        metrics_values = [
            eval_metrics.get("expert_utilization", 0),
            eval_metrics.get("load_balancing", 0),
            eval_metrics.get("expert_specialization", 0),
        ]
        plt.bar(index + i * bar_width, metrics_values, bar_width, label=mechanism)
    
    plt.xlabel("Metrics")
    plt.ylabel("Value")
    plt.title("Expert Metrics Comparison")
    plt.xticks(index + bar_width, ["Utilization", "Load Balancing", "Specialization"])
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "expert_metrics_comparison.png"))
    plt.close()


def main():
    """Main entry point for the script."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"comparison_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    
    # Run experiments for each mechanism
    results = {}
    for mechanism in args.mechanisms:
        logging.info(f"Running experiment for {mechanism}...")
        mechanism_results = run_experiment(mechanism, args)
        results[mechanism] = mechanism_results
    
    # Plot comparison
    plot_dir = os.path.join(args.output_dir, "plots")
    plot_comparison(results, plot_dir)
    
    # Log final results
    logging.info("Experiment results:")
    for mechanism, mechanism_results in results.items():
        eval_metrics = mechanism_results["eval_metrics"]
        logging.info(f"  {mechanism}:")
        for name, value in eval_metrics.items():
            logging.info(f"    {name}: {value}")


if __name__ == "__main__":
    main()
