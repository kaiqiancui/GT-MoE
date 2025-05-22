import argparse
import os
import yaml
import torch
import logging
from typing import Dict, Any

from models.custom_transformer_moe import CustomMoETransformer
from data_utils.c4_tiny_loader import create_c4_tiny_dataloaders
from data_utils.tokenizer_utils import load_tokenizer
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from training.training_utils import set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate MoE models with different routing mechanisms")
    
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "train_eval"],
        default="train_eval",
        help="Mode to run: train, eval, or both",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (overrides config)",
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
        default=None,
        help="Device to use (overrides config)",
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
    
    # Load checkpoint if specified
    checkpoint_path = config.get("checkpoint_path", None)
    if checkpoint_path and os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))
    
    return model


def main():
    """Main entry point for the script."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config_file)
    
    # Override configuration with command line arguments
    if args.output_dir:
        config["output_dir"] = args.output_dir
    
    if args.device:
        config["device"] = args.device
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Create output directory
    output_dir = config.get("output_dir", "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer_config = config.get("tokenizer", {})
    tokenizer = load_tokenizer(**tokenizer_config)
    
    # Create dataloaders
    data_config = config.get("data", {})
    train_dataloader, val_dataloader, test_dataloader = create_c4_tiny_dataloaders(
        tokenizer=tokenizer,
        **data_config,
    )
    
    # Create model
    model = create_model(config)
    
    # Train and/or evaluate
    if args.mode in ["train", "train_eval"]:
        # Create trainer
        trainer_config = config.get("trainer", {})
        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            config=trainer_config,
        )
        
        # Train model
        logging.info("Starting training...")
        trainer.train()
        
        # Save final model
        final_checkpoint_path = os.path.join(output_dir, "final_model.pt")
        torch.save(model.state_dict(), final_checkpoint_path)
        logging.info(f"Final model saved to {final_checkpoint_path}")
    
    if args.mode in ["eval", "train_eval"]:
        # Create evaluator
        evaluator_config = config.get("evaluator", {})
        evaluator = Evaluator(
            model=model,
            dataloader=test_dataloader,
            config=evaluator_config,
        )
        
        # Evaluate model
        logging.info("Starting evaluation...")
        metrics = evaluator.evaluate()
        
        # Log metrics
        logging.info("Evaluation results:")
        for name, value in metrics.items():
            logging.info(f"  {name}: {value}")


if __name__ == "__main__":
    main()
