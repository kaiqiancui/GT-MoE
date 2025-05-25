import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import os
import time
import logging
import numpy as np
from tqdm import tqdm
import wandb

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.custom_transformer_moe import CustomMoETransformer
from rd_esi.utils import calculate_load_metrics, plot_expert_metrics, plot_reputation_distribution, plot_expert_usage, save_metrics_to_json
from training.training_utils import get_lr_scheduler, get_optimizer, setup_wandb


class Trainer:
    """
    Trainer for the CustomMoETransformer model.
    
    This trainer handles the training loop, evaluation, and logging.
    
    Attributes:
        model (CustomMoETransformer): The model to train
        train_dataloader (DataLoader): DataLoader for training data
        val_dataloader (DataLoader): DataLoader for validation data
        test_dataloader (DataLoader): DataLoader for test data
        optimizer (torch.optim.Optimizer): Optimizer for training
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
        device (torch.device): Device to train on
        config (Dict[str, Any]): Training configuration
        logger (logging.Logger): Logger for training information
    """
    
    def __init__(
        self,
        model: CustomMoETransformer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: Optional[DataLoader] = None,
        config: Dict[str, Any] = None,
    ):
        """
        Initialize the Trainer.
        
        Args:
            model: The model to train
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            test_dataloader: Optional DataLoader for test data
            config: Training configuration
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.config = config or {}
        
        # Set device
        self.device = torch.device(self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        
        # Set up optimizer
        self.optimizer = get_optimizer(
            model=self.model,
            optimizer_name=self.config.get("optimizer", "adamw"),
            learning_rate=float(self.config.get("learning_rate", 5e-5)),
            weight_decay=float(self.config.get("weight_decay", 0.01)),
        )
        
        # Set up learning rate scheduler
        self.lr_scheduler = get_lr_scheduler(
            optimizer=self.optimizer,
            scheduler_name=self.config.get("lr_scheduler", "cosine"),
            num_warmup_steps=int(self.config.get("warmup_steps", 100)),
            num_training_steps=int(self.config.get("max_steps", 1000)),
        )
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        
        # Set up performance optimization options
        self.use_amp = self.config.get("use_amp", False)
        self.use_flash_attention = self.config.get("use_flash_attention", False)
        self.use_checkpoint = self.config.get("use_checkpoint", False)
        self.optimizer_states_on_cpu = self.config.get("optimizer_states_on_cpu", False)
        self.cuda_deterministic = self.config.get("cuda_deterministic", True)
        self.benchmark_cudnn = self.config.get("benchmark_cudnn", False)
        self.save_only_last = self.config.get("save_only_last", False)
        self.save_optimizer = self.config.get("save_optimizer", True)
        
        # Setup automatic mixed precision if enabled
        self.scaler = None
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')  # 使用新的推荐用法
            self.logger.info("Automatic Mixed Precision (AMP) training enabled")
        
        # Configure CUDA settings for performance
        if torch.cuda.is_available():
            if not self.cuda_deterministic:
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = self.benchmark_cudnn
                self.logger.info(f"CUDA non-deterministic mode enabled, cuDNN benchmark: {self.benchmark_cudnn}")
            else:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                self.logger.info("CUDA deterministic mode enabled (slower but reproducible)")
        
        # Set up wandb if enabled
        self.use_wandb = self.config.get("use_wandb", False)
        if self.use_wandb:
            setup_wandb(
                project_name=self.config.get("wandb_project", "rd-esi-moe"),
                run_name=self.config.get("wandb_run_name", "rd-esi-run"),
                config=self.config,
            )
        
        # Create output directory
        self.output_dir = self.config.get("output_dir", "results/rd_esi_run")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
        
        # Initialize metrics
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "val_ppl": [],
            "expert_load_variance": [],
            "expert_load_cv": [],
            "expert_load_entropy": [],
            "gpu_memory_usage": [],
            "training_speed": [],  # 记录每秒处理的样本数
            "eval_steps": [],     # 记录评估时的全局步数
        }
    
    def train(self) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Returns:
            Dictionary of training metrics
        """
        # Get training parameters
        num_epochs = self.config.get("num_epochs", 3)
        max_steps = self.config.get("max_steps", -1)
        gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 1)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        log_interval = self.config.get("log_interval", 10)
        eval_interval = self.config.get("eval_interval", 100)
        save_interval = self.config.get("save_interval", 500)
        
        # Initialize step counter and timing variables
        global_step = 0
        samples_processed = 0
        training_start_time = time.time()
        last_log_time = training_start_time
        
        # Training loop
        self.logger.info("Starting training...")
        self.logger.info(f"Using automatic mixed precision: {self.use_amp}")
        self.logger.info(f"Using flash attention: {self.use_flash_attention}")
        self.logger.info(f"Using gradient checkpointing: {self.use_checkpoint}")
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            step_loss = 0.0
            
            # Enable gradient checkpointing if configured
            if self.use_checkpoint and hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            
            # Progress bar for training
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}
                

                
                # Track batch size for throughput calculation
                batch_size = batch["input_ids"].size(0)
                samples_processed += batch_size
                
                # Forward and backward pass with AMP if enabled
                if self.use_amp:
                    with torch.amp.autocast('cuda'):  # 使用新的推荐用法
                        outputs = self.model(**batch)
                        loss = outputs["loss"]
                else:
                    # Standard forward pass
                    outputs = self.model(**batch)
                    loss = outputs["loss"]
                
                # --- IMPORTANT: Update MoE router states here ---
                # FIX: Update router state for EVERY batch, immediately after the forward pass.
                # This logic was moved from inside the gradient accumulation check.
                if "aux_outputs" in outputs:
                    for layer_id_str, layer_aux_outputs in outputs["aux_outputs"].items():
                        layer_id = int(layer_id_str.split('_')[-1])
                        ffn_module = self.model.layers[layer_id].ffn
                        
                        if hasattr(ffn_module, 'update_router_state') and layer_aux_outputs:
                            ffn_module.update_router_state(
                                expert_indices=layer_aux_outputs["expert_indices"],
                                performance_metrics=layer_aux_outputs["performance_metrics"],
                                batch_size=batch["input_ids"].size(0),
                                sequence_length=batch["input_ids"].size(1),
                            )
                # --- End of MoE router state update ---
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                
                # Backward pass
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update step loss
                step_loss += loss.item()
                
                # Update parameters if gradient accumulation is complete
                if (step + 1) % gradient_accumulation_steps == 0:
                    # Clip gradients (with AMP if enabled)
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        
                        # Update parameters with scaler
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # Standard gradient clipping and optimizer step
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        self.optimizer.step()
                    

                    
                    # Update learning rate
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Update global step
                    global_step += 1
                    
                    # The router state update logic has been moved outside the gradient accumulation check
                    # to ensure it runs for every batch. See the code above, after the forward pass.
                    
                    # Log training metrics
                    if global_step % log_interval == 0:
                        # Calculate expert load metrics
                        expert_loads = self._get_expert_loads()
                        load_metrics = calculate_load_metrics(expert_loads)
                        
                        # Calculate training speed (samples/second)
                        current_time = time.time()
                        elapsed_time = current_time - last_log_time
                        if elapsed_time > 0:
                            samples_per_second = (batch_size * gradient_accumulation_steps * log_interval) / elapsed_time
                            self.metrics["training_speed"].append(samples_per_second)
                        last_log_time = current_time
                        
                        # Get GPU memory usage
                        if torch.cuda.is_available():
                            gpu_memory_used = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)  # GB
                            self.metrics["gpu_memory_usage"].append(gpu_memory_used)
                            torch.cuda.reset_peak_memory_stats(self.device)
                        
                        # Update metrics
                        self.metrics["train_loss"].append(step_loss * gradient_accumulation_steps)
                        self.metrics["expert_load_variance"].append(load_metrics["variance"])
                        self.metrics["expert_load_cv"].append(load_metrics["cv"])
                        self.metrics["expert_load_entropy"].append(load_metrics["entropy"])
                        
                        # Log to progress bar
                        progress_bar.set_postfix({
                            "loss": f"{step_loss * gradient_accumulation_steps:.4f}",
                            "lr": f"{self.lr_scheduler.get_last_lr()[0]:.6f}",
                            "step": global_step,
                            "samples/s": f"{samples_per_second:.1f}" if 'samples_per_second' in locals() else "N/A",
                            "GPU mem": f"{gpu_memory_used:.2f}GB" if 'gpu_memory_used' in locals() else "N/A",
                        })
                        
                        # Log to wandb
                        if self.use_wandb:
                            log_dict = {
                                "train/loss": step_loss * gradient_accumulation_steps,
                                "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                                "train/expert_load_variance": load_metrics["variance"],
                                "train/expert_load_cv": load_metrics["cv"],
                                "train/expert_load_entropy": load_metrics["entropy"],
                                "train/step": global_step,
                            }
                            
                            if 'samples_per_second' in locals():
                                log_dict["train/samples_per_second"] = samples_per_second
                            
                            if 'gpu_memory_used' in locals():
                                log_dict["train/gpu_memory_used_gb"] = gpu_memory_used
                            
                            wandb.log(log_dict)
                        
                        # Reset step loss
                        step_loss = 0.0
                    
                    # Evaluate model
                    if global_step % eval_interval == 0:
                        eval_metrics = self.evaluate()
                        
                        # Update metrics
                        self.metrics["val_loss"].append(eval_metrics["loss"])
                        self.metrics["val_ppl"].append(eval_metrics["perplexity"])
                        self.metrics["eval_steps"].append(global_step)  # 记录评估时的全局步数
                        
                        # Log to wandb
                        if self.use_wandb:
                            wandb.log({
                                "eval/loss": eval_metrics["loss"],
                                "eval/perplexity": eval_metrics["perplexity"],
                                "eval/step": global_step,
                            })
                        
                        # Plot expert metrics
                        self._plot_expert_metrics(global_step)
                    
                    # Save model checkpoint
                    if global_step % save_interval == 0:
                        self.save_checkpoint(
                            os.path.join(self.output_dir, "checkpoints", f"checkpoint-{global_step}")
                        )
                        
                        # Save metrics
                        save_metrics_to_json(
                            self.metrics,
                            os.path.join(self.output_dir, "metrics.json"),
                        )
                
                # Check if max steps reached
                if max_steps > 0 and global_step >= max_steps:
                    break
            
            # Update epoch loss
            epoch_loss = epoch_loss / len(self.train_dataloader)
            
            # Log epoch metrics
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")
            
            # Check if max steps reached
            if max_steps > 0 and global_step >= max_steps:
                break
        
        # Final evaluation
        self.logger.info("Final evaluation...")
        eval_metrics = self.evaluate()
        
        # Update metrics
        self.metrics["val_loss"].append(eval_metrics["loss"])
        self.metrics["val_ppl"].append(eval_metrics["perplexity"])
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                "eval/loss": eval_metrics["loss"],
                "eval/perplexity": eval_metrics["perplexity"],
                "eval/step": global_step,
            })
        
        # Plot final expert metrics
        self._plot_expert_metrics(global_step)
        
        # Save final model
        self.save_checkpoint(
            os.path.join(self.output_dir, "checkpoints", "final")
        )
        
        # Save final metrics
        save_metrics_to_json(
            self.metrics,
            os.path.join(self.output_dir, "metrics.json"),
        )
        
        self.logger.info("Training complete!")
        
        return self.metrics
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the validation set.
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs["loss"]
                
                # Update total loss
                total_loss += loss.item() * batch["input_ids"].size(0)
                total_tokens += batch["input_ids"].size(0)
        
        # Calculate metrics
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        self.logger.info(f"Evaluation - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity,
        }
    
    def test(self) -> Dict[str, float]:
        """
        Test the model on the test set.
        
        Returns:
            Dictionary of test metrics
        """
        if self.test_dataloader is None:
            self.logger.warning("No test dataloader provided. Skipping testing.")
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc="Testing"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs["loss"]
                
                # Update total loss
                total_loss += loss.item() * batch["input_ids"].size(0)
                total_tokens += batch["input_ids"].size(0)
        
        # Calculate metrics
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        self.logger.info(f"Test - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity,
        }
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save a model checkpoint.
        
        Args:
            path: Path to save the checkpoint
        """
        os.makedirs(path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(path)
        
        # Save optimizer and scheduler if configured
        if self.save_optimizer:
            # Move optimizer states to CPU if configured to avoid OOM
            if self.optimizer_states_on_cpu:
                optimizer_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                                  for k, v in self.optimizer.state_dict().items()}
                lr_scheduler_state = self.lr_scheduler.state_dict()
            else:
                optimizer_state = self.optimizer.state_dict()
                lr_scheduler_state = self.lr_scheduler.state_dict()
            
            torch.save({
                "optimizer": optimizer_state,
                "lr_scheduler": lr_scheduler_state,
                "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            }, os.path.join(path, "optimizer.pt"))
        
        # Save training metrics
        save_metrics_to_json(
            self.metrics,
            os.path.join(path, "training_metrics.json"),
        )
        
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load a model checkpoint.
        
        Args:
            path: Path to load the checkpoint from
        """
        # Load model
        self.model.load_state_dict(torch.load(os.path.join(path, "pytorch_model.bin")))
        
        # Load optimizer and scheduler
        checkpoint = torch.load(os.path.join(path, "optimizer.pt"))
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        
        self.logger.info(f"Checkpoint loaded from {path}")
    
    def _get_expert_loads(self) -> torch.Tensor:
        """
        Get the current expert loads from all MoE layers.
        
        Returns:
            Tensor containing expert loads
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
            return torch.zeros(0)
    
    def _get_expert_counts(self) -> torch.Tensor:
        """
        Get the current expert selection counts from all MoE layers.
        
        Returns:
            Tensor containing expert counts
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
            return torch.zeros(0)
    
    def _get_reputation_scores(self) -> torch.Tensor:
        """
        Get the current reputation scores from all MoE layers.
        
        Returns:
            Tensor containing reputation scores
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
            return torch.zeros(0)
    
    def _plot_expert_metrics(self, step: int) -> None:
        """
        Plot expert metrics.
        
        Args:
            step: Current training step
        """
        # Plot expert metrics over time
        plot_expert_metrics(
            self.metrics,
            os.path.join(self.output_dir, "plots", "expert_metrics.png"),
            title="Expert Metrics Over Time",
        )
        
        # Plot reputation distribution
        reputation_scores = self._get_reputation_scores()
        if reputation_scores.numel() > 0:
            plot_reputation_distribution(
                reputation_scores,
                os.path.join(self.output_dir, "plots", f"reputation_distribution_{step}.png"),
                step=step,
            )
        
        # Plot expert usage
        expert_counts = self._get_expert_counts()
        if expert_counts.numel() > 0:
            plot_expert_usage(
                expert_counts,
                os.path.join(self.output_dir, "plots", f"expert_usage_{step}.png"),
                step=step,
            )
