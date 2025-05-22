import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, OneCycleLR
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import math
import wandb


def get_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """
    Get optimizer for training.
    
    Args:
        model: Model to optimize
        optimizer_name: Name of optimizer ('adam', 'adamw', or 'sgd')
        learning_rate: Learning rate
        weight_decay: Weight decay
        beta1: Beta1 for Adam-based optimizers
        beta2: Beta2 for Adam-based optimizers
        eps: Epsilon for Adam-based optimizers
        
    Returns:
        Optimizer
    """
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    if optimizer_name.lower() == "adam":
        return optim.Adam(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
        )
    elif optimizer_name.lower() == "adamw":
        return optim.AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
        )
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(
            optimizer_grouped_parameters,
            lr=learning_rate,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    
    Args:
        optimizer: Optimizer to apply the schedule to
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        last_epoch: The index of the last epoch when resuming training
        
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and
    the initial lr set in the optimizer.
    
    Args:
        optimizer: Optimizer to apply the schedule to
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cycles for cosine decay
        last_epoch: The index of the last epoch when resuming training
        
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "linear",
    num_warmup_steps: int = 100,
    num_training_steps: int = 1000,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get learning rate scheduler.
    
    Args:
        optimizer: Optimizer to apply the schedule to
        scheduler_name: Name of scheduler ('linear', 'cosine', or 'one_cycle')
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cycles for cosine decay
        last_epoch: The index of the last epoch when resuming training
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_name.lower() == "linear":
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, last_epoch
        )
    elif scheduler_name.lower() == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, num_cycles, last_epoch
        )
    elif scheduler_name.lower() == "one_cycle":
        return OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]["lr"],
            total_steps=num_training_steps,
            pct_start=float(num_warmup_steps) / float(num_training_steps),
            anneal_strategy="cos",
            cycle_momentum=True,
            div_factor=25.0,
            final_div_factor=10000.0,
            last_epoch=last_epoch,
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def setup_wandb(
    project_name: str = "rd-esi-moe",
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Set up Weights & Biases for experiment tracking.
    
    Args:
        project_name: Name of the W&B project
        run_name: Name of the W&B run
        config: Configuration to log
    """
    wandb.init(
        project=project_name,
        name=run_name,
        config=config,
    )


def setup_mixed_precision_training(
    use_fp16: bool = False,
    use_bf16: bool = False,
) -> Optional[torch.cuda.amp.GradScaler]:
    """
    Set up mixed precision training.
    
    Args:
        use_fp16: Whether to use FP16 precision
        use_bf16: Whether to use BF16 precision
        
    Returns:
        Gradient scaler for FP16 training, or None for full precision or BF16
    """
    if use_fp16:
        return torch.cuda.amp.GradScaler()
    elif use_bf16:
        # BF16 doesn't need a scaler
        return None
    else:
        # Full precision
        return None


def get_gradient_accumulation_steps(
    target_batch_size: int,
    actual_batch_size: int,
) -> int:
    """
    Calculate gradient accumulation steps to achieve a target effective batch size.
    
    Args:
        target_batch_size: Target effective batch size
        actual_batch_size: Actual batch size that fits in memory
        
    Returns:
        Number of gradient accumulation steps
    """
    return max(1, target_batch_size // actual_batch_size)


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
