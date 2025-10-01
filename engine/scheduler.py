"""
Learning rate schedulers.

Includes warmup and cosine annealing schedules.
"""

import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with warmup and cosine annealing.
    
    Linearly increases LR during warmup, then applies cosine annealing.
    """
    
    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        total_epochs: int,
        warmup_lr: float = 1e-6,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        """
        Initialize scheduler.
        
        Args:
            optimizer: Optimizer
            warmup_epochs: Number of warmup epochs
            total_epochs: Total number of training epochs
            warmup_lr: Initial learning rate for warmup
            min_lr: Minimum learning rate after cosine annealing
            last_epoch: Last epoch (for resuming)
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase: linear increase
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_lr + (base_lr - self.warmup_lr) * alpha
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (
                    1.0 + math.cos(math.pi * progress)
                )
                for base_lr in self.base_lrs
            ]


class WarmupMultiStepScheduler(_LRScheduler):
    """
    Learning rate scheduler with warmup and multi-step decay.
    
    Linearly increases LR during warmup, then applies step decay.
    """
    
    def __init__(
        self,
        optimizer,
        milestones,
        gamma: float = 0.1,
        warmup_epochs: int = 5,
        warmup_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        """
        Initialize scheduler.
        
        Args:
            optimizer: Optimizer
            milestones: List of epoch indices for LR decay
            gamma: Multiplicative factor for LR decay
            warmup_epochs: Number of warmup epochs
            warmup_lr: Initial learning rate for warmup
            last_epoch: Last epoch (for resuming)
        """
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_lr + (base_lr - self.warmup_lr) * alpha
                for base_lr in self.base_lrs
            ]
        else:
            # Multi-step decay
            decay_factor = self.gamma ** sum(
                self.last_epoch >= m for m in self.milestones
            )
            return [base_lr * decay_factor for base_lr in self.base_lrs]


def build_scheduler(optimizer, config: dict):
    """
    Build learning rate scheduler from configuration.
    
    Args:
        optimizer: Optimizer
        config: Training configuration
    
    Returns:
        Learning rate scheduler
    """
    scheduler_type = config.get('scheduler', 'cosine')
    
    if scheduler_type == 'cosine':
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=config.get('warmup_epochs', 5),
            total_epochs=config.get('epochs', 50),
            warmup_lr=config.get('warmup_lr', 1e-6),
            min_lr=config.get('min_lr', 1e-6),
        )
    elif scheduler_type == 'multistep':
        milestones = config.get('milestones', [30, 40])
        scheduler = WarmupMultiStepScheduler(
            optimizer,
            milestones=milestones,
            gamma=config.get('gamma', 0.1),
            warmup_epochs=config.get('warmup_epochs', 5),
            warmup_lr=config.get('warmup_lr', 1e-6),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler
