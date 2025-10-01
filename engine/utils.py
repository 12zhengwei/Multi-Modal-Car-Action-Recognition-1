"""
Utility functions for training and evaluation.

Includes metrics, logging, distributed training helpers, etc.
"""

import os
import random
import numpy as np
import torch
import torch.distributed as dist
from typing import Optional, Dict
import yaml
import json


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AccuracyMeter:
    """Computes Top-1 and Top-5 accuracy."""
    
    def __init__(self, topk=(1, 5)):
        self.topk = topk
        self.reset()
    
    def reset(self):
        self.correct = {k: 0 for k in self.topk}
        self.total = 0
    
    def update(self, output: torch.Tensor, target: torch.Tensor):
        """
        Update accuracy metrics.
        
        Args:
            output: Model predictions (B, num_classes)
            target: Ground truth labels (B,)
        """
        batch_size = target.size(0)
        self.total += batch_size
        
        _, pred = output.topk(max(self.topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        for k in self.topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            self.correct[k] += correct_k.item()
    
    def accuracy(self, k=1):
        """Get Top-k accuracy."""
        if self.total == 0:
            return 0.0
        return 100.0 * self.correct[k] / self.total
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all accuracy metrics."""
        return {f'top{k}': self.accuracy(k) for k in self.topk}


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    """
    Computes the accuracy over the k top predictions.
    
    Args:
        output: Model predictions (B, num_classes)
        target: Ground truth labels (B,)
        topk: Tuple of k values
    
    Returns:
        List of accuracies for each k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def set_random_seed(seed: int, deterministic: bool = False):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
        deterministic: Whether to use deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save YAML file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def save_checkpoint(
    state: Dict,
    is_best: bool,
    save_dir: str,
    filename: str = 'checkpoint.pth',
):
    """
    Save checkpoint.
    
    Args:
        state: State dictionary to save
        is_best: Whether this is the best checkpoint
        save_dir: Directory to save checkpoint
        filename: Checkpoint filename
    """
    os.makedirs(save_dir, exist_ok=True)
    
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_path = os.path.join(save_dir, 'best.pth')
        torch.save(state, best_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional = None,
    device: str = 'cuda',
) -> Dict:
    """
    Load checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state dict into
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state
        device: Device to load checkpoint to
    
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


# Distributed training helpers

def is_dist_available_and_initialized():
    """Check if distributed training is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    """Get world size (number of processes)."""
    if not is_dist_available_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """Get current process rank."""
    if not is_dist_available_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """Check if current process is main process."""
    return get_rank() == 0


def setup_distributed():
    """Setup distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
        )
        dist.barrier()
        return True
    return False


def cleanup_distributed():
    """Cleanup distributed training."""
    if is_dist_available_and_initialized():
        dist.destroy_process_group()


def reduce_dict(input_dict: Dict, average: bool = True):
    """
    Reduce dictionary of tensors across all processes.
    
    Args:
        input_dict: Dictionary of tensors
        average: Whether to average (vs sum)
    
    Returns:
        Reduced dictionary
    """
    if not is_dist_available_and_initialized():
        return input_dict
    
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        
        if average:
            values /= world_size
        
        reduced_dict = {k: v for k, v in zip(names, values)}
    
    return reduced_dict


class MetricLogger:
    """Logger for metrics during training."""
    
    def __init__(self, delimiter: str = "  "):
        self.meters = {}
        self.delimiter = delimiter
    
    def update(self, **kwargs):
        """Update meters with new values."""
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if k not in self.meters:
                self.meters[k] = AverageMeter()
            self.meters[k].update(v)
    
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")
    
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter.avg:.4f}")
        return self.delimiter.join(loss_str)
    
    def get_dict(self) -> Dict[str, float]:
        """Get dictionary of average values."""
        return {name: meter.avg for name, meter in self.meters.items()}
