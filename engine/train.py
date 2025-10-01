"""
Training script for multi-modal action recognition.

Supports:
- Mixed precision training (AMP)
- Distributed training (DDP)
- Gradient clipping
- Early stopping
- Checkpoint saving
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.datasets import build_dataloader
from models import build_model, print_model_info
from engine.utils import (
    AverageMeter,
    AccuracyMeter,
    set_random_seed,
    load_config,
    save_config,
    save_checkpoint,
    load_checkpoint,
    is_main_process,
    setup_distributed,
    cleanup_distributed,
)
from engine.scheduler import build_scheduler


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    epoch,
    config,
    device,
    writer=None,
):
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: GradScaler for AMP
        epoch: Current epoch number
        config: Training configuration
        device: Device to train on
        writer: TensorBoard writer
    
    Returns:
        Dictionary of training metrics
    """
    model.train()
    
    loss_meter = AverageMeter()
    acc_meter = AccuracyMeter(topk=(1, 5))
    
    # Get training config
    train_config = config['train']
    amp_enabled = train_config.get('amp', True)
    grad_clip = train_config.get('grad_clip', 20.0)
    log_freq = train_config.get('log_freq', 10)
    accumulation_steps = train_config.get('accumulation_steps', 1)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not is_main_process())
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(pbar):
        # Move data to device
        if isinstance(batch, tuple):
            # Single modality
            inputs, labels = batch
            inputs = inputs.to(device, non_blocking=True)
        else:
            # Multi-modal
            inputs = {k: v.to(device, non_blocking=True) 
                     for k, v in batch.items() if k != 'label'}
        
        labels = batch['label'] if isinstance(batch, dict) else labels
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass with AMP
        with autocast(enabled=amp_enabled):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
        
        # Backward pass
        if amp_enabled:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights
        if (step + 1) % accumulation_steps == 0:
            if amp_enabled:
                # Gradient clipping
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            
            optimizer.zero_grad()
        
        # Update metrics
        loss_meter.update(loss.item() * accumulation_steps, labels.size(0))
        acc_meter.update(outputs, labels)
        
        # Update progress bar
        if step % log_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc1': f'{acc_meter.accuracy(1):.2f}%',
                'lr': f'{lr:.6f}',
            })
            
            # Log to TensorBoard
            if writer is not None and is_main_process():
                global_step = epoch * len(dataloader) + step
                writer.add_scalar('train/loss', loss_meter.avg, global_step)
                writer.add_scalar('train/acc1', acc_meter.accuracy(1), global_step)
                writer.add_scalar('train/lr', lr, global_step)
    
    metrics = {
        'loss': loss_meter.avg,
        'acc1': acc_meter.accuracy(1),
        'acc5': acc_meter.accuracy(5),
    }
    
    return metrics


@torch.no_grad()
def validate(model, dataloader, criterion, epoch, config, device):
    """
    Validate model.
    
    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        epoch: Current epoch number
        config: Configuration
        device: Device to validate on
    
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    
    loss_meter = AverageMeter()
    acc_meter = AccuracyMeter(topk=(1, 5))
    
    pbar = tqdm(dataloader, desc=f"Validation", disable=not is_main_process())
    
    for batch in pbar:
        # Move data to device
        if isinstance(batch, tuple):
            inputs, labels = batch
            inputs = inputs.to(device, non_blocking=True)
        else:
            inputs = {k: v.to(device, non_blocking=True) 
                     for k, v in batch.items() if k != 'label'}
        
        labels = batch['label'] if isinstance(batch, dict) else labels
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Update metrics
        loss_meter.update(loss.item(), labels.size(0))
        acc_meter.update(outputs, labels)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc1': f'{acc_meter.accuracy(1):.2f}%',
        })
    
    metrics = {
        'loss': loss_meter.avg,
        'acc1': acc_meter.accuracy(1),
        'acc5': acc_meter.accuracy(5),
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train multi-modal action recognition')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override resume if provided
    if args.resume:
        config['train']['resume'] = args.resume
    
    # Setup distributed training
    distributed = setup_distributed()
    config['train']['distributed'] = distributed
    
    # Set random seed
    seed = config.get('seed', 42)
    set_random_seed(seed)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = config.get('device', {}).get('cudnn_benchmark', True)
    
    # Create output directory
    save_config_dict = config.get('save', {})
    out_dir = save_config_dict.get('out_dir', 'outputs/default')
    os.makedirs(out_dir, exist_ok=True)
    
    # Save configuration
    if is_main_process():
        save_config(config, os.path.join(out_dir, 'config.yaml'))
    
    # Build model
    if is_main_process():
        print("Building model...")
    model = build_model(config)
    model = model.to(device)
    
    if is_main_process():
        print_model_info(model)
    
    # Distributed training
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
        )
    
    # Build dataloaders
    if is_main_process():
        print("Building dataloaders...")
    
    data_config = config['data']
    train_config = config['train']
    
    train_loader = build_dataloader(
        index_file=data_config['train_index'],
        root_dir=data_config['root_dir'],
        modality=data_config['modality'],
        batch_size=train_config['batch_size'],
        num_workers=data_config.get('num_workers', 4),
        mode='train',
        config=data_config,
        pin_memory=data_config.get('pin_memory', True),
    )
    
    val_loader = build_dataloader(
        index_file=data_config['val_index'],
        root_dir=data_config['root_dir'],
        modality=data_config['modality'],
        batch_size=config['test']['batch_size'],
        num_workers=data_config.get('num_workers', 4),
        mode='val',
        config=data_config,
        pin_memory=data_config.get('pin_memory', True),
    )
    
    # Loss function
    label_smoothing = train_config.get('label_smoothing', 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # Optimizer
    optimizer_type = train_config.get('optimizer', 'adamw')
    if optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config['lr'],
            weight_decay=train_config.get('weight_decay', 0.05),
            betas=train_config.get('betas', [0.9, 0.999]),
            eps=train_config.get('eps', 1e-8),
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    # Learning rate scheduler
    scheduler = build_scheduler(optimizer, train_config)
    
    # AMP scaler
    scaler = GradScaler(enabled=train_config.get('amp', True))
    
    # TensorBoard writer
    writer = None
    if is_main_process():
        log_dir = os.path.join(out_dir, 'logs')
        writer = SummaryWriter(log_dir)
    
    # Resume training
    start_epoch = 0
    best_acc1 = 0.0
    
    if train_config.get('resume'):
        if is_main_process():
            print(f"Resuming from {train_config['resume']}...")
        checkpoint = load_checkpoint(
            train_config['resume'],
            model,
            optimizer,
            scheduler,
            device,
        )
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_acc1 = checkpoint.get('best_acc1', 0.0)
    
    # Training loop
    epochs = train_config.get('epochs', 50)
    val_freq = train_config.get('val_freq', 1)
    save_freq = train_config.get('save_freq', 1)
    early_stopping_patience = train_config.get('early_stopping_patience', 10)
    
    patience_counter = 0
    
    for epoch in range(start_epoch, epochs):
        if is_main_process():
            print(f"\nEpoch {epoch}/{epochs}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, epoch, config, device, writer
        )
        
        if is_main_process():
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc@1: {train_metrics['acc1']:.2f}%, "
                  f"Acc@5: {train_metrics['acc5']:.2f}%")
        
        # Validate
        if (epoch + 1) % val_freq == 0:
            val_metrics = validate(model, val_loader, criterion, epoch, config, device)
            
            if is_main_process():
                print(f"Val - Loss: {val_metrics['loss']:.4f}, "
                      f"Acc@1: {val_metrics['acc1']:.2f}%, "
                      f"Acc@5: {val_metrics['acc5']:.2f}%")
                
                # Log to TensorBoard
                writer.add_scalar('val/loss', val_metrics['loss'], epoch)
                writer.add_scalar('val/acc1', val_metrics['acc1'], epoch)
                writer.add_scalar('val/acc5', val_metrics['acc5'], epoch)
            
            # Check if best model
            is_best = val_metrics['acc1'] > best_acc1
            if is_best:
                best_acc1 = val_metrics['acc1']
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            if is_main_process() and (epoch + 1) % save_freq == 0:
                checkpoint_dir = os.path.join(out_dir, 'checkpoints')
                state = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if distributed else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc1': best_acc1,
                    'config': config,
                }
                save_checkpoint(state, is_best, checkpoint_dir, f'epoch_{epoch}.pth')
            
            # Early stopping
            if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                if is_main_process():
                    print(f"Early stopping triggered after {early_stopping_patience} epochs without improvement")
                break
        
        # Step scheduler
        scheduler.step()
    
    if is_main_process():
        print(f"\nTraining complete! Best Acc@1: {best_acc1:.2f}%")
        writer.close()
    
    # Cleanup
    if distributed:
        cleanup_distributed()


if __name__ == '__main__':
    main()
