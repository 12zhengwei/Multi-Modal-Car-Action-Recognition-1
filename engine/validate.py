"""
Validation and testing script for multi-modal action recognition.

Evaluates model on validation or test set with comprehensive metrics.
"""

import os
import sys
import argparse
from pathlib import Path
import json

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.datasets import build_dataloader
from models import build_model
from engine.utils import (
    AverageMeter,
    AccuracyMeter,
    set_random_seed,
    load_config,
    load_checkpoint,
)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, num_clips=1):
    """
    Evaluate model on dataset.
    
    Args:
        model: Model to evaluate
        dataloader: Dataloader
        criterion: Loss function
        device: Device
        num_clips: Number of clips for multi-clip testing
    
    Returns:
        Dictionary of metrics and predictions
    """
    model.eval()
    
    loss_meter = AverageMeter()
    acc_meter = AccuracyMeter(topk=(1, 5))
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(dataloader, desc="Evaluating")
    
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
        
        # Get predictions
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        
        # Update metrics
        loss_meter.update(loss.item(), labels.size(0))
        acc_meter.update(outputs, labels)
        
        # Store predictions
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc1': f'{acc_meter.accuracy(1):.2f}%',
        })
    
    # Compute per-class metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Per-class accuracy (mean class accuracy)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    mean_class_acc = np.mean(class_accuracies)
    
    metrics = {
        'loss': loss_meter.avg,
        'acc1': acc_meter.accuracy(1),
        'acc5': acc_meter.accuracy(5),
        'mean_class_acc': mean_class_acc * 100,
        'confusion_matrix': cm,
        'class_accuracies': class_accuracies,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
    }
    
    return metrics


def plot_confusion_matrix(cm, class_names, save_path):
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save plot
    """
    plt.figure(figsize=(20, 18))
    sns.heatmap(
        cm,
        annot=False,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved confusion matrix to {save_path}")


def save_results(metrics, class_names, save_dir):
    """
    Save evaluation results.
    
    Args:
        metrics: Dictionary of metrics
        class_names: List of class names
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics as JSON
    json_metrics = {
        'loss': float(metrics['loss']),
        'top1_accuracy': float(metrics['acc1']),
        'top5_accuracy': float(metrics['acc5']),
        'mean_class_accuracy': float(metrics['mean_class_acc']),
    }
    
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(json_metrics, f, indent=2)
    
    # Save per-class accuracies
    class_acc_dict = {
        class_names[i]: float(acc) * 100
        for i, acc in enumerate(metrics['class_accuracies'])
    }
    with open(os.path.join(save_dir, 'class_accuracies.json'), 'w') as f:
        json.dump(class_acc_dict, f, indent=2)
    
    # Save confusion matrix as CSV
    cm_path = os.path.join(save_dir, 'confusion_matrix.csv')
    np.savetxt(cm_path, metrics['confusion_matrix'], delimiter=',', fmt='%d')
    
    # Plot confusion matrix
    plot_path = os.path.join(save_dir, 'confusion_matrix.png')
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, plot_path)
    
    # Generate classification report
    report = classification_report(
        metrics['labels'],
        metrics['predictions'],
        target_names=class_names,
        digits=3,
    )
    
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    print(report)


def main():
    parser = argparse.ArgumentParser(description='Evaluate multi-modal action recognition')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                       help='Split to evaluate on')
    parser.add_argument('--num_clips', type=int, default=1, help='Number of clips for testing')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    seed = config.get('seed', 42)
    set_random_seed(seed)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Output directory
    if args.output is None:
        checkpoint_dir = Path(args.checkpoint).parent
        args.output = os.path.join(checkpoint_dir, f'eval_{args.split}')
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Results will be saved to: {args.output}")
    
    # Build model
    print("Building model...")
    model = build_model(config)
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    load_checkpoint(args.checkpoint, model, device=device)
    
    # Build dataloader
    print(f"Building dataloader for {args.split} split...")
    data_config = config['data']
    
    if args.split == 'val':
        index_file = data_config['val_index']
    else:
        index_file = data_config['test_index']
    
    dataloader = build_dataloader(
        index_file=index_file,
        root_dir=data_config['root_dir'],
        modality=data_config['modality'],
        batch_size=config['test']['batch_size'],
        num_workers=data_config.get('num_workers', 4),
        mode='val',
        config=data_config,
        pin_memory=False,
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Load class names
    import json
    class_mapping_path = os.path.join(
        os.path.dirname(index_file),
        'class_to_id.json'
    )
    with open(class_mapping_path, 'r') as f:
        class_to_id = json.load(f)
    
    # Sort by ID to get class names
    id_to_class = {v: k for k, v in class_to_id.items()}
    class_names = [id_to_class[i] for i in range(len(id_to_class))]
    
    # Evaluate
    print(f"\nEvaluating on {args.split} set...")
    metrics = evaluate(model, dataloader, criterion, device, args.num_clips)
    
    # Print summary
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Top-1 Accuracy: {metrics['acc1']:.2f}%")
    print(f"Top-5 Accuracy: {metrics['acc5']:.2f}%")
    print(f"Mean Class Accuracy: {metrics['mean_class_acc']:.2f}%")
    print("="*60)
    
    # Save results
    print("\nSaving results...")
    save_results(metrics, class_names, args.output)
    
    print(f"\n✓ Evaluation complete! Results saved to {args.output}")


if __name__ == '__main__':
    main()
