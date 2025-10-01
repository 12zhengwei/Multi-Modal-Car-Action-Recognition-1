"""
Unit tests for model forward pass.

Tests model building and forward propagation.
"""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import build_model
from engine.utils import load_config


def test_single_modality_forward():
    """Test forward pass for single modality model."""
    print("Testing single modality forward pass...")
    
    # Load config
    config_path = project_root / 'configs' / 'modality_rgb.yaml'
    config = load_config(str(config_path))
    
    # Override some settings for testing
    config['data']['modality'] = 'rgb'
    config['model']['fusion'] = 'none'
    
    # Build model
    model = build_model(config)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    num_frames = 16
    resolution = 224
    dummy_input = torch.randn(batch_size, 3, num_frames, resolution, resolution)
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    # Check output shape
    num_classes = config['model']['num_classes']
    assert output.shape == (batch_size, num_classes)
    print(f"  ✓ Output shape: {output.shape}")


def test_dual_modality_early_fusion():
    """Test forward pass for early fusion model."""
    print("Testing early fusion forward pass...")
    
    # Load config
    config_path = project_root / 'configs' / 'fusion_early.yaml'
    config = load_config(str(config_path))
    
    # Override some settings for testing
    config['data']['modality'] = 'rgb_kir'
    
    # Build model
    model = build_model(config)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    num_frames = 16
    resolution = 224
    dummy_input = {
        'rgb': torch.randn(batch_size, 3, num_frames, resolution, resolution),
        'kir': torch.randn(batch_size, 3, num_frames, resolution, resolution),
    }
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    # Check output shape
    num_classes = config['model']['num_classes']
    assert output.shape == (batch_size, num_classes)
    print(f"  ✓ Output shape: {output.shape}")


def test_dual_modality_late_fusion():
    """Test forward pass for late fusion model."""
    print("Testing late fusion forward pass...")
    
    # Load config
    config_path = project_root / 'configs' / 'fusion_late.yaml'
    config = load_config(str(config_path))
    
    # Override some settings for testing
    config['data']['modality'] = 'rgb_kir'
    
    # Build model
    model = build_model(config)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    num_frames = 16
    resolution = 224
    dummy_input = {
        'rgb': torch.randn(batch_size, 3, num_frames, resolution, resolution),
        'kir': torch.randn(batch_size, 3, num_frames, resolution, resolution),
    }
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    # Check output shape
    num_classes = config['model']['num_classes']
    assert output.shape == (batch_size, num_classes)
    print(f"  ✓ Output shape: {output.shape}")


def test_dual_modality_meta_fusion():
    """Test forward pass for META fusion model."""
    print("Testing META fusion forward pass...")
    
    # Load config
    config_path = project_root / 'configs' / 'fusion_meta.yaml'
    config = load_config(str(config_path))
    
    # Override some settings for testing
    config['data']['modality'] = 'rgb_kir'
    
    # Build model
    model = build_model(config)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    num_frames = 16
    resolution = 224
    dummy_input = {
        'rgb': torch.randn(batch_size, 3, num_frames, resolution, resolution),
        'kir': torch.randn(batch_size, 3, num_frames, resolution, resolution),
    }
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    # Check output shape
    num_classes = config['model']['num_classes']
    assert output.shape == (batch_size, num_classes)
    print(f"  ✓ Output shape: {output.shape}")


def test_model_parameters():
    """Test model parameter counting."""
    print("Testing model parameter counting...")
    
    from models import count_parameters
    
    # Load config
    config_path = project_root / 'configs' / 'default.yaml'
    config = load_config(str(config_path))
    config['data']['modality'] = 'rgb'
    config['model']['fusion'] = 'none'
    
    # Build model
    model = build_model(config)
    
    # Count parameters
    params = count_parameters(model)
    
    assert params['total'] > 0
    assert params['trainable'] > 0
    assert params['trainable'] <= params['total']
    
    print(f"  ✓ Total parameters: {params['total']:,}")
    print(f"  ✓ Trainable parameters: {params['trainable']:,}")


def test_loss_computation():
    """Test loss computation."""
    print("Testing loss computation...")
    
    import torch.nn as nn
    
    # Create dummy predictions and labels
    batch_size = 4
    num_classes = 34
    predictions = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Compute loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(predictions, labels)
    
    assert loss.item() > 0
    print(f"  ✓ Loss computed: {loss.item():.4f}")
    
    # Test with label smoothing
    criterion_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)
    loss_smooth = criterion_smooth(predictions, labels)
    
    assert loss_smooth.item() > 0
    print(f"  ✓ Loss with label smoothing: {loss_smooth.item():.4f}")


def main():
    """Run all tests."""
    print("="*60)
    print("Running Model Forward Pass Tests")
    print("="*60)
    
    tests = [
        test_single_modality_forward,
        test_dual_modality_early_fusion,
        test_dual_modality_late_fusion,
        test_dual_modality_meta_fusion,
        test_model_parameters,
        test_loss_computation,
    ]
    
    failed = 0
    for test in tests:
        try:
            print(f"\n{test.__name__}")
            test()
        except Exception as e:
            print(f"  ✗ Test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    if failed == 0:
        print("✓ All tests passed!")
    else:
        print(f"✗ {failed}/{len(tests)} tests failed")
    print("="*60)
    
    return failed


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
