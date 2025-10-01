"""
Export model to TorchScript format.

Supports both tracing and scripting methods.
"""

import os
import sys
import argparse
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import build_model
from engine.utils import load_config, load_checkpoint


def export_torchscript(
    model,
    output_path,
    config,
    method='trace',
):
    """
    Export model to TorchScript format.
    
    Args:
        model: PyTorch model
        output_path: Path to save TorchScript model
        config: Configuration dictionary
        method: Export method ('trace' or 'script')
    """
    model.eval()
    
    # Get input configuration
    data_config = config['data']
    modality = data_config.get('modality', 'rgb')
    num_frames = data_config.get('num_frames', 16)
    resolution = data_config.get('resolution', 224)
    
    # Create dummy input
    batch_size = 1
    dummy_input = {}
    
    if modality == 'rgb_kir':
        # Dual modality
        dummy_input = {
            'rgb': torch.randn(batch_size, 3, num_frames, resolution, resolution),
            'kir': torch.randn(batch_size, 3, num_frames, resolution, resolution),
        }
    else:
        # Single modality
        dummy_input = torch.randn(batch_size, 3, num_frames, resolution, resolution)
    
    # Move to same device as model
    device = next(model.parameters()).device
    if isinstance(dummy_input, dict):
        dummy_input = {k: v.to(device) for k, v in dummy_input.items()}
    else:
        dummy_input = dummy_input.to(device)
    
    # Export to TorchScript
    print(f"Exporting model to TorchScript using {method} method...")
    
    if method == 'trace':
        # Trace method
        with torch.no_grad():
            traced_model = torch.jit.trace(model, dummy_input)
        scripted_model = traced_model
    
    elif method == 'script':
        # Script method
        scripted_model = torch.jit.script(model)
    
    else:
        raise ValueError(f"Unknown export method: {method}")
    
    # Save model
    scripted_model.save(output_path)
    print(f"✓ Model exported to {output_path}")
    
    # Test inference
    print("\nTesting inference with TorchScript model...")
    loaded_model = torch.jit.load(output_path)
    loaded_model.eval()
    
    with torch.no_grad():
        ts_output = loaded_model(dummy_input)
    
    print(f"✓ TorchScript inference successful")
    print(f"  Output shape: {ts_output.shape}")
    
    # Compare with original model
    print("\nComparing with original PyTorch model...")
    with torch.no_grad():
        pt_output = model(dummy_input)
    
    max_diff = (pt_output - ts_output).abs().max().item()
    print(f"  Max difference: {max_diff}")
    
    if max_diff < 1e-5:
        print("✓ Outputs match (tolerance: 1e-5)")
    else:
        print(f"⚠ Warning: Difference detected ({max_diff})")
    
    # Print model info
    print("\nModel Information:")
    print(f"  Method: {method}")
    
    # Get file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  File size: {file_size:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Export model to TorchScript')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--output', type=str, default='model.torchscript', help='Output file')
    parser.add_argument('--method', type=str, default='trace', choices=['trace', 'script'],
                       help='Export method (trace or script)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build model
    print("Building model...")
    model = build_model(config)
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    load_checkpoint(args.checkpoint, model, device=device)
    
    # Export to TorchScript
    export_torchscript(
        model,
        args.output,
        config,
        method=args.method,
    )
    
    print(f"\n✓ Export complete!")
    print(f"  TorchScript model saved to: {args.output}")
    
    # Usage example
    print("\n" + "="*60)
    print("Usage Example:")
    print("="*60)
    print(f"""
import torch

# Load model
model = torch.jit.load('{args.output}')
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 16, 224, 224)

# Run inference
with torch.no_grad():
    output = model(dummy_input)

print(output.shape)  # (1, num_classes)
""")


if __name__ == '__main__':
    main()
