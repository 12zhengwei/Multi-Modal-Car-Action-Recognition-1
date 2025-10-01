"""
Export model to ONNX format.

Supports dynamic batch size and optional dynamic temporal dimension.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import onnx
import onnxruntime as ort

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import build_model
from engine.utils import load_config, load_checkpoint


def export_onnx(
    model,
    output_path,
    config,
    dynamic_batch=True,
    dynamic_temporal=False,
    opset_version=13,
):
    """
    Export model to ONNX format.
    
    Args:
        model: PyTorch model
        output_path: Path to save ONNX model
        config: Configuration dictionary
        dynamic_batch: Whether to use dynamic batch dimension
        dynamic_temporal: Whether to use dynamic temporal dimension
        opset_version: ONNX opset version
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
        input_names = ['rgb', 'kir']
    else:
        # Single modality
        dummy_input = torch.randn(batch_size, 3, num_frames, resolution, resolution)
        input_names = ['video']
    
    output_names = ['logits']
    
    # Define dynamic axes
    dynamic_axes = {}
    
    if dynamic_batch:
        for name in input_names:
            dynamic_axes[name] = {0: 'batch_size'}
        dynamic_axes['logits'] = {0: 'batch_size'}
    
    if dynamic_temporal:
        for name in input_names:
            if name not in dynamic_axes:
                dynamic_axes[name] = {}
            dynamic_axes[name][2] = 'num_frames'
    
    # Export to ONNX
    print(f"Exporting model to ONNX...")
    print(f"  Input names: {input_names}")
    print(f"  Output names: {output_names}")
    print(f"  Dynamic axes: {dynamic_axes}")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes if (dynamic_batch or dynamic_temporal) else None,
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False,
    )
    
    print(f"✓ Model exported to {output_path}")
    
    # Verify ONNX model
    print("\nVerifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")
    
    # Print model info
    print("\nModel Information:")
    print(f"  Opset version: {onnx_model.opset_import[0].version}")
    print(f"  IR version: {onnx_model.ir_version}")
    
    # Test inference with ONNX Runtime
    print("\nTesting inference with ONNX Runtime...")
    ort_session = ort.InferenceSession(output_path)
    
    # Prepare input
    if modality == 'rgb_kir':
        ort_inputs = {
            'rgb': dummy_input['rgb'].numpy(),
            'kir': dummy_input['kir'].numpy(),
        }
    else:
        ort_inputs = {'video': dummy_input.numpy()}
    
    # Run inference
    ort_outputs = ort_session.run(None, ort_inputs)
    
    print(f"✓ ONNX Runtime inference successful")
    print(f"  Output shape: {ort_outputs[0].shape}")
    
    # Compare with PyTorch
    print("\nComparing with PyTorch output...")
    with torch.no_grad():
        if modality == 'rgb_kir':
            pt_output = model({k: v.cuda() if torch.cuda.is_available() else v 
                              for k, v in dummy_input.items()})
        else:
            pt_output = model(dummy_input.cuda() if torch.cuda.is_available() else dummy_input)
        pt_output = pt_output.cpu().numpy()
    
    # Check similarity
    max_diff = abs(pt_output - ort_outputs[0]).max()
    print(f"  Max difference: {max_diff}")
    
    if max_diff < 1e-3:
        print("✓ Outputs match (tolerance: 1e-3)")
    else:
        print(f"⚠ Warning: Large difference detected ({max_diff})")


def main():
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--output', type=str, default='model.onnx', help='Output ONNX file')
    parser.add_argument('--dynamic_batch', action='store_true', help='Use dynamic batch size')
    parser.add_argument('--dynamic_temporal', action='store_true', help='Use dynamic temporal dimension')
    parser.add_argument('--opset', type=int, default=13, help='ONNX opset version')
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
    
    # Move model to CPU for export
    model = model.cpu()
    
    # Export to ONNX
    export_onnx(
        model,
        args.output,
        config,
        dynamic_batch=args.dynamic_batch,
        dynamic_temporal=args.dynamic_temporal,
        opset_version=args.opset,
    )
    
    print(f"\n✓ Export complete!")
    print(f"  ONNX model saved to: {args.output}")


if __name__ == '__main__':
    main()
