"""
Unit tests for data loading pipeline.

Tests video decoding, transforms, and dataset functionality.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.datasets.decoding import VideoDecoder, sample_frames, pad_or_truncate_frames
from data.datasets.transforms import VideoToTensor, VideoNormalize, VideoResize


def test_frame_sampling():
    """Test frame sampling functions."""
    print("Testing frame sampling...")
    
    # Test uniform sampling
    total_frames = 100
    num_frames = 16
    indices = sample_frames(total_frames, num_frames, mode='uniform')
    assert len(indices) == num_frames
    assert indices[0] == 0
    assert indices[-1] == total_frames - 1
    print("  ✓ Uniform sampling works")
    
    # Test random sampling
    indices = sample_frames(total_frames, num_frames, mode='random', temporal_stride=2)
    assert len(indices) == num_frames
    print("  ✓ Random sampling works")


def test_frame_padding():
    """Test frame padding functions."""
    print("Testing frame padding...")
    
    # Create dummy frames
    frames = np.random.rand(10, 224, 224, 3).astype(np.float32)
    
    # Test padding
    padded = pad_or_truncate_frames(frames, 16, mode='loop')
    assert padded.shape == (16, 224, 224, 3)
    print("  ✓ Frame padding works")
    
    # Test truncation
    truncated = pad_or_truncate_frames(frames, 5, mode='loop')
    assert truncated.shape == (5, 224, 224, 3)
    print("  ✓ Frame truncation works")


def test_video_transforms():
    """Test video transform functions."""
    print("Testing video transforms...")
    
    # Create dummy video (numpy array)
    video_np = np.random.randint(0, 256, (16, 224, 224, 3), dtype=np.uint8)
    
    # Test VideoToTensor
    to_tensor = VideoToTensor()
    video_tensor = to_tensor(video_np)
    assert video_tensor.shape == (3, 16, 224, 224)
    assert video_tensor.dtype == torch.float32
    assert 0 <= video_tensor.min() <= video_tensor.max() <= 1.0
    print("  ✓ VideoToTensor works")
    
    # Test VideoNormalize
    normalize = VideoNormalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    normalized = normalize(video_tensor)
    assert normalized.shape == video_tensor.shape
    print("  ✓ VideoNormalize works")
    
    # Test VideoResize
    resize = VideoResize((112, 112))
    resized = resize(video_tensor)
    assert resized.shape == (3, 16, 112, 112)
    print("  ✓ VideoResize works")


def test_decoder_initialization():
    """Test video decoder initialization."""
    print("Testing video decoder initialization...")
    
    try:
        # Test auto backend
        decoder = VideoDecoder(backend='auto')
        print(f"  ✓ VideoDecoder initialized with backend: {decoder.backend}")
    except RuntimeError as e:
        print(f"  ⚠ VideoDecoder initialization failed: {str(e)}")
        print("  This is expected if neither decord nor torchvision is available")


def test_config_loading():
    """Test configuration loading."""
    print("Testing configuration loading...")
    
    from engine.utils import load_config
    
    config_path = project_root / 'configs' / 'default.yaml'
    config = load_config(str(config_path))
    
    assert 'data' in config
    assert 'model' in config
    assert 'train' in config
    print("  ✓ Configuration loading works")


def main():
    """Run all tests."""
    print("="*60)
    print("Running Data Loading Tests")
    print("="*60)
    
    tests = [
        test_frame_sampling,
        test_frame_padding,
        test_video_transforms,
        test_decoder_initialization,
        test_config_loading,
    ]
    
    failed = 0
    for test in tests:
        try:
            print(f"\n{test.__name__}")
            test()
        except Exception as e:
            print(f"  ✗ Test failed: {str(e)}")
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
