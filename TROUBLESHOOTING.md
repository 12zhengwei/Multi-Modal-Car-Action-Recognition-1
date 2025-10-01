# Troubleshooting Guide

Common issues and their solutions for the Multi-Modal Car Action Recognition project.

## Table of Contents
- [Installation Issues](#installation-issues)
- [Data Loading Issues](#data-loading-issues)
- [Training Issues](#training-issues)
- [Memory Issues](#memory-issues)
- [Performance Issues](#performance-issues)
- [Model Issues](#model-issues)
- [Distributed Training Issues](#distributed-training-issues)

## Installation Issues

### Issue: pip install fails with "No matching distribution"
**Cause**: Incompatible Python version or package unavailable

**Solution**:
```bash
# Check Python version (need 3.8+)
python --version

# Upgrade pip
pip install --upgrade pip

# Install packages one by one to identify issue
pip install torch torchvision
pip install decord
```

### Issue: CUDA not available after installation
**Cause**: Wrong PyTorch version or CUDA driver mismatch

**Solution**:
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch version
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Issue: decord import error
**Cause**: decord not properly installed or incompatible

**Solution**:
```bash
# System will auto-fallback to torchvision, but to fix:
pip uninstall decord
pip install decord

# Or use conda:
conda install -c conda-forge decord
```

## Data Loading Issues

### Issue: "No videos found" when building indices
**Cause**: Incorrect directory structure or path

**Solution**:
```bash
# Check directory structure
ls /path/to/dataset/split0/video_train/rgb/
# Should show class folders

# Verify paths in config
cat configs/default.yaml | grep root_dir

# Build indices with verbose output
python data/build_indices.py --root_dir /path/to/dataset --out_dir cache
```

### Issue: "Failed to decode video"
**Cause**: Corrupted video file or unsupported codec

**Solution**:
```python
# Test video file with OpenCV
import cv2
cap = cv2.VideoCapture('/path/to/video.mp4')
if not cap.isOpened():
    print("Video cannot be opened")
else:
    print(f"Frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

# Re-encode video if needed
ffmpeg -i input.mp4 -c:v libx264 -preset fast output.mp4
```

### Issue: DataLoader hangs or is very slow
**Cause**: Too many/few workers, slow disk I/O

**Solution**:
```yaml
# In config.yaml, adjust num_workers
data:
  num_workers: 4  # Start with 4
  pin_memory: true

# Test different values (typically 2-8)
# If on HDD, reduce to 2
# If on SSD with fast CPU, increase to 8
```

### Issue: "Shape mismatch" in collate function
**Cause**: Inconsistent video dimensions or modality mismatch

**Solution**:
```python
# Check dataset outputs
from data.datasets import VideoDataset
dataset = VideoDataset(...)
sample = dataset[0]
print(type(sample), len(sample) if isinstance(sample, tuple) else sample.keys())

# Ensure modality setting matches data
# Single modality: modality: rgb or kir
# Dual modality: modality: rgb_kir
```

## Training Issues

### Issue: Loss is NaN
**Cause**: Learning rate too high, gradient explosion, bad initialization

**Solution**:
```yaml
# Reduce learning rate
train:
  lr: 1e-4  # Try 1e-4 instead of 2e-4
  
  # Enable gradient clipping
  grad_clip: 10.0  # Try lower value
  
  # Add warmup
  warmup_epochs: 10  # Increase warmup
```

### Issue: Loss not decreasing
**Cause**: Learning rate too low, wrong optimizer settings, data issues

**Solution**:
```yaml
# Check data loading
# Verify labels are correct in indices

# Increase learning rate
train:
  lr: 5e-4
  
# Check if model is updating
# Add this to training loop:
print(f"Grad norm: {torch.nn.utils.clip_grad_norm_(model.parameters(), 1e10)}")
```

### Issue: Training crashes with "CUDA out of memory"
**See Memory Issues section below**

### Issue: Validation accuracy stuck at 0
**Cause**: Wrong evaluation mode, data mismatch, model not loaded

**Solution**:
```python
# Ensure model is in eval mode
model.eval()

# Check if model is making predictions
with torch.no_grad():
    output = model(sample_input)
    print(output.shape, output.max(), output.min())

# Verify labels match classes
# Check class_to_id.json and dataset labels
```

### Issue: Training very slow
**See Performance Issues section below**

## Memory Issues

### Issue: CUDA out of memory during training
**Cause**: Batch size too large, model too big, memory leak

**Solution**:
```yaml
# Option 1: Reduce batch size
train:
  batch_size: 2  # Reduce from 8
  
# Option 2: Enable gradient accumulation
train:
  batch_size: 2
  accumulation_steps: 4  # Effective batch size = 2*4 = 8

# Option 3: Reduce video frames/resolution
data:
  num_frames: 8  # Reduce from 16
  resolution: 112  # Reduce from 224

# Option 4: Use smaller model
model:
  uniformerv2:
    depth: [3, 4, 8, 3]  # Small model
```

### Issue: Out of memory during validation
**Cause**: Larger validation batch, memory not freed

**Solution**:
```yaml
# Reduce test batch size
test:
  batch_size: 1

# Clear cache between batches
# Add to validation loop:
if i % 10 == 0:
    torch.cuda.empty_cache()
```

### Issue: Memory leak during training
**Cause**: Accumulating gradients, not detaching tensors

**Solution**:
```python
# Ensure proper gradient handling
optimizer.zero_grad()  # Always call before backward

# Don't accumulate loss values
loss_value = loss.item()  # Use .item() to detach

# Clear cache periodically
if epoch % 5 == 0:
    torch.cuda.empty_cache()
```

## Performance Issues

### Issue: Training is too slow
**Cause**: Various bottlenecks in data, model, or I/O

**Solution**:
```yaml
# 1. Enable AMP
train:
  amp: true

# 2. Enable cudnn benchmark
device:
  cudnn_benchmark: true

# 3. Optimize data loading
data:
  num_workers: 8  # Increase
  pin_memory: true

# 4. Use SSD for dataset
# 5. Reduce log frequency
train:
  log_freq: 50  # Log less often
```

### Issue: Data loading is the bottleneck
**Cause**: Slow video decoding, too few workers

**Solution**:
```bash
# Profile data loading
python -c "
from data.datasets import build_dataloader
import time
loader = build_dataloader(...)
start = time.time()
for i, batch in enumerate(loader):
    if i >= 10: break
print(f'Time per batch: {(time.time()-start)/10:.2f}s')
"

# Solutions:
# 1. Pre-decode videos and save as numpy arrays
# 2. Use faster storage (SSD)
# 3. Increase num_workers
# 4. Reduce video resolution during decoding
```

### Issue: GPU utilization is low
**Cause**: Data loading bottleneck, small batch size

**Solution**:
```bash
# Monitor GPU utilization
nvidia-smi -l 1

# If GPU usage < 50%:
# 1. Increase batch_size
# 2. Increase num_workers
# 3. Check if data loading is slow (see above)

# If GPU usage is high but training is slow:
# 1. Use a larger GPU
# 2. Reduce model size
# 3. Enable AMP
```

## Model Issues

### Issue: Model output shape mismatch
**Cause**: Wrong configuration, incorrect input

**Solution**:
```python
# Test model with dummy input
import torch
model = build_model(config)
model.eval()

# Single modality
dummy = torch.randn(1, 3, 16, 224, 224)
out = model(dummy)
print(out.shape)  # Should be (1, 34)

# Dual modality
dummy = {
    'rgb': torch.randn(1, 3, 16, 224, 224),
    'kir': torch.randn(1, 3, 16, 224, 224),
}
out = model(dummy)
print(out.shape)  # Should be (1, 34)
```

### Issue: "Forward pass failed" with fusion model
**Cause**: Input format mismatch, missing modality

**Solution**:
```python
# Check input format for fusion
print(f"Modality: {config['data']['modality']}")
print(f"Fusion: {config['model']['fusion']}")

# Ensure they match:
# rgb or kir → fusion: none
# rgb_kir → fusion: early/late/meta
```

### Issue: Cannot load checkpoint
**Cause**: Model architecture changed, state dict key mismatch

**Solution**:
```python
# Load with relaxed matching
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# Or manually map keys if needed
state_dict = checkpoint['model_state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    new_k = k.replace('module.', '')  # Remove DDP prefix
    new_state_dict[new_k] = v
model.load_state_dict(new_state_dict)
```

## Distributed Training Issues

### Issue: DDP initialization fails
**Cause**: Environment variables not set, wrong configuration

**Solution**:
```bash
# Ensure you're using torchrun
torchrun --nproc_per_node=2 engine/train.py --config config.yaml

# Not: python engine/train.py

# Check environment
echo $RANK $WORLD_SIZE $LOCAL_RANK

# Common fix: specify master address explicitly
torchrun --nproc_per_node=2 \
         --master_addr=localhost \
         --master_port=29500 \
         engine/train.py --config config.yaml
```

### Issue: Training hangs with multiple GPUs
**Cause**: Deadlock, mismatched batch sizes across GPUs

**Solution**:
```yaml
# Ensure drop_last=True for training
# Already implemented in code, but verify:
data:
  # Batch size should be divisible by num_GPUs
  train_batch_size: 8  # For 2 GPUs: 8/2 = 4 per GPU
  
# Check for operations that require synchronization
# Avoid operations like .item() in training loop
```

### Issue: Different results on different GPUs
**Cause**: Random seed not set properly, non-deterministic ops

**Solution**:
```python
# In config:
seed: 42

# Code already handles this, but verify:
set_random_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Export Issues

### Issue: ONNX export fails
**Cause**: Unsupported operations, dynamic shapes

**Solution**:
```python
# Try different opset version
python engine/export_onnx.py --opset 11  # Try 11, 12, 13

# Simplify model if needed
# Remove unsupported operations

# Use torch.onnx.export with verbose=True for debugging
```

### Issue: TorchScript trace fails
**Cause**: Control flow in model, dynamic operations

**Solution**:
```bash
# Try script method instead of trace
python engine/export_torchscript.py --method script

# Or simplify model to remove control flow
```

## Getting Help

If you're still stuck:

1. **Check logs**: Look at tensorboard logs and console output
2. **Enable debug mode**: Add `--debug` flag and print statements
3. **Test components**: Run unit tests to isolate issues
4. **Search issues**: Check existing GitHub issues
5. **Create issue**: Open a new issue with:
   - Python version
   - PyTorch version
   - CUDA version
   - Full error traceback
   - Minimal code to reproduce

## Performance Monitoring

### GPU Memory
```bash
# Monitor GPU memory
watch -n 1 nvidia-smi

# In Python
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
```

### Training Speed
```python
# Add profiling to training loop
import time
times = []
for epoch in range(epochs):
    start = time.time()
    # ... training code ...
    times.append(time.time() - start)
print(f"Avg epoch time: {sum(times)/len(times):.2f}s")
```

### Data Loading Speed
```python
# Time data loading
import time
loader = build_dataloader(...)
times = []
for i, batch in enumerate(loader):
    start = time.time()
    # Process batch
    times.append(time.time() - start)
    if i >= 100: break
print(f"Avg batch time: {sum(times)/len(times):.3f}s")
```
