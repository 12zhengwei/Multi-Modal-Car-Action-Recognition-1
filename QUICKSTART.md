# Quick Start Guide

This guide will help you get started with the Multi-Modal Car Action Recognition project in 5 minutes.

## Prerequisites

- Python 3.8+
- CUDA 11.7+ (optional, for GPU training)
- 16GB+ RAM
- 50GB+ free disk space (for dataset)

## Installation (2 minutes)

```bash
# Clone repository
git clone https://github.com/12zhengwei/Multi-Modal-Car-Action-Recognition-1.git
cd Multi-Modal-Car-Action-Recognition-1

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation (1 minute)

1. **Organize your dataset** following the structure:
```
/path/to/dataset/
├── split0/
│   ├── video_train/
│   │   ├── rgb/
│   │   │   └── [class_folders]/
│   │   └── kir/
│   │       └── [class_folders]/
│   ├── video_val/
│   └── video_test/
```

2. **Build indices**:
```bash
python data/build_indices.py \
    --root_dir /path/to/dataset \
    --out_dir cache
```

## Configuration (30 seconds)

Edit `configs/default.yaml`:

```yaml
data:
  root_dir: /path/to/dataset  # Change this
  split: split0
  modality: rgb_kir
  train_index: cache/split0_train.csv
  val_index: cache/split0_val.csv
```

## Training (1 minute to start)

```bash
# Quick training with default config
python engine/train.py --config configs/fusion_meta.yaml
```

Or use the shell script:
```bash
bash scripts/run_train.sh configs/fusion_meta.yaml 0
```

## Evaluation (30 seconds)

```bash
python engine/validate.py \
    --config configs/fusion_meta.yaml \
    --checkpoint outputs/split0_meta_rgbkir/checkpoints/best.pth \
    --split val
```

## Inference (30 seconds)

```bash
python engine/infer.py \
    --config configs/fusion_meta.yaml \
    --checkpoint outputs/split0_meta_rgbkir/checkpoints/best.pth \
    --video /path/to/video.mp4 \
    --visualize
```

## Tips for Quick Testing

### 1. Small Dataset Test
Create a minimal dataset with just 2-3 videos per class for quick testing:
```bash
data:
  num_frames: 8  # Reduce from 16
  resolution: 112  # Reduce from 224
train:
  batch_size: 2
  epochs: 5
```

### 2. Fast Model Configuration
Use smaller model for quick testing:
```yaml
model:
  uniformerv2:
    depth: [3, 4, 8, 3]  # Smaller model
```

### 3. Debug Mode
Add print statements or use Python debugger:
```bash
python -m pdb engine/train.py --config configs/default.yaml
```

## Common Issues

**Out of Memory?**
- Reduce `batch_size` to 1 or 2
- Reduce `num_frames` to 8
- Reduce `resolution` to 112

**No GPU?**
- Training works on CPU (slower)
- Set `train.amp: false` in config

**Missing decord?**
- System automatically uses torchvision
- Or install: `pip install decord`

## Next Steps

1. **Customize model**: Edit configs for your needs
2. **Try different fusion**: Test early, late, and META fusion
3. **Hyperparameter tuning**: Adjust learning rate, batch size, etc.
4. **Export model**: Use ONNX or TorchScript for deployment

## Support

- 📖 Full documentation: See [README.md](README.md)
- 🐛 Issues: Open an issue on GitHub
- 💬 Questions: Check existing issues or create new one

Happy training! 🚀
