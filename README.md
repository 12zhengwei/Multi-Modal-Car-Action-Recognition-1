# Multi-Modal Car Action Recognition

Multi-Modal Fusion for Action Recognition in Car Cabin Environment using PyTorch and UniFormerV2

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

This project implements a complete, production-ready multi-modal action recognition system for car cabin environments. It supports:

- **Multi-Modal Learning**: RGB and KIR (infrared) video modalities
- **Four Fusion Strategies**:
  - **Early Fusion**: Input/feature-level fusion with channel bottleneck
  - **Late Fusion**: Decision-level fusion with learnable weights
  - **META Fusion**: Advanced fusion with Motion Excitation, Multi-View Excitation, and Temporal Aggregation
  - **CMCF Fusion**: Cross-Modal Complementary Fusion with modality-specific enhancement and adaptive weighting
- **UniFormerV2 Backbone**: State-of-the-art video transformer with local and global attention
- **34 Action Classes**: Comprehensive action categories for car cabin activities
- **Production Features**: Mixed precision training (AMP), distributed training (DDP), model export (ONNX, TorchScript)

## 🚀 Features

### Data Pipeline
- ✅ Dual video decoder support (decord + torchvision fallback)
- ✅ Multi-modal synchronized sampling
- ✅ Comprehensive video augmentation
- ✅ Automatic dataset indexing from directory structure
- ✅ Robust handling of variable-length videos

### Model Architecture
- ✅ UniFormerV2 hierarchical video transformer
- ✅ Local and global attention mechanisms
- ✅ Four fusion strategies with clean interfaces
- ✅ Configurable model depth and width
- ✅ Support for single and dual modality training

### Training & Evaluation
- ✅ Mixed precision training (AMP)
- ✅ Distributed training (DDP)
- ✅ Gradient clipping and early stopping
- ✅ Warmup + Cosine learning rate schedule
- ✅ TensorBoard logging
- ✅ Comprehensive evaluation metrics (Top-1, Top-5, Mean Class Accuracy)
- ✅ Confusion matrix and classification reports

### Inference & Export
- ✅ Single video and batch inference
- ✅ Visualization with prediction overlays
- ✅ ONNX export with dynamic batching
- ✅ TorchScript export (trace/script)
- ✅ JSON and CSV output formats

## 📦 Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (for GPU training)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/12zhengwei/Multi-Modal-Car-Action-Recognition-1.git
cd Multi-Modal-Car-Action-Recognition-1
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment** (optional):
```bash
cp .env.example .env
# Edit .env with your settings
```

## 📊 Dataset Structure

The dataset should follow this directory structure:

```
root_dir/
├── split0/
│   ├── video_train/
│   │   ├── rgb/
│   │   │   ├── closing bottle/
│   │   │   │   ├── video1.mp4
│   │   │   │   └── video2.mp4
│   │   │   └── ...
│   │   └── kir/
│   │       ├── closing bottle/
│   │       └── ...
│   ├── video_val/
│   │   ├── rgb/
│   │   └── kir/
│   └── video_test/
│       ├── rgb/
│       └── kir/
├── split1/
│   └── ...
└── split2/
    └── ...
```

### Action Classes (34 classes)

The system recognizes 34 different actions in car cabin environments:

- closing bottle, closing door inside, closing door outside, closing laptop
- drinking, eating, entering car, exiting_car
- fastening seat belt, fetching an object
- interacting with phone, looking or moving around
- opening_backpack, opening bottle, opening door inside, opening door outside, opening_laptop
- placing an object, preparing_food, pressing automation button
- putting laptop into backpack, putting_on jacket, putting_on_sunglasses
- reading_magazine, reading_newspaper, sitting_still
- taking_laptop from backpack, taking_off jacket, taking off sunglasses
- talking_on_phone, unfastening seat belt
- using_multimedia display, working_on_ laptop, writing

## 🔧 Quick Start

### 1. Build Dataset Indices

First, scan your dataset and create index files:

```bash
python data/build_indices.py \
    --root_dir /path/to/dataset \
    --out_dir cache \
    --splits split0 split1 split2
```

This will create:
- `cache/split{0,1,2}_{train,val,test}.csv`: Dataset indices
- `cache/class_to_id.json`: Class name to ID mapping

### 2. Training

#### Single GPU Training

```bash
# RGB only
python engine/train.py --config configs/modality_rgb.yaml

# KIR only
python engine/train.py --config configs/modality_kir.yaml

# Early Fusion (RGB + KIR)
python engine/train.py --config configs/fusion_early.yaml

# Late Fusion (RGB + KIR)
python engine/train.py --config configs/fusion_late.yaml

# META Fusion (RGB + KIR)
python engine/train.py --config configs/fusion_meta.yaml

# CMCF Fusion (RGB + KIR)
python engine/train.py --config configs/fusion_cmcf.yaml
```

Or use the provided script:
```bash
bash scripts/run_train.sh configs/fusion_meta.yaml 0
```

#### Multi-GPU Training (DDP)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    engine/train.py \
    --config configs/fusion_meta.yaml
```

### 3. Evaluation

Evaluate on validation or test set:

```bash
python engine/validate.py \
    --config configs/fusion_meta.yaml \
    --checkpoint outputs/split0_meta_rgbkir/checkpoints/best.pth \
    --split val \
    --num_clips 4
```

Or use the script:
```bash
bash scripts/run_eval.sh \
    configs/fusion_meta.yaml \
    outputs/split0_meta_rgbkir/checkpoints/best.pth \
    val
```

### 4. Inference

#### Single Video

```bash
python engine/infer.py \
    --config configs/fusion_meta.yaml \
    --checkpoint outputs/checkpoints/best.pth \
    --video /path/to/video.mp4 \
    --visualize \
    --output inference_results
```

#### Batch Inference

```bash
python engine/infer.py \
    --config configs/fusion_meta.yaml \
    --checkpoint outputs/checkpoints/best.pth \
    --video /path/to/video/directory \
    --output inference_results \
    --format json
```

### 5. Model Export

#### Export to ONNX

```bash
python engine/export_onnx.py \
    --config configs/fusion_meta.yaml \
    --checkpoint outputs/checkpoints/best.pth \
    --output model.onnx \
    --dynamic_batch
```

#### Export to TorchScript

```bash
python engine/export_torchscript.py \
    --config configs/fusion_meta.yaml \
    --checkpoint outputs/checkpoints/best.pth \
    --output model.torchscript \
    --method trace
```

## 📝 Configuration

All training and model configurations are in YAML files under `configs/`. Key configuration sections:

### Data Configuration
```yaml
data:
  root_dir: /path/to/dataset
  split: split0
  modality: rgb_kir  # rgb | kir | rgb_kir
  num_frames: 16
  resolution: 224
  augmentation:
    horizontal_flip: 0.5
    color_jitter: true
```

### Model Configuration
```yaml
model:
  backbone: uniformerv2
  num_classes: 34
  fusion: meta  # none | early | late | meta | cmcf
  uniformerv2:
    depth: [5, 8, 20, 7]
    embed_dim: [64, 128, 320, 512]
    drop_path_rate: 0.1
```

### Training Configuration
```yaml
train:
  epochs: 50
  batch_size: 8
  lr: 2e-4
  weight_decay: 0.05
  amp: true  # Mixed precision
  grad_clip: 20.0
  early_stopping_patience: 10
```

## 🧪 Testing

Run unit tests to verify installation:

```bash
# Test data loading pipeline
python tests/test_data_loading.py

# Test model forward pass
python tests/test_forward.py
```

## 📈 Performance Benchmarks

Expected performance on car cabin action recognition (example baselines):

| Method | Modality | Top-1 Acc | Top-5 Acc | Params | FLOPs |
|--------|----------|-----------|-----------|--------|-------|
| UniFormerV2-B | RGB | 75-80% | 92-95% | 50M | 80G |
| UniFormerV2-B | KIR | 70-75% | 90-93% | 50M | 80G |
| Early Fusion | RGB+KIR | 78-83% | 93-96% | 50M | 85G |
| Late Fusion | RGB+KIR | 80-85% | 94-97% | 100M | 160G |
| META Fusion | RGB+KIR | 82-87% | 95-98% | 105M | 165G |
| CMCF Fusion | RGB+KIR | 83-88% | 95-98% | 110M | 170G |

*Note: Actual performance depends on dataset quality, hyperparameters, and training settings.*

## 🔍 Advanced Topics

### Custom Fusion Strategy

To implement a custom fusion strategy:

1. Create a new file in `models/fusion/custom.py`
2. Implement your fusion module
3. Register it in `models/fusion/factory.py`
4. Create a config file in `configs/fusion_custom.yaml`

### Handling Variable-Length Videos

The data pipeline automatically handles videos of different lengths:
- Short videos: Looped or padded to target length
- Long videos: Sampled to target length
- Configurable via `num_frames` and sampling strategy

### KIR (Infrared) Processing

KIR videos are automatically converted to pseudo-3-channel format if single-channel. You can customize normalization parameters:

```yaml
data:
  normalize:
    kir_mean: [0.5, 0.5, 0.5]
    kir_std: [0.5, 0.5, 0.5]
```

## 🛠️ Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
- Reduce `batch_size` in config
- Enable gradient accumulation: `accumulation_steps: 2`
- Reduce `num_frames` or `resolution`

**2. Decord not available**
- The system automatically falls back to torchvision
- To install decord: `pip install decord`

**3. Slow data loading**
- Increase `num_workers` in config
- Use SSD for dataset storage
- Pre-cache indices with `build_indices.py`

**4. Training instability**
- Enable gradient clipping (default: 20.0)
- Reduce learning rate
- Increase warmup epochs

## 📚 Project Structure

```
project-root/
├── configs/              # Configuration files
├── data/                 # Data processing modules
│   ├── build_indices.py  # Dataset indexing
│   └── datasets/         # Dataset implementations
├── models/               # Model architectures
│   ├── uniformerv2/      # UniFormerV2 backbone
│   ├── fusion/           # Fusion strategies
│   ├── heads/            # Classification heads
│   └── builder.py        # Model builder
├── engine/               # Training and evaluation
│   ├── train.py          # Training script
│   ├── validate.py       # Evaluation script
│   ├── infer.py          # Inference script
│   ├── export_onnx.py    # ONNX export
│   └── export_torchscript.py  # TorchScript export
├── scripts/              # Shell scripts
├── tests/                # Unit tests
└── README.md             # This file
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- UniFormerV2: [Official Repository](https://github.com/OpenGVLab/UniFormerV2)
- PyTorch: [https://pytorch.org/](https://pytorch.org/)
- Video understanding community

## 📧 Contact

For questions and feedback:
- Open an issue on GitHub
- Contact: 12zhengwei

## 🔗 Citation

If you use this code in your research, please cite:

```bibtex
@software{multimodal_car_action_recognition,
  title={Multi-Modal Car Action Recognition},
  author={12zhengwei},
  year={2024},
  url={https://github.com/12zhengwei/Multi-Modal-Car-Action-Recognition-1}
}
```