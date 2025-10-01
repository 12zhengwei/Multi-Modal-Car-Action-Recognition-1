# Architecture Documentation

This document provides a detailed overview of the system architecture for the Multi-Modal Car Action Recognition project.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Videos                             │
│                (RGB + KIR Modalities)                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Data Pipeline                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │Video Decoding│→ │Preprocessing │→ │Augmentation  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Model Architecture                          │
│  ┌──────────────────────────────────────────────────┐       │
│  │            UniFormerV2 Backbone                  │       │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐│       │
│  │  │Stage 1 │→ │Stage 2 │→ │Stage 3 │→ │Stage 4 ││       │
│  │  │(Local) │  │(Local) │  │(Local) │  │(Global)││       │
│  │  └────────┘  └────────┘  └────────┘  └────────┘│       │
│  └──────────────────────────────────────────────────┘       │
│                       │                                      │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────┐       │
│  │          Fusion Strategy                         │       │
│  │  (Early / Late / META)                           │       │
│  └──────────────────────────────────────────────────┘       │
│                       │                                      │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────┐       │
│  │        Classification Head                       │       │
│  └──────────────────────────────────────────────────┘       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Output Predictions                          │
│         (Top-1, Top-5, Confidence Scores)                    │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Pipeline

#### Video Decoding (`data/datasets/decoding.py`)
- **Primary**: decord (GPU-accelerated, fast)
- **Fallback**: torchvision.io (CPU-based, widely compatible)
- Handles variable-length videos
- Supports frame sampling strategies (uniform, random)

#### Preprocessing (`data/datasets/transforms.py`)
- Frame sampling (16 frames default)
- Spatial resizing (224x224 default)
- Normalization (ImageNet-style or custom)
- KIR to pseudo-RGB conversion

#### Augmentation
- **Training**: Random crop, horizontal flip, color jitter
- **Validation/Testing**: Center crop only
- Synchronized across modalities

### 2. Model Architecture

#### UniFormerV2 Backbone (`models/uniformerv2/`)

**Design Philosophy**: Hierarchical video transformer with local and global attention

**Architecture**:
```
Input: (B, C=3, T=16, H=224, W=224)

Stage 1 (Local):
  - Patch Embedding: 4×4 spatial, 2× temporal
  - Blocks: 5× Local MHRA + FFN
  - Output: (B, 64, 8, 56, 56)

Stage 2 (Local):
  - Downsample: 2×2 spatial, 2× temporal
  - Blocks: 8× Local MHRA + FFN
  - Output: (B, 128, 4, 28, 28)

Stage 3 (Local):
  - Downsample: 2×2 spatial, 2× temporal
  - Blocks: 20× Local MHRA + FFN
  - Output: (B, 320, 2, 14, 14)

Stage 4 (Global):
  - Downsample: 2×2 spatial, 1× temporal
  - Blocks: 7× Global MHRA + FFN
  - Output: (B, 512, 2, 7, 7)
```

**Key Components**:
- **Local MHRA**: Multi-Head Relation Aggregator with 3D convolution
- **Global MHRA**: Standard multi-head self-attention
- **FFN**: Feed-forward network with GELU activation
- **DropPath**: Stochastic depth for regularization

### 3. Fusion Strategies

#### Early Fusion (`models/fusion/early.py`)

```
RGB Video (B,3,T,H,W) ──┐
                        ├─→ Concat ──→ 1x1 Conv ──→ Shared Backbone ──→ Head
KIR Video (B,3,T,H,W) ──┘   (B,6,T,H,W)  (B,3,T,H,W)
```

**Features**:
- Input/feature level fusion
- Optional 1x1 conv bottleneck for dimension reduction
- Single backbone (parameter efficient)

#### Late Fusion (`models/fusion/late.py`)

```
RGB Video ──→ Backbone₁ ──→ Head₁ ──→ Logits₁ ──┐
                                                  ├─→ Weighted Fusion ──→ Final Logits
KIR Video ──→ Backbone₂ ──→ Head₂ ──→ Logits₂ ──┘
```

**Features**:
- Decision-level fusion
- Independent processing per modality
- Learnable or fixed fusion weights
- Three fusion methods: logits averaging, probability averaging, weighted

#### META Fusion (`models/fusion/meta.py`)

```
RGB Video ──→ Backbone₁ ──→ Features₁ ──┐
                                         │
                                         ├─→ Motion Excitation
                                         │       ↓
                                         ├─→ Multi-View Excitation
                                         │       ↓
                                         ├─→ Temporal Aggregation
                                         │       ↓
KIR Video ──→ Backbone₂ ──→ Features₂ ──┘   Fused Features ──→ Head
```

**Components**:

1. **Motion Excitation**:
   - Computes frame-to-frame differences
   - Extracts temporal motion patterns
   - 3D convolution with sigmoid gating

2. **Multi-View Excitation**:
   - Cross-modal attention mechanism
   - Adaptive weighting of modalities
   - Channel-wise excitation

3. **Temporal Aggregation**:
   - Hierarchical temporal modeling
   - Grouped 1D convolutions
   - Multi-scale temporal receptive fields

### 4. Training Pipeline

#### Optimizer: AdamW
```python
optimizer = AdamW(
    params=model.parameters(),
    lr=2e-4,
    weight_decay=0.05,
    betas=(0.9, 0.999)
)
```

#### Learning Rate Schedule: Warmup + Cosine
```
LR │     ╱────────╲
   │    ╱          ╲
   │   ╱            ╲___
   │  ╱                 ╲___
   │ ╱                      ╲___
   │╱___________________________╲___
   └─────────────────────────────────→ Epoch
     │←Warmup→│←─ Cosine Decay ─→│
```

#### Loss Function
- CrossEntropyLoss with optional label smoothing
- Standard classification objective
- No auxiliary losses (clean training)

#### Regularization
- DropPath (stochastic depth): 0.1
- Dropout in classification head: 0.5
- Weight decay: 0.05
- Gradient clipping: max_norm=20.0

### 5. Evaluation Metrics

#### Top-K Accuracy
```python
Top-1 Accuracy = (# correct predictions) / (# total samples)
Top-5 Accuracy = (# samples with correct label in top-5) / (# total samples)
```

#### Mean Class Accuracy
```python
Per-class Accuracy = (TP for class i) / (Total samples of class i)
Mean Class Accuracy = Average of all per-class accuracies
```

#### Confusion Matrix
- NxN matrix (N=34 classes)
- Visualized as heatmap
- Identifies common misclassifications

## Data Flow

### Training Loop

```
1. Load batch from DataLoader
   ├─ Video decoding (decord/torchvision)
   ├─ Frame sampling (random)
   ├─ Augmentation (random crop, flip, color jitter)
   └─ Normalization

2. Forward pass
   ├─ Model(inputs) → logits
   └─ criterion(logits, labels) → loss

3. Backward pass (with AMP)
   ├─ scaler.scale(loss).backward()
   ├─ scaler.unscale_(optimizer)
   ├─ clip_grad_norm_(parameters, max_norm=20)
   └─ scaler.step(optimizer)

4. Update metrics
   ├─ Loss averaging
   ├─ Accuracy computation
   └─ TensorBoard logging

5. Scheduler step
   └─ lr = cosine_schedule(epoch)
```

### Inference Flow

```
1. Load video
   └─ VideoDecoder.decode_video(path)

2. Preprocess
   ├─ Sample frames (uniform)
   ├─ Resize & crop
   └─ Normalize

3. Model inference
   ├─ model.eval()
   ├─ with torch.no_grad():
   └─   logits = model(video)

4. Post-process
   ├─ probs = softmax(logits)
   ├─ top5 = topk(probs, 5)
   └─ Format output (JSON/CSV)

5. Optional visualization
   └─ Overlay predictions on video
```

## Model Variants

### Size Configurations

| Variant | Depth | Embed Dim | Heads | Params | FLOPs |
|---------|-------|-----------|-------|--------|-------|
| Small   | [3,4,8,3] | [64,128,320,512] | [2,4,10,16] | ~30M | ~50G |
| Base    | [5,8,20,7] | [64,128,320,512] | [2,4,10,16] | ~50M | ~80G |
| Large   | [5,10,25,10] | [96,192,480,768] | [3,6,15,24] | ~90M | ~140G |

### Fusion Configurations

| Fusion | Backbones | Heads | Total Params |
|--------|-----------|-------|--------------|
| None (Single) | 1 | 1 | ~50M |
| Early | 1 | 1 | ~50M |
| Late | 2 | 2 | ~100M |
| META | 2 | 1 | ~105M |

## Performance Optimization

### Memory Optimization
1. **AMP (Automatic Mixed Precision)**: Reduces memory by ~40%
2. **Gradient Checkpointing**: Trade compute for memory
3. **Gradient Accumulation**: Simulate larger batch sizes

### Speed Optimization
1. **CuDNN Benchmark**: Auto-tune convolution algorithms
2. **Pin Memory**: Faster CPU→GPU transfer
3. **DataLoader Workers**: Parallel data loading

### Distributed Training
```
# Single Node, Multiple GPUs
torchrun --nproc_per_node=4 engine/train.py --config config.yaml

# Multiple Nodes
torchrun --nnodes=2 --nproc_per_node=4 \
         --master_addr=... --master_port=... \
         engine/train.py --config config.yaml
```

## Configuration System

### Hierarchy
```
default.yaml (base config)
    ↓
fusion_*.yaml (fusion-specific overrides)
    ↓
modality_*.yaml (modality-specific overrides)
    ↓
Command-line arguments (runtime overrides)
```

### Config Sections
- **data**: Dataset paths, preprocessing, augmentation
- **model**: Architecture, fusion strategy, hyperparameters
- **train**: Optimizer, scheduler, training settings
- **test**: Evaluation settings
- **save**: Output directories

## Extension Points

### Adding New Fusion Strategy
1. Create `models/fusion/new_fusion.py`
2. Implement `NewFusion` and `NewFusionModel` classes
3. Register in `models/fusion/factory.py`
4. Create config `configs/fusion_new.yaml`

### Adding New Backbone
1. Create `models/new_backbone/`
2. Implement backbone architecture
3. Update `models/builder.py`
4. Test forward pass

### Custom Dataset
1. Subclass `VideoDataset` in `data/datasets/`
2. Override `__getitem__` for custom loading
3. Update `build_dataloader` if needed

## Debugging Tips

### Common Issues
1. **NaN Loss**: Reduce learning rate, check data normalization
2. **OOM**: Reduce batch size, enable gradient accumulation
3. **Slow Training**: Increase num_workers, use SSD for data

### Profiling
```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(inputs)

print(prof.key_averages().table())
```

## References

- UniFormerV2 Paper: [Link](https://arxiv.org/abs/2211.09552)
- PyTorch Documentation: [Link](https://pytorch.org/docs/)
- Video Understanding: Best practices from TimeSformer, VideoMAE, etc.
