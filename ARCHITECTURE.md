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

**META Fusion 详细介绍 (Detailed Introduction)**:

META Fusion是一种先进的多模态融合策略，专门设计用于处理RGB和KIR（近红外）视频模态。该方法通过三个核心模块实现高效的特征融合：

**1. 运动激励模块 (Motion Excitation Module)**
- **原理**：通过计算相邻帧之间的差分来捕获时序运动模式
- **实现**：
  - 计算帧间差分：`diff = frame(t+1) - frame(t)`
  - 使用3D卷积提取运动特征，包含时间维度和空间维度
  - 通过Sigmoid门控机制生成运动权重
  - 将运动权重应用于原始特征：`enhanced = features × motion_weights`
- **作用**：强化动作相关的时序信息，抑制静态背景

**2. 多视角激励模块 (Multi-View Excitation Module)**
- **原理**：利用跨模态注意力机制自适应地融合不同模态的信息
- **实现**：
  - 全局平均池化获取模态间的全局上下文
  - 通过1×1×1卷积计算跨模态注意力权重
  - Softmax归一化确保模态权重和为1
  - 为每个模态生成特定的门控信号
  - 结合注意力权重和门控信号调节特征：`modality_out = modality × gate × attention`
- **作用**：
  - RGB模态捕获丰富的外观和颜色信息
  - KIR模态在低光照条件下提供稳定的热信息
  - 自适应权重根据场景条件动态调整两种模态的贡献

**3. 时序聚合模块 (Temporal Aggregation Module)**
- **原理**：通过分组卷积实现多尺度时序建模
- **实现**：
  - 使用多个并行的深度可分离3D卷积（仅在时间维度）
  - 每组捕获不同尺度的时序依赖关系
  - 将所有组的输出拼接后通过1×1×1卷积聚合
  - 添加残差连接保留原始时序信息
- **作用**：
  - 建立长短期时序依赖
  - 捕获不同时间尺度的动作模式
  - 增强模型对动作持续时间变化的鲁棒性

**整体流程**：
1. RGB和KIR视频分别通过独立的骨干网络提取特征
2. 每个模态的特征经过运动激励增强时序信息
3. 多视角激励模块进行跨模态信息交互和自适应加权
4. 拼接增强后的多模态特征
5. 时序聚合模块建模时序依赖关系
6. 最终通过1×1×1卷积融合为统一特征表示
7. 共享分类头输出动作类别

**优势**：
- 充分利用RGB和KIR的互补特性
- 运动激励强化动作相关的时序特征
- 跨模态注意力实现自适应融合
- 多尺度时序建模增强时序推理能力
- 端到端训练，融合策略可学习优化

#### CMCF Fusion (`models/fusion/cmcf.py`)

```
RGB Video ──→ Backbone₁ ──→ Features₁ ──┐
                                         │
                                         ├─→ Modality-Specific Enhancement
                                         │       ↓
                                         ├─→ Complementary Attention
                                         │       ↓
                                         ├─→ Adaptive Weighting
                                         │       ↓
KIR Video ──→ Backbone₂ ──→ Features₂ ──┘   Fused Features ──→ Head
```

**Components**:

1. **Modality-Specific Enhancement**:
   - RGB-specific: Color and texture enhancement
   - KIR-specific: Thermal pattern enhancement
   - Channel attention for each modality

2. **Complementary Attention**:
   - Cross-modal query-key-value attention
   - RGB features query KIR features and vice versa
   - Captures complementary information

3. **Adaptive Weighting**:
   - Reliability estimation for each modality
   - Dynamic fusion weights based on scene conditions
   - Combines enhanced features with learned weights

**CMCF Fusion 详细介绍 (Detailed Introduction)**:

CMCF（Cross-Modal Complementary Fusion，跨模态互补融合）是一种专门为RGB和KIR视频设计的融合策略，充分利用两种模态的互补特性。

**设计理念**：
- RGB模态：提供丰富的颜色、纹理和外观信息，在光照良好时表现优异
- KIR模态：提供热辐射和低光照信息，在暗光或逆光条件下稳定可靠
- 互补融合：动态平衡两种模态的贡献，根据场景条件自适应调整

**核心模块**：

**1. 模态特定增强 (Modality-Specific Enhancement)**
- 为RGB和KIR分别设计专门的增强网络
- RGB增强：强化颜色对比度和纹理细节
- KIR增强：强化热模式和边缘信息
- 使用通道注意力机制选择性增强有用特征

**2. 互补注意力 (Complementary Attention)**
- RGB特征查询KIR特征，获取热信息补充
- KIR特征查询RGB特征，获取外观细节
- 双向交互确保信息充分交换
- 捕获模态间的互补关系

**3. 自适应加权 (Adaptive Weighting)**
- 评估每个模态在当前场景下的可靠性
- 基于特征统计量（均值、方差）估计质量
- 动态生成融合权重
- 在光照良好时侧重RGB，暗光条件下侧重KIR

**适用场景**：
- 车载环境中光照条件变化大
- 需要在不同时段（白天/夜晚）保持稳定性能
- RGB和KIR具有明显的互补特性

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
| CMCF | 2 | 1 | ~104M |

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
