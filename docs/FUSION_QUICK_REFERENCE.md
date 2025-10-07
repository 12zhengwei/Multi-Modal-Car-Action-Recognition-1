# Fusion Methods Quick Reference 融合方法快速参考

## When to Use Which Fusion? 何时使用哪种融合方法？

### 🎯 Decision Tree 决策树

```
Start: Need to fuse RGB + KIR modalities?
│
├─ Limited computational resources? → Early Fusion
│   Budget constrained, ~50M params
│
├─ Modalities have very different quality? → Late Fusion  
│   Independent optimization, ~100M params
│
├─ Complex temporal action patterns? → META Fusion
│   Motion analysis crucial, ~105M params
│
└─ Varying lighting conditions? → CMCF Fusion ⭐
    Adaptive to scene conditions, ~104M params
```

## Quick Comparison 快速对比

| Aspect 方面 | Early | Late | META | CMCF ⭐ |
|------------|-------|------|------|--------|
| **Fusion Stage 融合阶段** | Input 输入级 | Decision 决策级 | Feature 特征级 | Feature 特征级 |
| **Parameters 参数量** | ~50M | ~100M | ~105M | ~104M |
| **Speed 速度** | ⚡⚡⚡ Fast | ⚡⚡ Medium | ⚡⚡ Medium | ⚡⚡ Medium |
| **Accuracy 准确率** | ⭐⭐⭐ Good | ⭐⭐⭐⭐ Better | ⭐⭐⭐⭐⭐ Best | ⭐⭐⭐⭐⭐ Best |
| **RGB-KIR Complementarity RGB-KIR互补性** | ⭐ Low | ⭐⭐ Medium | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ **Excellent** |
| **Lighting Robustness 光照鲁棒性** | ⭐⭐ Medium | ⭐⭐⭐ Good | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ **Excellent** |
| **Temporal Modeling 时序建模** | ⭐⭐⭐ Good | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ **Excellent** | ⭐⭐⭐ Good |

## Configuration Examples 配置示例

### Early Fusion

```yaml
model:
  fusion: early
  fusion_config:
    early:
      bottleneck_dim: 192
      use_bottleneck: true
```

**Use when 适用于**: Limited budget, small dataset

### Late Fusion

```yaml
model:
  fusion: late
  fusion_config:
    late:
      fusion_method: logits  # or 'probs', 'weighted'
      learnable_weights: true
```

**Use when 适用于**: Modalities trained separately, different quality

### META Fusion

```yaml
model:
  fusion: meta
  fusion_config:
    meta:
      motion_kernel_size: 3
      temporal_groups: 4
      excitation_reduction: 4
      use_motion_excitation: true
      use_multiview_excitation: true
      use_temporal_aggregation: true
```

**Use when 适用于**: Complex temporal patterns, motion is key

### CMCF Fusion ⭐ (NEW)

```yaml
model:
  fusion: cmcf
  fusion_config:
    cmcf:
      enhancement_reduction: 8
      attention_heads: 4
```

**Use when 适用于**: Variable lighting, need adaptive RGB-KIR balance

## Performance Guidelines 性能指南

### Dataset Size 数据集大小

| Dataset Size | Recommended Fusion |
|--------------|-------------------|
| < 5K videos | Early Fusion |
| 5K - 20K videos | Late Fusion or CMCF |
| > 20K videos | META or CMCF |

### Use Case Scenarios 使用场景

#### 🌞 Good Lighting (Daytime) 光照良好（白天）
- Early Fusion: ⭐⭐⭐⭐
- Late Fusion: ⭐⭐⭐⭐
- META Fusion: ⭐⭐⭐⭐⭐
- **CMCF Fusion: ⭐⭐⭐⭐⭐** (Auto-weights RGB higher)

#### 🌙 Low Light (Nighttime) 低光照（夜间）
- Early Fusion: ⭐⭐
- Late Fusion: ⭐⭐⭐
- META Fusion: ⭐⭐⭐⭐
- **CMCF Fusion: ⭐⭐⭐⭐⭐** (Auto-weights KIR higher)

#### 🌗 Mixed Lighting (Dawn/Dusk) 混合光照（黄昏/黎明）
- Early Fusion: ⭐⭐
- Late Fusion: ⭐⭐⭐
- META Fusion: ⭐⭐⭐⭐
- **CMCF Fusion: ⭐⭐⭐⭐⭐** (Adaptive weighting)

#### ⚡ Fast Actions 快速动作
- Early Fusion: ⭐⭐⭐
- Late Fusion: ⭐⭐⭐
- **META Fusion: ⭐⭐⭐⭐⭐** (Motion excitation)
- CMCF Fusion: ⭐⭐⭐⭐

#### 🎬 Slow/Subtle Actions 缓慢/细微动作
- Early Fusion: ⭐⭐⭐
- Late Fusion: ⭐⭐⭐⭐
- META Fusion: ⭐⭐⭐⭐
- **CMCF Fusion: ⭐⭐⭐⭐⭐** (Feature enhancement)

## Training Tips 训练技巧

### For CMCF Fusion 针对CMCF融合

```python
# Recommended hyperparameters
learning_rate = 2e-4
batch_size = 8  # per GPU
num_epochs = 100
warmup_epochs = 10

# If unstable training
learning_rate = 1e-4  # Lower LR
attention_heads = 2   # Reduce complexity

# If limited memory
enhancement_reduction = 16  # More compression
use_amp = True              # Mixed precision
```

### For META Fusion 针对META融合

```python
# Recommended hyperparameters
learning_rate = 2e-4
batch_size = 6  # Slightly smaller due to complexity
temporal_groups = 4

# For fast actions
motion_kernel_size = 5     # Larger kernel
use_motion_excitation = True

# For slow actions
temporal_groups = 8        # More temporal scales
use_temporal_aggregation = True
```

## Common Issues 常见问题

### Q1: Which fusion for production deployment? 生产部署推荐？

**A**: 
- **CMCF Fusion** - Best balance of accuracy and adaptability
- **Late Fusion** - If need to update modalities independently
- **Early Fusion** - If extremely resource constrained

### Q2: Can I mix fusion methods? 可以混合使用吗？

**A**: No, choose one fusion strategy per model. But you can ensemble different fusion models.

### Q3: CMCF vs META, which is better? CMCF和META哪个更好？

**A**: 
- **CMCF**: Better for lighting variation, simpler, faster
- **META**: Better for complex temporal patterns, motion analysis
- **Tip**: Try both and evaluate on your specific dataset

### Q4: Training time comparison? 训练时间对比？

Relative training time (batch_size=8, 16 frames):
- Early Fusion: 1.0x (baseline)
- Late Fusion: 1.8x
- META Fusion: 2.2x
- CMCF Fusion: 2.0x

## Commands Cheatsheet 命令速查

```bash
# Train with CMCF
python scripts/train.py --config configs/fusion_cmcf.yaml

# Train with META
python scripts/train.py --config configs/fusion_meta.yaml

# Evaluate
python scripts/eval.py --config configs/fusion_cmcf.yaml \
    --checkpoint outputs/split0_cmcf_rgbkir/best_model.pth

# Export to ONNX
python scripts/export.py --config configs/fusion_cmcf.yaml \
    --checkpoint outputs/split0_cmcf_rgbkir/best_model.pth \
    --format onnx --output models/cmcf.onnx

# Inference
python scripts/infer.py --config configs/fusion_cmcf.yaml \
    --checkpoint outputs/split0_cmcf_rgbkir/best_model.pth \
    --video_rgb path/to/rgb.mp4 --video_kir path/to/kir.mp4
```

## Further Reading 延伸阅读

- [CMCF Fusion Guide](./CMCF_FUSION_GUIDE.md) - Comprehensive CMCF documentation
- [ARCHITECTURE.md](../ARCHITECTURE.md) - Full architecture details
- [QUICKSTART.md](../QUICKSTART.md) - Getting started guide

---

**Recommendation 推荐**: For vehicle action recognition with varying lighting conditions, start with **CMCF Fusion**. It provides the best balance of accuracy, speed, and adaptability to lighting changes. 对于光照条件变化的车载动作识别，推荐使用**CMCF融合**，它在准确率、速度和光照适应性之间达到最佳平衡。
