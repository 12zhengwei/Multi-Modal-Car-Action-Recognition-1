"""
Model builder.

Constructs complete models from configuration.
"""

import torch.nn as nn
from typing import Dict, Optional

from .uniformerv2 import uniformerv2_b16, uniformerv2_s16, uniformerv2_l16
from .heads import build_classification_head
from .fusion import create_fusion_model


def build_backbone(config: Dict) -> nn.Module:
    """
    Build backbone network from configuration.
    
    Args:
        config: Model configuration
    
    Returns:
        Backbone network
    """
    backbone_type = config.get('backbone', 'uniformerv2')
    
    if backbone_type == 'uniformerv2':
        # Get UniFormerV2 configuration
        uniformer_config = config.get('uniformerv2', {})
        
        # Determine model size
        depth = uniformer_config.get('depth', [5, 8, 20, 7])
        
        if depth == [3, 4, 8, 3]:
            model = uniformerv2_s16(
                drop_rate=uniformer_config.get('drop_rate', 0.0),
                attn_drop_rate=uniformer_config.get('attn_drop_rate', 0.0),
                drop_path_rate=uniformer_config.get('drop_path_rate', 0.1),
            )
        elif depth == [5, 8, 20, 7]:
            model = uniformerv2_b16(
                drop_rate=uniformer_config.get('drop_rate', 0.0),
                attn_drop_rate=uniformer_config.get('attn_drop_rate', 0.0),
                drop_path_rate=uniformer_config.get('drop_path_rate', 0.1),
            )
        elif depth == [5, 10, 25, 10]:
            model = uniformerv2_l16(
                drop_rate=uniformer_config.get('drop_rate', 0.0),
                attn_drop_rate=uniformer_config.get('attn_drop_rate', 0.0),
                drop_path_rate=uniformer_config.get('drop_path_rate', 0.1),
            )
        else:
            # Custom configuration
            from .uniformerv2 import UniFormerV2
            model = UniFormerV2(
                depth=depth,
                embed_dim=uniformer_config.get('embed_dim', [64, 128, 320, 512]),
                **uniformer_config
            )
        
        return model
    
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")


def build_model(config: Dict) -> nn.Module:
    """
    Build complete model from configuration.
    
    Args:
        config: Complete configuration dictionary
    
    Returns:
        Complete model (backbone + fusion + head)
    """
    model_config = config['model']
    data_config = config['data']
    
    modality = data_config.get('modality', 'rgb')
    fusion_type = model_config.get('fusion', 'none')
    num_classes = model_config.get('num_classes', 34)
    
    # Build backbones for each required modality
    backbones = {}
    if modality == 'rgb':
        backbones['rgb'] = build_backbone(model_config)
    elif modality == 'kir':
        backbones['kir'] = build_backbone(model_config)
    elif modality == 'rgb_kir':
        # Build separate backbones for each modality
        backbones['rgb'] = build_backbone(model_config)
        backbones['kir'] = build_backbone(model_config)
    else:
        raise ValueError(f"Unknown modality: {modality}")
    
    # Get backbone output dimension
    backbone = list(backbones.values())[0]
    if hasattr(backbone, 'embed_dim'):
        if isinstance(backbone.embed_dim, list):
            embed_dim = backbone.embed_dim[-1]
        else:
            embed_dim = backbone.embed_dim
    else:
        embed_dim = 512  # Default
    
    # Build classification heads
    heads = {}
    head_config = model_config.get('head', {})
    
    for modality_name in backbones.keys():
        heads[modality_name] = build_classification_head(
            in_channels=embed_dim,
            num_classes=num_classes,
            config=head_config,
        )
    
    # Build fusion model
    fusion_config = model_config.get('fusion_config', {})
    model = create_fusion_model(
        fusion_type=fusion_type,
        backbones=backbones,
        heads=heads,
        config=fusion_config,
    )
    
    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params,
    }


def print_model_info(model: nn.Module):
    """
    Print model information.
    
    Args:
        model: PyTorch model
    """
    params = count_parameters(model)
    
    print("=" * 60)
    print("Model Information")
    print("=" * 60)
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Non-trainable parameters: {params['non_trainable']:,}")
    print(f"Model size (MB): {params['total'] * 4 / 1024 / 1024:.2f}")
    print("=" * 60)
