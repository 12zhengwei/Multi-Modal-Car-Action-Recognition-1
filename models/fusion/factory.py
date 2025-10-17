"""
Fusion strategy factory.

Creates fusion modules based on configuration.
"""

from typing import Dict, Optional
import torch.nn as nn

from .early import EarlyFusionModel
from .late import LateFusionModel
from .meta import METAFusionModel
from .cmcf import CMCFFusionModel


def create_fusion_model(
    fusion_type: str,
    backbones: Dict[str, nn.Module],
    heads: Dict[str, nn.Module],
    config: Dict,
) -> nn.Module:
    """
    Create fusion model based on configuration.
    
    Args:
        fusion_type: Type of fusion ('early', 'late', 'meta', or 'none')
        backbones: Dictionary of backbone networks for each modality
        heads: Dictionary of classification heads for each modality
        config: Fusion configuration dictionary
    
    Returns:
        Fusion model instance
    """
    if fusion_type == 'none':
        # Single modality - no fusion needed
        # Return the first available backbone + head
        modality = list(backbones.keys())[0]
        return SingleModalityModel(backbones[modality], heads[modality])
    
    elif fusion_type == 'early':
        # Early fusion - single backbone and head
        # Use first backbone and head (will be adapted for fused input)
        backbone = list(backbones.values())[0]
        head = list(heads.values())[0]
        
        fusion_config = config.get('early', {})
        model = EarlyFusionModel(
            backbone=backbone,
            head=head,
            in_channels=3,
            num_modalities=len(backbones),
            bottleneck_dim=fusion_config.get('bottleneck_dim', 192),
            use_bottleneck=True,
        )
        return model
    
    elif fusion_type == 'late':
        # Late fusion - separate backbones and heads
        fusion_config = config.get('late', {})
        model = LateFusionModel(
            backbones=backbones,
            heads=heads,
            fusion_method=fusion_config.get('fusion_method', 'logits'),
            learnable_weights=fusion_config.get('learnable_weights', True),
        )
        return model
    
    elif fusion_type == 'meta':
        # META fusion - separate backbones, shared head
        head = list(heads.values())[0]
        fusion_config = config.get('meta', {})
        
        # Get fusion dimension from backbone output
        fusion_dim = get_backbone_output_dim(list(backbones.values())[0])
        
        model = METAFusionModel(
            backbones=backbones,
            head=head,
            fusion_dim=fusion_dim,
            motion_kernel_size=fusion_config.get('motion_kernel_size', 3),
            temporal_groups=fusion_config.get('temporal_groups', 4),
            excitation_reduction=fusion_config.get('excitation_reduction', 4),
            use_motion_excitation=fusion_config.get('use_motion_excitation', True),
            use_multiview_excitation=fusion_config.get('use_multiview_excitation', True),
            use_temporal_aggregation=fusion_config.get('use_temporal_aggregation', True),
        )
        return model
    
    elif fusion_type == 'cmcf':
        # CMCF fusion - separate backbones, shared head
        head = list(heads.values())[0]
        fusion_config = config.get('cmcf', {})
        
        # Get fusion dimension from backbone output
        fusion_dim = get_backbone_output_dim(list(backbones.values())[0])
        
        model = CMCFFusionModel(
            backbones=backbones,
            head=head,
            fusion_dim=fusion_dim,
            enhancement_hidden_channels=fusion_config.get('enhancement_hidden_channels', None),
            attention_reduction=fusion_config.get('attention_reduction', 4),
            weighting_hidden_dim=fusion_config.get('weighting_hidden_dim', 64),
            dropout=fusion_config.get('dropout', 0.1),
        )
        return model
    
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")


def get_backbone_output_dim(backbone: nn.Module) -> int:
    """
    Get output dimension of backbone network.
    
    Args:
        backbone: Backbone network
    
    Returns:
        Output channel dimension
    """
    if hasattr(backbone, 'embed_dim'):
        # UniFormerV2 case
        if isinstance(backbone.embed_dim, list):
            return backbone.embed_dim[-1]
        return backbone.embed_dim
    elif hasattr(backbone, 'num_features'):
        return backbone.num_features
    else:
        # Default guess
        return 512


class SingleModalityModel(nn.Module):
    """
    Simple wrapper for single modality model.
    
    No fusion needed.
    """
    
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, T, H, W) or dict with single modality
        
        Returns:
            Class logits (B, num_classes)
        """
        # Handle dict input (from multi-modal dataloader)
        if isinstance(x, dict):
            # Get first available modality
            for key in ['rgb', 'kir']:
                if key in x and x[key] is not None:
                    x = x[key]
                    break
        
        features = self.backbone(x)
        logits = self.head(features)
        return logits
