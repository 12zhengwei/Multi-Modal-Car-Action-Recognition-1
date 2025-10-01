"""
Early Fusion strategy for multi-modal video understanding.

Fuses modalities at input level or early feature level.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class EarlyFusion(nn.Module):
    """
    Early Fusion module.
    
    Fuses multiple modalities at input or early feature level through:
    1. Channel concatenation
    2. Optional 1x1 convolution bottleneck for dimension reduction
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_modalities: int = 2,
        bottleneck_dim: Optional[int] = None,
        use_bottleneck: bool = True,
    ):
        """
        Initialize Early Fusion module.
        
        Args:
            in_channels: Number of channels per modality (typically 3 for RGB/KIR)
            num_modalities: Number of modalities to fuse (typically 2 for RGB+KIR)
            bottleneck_dim: Dimension of bottleneck layer (if None, use in_channels)
            use_bottleneck: Whether to use 1x1 conv bottleneck
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_modalities = num_modalities
        self.fused_channels = in_channels * num_modalities
        
        # Bottleneck for dimension reduction
        if use_bottleneck:
            if bottleneck_dim is None:
                bottleneck_dim = in_channels  # Reduce back to single modality size
            
            self.bottleneck = nn.Sequential(
                nn.Conv3d(self.fused_channels, bottleneck_dim, kernel_size=1),
                nn.BatchNorm3d(bottleneck_dim),
                nn.ReLU(inplace=True),
            )
            self.out_channels = bottleneck_dim
        else:
            self.bottleneck = nn.Identity()
            self.out_channels = self.fused_channels
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for early fusion.
        
        Args:
            inputs: Dictionary containing modality tensors
                - 'rgb': RGB video tensor (B, C, T, H, W)
                - 'kir': KIR video tensor (B, C, T, H, W)
        
        Returns:
            Fused tensor (B, out_channels, T, H, W)
        """
        # Collect modality tensors
        modalities = []
        for key in ['rgb', 'kir']:
            if key in inputs and inputs[key] is not None:
                modalities.append(inputs[key])
        
        if len(modalities) == 0:
            raise ValueError("No modalities provided for fusion")
        
        # Concatenate along channel dimension
        fused = torch.cat(modalities, dim=1)  # (B, C*num_modalities, T, H, W)
        
        # Apply bottleneck if configured
        fused = self.bottleneck(fused)
        
        return fused


class EarlyFusionModel(nn.Module):
    """
    Complete model with Early Fusion.
    
    Combines early fusion with a shared backbone.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        in_channels: int = 3,
        num_modalities: int = 2,
        bottleneck_dim: Optional[int] = None,
        use_bottleneck: bool = True,
    ):
        """
        Initialize Early Fusion model.
        
        Args:
            backbone: Video backbone network (e.g., UniFormerV2)
            head: Classification head
            in_channels: Number of channels per modality
            num_modalities: Number of modalities to fuse
            bottleneck_dim: Dimension of bottleneck layer
            use_bottleneck: Whether to use bottleneck
        """
        super().__init__()
        
        self.fusion = EarlyFusion(
            in_channels=in_channels,
            num_modalities=num_modalities,
            bottleneck_dim=bottleneck_dim,
            use_bottleneck=use_bottleneck,
        )
        
        # Update backbone input channels if needed
        if hasattr(backbone, 'patch_embed') and hasattr(backbone.patch_embed, 'proj'):
            original_conv = backbone.patch_embed.proj
            if original_conv.in_channels != self.fusion.out_channels:
                # Replace first conv layer to match fused channels
                new_conv = nn.Conv3d(
                    self.fusion.out_channels,
                    original_conv.out_channels,
                    kernel_size=original_conv.kernel_size,
                    stride=original_conv.stride,
                    padding=original_conv.padding,
                    bias=original_conv.bias is not None,
                )
                # Initialize with average of original weights
                with torch.no_grad():
                    if use_bottleneck and bottleneck_dim == in_channels:
                        # If bottleneck reduces to original size, use original weights
                        new_conv.weight.copy_(original_conv.weight)
                    else:
                        # Initialize new weights
                        nn.init.kaiming_normal_(new_conv.weight, mode='fan_out')
                    if new_conv.bias is not None:
                        nn.init.constant_(new_conv.bias, 0)
                
                backbone.patch_embed.proj = new_conv
        
        self.backbone = backbone
        self.head = head
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Dictionary containing modality tensors
        
        Returns:
            Class logits (B, num_classes)
        """
        # Fuse modalities
        x = self.fusion(inputs)
        
        # Forward through backbone
        x = self.backbone(x)
        
        # Forward through head
        logits = self.head(x)
        
        return logits
