"""
UniFormerV2 backbone for video understanding.

A hierarchical video transformer with local and global attention.
"""

import torch
import torch.nn as nn
from typing import List, Optional
import numpy as np

from .blocks import LocalBlock, GlobalBlock, DropPath


class PatchEmbed3D(nn.Module):
    """
    3D patch embedding for video.
    
    Converts video (B, C, T, H, W) to patch tokens.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 64,
        temporal_patch_size: int = 2,
        spatial_patch_size: int = 4,
    ):
        super().__init__()
        self.temporal_patch_size = temporal_patch_size
        self.spatial_patch_size = spatial_patch_size
        
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(temporal_patch_size, spatial_patch_size, spatial_patch_size),
            stride=(temporal_patch_size, spatial_patch_size, spatial_patch_size),
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input video (B, C, T, H, W)
        
        Returns:
            Patch embeddings (B, embed_dim, T', H', W')
        """
        x = self.proj(x)  # (B, embed_dim, T', H', W')
        
        # Apply layer norm
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, T'*H'*W', C)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, T, H, W)
        
        return x


class Downsample3D(nn.Module):
    """
    3D downsampling layer for hierarchical architecture.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        temporal_stride: int = 2,
        spatial_stride: int = 2,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_dim,
            out_dim,
            kernel_size=(temporal_stride, spatial_stride, spatial_stride),
            stride=(temporal_stride, spatial_stride, spatial_stride),
        )
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, T, H, W)
        
        Returns:
            Downsampled tensor (B, out_dim, T', H', W')
        """
        x = self.conv(x)
        
        # Apply layer norm
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, T, H, W)
        
        return x


class UniFormerV2(nn.Module):
    """
    UniFormerV2 backbone for video understanding.
    
    A hierarchical video transformer combining local and global attention.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        depth: List[int] = [5, 8, 20, 7],
        embed_dim: List[int] = [64, 128, 320, 512],
        num_heads: List[int] = [2, 4, 10, 16],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        use_checkpoint: bool = False,
        checkpoint_num: List[int] = [0, 0, 0, 0],
        temporal_downsample: List[int] = [2, 2, 2, 1],  # Temporal stride for each stage
        spatial_downsample: List[int] = [4, 2, 2, 2],  # Spatial stride for each stage
    ):
        super().__init__()
        
        self.num_stages = len(depth)
        self.depth = depth
        self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        
        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            in_channels=in_channels,
            embed_dim=embed_dim[0],
            temporal_patch_size=temporal_downsample[0],
            spatial_patch_size=spatial_downsample[0],
        )
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        
        # Build stages
        self.stages = nn.ModuleList()
        cur_depth = 0
        
        for i in range(self.num_stages):
            # Downsampling between stages (except first stage)
            if i > 0:
                downsample = Downsample3D(
                    embed_dim[i - 1],
                    embed_dim[i],
                    temporal_stride=temporal_downsample[i],
                    spatial_stride=spatial_downsample[i],
                )
            else:
                downsample = None
            
            # Build blocks for this stage
            blocks = []
            for j in range(depth[i]):
                # Use local blocks for early stages, global for later stages
                use_global = (i >= self.num_stages - 1)  # Only last stage uses global
                
                if use_global:
                    block = GlobalBlock(
                        dim=embed_dim[i],
                        num_heads=num_heads[i],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[cur_depth + j],
                        norm_layer=norm_layer,
                    )
                else:
                    block = LocalBlock(
                        dim=embed_dim[i],
                        num_heads=num_heads[i],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[cur_depth + j],
                        norm_layer=norm_layer,
                    )
                
                blocks.append(block)
            
            cur_depth += depth[i]
            
            stage = nn.ModuleDict({
                'downsample': downsample,
                'blocks': nn.ModuleList(blocks),
            })
            self.stages.append(stage)
        
        # Final norm
        self.norm = norm_layer(embed_dim[-1])
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature extractor.
        
        Args:
            x: Input video (B, C, T, H, W)
        
        Returns:
            Features (B, embed_dim[-1], T', H', W')
        """
        # Patch embedding
        x = self.patch_embed(x)
        
        # Forward through stages
        for stage in self.stages:
            # Downsample
            if stage['downsample'] is not None:
                x = stage['downsample'](x)
            
            # Forward through blocks
            for block in stage['blocks']:
                x = block(x)
        
        # Final norm
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, T*H*W, C)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, T, H, W)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input video (B, C, T, H, W)
        
        Returns:
            Features (B, embed_dim[-1], T', H', W')
        """
        return self.forward_features(x)


def uniformerv2_b16(pretrained: bool = False, **kwargs) -> UniFormerV2:
    """
    UniFormerV2-B16 model.
    
    Base model with 16-frame input.
    """
    model = UniFormerV2(
        depth=[5, 8, 20, 7],
        embed_dim=[64, 128, 320, 512],
        num_heads=[2, 4, 10, 16],
        mlp_ratio=4.0,
        qkv_bias=True,
        **kwargs
    )
    return model


def uniformerv2_s16(pretrained: bool = False, **kwargs) -> UniFormerV2:
    """
    UniFormerV2-S16 model.
    
    Small model with 16-frame input.
    """
    model = UniFormerV2(
        depth=[3, 4, 8, 3],
        embed_dim=[64, 128, 320, 512],
        num_heads=[2, 4, 10, 16],
        mlp_ratio=4.0,
        qkv_bias=True,
        **kwargs
    )
    return model


def uniformerv2_l16(pretrained: bool = False, **kwargs) -> UniFormerV2:
    """
    UniFormerV2-L16 model.
    
    Large model with 16-frame input.
    """
    model = UniFormerV2(
        depth=[5, 10, 25, 10],
        embed_dim=[96, 192, 480, 768],
        num_heads=[3, 6, 15, 24],
        mlp_ratio=4.0,
        qkv_bias=True,
        **kwargs
    )
    return model
