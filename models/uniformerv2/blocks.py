"""
UniFormerV2 building blocks.

Implements the core components of UniFormerV2:
- Local MHRA (Multi-Head Relation Aggregator)
- Global MHRA with cross-attention
- MLP blocks
- DropPath for stochastic depth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    
    Used for regularization in vision transformers.
    """
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class Mlp(nn.Module):
    """
    MLP (Multi-Layer Perceptron) block.
    
    Two linear layers with GELU activation and dropout.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CMlp(nn.Module):
    """
    Convolutional MLP for local feature processing.
    
    Uses 1x1 convolutions instead of linear layers.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LocalMHRA(nn.Module):
    """
    Local Multi-Head Relation Aggregator.
    
    Performs local attention within spatial-temporal neighborhoods.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        # Convolutional QKV projection
        padding = kernel_size // 2
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size, padding=padding, groups=dim)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, T, H, W)
        
        Returns:
            Output tensor (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape
        
        # Generate QKV using 3D convolution
        qkv = self.qkv(x)  # (B, 3*C, T, H, W)
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, T * H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, num_heads, T*H*W, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, T * H * W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Reshape back
        x = x.transpose(1, 2).reshape(B, C, T, H, W)
        return x


class GlobalMHRA(nn.Module):
    """
    Global Multi-Head Relation Aggregator.
    
    Performs global attention across entire spatial-temporal space.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, T, H, W)
        
        Returns:
            Output tensor (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape
        N = T * H * W
        
        # Reshape to sequence format
        x = x.flatten(2).transpose(1, 2)  # (B, T*H*W, C)
        
        # Generate QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Reshape back
        x = x.transpose(1, 2).reshape(B, C, T, H, W)
        return x


class LocalBlock(nn.Module):
    """
    Local UniFormer block with local MHRA.
    
    Combines local attention with feedforward network.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = LocalMHRA(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, T, H, W)
        
        Returns:
            Output tensor (B, C, T, H, W)
        """
        # Attention block with residual
        B, C, T, H, W = x.shape
        x_norm = x.flatten(2).transpose(1, 2)  # (B, T*H*W, C)
        x_norm = self.norm1(x_norm).transpose(1, 2).reshape(B, C, T, H, W)
        x = x + self.drop_path(self.attn(x_norm))
        
        # MLP block with residual
        x_norm = x.flatten(2).transpose(1, 2)  # (B, T*H*W, C)
        x_norm = self.norm2(x_norm).transpose(1, 2).reshape(B, C, T, H, W)
        x = x + self.drop_path(self.mlp(x_norm))
        
        return x


class GlobalBlock(nn.Module):
    """
    Global UniFormer block with global MHRA.
    
    Combines global attention with feedforward network.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalMHRA(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, T, H, W)
        
        Returns:
            Output tensor (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape
        
        # Attention block with residual
        x_norm = x.flatten(2).transpose(1, 2)  # (B, T*H*W, C)
        x_norm = self.norm1(x_norm)
        x_norm = x_norm.transpose(1, 2).reshape(B, C, T, H, W)
        x = x + self.drop_path(self.attn(x_norm))
        
        # MLP block with residual
        x_flat = x.flatten(2).transpose(1, 2)  # (B, T*H*W, C)
        x_flat = x_flat + self.drop_path(self.mlp(self.norm2(x_flat)))
        x = x_flat.transpose(1, 2).reshape(B, C, T, H, W)
        
        return x
