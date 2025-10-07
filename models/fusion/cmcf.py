"""
CMCF Fusion: Cross-Modal Complementary Fusion.

Fusion strategy designed for RGB and KIR (infrared) video modalities that:
1. Modality-Specific Enhancement: Enhances features specific to each modality
2. Complementary Attention: Cross-modal attention to capture complementary information
3. Adaptive Weighting: Dynamic fusion weights based on modality reliability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class ModalitySpecificEnhancement(nn.Module):
    """
    Modality-Specific Enhancement module.
    
    Enhances features specific to each modality using channel attention.
    """
    
    def __init__(
        self,
        in_channels: int,
        reduction: int = 8,
    ):
        """
        Initialize Modality-Specific Enhancement.
        
        Args:
            in_channels: Number of input channels
            reduction: Channel reduction ratio for attention
        """
        super().__init__()
        
        reduced_channels = max(in_channels // reduction, 16)
        
        # Channel attention for feature enhancement
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, reduced_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduced_channels, in_channels, 1),
            nn.Sigmoid(),
        )
        
        # Spatial enhancement
        self.spatial_enhance = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 3), 
                     padding=(0, 1, 1), groups=in_channels),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, T, H, W)
        
        Returns:
            Enhanced features (B, C, T, H, W)
        """
        # Channel attention
        channel_weights = self.channel_attention(x)
        enhanced = x * channel_weights
        
        # Spatial enhancement
        enhanced = self.spatial_enhance(enhanced)
        
        # Residual connection
        output = x + enhanced
        
        return output


class ComplementaryAttention(nn.Module):
    """
    Complementary Attention module.
    
    Applies cross-modal attention to capture complementary information between modalities.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_heads: int = 4,
    ):
        """
        Initialize Complementary Attention.
        
        Args:
            in_channels: Number of input channels
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        
        # Query, Key, Value projections for cross-attention
        self.query_proj = nn.Conv3d(in_channels, in_channels, 1)
        self.key_proj = nn.Conv3d(in_channels, in_channels, 1)
        self.value_proj = nn.Conv3d(in_channels, in_channels, 1)
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 1),
            nn.BatchNorm3d(in_channels),
        )
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query_feat: torch.Tensor, key_value_feat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for cross-modal attention.
        
        Args:
            query_feat: Query features from one modality (B, C, T, H, W)
            key_value_feat: Key/Value features from another modality (B, C, T, H, W)
        
        Returns:
            Attended features (B, C, T, H, W)
        """
        B, C, T, H, W = query_feat.shape
        
        # Project to Q, K, V
        Q = self.query_proj(query_feat)  # (B, C, T, H, W)
        K = self.key_proj(key_value_feat)  # (B, C, T, H, W)
        V = self.value_proj(key_value_feat)  # (B, C, T, H, W)
        
        # Reshape for multi-head attention
        # (B, C, T, H, W) -> (B, num_heads, head_dim, T*H*W)
        Q = Q.view(B, self.num_heads, self.head_dim, -1)
        K = K.view(B, self.num_heads, self.head_dim, -1)
        V = V.view(B, self.num_heads, self.head_dim, -1)
        
        # Compute attention scores
        # (B, num_heads, head_dim, T*H*W) x (B, num_heads, T*H*W, head_dim)
        # -> (B, num_heads, head_dim, head_dim)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        # (B, num_heads, head_dim, head_dim) x (B, num_heads, head_dim, T*H*W)
        # -> (B, num_heads, head_dim, T*H*W)
        attended = torch.matmul(attention_weights, V)
        
        # Reshape back
        attended = attended.view(B, C, T, H, W)
        
        # Output projection
        output = self.out_proj(attended)
        
        return output


class AdaptiveWeighting(nn.Module):
    """
    Adaptive Weighting module.
    
    Computes dynamic fusion weights based on modality reliability.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_modalities: int = 2,
    ):
        """
        Initialize Adaptive Weighting.
        
        Args:
            in_channels: Number of input channels
            num_modalities: Number of modalities
        """
        super().__init__()
        
        self.num_modalities = num_modalities
        
        # Reliability estimation network
        self.reliability_net = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels * num_modalities, in_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, num_modalities, 1),
            nn.Softmax(dim=1),
        )
    
    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            modalities: List of modality tensors [(B, C, T, H, W), ...]
        
        Returns:
            Weighted fusion of modalities (B, C, T, H, W)
        """
        # Concatenate modalities
        concat = torch.cat(modalities, dim=1)  # (B, C*num_modalities, T, H, W)
        
        # Compute reliability weights
        weights = self.reliability_net(concat)  # (B, num_modalities, 1, 1, 1)
        
        # Apply weights and sum
        weighted_sum = sum(
            w.expand_as(m) * m 
            for w, m in zip(weights.split(1, dim=1), modalities)
        )
        
        return weighted_sum


class CMCFFusion(nn.Module):
    """
    CMCF Fusion module.
    
    Cross-Modal Complementary Fusion for RGB and KIR modalities.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_modalities: int = 2,
        enhancement_reduction: int = 8,
        attention_heads: int = 4,
    ):
        """
        Initialize CMCF Fusion.
        
        Args:
            in_channels: Number of channels per modality
            num_modalities: Number of modalities (default: 2 for RGB+KIR)
            enhancement_reduction: Reduction ratio for modality enhancement
            attention_heads: Number of attention heads for complementary attention
        """
        super().__init__()
        
        self.num_modalities = num_modalities
        
        # Modality-Specific Enhancement for each modality
        self.modality_enhancement = nn.ModuleList([
            ModalitySpecificEnhancement(
                in_channels=in_channels,
                reduction=enhancement_reduction,
            )
            for _ in range(num_modalities)
        ])
        
        # Complementary Attention modules
        # Each modality attends to other modalities
        self.complementary_attention = nn.ModuleList([
            ComplementaryAttention(
                in_channels=in_channels,
                num_heads=attention_heads,
            )
            for _ in range(num_modalities)
        ])
        
        # Adaptive Weighting
        self.adaptive_weighting = AdaptiveWeighting(
            in_channels=in_channels,
            num_modalities=num_modalities,
        )
        
        # Final fusion layer
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            modalities: List of modality tensors [(B, C, T, H, W), ...]
                       Typically [rgb_features, kir_features]
        
        Returns:
            Fused tensor (B, C, T, H, W)
        """
        if len(modalities) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modalities, got {len(modalities)}"
            )
        
        # Step 1: Modality-Specific Enhancement
        enhanced_modalities = [
            enhance_module(modality)
            for enhance_module, modality in zip(self.modality_enhancement, modalities)
        ]
        
        # Step 2: Complementary Attention
        # Each modality attends to the other modality
        complementary_features = []
        for i, (attn_module, query_modality) in enumerate(
            zip(self.complementary_attention, enhanced_modalities)
        ):
            # Get key-value modality (the other modality)
            # For 2 modalities: if i=0, kv_idx=1; if i=1, kv_idx=0
            kv_idx = (i + 1) % self.num_modalities
            key_value_modality = enhanced_modalities[kv_idx]
            
            # Apply cross-modal attention
            attended = attn_module(query_modality, key_value_modality)
            
            # Add residual connection
            complementary = query_modality + attended
            complementary_features.append(complementary)
        
        # Step 3: Adaptive Weighting
        fused = self.adaptive_weighting(complementary_features)
        
        # Step 4: Final fusion
        fused = self.fusion_conv(fused)
        
        return fused


class CMCFFusionModel(nn.Module):
    """
    Complete model with CMCF Fusion.
    
    Uses separate backbones for feature extraction, then applies CMCF fusion.
    """
    
    def __init__(
        self,
        backbones: Dict[str, nn.Module],
        head: nn.Module,
        fusion_dim: int,
        enhancement_reduction: int = 8,
        attention_heads: int = 4,
    ):
        """
        Initialize CMCF Fusion model.
        
        Args:
            backbones: Dictionary of backbones for each modality
            head: Classification head
            fusion_dim: Dimension for fusion (output channels of backbones)
            enhancement_reduction: Reduction ratio for modality enhancement
            attention_heads: Number of attention heads
        """
        super().__init__()
        
        self.backbones = nn.ModuleDict(backbones)
        
        self.fusion = CMCFFusion(
            in_channels=fusion_dim,
            num_modalities=len(backbones),
            enhancement_reduction=enhancement_reduction,
            attention_heads=attention_heads,
        )
        
        self.head = head
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Dictionary containing modality tensors
                - 'rgb': RGB video tensor (B, C, T, H, W)
                - 'kir': KIR video tensor (B, C, T, H, W)
        
        Returns:
            Class logits (B, num_classes)
        """
        # Extract features from each modality
        features = []
        for modality in ['rgb', 'kir']:
            if modality in inputs and inputs[modality] is not None:
                feat = self.backbones[modality](inputs[modality])
                features.append(feat)
        
        # Apply CMCF fusion
        fused_features = self.fusion(features)
        
        # Classification
        logits = self.head(fused_features)
        
        return logits
