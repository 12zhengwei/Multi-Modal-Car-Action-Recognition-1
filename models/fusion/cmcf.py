"""
CMCF Fusion: Cross-Modal Complementary Fusion.

Features modality-specific enhancement and adaptive weighting for
robust multi-modal fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ModalitySpecificEnhancement(nn.Module):
    """
    Modality-Specific Enhancement module.
    
    Learns modality-specific representations through specialized pathways.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Initialize Modality-Specific Enhancement.
        
        Args:
            in_channels: Number of input channels
            hidden_channels: Number of hidden channels (default: in_channels)
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        if hidden_channels is None:
            hidden_channels = in_channels
        
        # Modality-specific transformation pathway
        self.transform = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(hidden_channels, in_channels, kernel_size=1),
            nn.BatchNorm3d(in_channels),
        )
        
        # Channel attention for feature recalibration
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, max(in_channels // 4, 1), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(in_channels // 4, 1), in_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        
        # Spatial attention for spatial recalibration
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, T, H, W)
        
        Returns:
            Enhanced features (B, C, T, H, W)
        """
        # Modality-specific transformation
        transformed = self.transform(x)
        
        # Channel attention
        channel_weights = self.channel_attention(transformed)
        channel_enhanced = transformed * channel_weights
        
        # Spatial attention
        spatial_weights = self.spatial_attention(channel_enhanced)
        spatial_enhanced = channel_enhanced * spatial_weights
        
        # Residual connection
        output = x + spatial_enhanced
        
        return output


class ComplementaryAttention(nn.Module):
    """
    Complementary Attention module.
    
    Identifies and emphasizes complementary information between modalities.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_modalities: int = 2,
        reduction: int = 4,
    ):
        """
        Initialize Complementary Attention.
        
        Args:
            in_channels: Number of input channels per modality
            num_modalities: Number of modalities
            reduction: Channel reduction ratio for attention
        """
        super().__init__()
        
        self.num_modalities = num_modalities
        reduced_channels = max(in_channels // reduction, 16)
        
        # Cross-modal complementary attention
        self.complementary_attention = nn.Sequential(
            nn.Conv3d(in_channels * num_modalities, reduced_channels, kernel_size=1),
            nn.BatchNorm3d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduced_channels, in_channels * num_modalities, kernel_size=1),
        )
        
        # Per-modality complementary gates
        self.modality_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels * 2, in_channels, kernel_size=1),
                nn.BatchNorm3d(in_channels),
                nn.Sigmoid(),
            )
            for _ in range(num_modalities)
        ])
    
    def forward(self, modalities: list) -> list:
        """
        Forward pass.
        
        Args:
            modalities: List of modality tensors [(B, C, T, H, W), ...]
        
        Returns:
            List of complementary-enhanced modality tensors
        """
        if len(modalities) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modalities, got {len(modalities)}"
            )
        
        # Concatenate all modalities for global context
        concat = torch.cat(modalities, dim=1)  # (B, C*num_modalities, T, H, W)
        
        # Compute complementary attention scores
        attention_map = self.complementary_attention(concat)
        
        # Split attention for each modality
        attention_splits = torch.split(
            attention_map,
            attention_map.size(1) // self.num_modalities,
            dim=1
        )
        
        # Apply complementary gates to each modality
        enhanced_modalities = []
        for i, (modality, gate, attn) in enumerate(
            zip(modalities, self.modality_gates, attention_splits)
        ):
            # Compute complementary information (other modalities)
            other_modalities = [m for j, m in enumerate(modalities) if j != i]
            complementary_info = torch.mean(
                torch.stack(other_modalities, dim=0),
                dim=0
            )
            
            # Combine current modality with complementary information
            combined = torch.cat([modality, complementary_info], dim=1)
            gate_weight = gate(combined)
            
            # Apply attention and gate
            enhanced = modality + (attn * gate_weight * modality)
            enhanced_modalities.append(enhanced)
        
        return enhanced_modalities


class AdaptiveWeighting(nn.Module):
    """
    Adaptive Weighting module.
    
    Learns dynamic fusion weights based on input content.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_modalities: int = 2,
        hidden_dim: int = 64,
    ):
        """
        Initialize Adaptive Weighting.
        
        Args:
            in_channels: Number of input channels per modality
            num_modalities: Number of modalities
            hidden_dim: Hidden dimension for weight prediction
        """
        super().__init__()
        
        self.num_modalities = num_modalities
        
        # Global context extraction
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Weight prediction network
        self.weight_predictor = nn.Sequential(
            nn.Linear(in_channels * num_modalities, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_modalities),
        )
    
    def forward(self, modalities: list) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            modalities: List of modality tensors [(B, C, T, H, W), ...]
        
        Returns:
            Weighted fusion result (B, C, T, H, W)
        """
        B = modalities[0].size(0)
        
        # Extract global context from each modality
        contexts = []
        for modality in modalities:
            context = self.global_pool(modality)  # (B, C, 1, 1, 1)
            contexts.append(context.view(B, -1))
        
        # Concatenate contexts
        concat_context = torch.cat(contexts, dim=1)  # (B, C*num_modalities)
        
        # Predict adaptive weights
        weights = self.weight_predictor(concat_context)  # (B, num_modalities)
        weights = F.softmax(weights, dim=1)  # Normalize
        
        # Apply weights to modalities
        weighted_modalities = []
        for i, modality in enumerate(modalities):
            # Shape: (B, 1, 1, 1, 1) broadcasts to (B, C, T, H, W)
            weight = weights[:, i:i+1, None, None, None]
            weighted = modality * weight
            weighted_modalities.append(weighted)
        
        # Sum weighted modalities
        fused = sum(weighted_modalities)
        
        return fused


class CMCFFusion(nn.Module):
    """
    CMCF (Cross-Modal Complementary Fusion) module.
    
    Combines modality-specific enhancement, complementary attention,
    and adaptive weighting for robust multi-modal fusion.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_modalities: int = 2,
        enhancement_hidden_channels: Optional[int] = None,
        attention_reduction: int = 4,
        weighting_hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        """
        Initialize CMCF Fusion.
        
        Args:
            in_channels: Number of channels per modality
            num_modalities: Number of modalities
            enhancement_hidden_channels: Hidden channels for enhancement
            attention_reduction: Reduction ratio for attention
            weighting_hidden_dim: Hidden dimension for adaptive weighting
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_modalities = num_modalities
        
        # Modality-Specific Enhancement for each modality
        self.enhancements = nn.ModuleList([
            ModalitySpecificEnhancement(
                in_channels=in_channels,
                hidden_channels=enhancement_hidden_channels,
                dropout=dropout,
            )
            for _ in range(num_modalities)
        ])
        
        # Complementary Attention
        self.complementary_attention = ComplementaryAttention(
            in_channels=in_channels,
            num_modalities=num_modalities,
            reduction=attention_reduction,
        )
        
        # Adaptive Weighting
        self.adaptive_weighting = AdaptiveWeighting(
            in_channels=in_channels,
            num_modalities=num_modalities,
            hidden_dim=weighting_hidden_dim,
        )
        
        # Final refinement
        self.refinement = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, modalities: list) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            modalities: List of modality tensors [(B, C, T, H, W), ...]
        
        Returns:
            Fused tensor (B, C, T, H, W)
        """
        if len(modalities) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modalities, got {len(modalities)}"
            )
        
        # Step 1: Modality-Specific Enhancement
        enhanced_modalities = [
            enhancement(modality)
            for enhancement, modality in zip(self.enhancements, modalities)
        ]
        
        # Step 2: Complementary Attention
        complementary_modalities = self.complementary_attention(enhanced_modalities)
        
        # Step 3: Adaptive Weighting and Fusion
        fused = self.adaptive_weighting(complementary_modalities)
        
        # Step 4: Final Refinement
        refined = self.refinement(fused)
        
        return refined


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
        enhancement_hidden_channels: Optional[int] = None,
        attention_reduction: int = 4,
        weighting_hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        """
        Initialize CMCF Fusion model.
        
        Args:
            backbones: Dictionary of backbones for each modality
            head: Classification head
            fusion_dim: Dimension for fusion (output channels of backbones)
            enhancement_hidden_channels: Hidden channels for enhancement
            attention_reduction: Reduction ratio for attention
            weighting_hidden_dim: Hidden dimension for adaptive weighting
            dropout: Dropout rate
        """
        super().__init__()
        
        self.backbones = nn.ModuleDict(backbones)
        
        self.fusion = CMCFFusion(
            in_channels=fusion_dim,
            num_modalities=len(backbones),
            enhancement_hidden_channels=enhancement_hidden_channels,
            attention_reduction=attention_reduction,
            weighting_hidden_dim=weighting_hidden_dim,
            dropout=dropout,
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
        for modality in self.backbones.keys():
            if modality in inputs and inputs[modality] is not None:
                feat = self.backbones[modality](inputs[modality])
                features.append(feat)
        
        # Apply CMCF fusion
        fused_features = self.fusion(features)
        
        # Classification
        logits = self.head(fused_features)
        
        return logits
