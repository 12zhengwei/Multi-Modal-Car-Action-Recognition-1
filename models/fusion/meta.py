"""
META Fusion: Motion and Multi-View Excitation with Temporal Aggregation.

Advanced fusion strategy that incorporates:
1. Motion Excitation: Temporal motion features from frame differences
2. Multi-View Excitation: Cross-modal attention for adaptive weighting
3. Temporal Aggregation: Hierarchical temporal modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class MotionExcitation(nn.Module):
    """
    Motion Excitation module.
    
    Extracts motion features from temporal frame differences.
    """
    
    def __init__(
        self,
        in_channels: int,
        reduction: int = 4,
        kernel_size: int = 3,
    ):
        """
        Initialize Motion Excitation.
        
        Args:
            in_channels: Number of input channels
            reduction: Channel reduction ratio
            kernel_size: Kernel size for motion convolution
        """
        super().__init__()
        
        reduced_channels = max(in_channels // reduction, 16)
        padding = kernel_size // 2
        
        # Motion extraction via 3D convolution on differences
        self.motion_conv = nn.Sequential(
            nn.Conv3d(in_channels, reduced_channels, 
                     kernel_size=(3, kernel_size, kernel_size),
                     padding=(1, padding, padding)),
            nn.BatchNorm3d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduced_channels, in_channels, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, T, H, W)
        
        Returns:
            Motion-excited features (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape
        
        # Compute frame differences (temporal gradient)
        if T > 1:
            # Forward difference
            diff_forward = x[:, :, 1:] - x[:, :, :-1]
            # Pad to maintain temporal dimension
            diff_forward = F.pad(diff_forward, (0, 0, 0, 0, 0, 1), mode='replicate')
        else:
            diff_forward = torch.zeros_like(x)
        
        # Extract motion features
        motion_weights = self.motion_conv(diff_forward)
        
        # Apply motion excitation
        excited = x * motion_weights
        
        return excited


class MultiViewExcitation(nn.Module):
    """
    Multi-View (Multi-Modal) Excitation module.
    
    Uses cross-modal attention to adaptively weight different modalities.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_modalities: int = 2,
        reduction: int = 4,
    ):
        """
        Initialize Multi-View Excitation.
        
        Args:
            in_channels: Number of input channels per modality
            num_modalities: Number of modalities
            reduction: Channel reduction ratio for attention
        """
        super().__init__()
        
        self.num_modalities = num_modalities
        reduced_channels = max(in_channels // reduction, 16)
        
        # Global pooling followed by attention network
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels * num_modalities, reduced_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduced_channels, in_channels * num_modalities, 1),
        )
        
        # Per-modality gates
        self.modality_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels, in_channels, 1),
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
            List of excited modality tensors
        """
        if len(modalities) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modalities, got {len(modalities)}"
            )
        
        # Concatenate all modalities
        concat = torch.cat(modalities, dim=1)  # (B, C*num_modalities, T, H, W)
        
        # Compute cross-modal attention
        attention_weights = self.attention(concat)  # (B, C*num_modalities, 1, 1, 1)
        attention_weights = torch.softmax(
            attention_weights.view(attention_weights.size(0), self.num_modalities, -1, 1, 1, 1),
            dim=1
        )
        
        # Apply attention to each modality
        excited_modalities = []
        for i, (modality, gate) in enumerate(zip(modalities, self.modality_gates)):
            # Extract attention for this modality
            attn = attention_weights[:, i]  # (B, C, 1, 1, 1)
            
            # Apply gate and attention
            gate_weight = gate(modality)
            excited = modality * gate_weight * attn
            excited_modalities.append(excited)
        
        return excited_modalities


class TemporalAggregation(nn.Module):
    """
    Temporal Aggregation module.
    
    Hierarchical temporal modeling with grouped convolutions.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_groups: int = 4,
    ):
        """
        Initialize Temporal Aggregation.
        
        Args:
            in_channels: Number of input channels
            num_groups: Number of groups for temporal modeling
        """
        super().__init__()
        
        self.num_groups = num_groups
        
        # Hierarchical temporal convolutions
        self.temporal_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    in_channels,
                    kernel_size=(3, 1, 1),
                    padding=(1, 0, 0),
                    groups=in_channels,  # Depthwise
                ),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_groups)
        ])
        
        # Aggregation layer
        self.aggregation = nn.Sequential(
            nn.Conv3d(in_channels * num_groups, in_channels, 1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, T, H, W)
        
        Returns:
            Temporally aggregated features (B, C, T, H, W)
        """
        # Apply temporal convolutions at different scales
        temporal_features = []
        for conv in self.temporal_convs:
            feat = conv(x)
            temporal_features.append(feat)
        
        # Concatenate and aggregate
        concat_features = torch.cat(temporal_features, dim=1)
        aggregated = self.aggregation(concat_features)
        
        # Residual connection
        output = x + aggregated
        
        return output


class METAFusion(nn.Module):
    """
    META Fusion module.
    
    Combines Motion Excitation, Multi-View Excitation, and Temporal Aggregation.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_modalities: int = 2,
        motion_kernel_size: int = 3,
        temporal_groups: int = 4,
        excitation_reduction: int = 4,
        use_motion_excitation: bool = True,
        use_multiview_excitation: bool = True,
        use_temporal_aggregation: bool = True,
    ):
        """
        Initialize META Fusion.
        
        Args:
            in_channels: Number of channels per modality
            num_modalities: Number of modalities
            motion_kernel_size: Kernel size for motion extraction
            temporal_groups: Number of groups for temporal aggregation
            excitation_reduction: Reduction ratio for excitation modules
            use_motion_excitation: Whether to use motion excitation
            use_multiview_excitation: Whether to use multi-view excitation
            use_temporal_aggregation: Whether to use temporal aggregation
        """
        super().__init__()
        
        self.use_motion_excitation = use_motion_excitation
        self.use_multiview_excitation = use_multiview_excitation
        self.use_temporal_aggregation = use_temporal_aggregation
        
        # Motion Excitation
        if use_motion_excitation:
            self.motion_excitation = nn.ModuleList([
                MotionExcitation(
                    in_channels=in_channels,
                    reduction=excitation_reduction,
                    kernel_size=motion_kernel_size,
                )
                for _ in range(num_modalities)
            ])
        
        # Multi-View Excitation
        if use_multiview_excitation:
            self.multiview_excitation = MultiViewExcitation(
                in_channels=in_channels,
                num_modalities=num_modalities,
                reduction=excitation_reduction,
            )
        
        # Temporal Aggregation
        if use_temporal_aggregation:
            self.temporal_aggregation = TemporalAggregation(
                in_channels=in_channels * num_modalities,
                num_groups=temporal_groups,
            )
        
        # Final fusion layer
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(in_channels * num_modalities, in_channels, 1),
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
        # Motion Excitation
        if self.use_motion_excitation:
            modalities = [
                motion_module(modality)
                for motion_module, modality in zip(self.motion_excitation, modalities)
            ]
        
        # Multi-View Excitation
        if self.use_multiview_excitation:
            modalities = self.multiview_excitation(modalities)
        
        # Concatenate modalities
        fused = torch.cat(modalities, dim=1)  # (B, C*num_modalities, T, H, W)
        
        # Temporal Aggregation
        if self.use_temporal_aggregation:
            fused = self.temporal_aggregation(fused)
        
        # Final fusion
        fused = self.fusion_conv(fused)
        
        return fused


class METAFusionModel(nn.Module):
    """
    Complete model with META Fusion.
    
    Uses separate backbones for feature extraction, then applies META fusion.
    """
    
    def __init__(
        self,
        backbones: Dict[str, nn.Module],
        head: nn.Module,
        fusion_dim: int,
        motion_kernel_size: int = 3,
        temporal_groups: int = 4,
        excitation_reduction: int = 4,
        use_motion_excitation: bool = True,
        use_multiview_excitation: bool = True,
        use_temporal_aggregation: bool = True,
    ):
        """
        Initialize META Fusion model.
        
        Args:
            backbones: Dictionary of backbones for each modality
            head: Classification head
            fusion_dim: Dimension for fusion (output channels of backbones)
            motion_kernel_size: Kernel size for motion extraction
            temporal_groups: Number of groups for temporal aggregation
            excitation_reduction: Reduction ratio
            use_motion_excitation: Whether to use motion excitation
            use_multiview_excitation: Whether to use multi-view excitation
            use_temporal_aggregation: Whether to use temporal aggregation
        """
        super().__init__()
        
        self.backbones = nn.ModuleDict(backbones)
        
        self.fusion = METAFusion(
            in_channels=fusion_dim,
            num_modalities=len(backbones),
            motion_kernel_size=motion_kernel_size,
            temporal_groups=temporal_groups,
            excitation_reduction=excitation_reduction,
            use_motion_excitation=use_motion_excitation,
            use_multiview_excitation=use_multiview_excitation,
            use_temporal_aggregation=use_temporal_aggregation,
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
        
        # Apply META fusion
        fused_features = self.fusion(features)
        
        # Classification
        logits = self.head(fused_features)
        
        return logits
