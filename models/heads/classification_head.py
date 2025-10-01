"""
Classification head for action recognition.

Performs spatial-temporal pooling and classification.
"""

import torch
import torch.nn as nn
from typing import Optional


class ClassificationHead(nn.Module):
    """
    Classification head for video understanding.
    
    Performs global pooling followed by classification.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.5,
        pooling: str = 'avg',
    ):
        """
        Initialize classification head.
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            dropout: Dropout rate
            pooling: Pooling type ('avg' or 'max')
        """
        super().__init__()
        
        self.pooling = pooling
        
        # Global pooling
        if pooling == 'avg':
            self.global_pool = nn.AdaptiveAvgPool3d(1)
        elif pooling == 'max':
            self.global_pool = nn.AdaptiveMaxPool3d(1)
        else:
            raise ValueError(f"Unknown pooling type: {pooling}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Classification layer
        self.fc = nn.Linear(in_channels, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (B, C, T, H, W)
        
        Returns:
            Class logits (B, num_classes)
        """
        # Global pooling
        x = self.global_pool(x)  # (B, C, 1, 1, 1)
        x = x.flatten(1)  # (B, C)
        
        # Dropout
        x = self.dropout(x)
        
        # Classification
        logits = self.fc(x)  # (B, num_classes)
        
        return logits


class MultiHeadClassificationHead(nn.Module):
    """
    Multi-head classification with auxiliary outputs.
    
    Can produce multiple predictions at different temporal scales.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_heads: int = 1,
        dropout: float = 0.5,
        pooling: str = 'avg',
    ):
        """
        Initialize multi-head classification.
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            num_heads: Number of classification heads
            dropout: Dropout rate
            pooling: Pooling type
        """
        super().__init__()
        
        self.num_heads = num_heads
        
        # Create multiple heads
        self.heads = nn.ModuleList([
            ClassificationHead(
                in_channels=in_channels,
                num_classes=num_classes,
                dropout=dropout,
                pooling=pooling,
            )
            for _ in range(num_heads)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (B, C, T, H, W)
        
        Returns:
            Primary head logits (B, num_classes) or list of all heads during training
        """
        if self.num_heads == 1:
            return self.heads[0](x)
        
        # Multiple heads
        outputs = [head(x) for head in self.heads]
        
        if self.training:
            # Return all outputs during training for multi-task loss
            return outputs
        else:
            # Return primary head output during inference
            return outputs[0]


def build_classification_head(
    in_channels: int,
    num_classes: int,
    config: Optional[dict] = None,
) -> nn.Module:
    """
    Build classification head from configuration.
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        config: Configuration dictionary
    
    Returns:
        Classification head module
    """
    config = config or {}
    
    dropout = config.get('dropout', 0.5)
    pooling = config.get('pooling', 'avg')
    num_heads = config.get('num_heads', 1)
    
    if num_heads > 1:
        return MultiHeadClassificationHead(
            in_channels=in_channels,
            num_classes=num_classes,
            num_heads=num_heads,
            dropout=dropout,
            pooling=pooling,
        )
    else:
        return ClassificationHead(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
            pooling=pooling,
        )
