"""
Late Fusion strategy for multi-modal video understanding.

Fuses modalities at decision level (logits or probabilities).
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import torch.nn.functional as F


class LateFusion(nn.Module):
    """
    Late Fusion module.
    
    Fuses modalities at logits or probability level with optional learnable weights.
    """
    
    def __init__(
        self,
        num_modalities: int = 2,
        fusion_method: str = 'logits',
        learnable_weights: bool = True,
    ):
        """
        Initialize Late Fusion module.
        
        Args:
            num_modalities: Number of modalities to fuse
            fusion_method: Fusion method ('logits', 'prob', or 'weighted')
            learnable_weights: Whether to use learnable fusion weights
        """
        super().__init__()
        
        self.num_modalities = num_modalities
        self.fusion_method = fusion_method
        self.learnable_weights = learnable_weights
        
        if learnable_weights:
            # Initialize learnable weights (will be softmaxed)
            self.fusion_weights = nn.Parameter(
                torch.ones(num_modalities) / num_modalities
            )
        else:
            # Equal weights
            self.register_buffer(
                'fusion_weights',
                torch.ones(num_modalities) / num_modalities
            )
    
    def forward(self, logits_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for late fusion.
        
        Args:
            logits_dict: Dictionary containing logits from each modality
                - 'rgb': RGB logits (B, num_classes)
                - 'kir': KIR logits (B, num_classes)
        
        Returns:
            Fused logits (B, num_classes)
        """
        # Collect logits from available modalities
        logits_list = []
        modality_names = ['rgb', 'kir']
        
        for name in modality_names:
            if name in logits_dict and logits_dict[name] is not None:
                logits_list.append(logits_dict[name])
        
        if len(logits_list) == 0:
            raise ValueError("No modalities provided for fusion")
        
        # Single modality case
        if len(logits_list) == 1:
            return logits_list[0]
        
        # Get fusion weights
        if self.learnable_weights:
            weights = F.softmax(self.fusion_weights[:len(logits_list)], dim=0)
        else:
            weights = self.fusion_weights[:len(logits_list)]
            weights = weights / weights.sum()  # Normalize
        
        # Fuse based on method
        if self.fusion_method == 'logits':
            # Weighted average of logits
            fused_logits = sum(w * logits for w, logits in zip(weights, logits_list))
            return fused_logits
        
        elif self.fusion_method == 'prob':
            # Weighted average of probabilities
            probs_list = [F.softmax(logits, dim=-1) for logits in logits_list]
            fused_probs = sum(w * probs for w, probs in zip(weights, probs_list))
            # Convert back to logits
            fused_logits = torch.log(fused_probs + 1e-8)
            return fused_logits
        
        elif self.fusion_method == 'weighted':
            # Learnable weighted fusion (same as logits method)
            fused_logits = sum(w * logits for w, logits in zip(weights, logits_list))
            return fused_logits
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")


class LateFusionModel(nn.Module):
    """
    Complete model with Late Fusion.
    
    Uses separate backbones for each modality and fuses at decision level.
    """
    
    def __init__(
        self,
        backbones: Dict[str, nn.Module],
        heads: Dict[str, nn.Module],
        fusion_method: str = 'logits',
        learnable_weights: bool = True,
    ):
        """
        Initialize Late Fusion model.
        
        Args:
            backbones: Dictionary of backbones for each modality
                - 'rgb': RGB backbone
                - 'kir': KIR backbone
            heads: Dictionary of classification heads for each modality
                - 'rgb': RGB head
                - 'kir': KIR head
            fusion_method: Fusion method ('logits', 'prob', or 'weighted')
            learnable_weights: Whether to use learnable fusion weights
        """
        super().__init__()
        
        self.backbones = nn.ModuleDict(backbones)
        self.heads = nn.ModuleDict(heads)
        
        self.fusion = LateFusion(
            num_modalities=len(backbones),
            fusion_method=fusion_method,
            learnable_weights=learnable_weights,
        )
    
    def forward_modality(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        """
        Forward pass for single modality.
        
        Args:
            x: Input tensor (B, C, T, H, W)
            modality: Modality name ('rgb' or 'kir')
        
        Returns:
            Logits (B, num_classes)
        """
        features = self.backbones[modality](x)
        logits = self.heads[modality](features)
        return logits
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        return_individual: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Dictionary containing modality tensors
                - 'rgb': RGB video tensor (B, C, T, H, W)
                - 'kir': KIR video tensor (B, C, T, H, W)
            return_individual: Whether to return individual modality logits
        
        Returns:
            Fused logits (B, num_classes) or dict with individual and fused logits
        """
        logits_dict = {}
        
        # Forward through each modality
        for modality in ['rgb', 'kir']:
            if modality in inputs and inputs[modality] is not None:
                logits_dict[modality] = self.forward_modality(
                    inputs[modality],
                    modality
                )
        
        # Fuse logits
        fused_logits = self.fusion(logits_dict)
        
        if return_individual:
            return {
                'fused': fused_logits,
                'individual': logits_dict,
            }
        
        return fused_logits
