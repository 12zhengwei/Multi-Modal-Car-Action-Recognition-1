"""
Custom collate functions for video data batching.

Handles variable-length videos and multi-modal data.
"""

import torch
from typing import List, Dict, Tuple, Optional


def video_collate_fn(batch: List) -> Tuple:
    """
    Collate function for single-modality video batches.
    
    Args:
        batch: List of (video, label) tuples
            video: torch.Tensor (C, T, H, W)
            label: int
    
    Returns:
        Batched videos and labels
    """
    videos = []
    labels = []
    
    for item in batch:
        video, label = item
        videos.append(video)
        labels.append(label)
    
    # Stack videos
    videos = torch.stack(videos, dim=0)  # (B, C, T, H, W)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return videos, labels


def multimodal_collate_fn(batch: List) -> Tuple:
    """
    Collate function for multi-modal video batches.
    
    Args:
        batch: List of dictionaries with keys:
            'rgb': torch.Tensor (C, T, H, W) or None
            'kir': torch.Tensor (C, T, H, W) or None
            'label': int
    
    Returns:
        Dictionary with batched data
    """
    rgb_videos = []
    kir_videos = []
    labels = []
    has_rgb = False
    has_kir = False
    
    for item in batch:
        if 'rgb' in item and item['rgb'] is not None:
            rgb_videos.append(item['rgb'])
            has_rgb = True
        if 'kir' in item and item['kir'] is not None:
            kir_videos.append(item['kir'])
            has_kir = True
        labels.append(item['label'])
    
    # Prepare output dictionary
    output = {}
    
    if has_rgb and len(rgb_videos) == len(batch):
        output['rgb'] = torch.stack(rgb_videos, dim=0)  # (B, C, T, H, W)
    
    if has_kir and len(kir_videos) == len(batch):
        output['kir'] = torch.stack(kir_videos, dim=0)  # (B, C, T, H, W)
    
    output['label'] = torch.tensor(labels, dtype=torch.long)
    
    return output


def get_collate_fn(modality: str):
    """
    Get appropriate collate function based on modality.
    
    Args:
        modality: 'rgb', 'kir', or 'rgb_kir'
    
    Returns:
        Collate function
    """
    if modality in ['rgb', 'kir']:
        return video_collate_fn
    elif modality == 'rgb_kir':
        return multimodal_collate_fn
    else:
        raise ValueError(f"Unknown modality: {modality}")
