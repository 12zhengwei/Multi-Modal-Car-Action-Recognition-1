"""
Multi-modal video dataset for action recognition.

Supports single modality (RGB or KIR) and dual modality (RGB + KIR) training.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np

from .decoding import VideoDecoder, sample_frames, pad_or_truncate_frames
from .transforms import (
    get_train_transforms, 
    get_val_transforms, 
    get_normalization,
    KIRToRGB,
)


class VideoDataset(Dataset):
    """
    Multi-modal video dataset for action recognition.
    
    Supports:
    - Single modality: RGB or KIR only
    - Dual modality: RGB + KIR with synchronized sampling
    """
    
    def __init__(
        self,
        index_file: str,
        root_dir: str,
        modality: str,
        num_frames: int = 16,
        frame_stride: int = 2,
        mode: str = 'train',
        config: Optional[Dict] = None,
        decoder_backend: str = 'auto',
    ):
        """
        Initialize video dataset.
        
        Args:
            index_file: Path to CSV index file
            root_dir: Root directory of dataset
            modality: 'rgb', 'kir', or 'rgb_kir'
            num_frames: Number of frames per clip
            frame_stride: Stride for temporal sampling
            mode: 'train', 'val', or 'test'
            config: Configuration dictionary
            decoder_backend: Video decoder backend
        """
        self.root_dir = Path(root_dir)
        self.modality = modality
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.mode = mode
        self.config = config or {}
        
        # Load index
        self.index = pd.read_csv(index_file)
        print(f"Loaded {len(self.index)} videos from {index_file}")
        
        # Filter by modality if single modality
        if modality in ['rgb', 'kir']:
            self.index = self.index[self.index['modality'] == modality].reset_index(drop=True)
            print(f"Filtered to {len(self.index)} {modality} videos")
        elif modality == 'rgb_kir':
            # For dual modality, organize by video pairs
            self._organize_dual_modality()
        else:
            raise ValueError(f"Unknown modality: {modality}")
        
        # Initialize decoder
        self.decoder = VideoDecoder(backend=decoder_backend)
        
        # Initialize transforms
        if mode == 'train':
            self.transform = get_train_transforms(self.config)
        else:
            self.transform = get_val_transforms(self.config)
        
        # Initialize normalization
        if modality in ['rgb', 'kir']:
            self.normalize = get_normalization(self.config, modality)
        else:
            self.normalize_rgb = get_normalization(self.config, 'rgb')
            self.normalize_kir = get_normalization(self.config, 'kir')
        
        # KIR to RGB converter
        self.kir_to_rgb = KIRToRGB()
    
    def _organize_dual_modality(self):
        """Organize index for dual modality (RGB + KIR pairs)."""
        # Group by video identifier (path without modality)
        rgb_videos = self.index[self.index['modality'] == 'rgb'].copy()
        kir_videos = self.index[self.index['modality'] == 'kir'].copy()
        
        # Create unique identifiers (path without modality folder)
        def get_video_id(path):
            parts = Path(path).parts
            # Remove modality from path
            new_parts = [p for p in parts if p not in ['rgb', 'kir']]
            return str(Path(*new_parts))
        
        rgb_videos['video_id'] = rgb_videos['path'].apply(get_video_id)
        kir_videos['video_id'] = kir_videos['path'].apply(get_video_id)
        
        # Merge RGB and KIR videos
        merged = rgb_videos.merge(
            kir_videos,
            on=['video_id', 'label_name', 'label_id', 'split'],
            suffixes=('_rgb', '_kir')
        )
        
        self.dual_index = merged
        print(f"Created {len(self.dual_index)} RGB-KIR pairs")
    
    def __len__(self) -> int:
        if self.modality == 'rgb_kir':
            return len(self.dual_index)
        return len(self.index)
    
    def _load_video(
        self,
        video_path: str,
        is_kir: bool = False,
    ) -> torch.Tensor:
        """
        Load and preprocess a single video.
        
        Args:
            video_path: Relative path to video file
            is_kir: Whether this is a KIR video
        
        Returns:
            Preprocessed video tensor
        """
        full_path = str(self.root_dir / video_path)
        
        try:
            # Decode video
            frames, total_frames = self.decoder.decode_video(full_path)
            
            # Sample frames
            if self.mode == 'train':
                frame_indices = sample_frames(
                    total_frames,
                    self.num_frames,
                    mode='random',
                    temporal_stride=self.frame_stride,
                )
            else:
                frame_indices = sample_frames(
                    total_frames,
                    self.num_frames,
                    mode='uniform',
                    temporal_stride=1,
                )
            
            frames, _ = self.decoder.decode_video(full_path, frame_indices=frame_indices)
            
            # Pad or truncate to exact number of frames
            frames = pad_or_truncate_frames(frames, self.num_frames, mode='loop')
            
            # Convert KIR to pseudo-3-channel if needed
            if is_kir and frames.shape[-1] == 1:
                frames = self.kir_to_rgb(frames)
            
            # Apply transforms
            video = self.transform(frames)  # (C, T, H, W)
            
            # Apply normalization
            if is_kir:
                video = self.normalize_kir(video)
            else:
                if self.modality == 'rgb_kir':
                    video = self.normalize_rgb(video)
                else:
                    video = self.normalize(video)
            
            return video
            
        except Exception as e:
            print(f"Error loading video {video_path}: {str(e)}")
            # Return zero tensor as fallback
            return torch.zeros(3, self.num_frames, 224, 224)
    
    def __getitem__(self, idx: int):
        """Get a single item or pair of items."""
        if self.modality == 'rgb_kir':
            return self._get_dual_item(idx)
        else:
            return self._get_single_item(idx)
    
    def _get_single_item(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get single modality item."""
        item = self.index.iloc[idx]
        
        video = self._load_video(
            item['path'],
            is_kir=(self.modality == 'kir')
        )
        label = int(item['label_id'])
        
        return video, label
    
    def _get_dual_item(self, idx: int) -> Dict:
        """Get dual modality item (RGB + KIR pair)."""
        item = self.dual_index.iloc[idx]
        
        # Load RGB video
        rgb_video = self._load_video(item['path_rgb'], is_kir=False)
        
        # Load KIR video
        kir_video = self._load_video(item['path_kir'], is_kir=True)
        
        label = int(item['label_id'])
        
        return {
            'rgb': rgb_video,
            'kir': kir_video,
            'label': label,
        }


def build_dataloader(
    index_file: str,
    root_dir: str,
    modality: str,
    batch_size: int,
    num_workers: int,
    mode: str = 'train',
    config: Optional[Dict] = None,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """
    Build a dataloader for video dataset.
    
    Args:
        index_file: Path to CSV index file
        root_dir: Root directory of dataset
        modality: 'rgb', 'kir', or 'rgb_kir'
        batch_size: Batch size
        num_workers: Number of data loading workers
        mode: 'train', 'val', or 'test'
        config: Configuration dictionary
        pin_memory: Whether to pin memory
    
    Returns:
        DataLoader instance
    """
    from .collate import get_collate_fn
    
    dataset = VideoDataset(
        index_file=index_file,
        root_dir=root_dir,
        modality=modality,
        num_frames=config.get('num_frames', 16),
        frame_stride=config.get('frame_stride', 2),
        mode=mode,
        config=config,
    )
    
    collate_fn = get_collate_fn(modality)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=(mode == 'train'),
    )
    
    return dataloader
