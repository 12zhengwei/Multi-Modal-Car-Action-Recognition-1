"""
Video data augmentation and preprocessing transforms.

Supports both spatial and temporal augmentations for video data.
"""

import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from typing import List, Tuple, Optional
import random


class VideoCompose:
    """Compose multiple video transforms."""
    
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            video = transform(video)
        return video


class VideoToTensor:
    """
    Convert video from numpy array (T, H, W, C) to torch tensor (C, T, H, W).
    Also converts from [0, 255] to [0.0, 1.0].
    """
    
    def __call__(self, video: np.ndarray) -> torch.Tensor:
        """
        Args:
            video: numpy array (T, H, W, C) in [0, 255]
        
        Returns:
            torch tensor (C, T, H, W) in [0.0, 1.0]
        """
        # Convert to float and normalize to [0, 1]
        video = video.astype(np.float32) / 255.0
        
        # Transpose from (T, H, W, C) to (C, T, H, W)
        video = torch.from_numpy(video).permute(3, 0, 1, 2)
        
        return video


class VideoNormalize:
    """Normalize video with mean and standard deviation."""
    
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)
    
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: torch tensor (C, T, H, W)
        
        Returns:
            Normalized video
        """
        return (video - self.mean) / self.std


class VideoResize:
    """Resize video frames to target size."""
    
    def __init__(self, size: Tuple[int, int]):
        """
        Args:
            size: Target size (H, W)
        """
        self.size = size
    
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: torch tensor (C, T, H, W)
        
        Returns:
            Resized video
        """
        C, T, H, W = video.shape
        
        # Resize each frame
        resized_frames = []
        for t in range(T):
            frame = video[:, t, :, :]  # (C, H, W)
            resized_frame = F.resize(frame, self.size)
            resized_frames.append(resized_frame)
        
        return torch.stack(resized_frames, dim=1)  # (C, T, H, W)


class VideoRandomResizedCrop:
    """Random resized crop for video."""
    
    def __init__(
        self,
        size: Tuple[int, int],
        scale: Tuple[float, float] = (0.8, 1.0),
        ratio: Tuple[float, float] = (3./4., 4./3.),
    ):
        self.size = size
        self.scale = scale
        self.ratio = ratio
    
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: torch tensor (C, T, H, W)
        
        Returns:
            Cropped and resized video
        """
        C, T, H, W = video.shape
        
        # Get crop parameters (same for all frames)
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            video[:, 0, :, :],  # Use first frame to get params
            self.scale,
            self.ratio
        )
        
        # Apply crop to all frames
        cropped_frames = []
        for t in range(T):
            frame = video[:, t, :, :]
            cropped_frame = F.resized_crop(frame, i, j, h, w, self.size)
            cropped_frames.append(cropped_frame)
        
        return torch.stack(cropped_frames, dim=1)


class VideoCenterCrop:
    """Center crop for video."""
    
    def __init__(self, size: Tuple[int, int]):
        self.size = size
    
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: torch tensor (C, T, H, W)
        
        Returns:
            Center cropped video
        """
        C, T, H, W = video.shape
        
        cropped_frames = []
        for t in range(T):
            frame = video[:, t, :, :]
            cropped_frame = F.center_crop(frame, self.size)
            cropped_frames.append(cropped_frame)
        
        return torch.stack(cropped_frames, dim=1)


class VideoRandomHorizontalFlip:
    """Random horizontal flip for video."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: torch tensor (C, T, H, W)
        
        Returns:
            Flipped video (or original)
        """
        if random.random() < self.p:
            return torch.flip(video, dims=[3])  # Flip width dimension
        return video


class VideoColorJitter:
    """Color jitter for video (brightness, contrast, saturation, hue)."""
    
    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
    ):
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
    
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: torch tensor (C, T, H, W)
        
        Returns:
            Color jittered video
        """
        C, T, H, W = video.shape
        
        # Apply same color jitter to all frames
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            transforms.ColorJitter.get_params(
                self.color_jitter.brightness,
                self.color_jitter.contrast,
                self.color_jitter.saturation,
                self.color_jitter.hue,
            )
        
        jittered_frames = []
        for t in range(T):
            frame = video[:, t, :, :]
            # Apply transforms in the order specified by fn_idx
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    frame = F.adjust_brightness(frame, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    frame = F.adjust_contrast(frame, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    frame = F.adjust_saturation(frame, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    frame = F.adjust_hue(frame, hue_factor)
            jittered_frames.append(frame)
        
        return torch.stack(jittered_frames, dim=1)


class KIRToRGB:
    """Convert single-channel KIR (infrared) to pseudo-3-channel."""
    
    def __call__(self, video: np.ndarray) -> np.ndarray:
        """
        Args:
            video: numpy array (T, H, W) or (T, H, W, 1)
        
        Returns:
            Pseudo-3-channel video (T, H, W, 3)
        """
        if video.ndim == 3:
            # (T, H, W) -> (T, H, W, 1)
            video = video[..., np.newaxis]
        
        if video.shape[-1] == 1:
            # Repeat channel dimension
            video = np.repeat(video, 3, axis=-1)
        
        return video


def get_train_transforms(config: dict) -> VideoCompose:
    """
    Get training transforms from config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Composed video transforms
    """
    aug_config = config.get('augmentation', {})
    norm_config = config.get('normalize', {})
    resolution = config.get('resolution', 224)
    
    transforms_list = [
        VideoToTensor(),
        VideoRandomResizedCrop(
            size=(resolution, resolution),
            scale=tuple(aug_config.get('scale_range', [0.8, 1.0])),
        ),
        VideoRandomHorizontalFlip(p=aug_config.get('horizontal_flip', 0.5)),
    ]
    
    # Add color jitter if enabled
    if aug_config.get('color_jitter', True):
        transforms_list.append(
            VideoColorJitter(
                brightness=aug_config.get('brightness', 0.2),
                contrast=aug_config.get('contrast', 0.2),
                saturation=aug_config.get('saturation', 0.2),
                hue=aug_config.get('hue', 0.1),
            )
        )
    
    return VideoCompose(transforms_list)


def get_val_transforms(config: dict) -> VideoCompose:
    """
    Get validation/test transforms from config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Composed video transforms
    """
    resolution = config.get('resolution', 224)
    
    transforms_list = [
        VideoToTensor(),
        VideoResize((resolution, resolution)),
        VideoCenterCrop((resolution, resolution)),
    ]
    
    return VideoCompose(transforms_list)


def get_normalization(config: dict, modality: str) -> VideoNormalize:
    """
    Get normalization transform for specific modality.
    
    Args:
        config: Configuration dictionary
        modality: 'rgb' or 'kir'
    
    Returns:
        Normalization transform
    """
    norm_config = config.get('normalize', {})
    
    if modality == 'rgb':
        mean = norm_config.get('rgb_mean', [0.485, 0.456, 0.406])
        std = norm_config.get('rgb_std', [0.229, 0.224, 0.225])
    elif modality == 'kir':
        mean = norm_config.get('kir_mean', [0.5, 0.5, 0.5])
        std = norm_config.get('kir_std', [0.5, 0.5, 0.5])
    else:
        raise ValueError(f"Unknown modality: {modality}")
    
    return VideoNormalize(mean, std)
