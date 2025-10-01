"""
Video decoding utilities with fallback support.

Supports both decord and torchvision.io for video decoding.
Automatically falls back to torchvision if decord is not available.
"""

import numpy as np
import torch
from typing import Optional, Tuple
import warnings


# Try to import decord
try:
    import decord
    from decord import VideoReader, cpu, gpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    warnings.warn("decord not available, falling back to torchvision")

# Try to import torchvision video
try:
    import torchvision
    from torchvision.io import read_video
    TORCHVISION_VIDEO_AVAILABLE = True
except ImportError:
    TORCHVISION_VIDEO_AVAILABLE = False
    warnings.warn("torchvision.io not available")


class VideoDecoder:
    """
    Universal video decoder with automatic fallback.
    
    Supports decord (preferred) and torchvision.io as fallback.
    """
    
    def __init__(self, backend: str = 'auto', device: str = 'cpu'):
        """
        Initialize video decoder.
        
        Args:
            backend: Decoding backend ('decord', 'torchvision', or 'auto')
            device: Device for decoding ('cpu' or 'cuda')
        """
        self.device = device
        
        # Select backend
        if backend == 'auto':
            if DECORD_AVAILABLE:
                self.backend = 'decord'
            elif TORCHVISION_VIDEO_AVAILABLE:
                self.backend = 'torchvision'
            else:
                raise RuntimeError("No video decoding backend available")
        else:
            self.backend = backend
            
        # Validate backend availability
        if self.backend == 'decord' and not DECORD_AVAILABLE:
            raise RuntimeError("decord backend requested but not available")
        if self.backend == 'torchvision' and not TORCHVISION_VIDEO_AVAILABLE:
            raise RuntimeError("torchvision backend requested but not available")
    
    def decode_video(
        self,
        video_path: str,
        num_frames: Optional[int] = None,
        frame_indices: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Decode video from file.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to decode (if frame_indices not provided)
            frame_indices: Specific frame indices to decode
        
        Returns:
            frames: Video frames as numpy array (T, H, W, C)
            total_frames: Total number of frames in the video
        """
        if self.backend == 'decord':
            return self._decode_decord(video_path, num_frames, frame_indices)
        elif self.backend == 'torchvision':
            return self._decode_torchvision(video_path, num_frames, frame_indices)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _decode_decord(
        self,
        video_path: str,
        num_frames: Optional[int] = None,
        frame_indices: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, int]:
        """Decode video using decord."""
        try:
            # Use CPU context by default (GPU context can be flaky)
            ctx = cpu(0)
            vr = VideoReader(video_path, ctx=ctx)
            total_frames = len(vr)
            
            if frame_indices is not None:
                # Clip indices to valid range
                frame_indices = np.clip(frame_indices, 0, total_frames - 1)
                frames = vr.get_batch(frame_indices.tolist()).asnumpy()
            elif num_frames is not None:
                # Uniform sampling
                frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
                frames = vr.get_batch(frame_indices.tolist()).asnumpy()
            else:
                # Get all frames
                frames = vr.get_batch(range(total_frames)).asnumpy()
            
            return frames, total_frames
            
        except Exception as e:
            raise RuntimeError(f"Failed to decode video {video_path}: {str(e)}")
    
    def _decode_torchvision(
        self,
        video_path: str,
        num_frames: Optional[int] = None,
        frame_indices: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, int]:
        """Decode video using torchvision."""
        try:
            # Read entire video
            video, audio, info = read_video(video_path, pts_unit='sec')
            
            # Convert from (T, H, W, C) torch tensor to numpy
            frames = video.numpy()
            total_frames = len(frames)
            
            if frame_indices is not None:
                # Clip indices to valid range
                frame_indices = np.clip(frame_indices, 0, total_frames - 1)
                frames = frames[frame_indices]
            elif num_frames is not None:
                # Uniform sampling
                frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
                frames = frames[frame_indices]
            
            return frames, total_frames
            
        except Exception as e:
            raise RuntimeError(f"Failed to decode video {video_path}: {str(e)}")
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get video metadata without decoding all frames.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dictionary with video metadata (num_frames, fps, duration, etc.)
        """
        if self.backend == 'decord':
            try:
                vr = VideoReader(video_path, ctx=cpu(0))
                info = {
                    'num_frames': len(vr),
                    'fps': vr.get_avg_fps(),
                    'duration': len(vr) / vr.get_avg_fps(),
                    'height': vr[0].shape[0],
                    'width': vr[0].shape[1],
                }
                return info
            except Exception as e:
                raise RuntimeError(f"Failed to get video info {video_path}: {str(e)}")
        
        elif self.backend == 'torchvision':
            try:
                # torchvision requires reading video to get info
                video, audio, info = read_video(video_path, pts_unit='sec')
                return {
                    'num_frames': len(video),
                    'fps': info['video_fps'],
                    'duration': len(video) / info['video_fps'],
                    'height': video.shape[1],
                    'width': video.shape[2],
                }
            except Exception as e:
                raise RuntimeError(f"Failed to get video info {video_path}: {str(e)}")


def sample_frames(
    total_frames: int,
    num_frames: int,
    mode: str = 'uniform',
    temporal_stride: int = 1,
) -> np.ndarray:
    """
    Sample frame indices from a video.
    
    Args:
        total_frames: Total number of frames in the video
        num_frames: Number of frames to sample
        mode: Sampling mode ('uniform' or 'random')
        temporal_stride: Stride between consecutive frames
    
    Returns:
        Array of frame indices
    """
    if mode == 'uniform':
        # Uniform sampling across entire video
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    elif mode == 'random':
        # Random clip sampling with stride
        clip_length = num_frames * temporal_stride
        
        if clip_length > total_frames:
            # Video too short, use uniform sampling with padding
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            # Random start position
            max_start = total_frames - clip_length
            start_idx = np.random.randint(0, max_start + 1)
            indices = np.arange(start_idx, start_idx + clip_length, temporal_stride)
            indices = indices[:num_frames]
    
    else:
        raise ValueError(f"Unknown sampling mode: {mode}")
    
    return indices


def pad_or_truncate_frames(
    frames: np.ndarray,
    num_frames: int,
    mode: str = 'loop',
) -> np.ndarray:
    """
    Pad or truncate frames to desired length.
    
    Args:
        frames: Video frames (T, H, W, C)
        num_frames: Desired number of frames
        mode: Padding mode ('loop', 'repeat', 'zero')
    
    Returns:
        Padded or truncated frames
    """
    current_frames = len(frames)
    
    if current_frames == num_frames:
        return frames
    
    elif current_frames > num_frames:
        # Truncate
        return frames[:num_frames]
    
    else:
        # Pad
        pad_length = num_frames - current_frames
        
        if mode == 'loop':
            # Repeat video from beginning
            repeat_times = (num_frames + current_frames - 1) // current_frames
            frames = np.tile(frames, (repeat_times, 1, 1, 1))
            return frames[:num_frames]
        
        elif mode == 'repeat':
            # Repeat last frame
            last_frame = frames[-1:].repeat(pad_length, axis=0)
            return np.concatenate([frames, last_frame], axis=0)
        
        elif mode == 'zero':
            # Zero padding
            pad_frames = np.zeros((pad_length,) + frames.shape[1:], dtype=frames.dtype)
            return np.concatenate([frames, pad_frames], axis=0)
        
        else:
            raise ValueError(f"Unknown padding mode: {mode}")
