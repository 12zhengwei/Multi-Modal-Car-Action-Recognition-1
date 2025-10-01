"""
Inference script for multi-modal action recognition.

Supports:
- Single video or batch inference
- Visualization of predictions
- Multiple output formats (JSON, CSV)
"""

import os
import sys
import argparse
from pathlib import Path
import json
import csv

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.datasets.decoding import VideoDecoder, sample_frames, pad_or_truncate_frames
from data.datasets.transforms import get_val_transforms, get_normalization, KIRToRGB
from models import build_model
from engine.utils import load_config, load_checkpoint


class VideoInference:
    """Video inference engine."""
    
    def __init__(
        self,
        model,
        config,
        class_names,
        device='cuda',
    ):
        """
        Initialize inference engine.
        
        Args:
            model: Trained model
            config: Configuration dictionary
            class_names: List of class names
            device: Device for inference
        """
        self.model = model
        self.model.eval()
        
        self.config = config
        self.class_names = class_names
        self.device = device
        
        # Setup decoder and transforms
        self.decoder = VideoDecoder(backend='auto')
        self.transform = get_val_transforms(config['data'])
        
        # Get modality and normalization
        self.modality = config['data'].get('modality', 'rgb')
        if self.modality in ['rgb', 'kir']:
            self.normalize = get_normalization(config['data'], self.modality)
        else:
            self.normalize_rgb = get_normalization(config['data'], 'rgb')
            self.normalize_kir = get_normalization(config['data'], 'kir')
        
        self.num_frames = config['data'].get('num_frames', 16)
        self.kir_to_rgb = KIRToRGB()
    
    def preprocess_video(self, video_path, is_kir=False):
        """
        Preprocess video for inference.
        
        Args:
            video_path: Path to video file
            is_kir: Whether this is a KIR video
        
        Returns:
            Preprocessed video tensor
        """
        # Decode video
        frames, total_frames = self.decoder.decode_video(video_path)
        
        # Sample frames
        frame_indices = sample_frames(
            total_frames,
            self.num_frames,
            mode='uniform',
            temporal_stride=1,
        )
        frames, _ = self.decoder.decode_video(video_path, frame_indices=frame_indices)
        
        # Pad or truncate
        frames = pad_or_truncate_frames(frames, self.num_frames, mode='loop')
        
        # Convert KIR to RGB if needed
        if is_kir and frames.shape[-1] == 1:
            frames = self.kir_to_rgb(frames)
        
        # Apply transforms
        video = self.transform(frames)  # (C, T, H, W)
        
        # Apply normalization
        if is_kir:
            video = self.normalize_kir(video) if self.modality == 'rgb_kir' else self.normalize(video)
        else:
            video = self.normalize_rgb(video) if self.modality == 'rgb_kir' else self.normalize(video)
        
        return video
    
    @torch.no_grad()
    def predict(self, video_path, kir_path=None):
        """
        Predict action for video.
        
        Args:
            video_path: Path to RGB video (or single modality video)
            kir_path: Optional path to KIR video
        
        Returns:
            Dictionary with predictions
        """
        # Preprocess videos
        if self.modality == 'rgb_kir':
            if kir_path is None:
                raise ValueError("KIR video path required for dual modality")
            
            rgb_video = self.preprocess_video(video_path, is_kir=False)
            kir_video = self.preprocess_video(kir_path, is_kir=True)
            
            inputs = {
                'rgb': rgb_video.unsqueeze(0).to(self.device),
                'kir': kir_video.unsqueeze(0).to(self.device),
            }
        else:
            video = self.preprocess_video(
                video_path,
                is_kir=(self.modality == 'kir')
            )
            inputs = video.unsqueeze(0).to(self.device)
        
        # Forward pass
        outputs = self.model(inputs)
        probs = F.softmax(outputs, dim=1)
        
        # Get top-5 predictions
        top5_probs, top5_indices = torch.topk(probs, 5, dim=1)
        top5_probs = top5_probs[0].cpu().numpy()
        top5_indices = top5_indices[0].cpu().numpy()
        
        # Format predictions
        predictions = {
            'top1': {
                'class': self.class_names[top5_indices[0]],
                'confidence': float(top5_probs[0]),
            },
            'top5': [
                {
                    'class': self.class_names[idx],
                    'confidence': float(prob),
                }
                for idx, prob in zip(top5_indices, top5_probs)
            ],
            'all_probs': probs[0].cpu().numpy().tolist(),
        }
        
        return predictions


def visualize_prediction(video_path, predictions, output_path, class_names):
    """
    Visualize predictions on video.
    
    Args:
        video_path: Path to video file
        predictions: Prediction dictionary
        output_path: Path to save visualization
        class_names: List of class names
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Prepare text
    top1 = predictions['top1']
    text = f"{top1['class']}: {top1['confidence']*100:.1f}%"
    
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add text overlay
        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        
        # Write frame
        out.write(frame)
    
    cap.release()
    out.release()
    
    print(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Inference for multi-modal action recognition')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--video', type=str, required=True, help='Path to video file or directory')
    parser.add_argument('--kir', type=str, default=None, help='Path to KIR video (for dual modality)')
    parser.add_argument('--output', type=str, default='inference_results', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Create visualization video')
    parser.add_argument('--format', type=str, default='json', choices=['json', 'csv'],
                       help='Output format')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Build model
    print("Loading model...")
    model = build_model(config)
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    load_checkpoint(args.checkpoint, model, device=device)
    
    # Load class names
    data_config = config['data']
    class_mapping_path = Path(data_config['train_index']).parent / 'class_to_id.json'
    with open(class_mapping_path, 'r') as f:
        class_to_id = json.load(f)
    
    id_to_class = {v: k for k, v in class_to_id.items()}
    class_names = [id_to_class[i] for i in range(len(id_to_class))]
    
    # Create inference engine
    inference = VideoInference(model, config, class_names, device)
    
    # Process video(s)
    video_path = Path(args.video)
    
    if video_path.is_file():
        # Single video
        print(f"\nProcessing: {video_path}")
        predictions = inference.predict(str(video_path), args.kir)
        
        # Print results
        print("\nPrediction:")
        print(f"  Top-1: {predictions['top1']['class']} "
              f"({predictions['top1']['confidence']*100:.2f}%)")
        print("\nTop-5:")
        for i, pred in enumerate(predictions['top5'], 1):
            print(f"  {i}. {pred['class']} ({pred['confidence']*100:.2f}%)")
        
        # Save results
        if args.format == 'json':
            output_file = os.path.join(args.output, 'predictions.json')
            with open(output_file, 'w') as f:
                json.dump({str(video_path): predictions}, f, indent=2)
        else:
            output_file = os.path.join(args.output, 'predictions.csv')
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['video', 'predicted_class', 'confidence'])
                writer.writerow([
                    str(video_path),
                    predictions['top1']['class'],
                    predictions['top1']['confidence']
                ])
        
        print(f"\nResults saved to {output_file}")
        
        # Visualize
        if args.visualize:
            vis_path = os.path.join(args.output, f"{video_path.stem}_pred.mp4")
            visualize_prediction(str(video_path), predictions, vis_path, class_names)
    
    elif video_path.is_dir():
        # Batch processing
        video_files = list(video_path.glob('*.mp4'))
        print(f"\nProcessing {len(video_files)} videos...")
        
        all_predictions = {}
        
        for vf in tqdm(video_files):
            try:
                predictions = inference.predict(str(vf))
                all_predictions[str(vf)] = predictions
            except Exception as e:
                print(f"Error processing {vf}: {str(e)}")
        
        # Save results
        if args.format == 'json':
            output_file = os.path.join(args.output, 'predictions.json')
            with open(output_file, 'w') as f:
                json.dump(all_predictions, f, indent=2)
        else:
            output_file = os.path.join(args.output, 'predictions.csv')
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['video', 'predicted_class', 'confidence'])
                for vf, pred in all_predictions.items():
                    writer.writerow([
                        vf,
                        pred['top1']['class'],
                        pred['top1']['confidence']
                    ])
        
        print(f"\nResults saved to {output_file}")
    
    else:
        raise ValueError(f"Invalid video path: {video_path}")
    
    print("\n✓ Inference complete!")


if __name__ == '__main__':
    main()
