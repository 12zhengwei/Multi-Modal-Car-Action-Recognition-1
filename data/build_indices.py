"""
Build indices for the multi-modal action recognition dataset.

This script scans the dataset directory structure and generates CSV index files
for training, validation, and testing sets. It also creates a class-to-ID mapping.

Directory structure expected:
    root_dir/split_x/video_type/modality/class_name/video.mp4

Where:
    - split_x: split0, split1, split2
    - video_type: video_train, video_val, video_test
    - modality: rgb, kir
    - class_name: one of 34 action classes
    - video.mp4: actual video file
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm


# 34 action classes (preserve exact naming with spaces and underscores)
ACTION_CLASSES = [
    "closing bottle",
    "closing door inside",
    "closing door outside",
    "closing laptop",
    "drinking",
    "eating",
    "entering car",
    "exiting_car",
    "fastening seat belt",
    "fetching an object",
    "interacting with phone",
    "looking or moving around",
    "opening_backpack",
    "opening bottle",
    "opening door inside",
    "opening door outside",
    "opening_laptop",
    "placing an object",
    "preparing_food",
    "pressing automation button",
    "putting laptop into backpack",
    "putting_on jacket",
    "putting_on_sunglasses",
    "reading_magazine",
    "reading_newspaper",
    "sitting_still",
    "taking_laptop from backpack",
    "taking_off jacket",
    "taking off sunglasses",
    "talking_on_phone",
    "unfastening seat belt",
    "using_multimedia display",
    "working_on_ laptop",
    "writing",
]


def scan_videos(root_dir: str, split: str, video_type: str) -> List[Dict]:
    """
    Scan videos in a specific split and video type.
    
    Args:
        root_dir: Root directory of the dataset
        split: Split name (e.g., 'split0')
        video_type: Video type ('video_train', 'video_val', 'video_test')
    
    Returns:
        List of dictionaries containing video information
    """
    videos = []
    split_path = Path(root_dir) / split / video_type
    
    if not split_path.exists():
        print(f"Warning: {split_path} does not exist")
        return videos
    
    # Iterate through modalities (rgb, kir)
    for modality in ['rgb', 'kir']:
        modality_path = split_path / modality
        if not modality_path.exists():
            print(f"Warning: {modality_path} does not exist")
            continue
        
        # Iterate through class directories
        for class_dir in modality_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            
            # Check if class name is valid
            if class_name not in ACTION_CLASSES:
                print(f"Warning: Unknown class '{class_name}' in {class_dir}")
                continue
            
            # Get class ID
            class_id = ACTION_CLASSES.index(class_name)
            
            # Iterate through video files
            for video_file in class_dir.glob('*.mp4'):
                video_info = {
                    'path': str(video_file.relative_to(root_dir)),
                    'label_name': class_name,
                    'label_id': class_id,
                    'modality': modality,
                    'split': split,
                }
                videos.append(video_info)
    
    return videos


def build_indices(root_dir: str, out_dir: str, splits: List[str] = None):
    """
    Build index files for all splits and video types.
    
    Args:
        root_dir: Root directory of the dataset
        out_dir: Output directory for index files
        splits: List of splits to process (default: ['split0', 'split1', 'split2'])
    """
    if splits is None:
        splits = ['split0', 'split1', 'split2']
    
    # Create output directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Build class-to-ID mapping
    class_to_id = {class_name: idx for idx, class_name in enumerate(ACTION_CLASSES)}
    
    # Save class mapping
    class_mapping_path = out_path / 'class_to_id.json'
    with open(class_mapping_path, 'w') as f:
        json.dump(class_to_id, f, indent=2)
    print(f"Saved class mapping to {class_mapping_path}")
    
    # Process each split
    for split in splits:
        print(f"\nProcessing {split}...")
        
        # Process each video type
        for video_type, output_name in [
            ('video_train', 'train'),
            ('video_val', 'val'),
            ('video_test', 'test'),
        ]:
            print(f"  Scanning {video_type}...")
            videos = scan_videos(root_dir, split, video_type)
            
            if len(videos) == 0:
                print(f"  Warning: No videos found for {split}/{video_type}")
                continue
            
            # Create DataFrame
            df = pd.DataFrame(videos)
            
            # Save to CSV
            csv_path = out_path / f'{split}_{output_name}.csv'
            df.to_csv(csv_path, index=False)
            
            # Print statistics
            print(f"  Found {len(videos)} videos")
            print(f"  RGB: {len(df[df['modality'] == 'rgb'])}, "
                  f"KIR: {len(df[df['modality'] == 'kir'])}")
            print(f"  Classes: {df['label_name'].nunique()}")
            print(f"  Saved to {csv_path}")
    
    print(f"\n✓ Index building complete!")
    print(f"  Class mapping: {class_mapping_path}")
    print(f"  Index files: {out_path}/*.csv")


def main():
    parser = argparse.ArgumentParser(
        description='Build indices for multi-modal action recognition dataset'
    )
    parser.add_argument(
        '--root_dir',
        type=str,
        required=True,
        help='Root directory of the dataset'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='cache',
        help='Output directory for index files (default: cache)'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['split0', 'split1', 'split2'],
        help='Splits to process (default: split0 split1 split2)'
    )
    
    args = parser.parse_args()
    
    # Validate root directory
    if not os.path.exists(args.root_dir):
        raise ValueError(f"Root directory does not exist: {args.root_dir}")
    
    build_indices(args.root_dir, args.out_dir, args.splits)


if __name__ == '__main__':
    main()
