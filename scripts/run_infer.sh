#!/bin/bash
# Inference script for multi-modal action recognition

# Usage: bash scripts/run_infer.sh configs/fusion_meta.yaml outputs/checkpoints/best.pth /path/to/video.mp4

CONFIG=${1:-configs/default.yaml}
CHECKPOINT=${2:-outputs/default/checkpoints/best.pth}
VIDEO=${3:-/path/to/video.mp4}
GPU=${4:-0}

echo "Inference with config: $CONFIG"
echo "Using checkpoint: $CHECKPOINT"
echo "Video: $VIDEO"
echo "Using GPU: $GPU"

CUDA_VISIBLE_DEVICES=$GPU python engine/infer.py \
    --config $CONFIG \
    --checkpoint $CHECKPOINT \
    --video $VIDEO \
    --output inference_results \
    --visualize \
    --format json
