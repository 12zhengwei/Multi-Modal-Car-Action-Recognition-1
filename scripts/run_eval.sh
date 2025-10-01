#!/bin/bash
# Evaluation script for multi-modal action recognition

# Usage: bash scripts/run_eval.sh configs/fusion_meta.yaml outputs/split0_meta_rgbkir/checkpoints/best.pth

CONFIG=${1:-configs/default.yaml}
CHECKPOINT=${2:-outputs/default/checkpoints/best.pth}
SPLIT=${3:-val}
GPU=${4:-0}

echo "Evaluating with config: $CONFIG"
echo "Using checkpoint: $CHECKPOINT"
echo "Split: $SPLIT"
echo "Using GPU: $GPU"

CUDA_VISIBLE_DEVICES=$GPU python engine/validate.py \
    --config $CONFIG \
    --checkpoint $CHECKPOINT \
    --split $SPLIT \
    --num_clips 4
