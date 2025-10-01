#!/bin/bash
# Training script for multi-modal action recognition

# Single GPU training
# Usage: bash scripts/run_train.sh

CONFIG=${1:-configs/default.yaml}
GPU=${2:-0}

echo "Training with config: $CONFIG"
echo "Using GPU: $GPU"

CUDA_VISIBLE_DEVICES=$GPU python engine/train.py \
    --config $CONFIG

# Multi-GPU training with DDP
# Usage: bash scripts/run_train.sh configs/fusion_meta.yaml 0,1,2,3
#
# CUDA_VISIBLE_DEVICES=$GPU torchrun \
#     --nproc_per_node=4 \
#     --master_port=29500 \
#     engine/train.py \
#     --config $CONFIG
