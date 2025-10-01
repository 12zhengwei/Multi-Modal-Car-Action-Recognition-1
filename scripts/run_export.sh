#!/bin/bash
# Model export script

# Usage: bash scripts/run_export.sh configs/fusion_meta.yaml outputs/checkpoints/best.pth

CONFIG=${1:-configs/default.yaml}
CHECKPOINT=${2:-outputs/default/checkpoints/best.pth}
FORMAT=${3:-onnx}  # onnx or torchscript

echo "Exporting model with config: $CONFIG"
echo "Using checkpoint: $CHECKPOINT"
echo "Format: $FORMAT"

if [ "$FORMAT" = "onnx" ]; then
    python engine/export_onnx.py \
        --config $CONFIG \
        --checkpoint $CHECKPOINT \
        --output model.onnx \
        --dynamic_batch
elif [ "$FORMAT" = "torchscript" ]; then
    python engine/export_torchscript.py \
        --config $CONFIG \
        --checkpoint $CHECKPOINT \
        --output model.torchscript \
        --method trace
else
    echo "Unknown format: $FORMAT"
    echo "Use 'onnx' or 'torchscript'"
    exit 1
fi

echo "Export complete!"
