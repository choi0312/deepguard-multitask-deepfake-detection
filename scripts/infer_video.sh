#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${1:-outputs/deepguard_resnet34_multitask/checkpoints/best.pt}
VIDEO_PATH=${2:-samples/sample.mp4}

python -m src.inference.predict_video \
  --config configs/default.yaml \
  --checkpoint "$CHECKPOINT" \
  --video_path "$VIDEO_PATH"
