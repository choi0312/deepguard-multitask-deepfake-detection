#!/usr/bin/env bash
set -e

python -m src.inference.infer_video \
  --config configs/default.yaml \
  --checkpoint outputs/deepguard_resnet34_multitask/checkpoints/best.pt \
  --video_path samples/sample.mp4
