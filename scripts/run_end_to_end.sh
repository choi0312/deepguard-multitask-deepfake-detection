#!/usr/bin/env bash
set -e

python -m src.data.build_manifest \
  --data_root data/processed_frames \
  --output_dir data/manifests \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --split_by video_id

python -m src.training.train --config configs/default.yaml

python -m src.evaluation.evaluate \
  --config configs/default.yaml \
  --checkpoint outputs/deepguard_resnet34_multitask/checkpoints/best.pt \
  --split test
