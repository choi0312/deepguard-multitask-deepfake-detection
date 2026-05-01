#!/usr/bin/env bash
set -e

python -m src.evaluation.evaluate \
  --config configs/default.yaml \
  --checkpoint outputs/deepguard_resnet34_multitask/checkpoints/best.pt \
  --split test
