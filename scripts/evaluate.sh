#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${1:-outputs/deepguard_resnet34_multitask/checkpoints/best.pt}
SPLIT=${2:-test}

python -m src.evaluation.evaluate \
  --config configs/default.yaml \
  --checkpoint "$CHECKPOINT" \
  --split "$SPLIT"
