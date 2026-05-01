#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-data/processed_frames}
MANIFEST_DIR=${MANIFEST_DIR:-data/manifests}
CHECKPOINT=${CHECKPOINT:-outputs/deepguard_resnet34_multitask/checkpoints/best.pt}

if [ ! -d "$DATA_ROOT" ]; then
  echo "[ERROR] DATA_ROOT not found: $DATA_ROOT"
  echo "Expected folder structure:"
  echo "  $DATA_ROOT/Real"
  echo "  $DATA_ROOT/Deepfakes"
  echo "  $DATA_ROOT/FaceSwap"
  echo "  $DATA_ROOT/Face2Face"
  echo "  $DATA_ROOT/NeuralTextures"
  exit 1
fi

python -m src.data.build_manifest \
  --data_root "$DATA_ROOT" \
  --output_dir "$MANIFEST_DIR" \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --split_by video_id

python -m src.training.train --config configs/default.yaml

python -m src.evaluation.evaluate \
  --config configs/default.yaml \
  --checkpoint "$CHECKPOINT" \
  --split test

if [ -f samples/sample.mp4 ]; then
  python -m src.inference.predict_video \
    --config configs/default.yaml \
    --checkpoint "$CHECKPOINT" \
    --video_path samples/sample.mp4
else
  echo "[INFO] samples/sample.mp4 not found. Skipping sample inference."
fi
