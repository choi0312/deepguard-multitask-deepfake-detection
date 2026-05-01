# DeepGuard: Multi-Task Deepfake Detection & Manipulation Method Classification

DeepGuard is a portfolio-ready deep learning project for detecting deepfake videos and classifying the manipulation method used to generate them.

The project is built around a multi-task learning setting:

- **Task 1:** Real/Fake binary classification
- **Task 2:** Manipulation-method classification among Deepfakes, FaceSwap, Face2Face, and NeuralTextures

## 1. Project Overview

As deepfake generation techniques become increasingly realistic, simple real/fake detection is not always sufficient. DeepGuard extends the detection pipeline by attempting to identify the manipulation method behind fake videos.

The system follows a practical video-forensics pipeline:

```text
Raw Video
  -> Frame Extraction
  -> Face Detection / Alignment
  -> Crop & Resize
  -> Multi-task Model Inference
  -> Video-level Aggregation
  -> Final Decision
```

## 2. Key Features

- PyTorch-based multi-task deep learning pipeline
- Shared CNN backbone with task-specific classification heads
- Real/Fake detection and manipulation-method classification
- Frame-level inference and video-level aggregation
- YAML-based experiment configuration
- Reproducible training, evaluation, and inference scripts
- Failure analysis for multi-task learning limitations

## 3. Dataset

This project assumes a FaceForensics++-style directory structure.

```text
data/processed_frames/
├─ Real/
├─ Deepfakes/
├─ FaceSwap/
├─ Face2Face/
└─ NeuralTextures/
```

Raw videos, processed frames, and trained checkpoints are not included in this repository due to dataset licensing and storage constraints.

## 4. Model Architecture

DeepGuard uses a shared CNN backbone and two task-specific heads:

- **Binary head:** predicts whether the input frame is real or fake
- **Method head:** predicts the manipulation method for fake samples

The total loss is defined as:

```text
L_total = L_real_fake + lambda * L_method
```

where:

- `L_real_fake`: Binary cross-entropy loss for Real/Fake classification
- `L_method`: Cross-entropy loss for manipulation-method classification
- `lambda`: task balancing coefficient

## 5. Experimental Summary

| Task | Metric | Result Summary |
|---|---:|---|
| Real/Fake Detection | Accuracy / F1 | Strong performance, above 95% in the course experiment |
| Method Classification | Macro F1 | Meaningful but still limited signal; requires further improvement |

The main finding is that the model learns robust real/fake discrimination, while manipulation-method classification remains more challenging due to visual similarity between manipulation techniques and task interference in the shared backbone.

## 6. Error Analysis

Task 2 is intentionally presented as an open improvement target rather than a fully solved problem. Main limitations include:

- Similar visual artifacts across manipulation methods
- Low-resolution or compressed face frames
- Weak supervision because method loss is computed only on fake samples
- Potential gradient interference between detection and method-classification objectives

Planned improvements:

- Two-stage architecture: detection model followed by method-classification model
- Frequency-domain features such as FFT or DCT
- Facial landmark and geometry-based auxiliary features
- Task-specific branches or attention modules
- Cross-dataset validation with DFDC or Celeb-DF

## 7. Repository Structure

```text
deepguard-multitask-deepfake-detection/
├─ configs/
├─ scripts/
├─ src/
│  ├─ data/
│  ├─ models/
│  ├─ training/
│  ├─ evaluation/
│  ├─ inference/
│  └─ utils/
├─ docs/
├─ reports/
├─ assets/
├─ samples/
└─ tests/
```

## 8. Installation

```bash
pip install -r requirements.txt
```

## 9. How to Run

Build dataset manifest:

```bash
python -m src.data.build_manifest \
  --data_root data/processed_frames \
  --output_dir data/manifests \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --split_by video_id
```

Train:

```bash
python -m src.training.train --config configs/default.yaml
```

Evaluate:

```bash
python -m src.evaluation.evaluate \
  --config configs/default.yaml \
  --checkpoint outputs/deepguard_resnet34_multitask/checkpoints/best.pt \
  --split test
```

Run video inference:

```bash
python -m src.inference.infer_video \
  --config configs/default.yaml \
  --checkpoint outputs/deepguard_resnet34_multitask/checkpoints/best.pt \
  --video_path samples/sample.mp4
```

## 10. My Contribution

- Designed and organized the preprocessing pipeline for video-to-frame conversion
- Structured labels for real/fake detection and manipulation-method classification
- Conducted preprocessing visualization and frame-level quality inspection
- Analyzed Task 1 and Task 2 evaluation results
- Interpreted the limitations of multi-task learning for manipulation-method classification
- Refactored the original course project into a reproducible portfolio-ready pipeline

## 11. Responsible Use

This repository is intended for educational and research purposes. It should not be used for surveillance, harassment, or unauthorized biometric analysis.

## 12. License

This project is released under the MIT License unless otherwise specified.
