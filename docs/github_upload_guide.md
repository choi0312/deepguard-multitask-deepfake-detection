# GitHub Upload Guide

## 1. Create an Empty Repository

Recommended repository name:

```text
deepguard-multitask-deepfake-detection
```

Repository description:

```text
Multi-task deepfake detection pipeline for real/fake classification and manipulation-method recognition using PyTorch.
```

Recommended topics:

```text
deepfake-detection, computer-vision, pytorch, multitask-learning, resnet, faceforensics, ai-safety
```

## 2. Push from Terminal

```bash
cd deepguard-multitask-deepfake-detection

git init
git add .
git commit -m "Refactor DeepGuard into reproducible multi-task deepfake pipeline"
git branch -M main
git remote add origin https://github.com/choi0312/deepguard-multitask-deepfake-detection.git
git push -u origin main
```

Or use:

```bash
bash scripts/push_to_github.sh choi0312 deepguard-multitask-deepfake-detection
```

## 3. Do Not Upload

- Raw FaceForensics++ videos
- Preprocessed frame folders
- `.pt`, `.pth`, `.ckpt` model checkpoints
- Colab runtime logs
- Large zip files
- Personal Google Drive paths

## 4. Upload Instead

- Clean code
- README
- Config files
- Evaluation summary images
- Small sample images only if license allows
- Report/presentation PDF if appropriate
