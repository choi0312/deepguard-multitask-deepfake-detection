# Failure Analysis: Manipulation Method Classification

## Observation

The original DeepGuard experiment achieved strong binary real/fake detection, while method classification remained substantially harder.

In portfolio language, this should be presented as:

> DeepGuard achieved reliable real/fake detection and showed preliminary class-level signal for manipulation-method recognition. However, the method-classification task remains an open improvement target due to visual similarity between fake-generation methods and multi-task optimization interference.

This is more accurate and professional than claiming that Task 2 is already fully solved.

---

## Why Task 2 Is Hard

### 1. Visual Similarity Across Manipulation Methods

Deepfakes, FaceSwap, Face2Face, and NeuralTextures all manipulate facial regions. At low resolution or under video compression, the method-specific artifacts become subtle.

### 2. Task Interference

The shared backbone can learn features that are sufficient for real/fake detection but not discriminative enough for method-level recognition. In other words, the model may learn "fake artifact present" rather than "which manipulation mechanism produced this artifact."

### 3. Fake-only Supervision

Task 2 loss is computed only for fake samples. Real samples do not contribute to the method head, so the method-classification branch receives fewer effective updates than the binary head.

### 4. Domain Mismatch

ImageNet-pretrained backbones are optimized for object recognition, not subtle forensic artifacts. Fine-grained manipulation classification may require frequency-domain features or face-specific pretraining.

---

## Improvements Added in This Refactor

- Method loss weight increased through `lambda_method`
- Focal-style cross entropy option for difficult fake samples
- Method class weighting for imbalanced fake classes
- Composite checkpoint selection using both Task 1 and Task 2 metrics
- Modular codebase for easier follow-up experiments

---

## Future Work

1. Two-stage architecture: binary detector first, method classifier second
2. Frequency branch: FFT/DCT features for artifact-sensitive representation
3. Facial landmark branch: geometry-aware manipulation cues
4. Task-specific mid-level branches to reduce interference
5. Cross-dataset generalization experiments
