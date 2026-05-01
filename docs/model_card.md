# Model Card: DeepGuard Multi-Task Detector

## Intended Use

DeepGuard is intended for education, research, and defensive deepfake analysis. It estimates whether an input face image/video is real or fake and provides a tentative manipulation-method prediction for fake samples.

## Not Intended For

- Making final legal or forensic judgments without human review
- Surveillance or identity-based decision-making
- Generating or improving deepfake content

## Inputs

- Face-centered image frames, preferably 224x224 RGB
- Video files for frame extraction and aggregation inference

## Outputs

- Fake probability
- Binary real/fake prediction
- Manipulation-method probability distribution
- Video-level aggregate decision

## Limitations

- Method classification is exploratory and not production-grade
- Performance can degrade under unseen compression, resolution, camera, or dataset conditions
- The model may learn dataset artifacts rather than universal deepfake cues

## Recommended Evaluation

- Report Task 1 and Task 2 separately
- Use fake-only macro F1 for method classification
- Include confusion matrix analysis
- Evaluate at both frame and video levels
