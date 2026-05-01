# DeepGuard: 멀티태스크 기반 딥페이크 탐지 및 조작 기법 분류

> FaceForensics++ 기반의 딥페이크 진위 여부 탐지와 조작 기법 분류를 동시에 수행하는 PyTorch 기반 멀티태스크 딥러닝 프로젝트입니다.

## 1. 프로젝트 개요

**DeepGuard**는 입력 영상이 실제 영상인지 딥페이크 영상인지 판별하고, 딥페이크로 판단되는 경우 어떤 조작 기법이 사용되었는지를 추가로 분류하는 멀티태스크 딥러닝 프로젝트입니다.

본 프로젝트는 딥러닝 교과목 팀 프로젝트로 시작되었으며, GitHub 포트폴리오 제출 및 대학원 컨택에 활용할 수 있도록 코드 구조, 실험 파이프라인, 문서화 방식을 재정비한 버전입니다. 단순한 모델 학습 결과 제시에 그치지 않고, 전처리 파이프라인, 모델 구조, 평가 방식, 실패 원인 분석, 향후 개선 방향까지 포함하는 연구형 프로젝트로 정리했습니다.

## 2. 문제 정의

딥페이크 탐지는 일반적으로 실제 영상과 조작 영상을 구분하는 이진 분류 문제로 다뤄집니다. 그러나 실제 활용 환경에서는 단순히 진위 여부를 판별하는 것뿐만 아니라, 조작 영상이 어떤 방식으로 생성되었는지를 분석하는 것도 중요합니다.

본 프로젝트는 다음 두 가지 과제를 동시에 수행합니다.

| 구분 | 과제 | 설명 |
|---|---|---|
| Task 1 | Real/Fake Classification | 입력 프레임 또는 영상이 실제인지 딥페이크인지 판별 |
| Task 2 | Manipulation Method Classification | 딥페이크로 판단된 경우 사용된 조작 기법을 분류 |

Task 2의 분류 대상은 다음 네 가지 조작 기법입니다.

- Deepfakes
- FaceSwap
- Face2Face
- NeuralTextures

## 3. 전체 파이프라인

DeepGuard는 영상 단위 입력을 바로 분류하는 대신, 영상에서 프레임을 추출한 뒤 프레임 단위 예측을 수행하고 이를 영상 단위 결과로 집계하는 구조를 사용합니다.

```text
Raw Video
  -> Frame Extraction
  -> Face Detection / Alignment
  -> Crop & Resize
  -> Multi-task Model Inference
  -> Video-level Aggregation
  -> Final Decision
```

이러한 구조는 영상 내 일부 프레임에서 발생할 수 있는 오분류의 영향을 줄이고, 여러 프레임의 평균적인 예측 확률을 활용해 더 안정적인 영상 단위 판단을 가능하게 합니다.

## 4. 데이터셋 구성

본 프로젝트는 FaceForensics++ 계열 데이터셋 구조를 가정합니다. 원본 영상과 전처리된 전체 프레임은 용량 및 라이선스 문제로 저장소에 포함하지 않았습니다.

예상되는 전처리 프레임 구조는 다음과 같습니다.

```text
data/processed_frames/
├─ Real/
├─ Deepfakes/
├─ FaceSwap/
├─ Face2Face/
└─ NeuralTextures/
```

각 이미지는 다음 두 종류의 레이블을 갖습니다.

| 레이블 | 설명 |
|---|---|
| `label_rf` | Real/Fake 이진 레이블. Real=0, Fake=1 |
| `label_method` | Fake 조작 기법 레이블. Real 샘플은 -1로 처리 |

## 5. 모델 구조

모델은 하나의 공유 CNN 백본과 두 개의 분류 헤드로 구성됩니다.

```text
Input Frame
  -> Shared ResNet Backbone
  -> Feature Vector
      -> Binary Head: Real/Fake
      -> Method Head: Manipulation Method
```

구현상 기본 백본은 `ResNet34`이며, ImageNet 사전학습 가중치를 활용할 수 있도록 구성했습니다.

### 손실 함수

전체 손실은 다음과 같이 정의합니다.

```text
L_total = L_real_fake + lambda * L_method
```

| 손실 | 설명 |
|---|---|
| `L_real_fake` | 모든 샘플에 대해 계산되는 Real/Fake 이진 분류 손실 |
| `L_method` | Fake 샘플에 대해서만 계산되는 조작 기법 다중 분류 손실 |
| `lambda` | 두 task의 상대적 중요도를 조정하는 가중치 |

Real 샘플에는 조작 기법 레이블이 존재하지 않기 때문에, Task 2 손실은 Fake 샘플에 대해서만 마스킹하여 계산합니다.

## 6. 실험 결과 요약

교과목 프로젝트 실험에서는 Task 1의 진위 여부 탐지 성능이 높게 관찰되었습니다. 반면 Task 2의 조작 기법 분류는 일부 클래스 구분 신호를 확인할 수 있었으나, 조작 기법 간 시각적 유사성과 멀티태스크 학습 간섭으로 인해 추가 개선이 필요한 것으로 분석했습니다.

| 과제 | 주요 지표 | 결과 해석 |
|---|---|---|
| Task 1: Real/Fake Detection | Accuracy / F1-score | 높은 수준의 진위 판별 성능 확인 |
| Task 2: Method Classification | Macro F1 / Confusion Matrix | 일부 유의미한 분류 신호는 있으나, 향후 구조적 개선 필요 |
| Joint 5-Class Classification | Accuracy | Task 2 성능에 영향을 크게 받는 종합 지표 |

본 저장소에서는 Task 2를 완전히 해결된 문제로 제시하지 않고, **딥페이크 기법 간 미세한 차이를 어떻게 학습할 것인가**라는 후속 연구 문제로 정리했습니다.

## 7. Task 2 한계 분석

조작 기법 분류가 어려운 주요 원인은 다음과 같습니다.

1. **조작 기법 간 시각적 유사성**  
   Deepfakes, FaceSwap, Face2Face, NeuralTextures는 모두 얼굴 영역 조작을 기반으로 하므로 저해상도 프레임에서는 기법별 차이가 뚜렷하게 드러나지 않을 수 있습니다.

2. **멀티태스크 학습 간섭**  
   공유 백본은 상대적으로 쉬운 Real/Fake 구분 특징을 우선적으로 학습할 수 있습니다. 이 경우 조작 기법 간 세밀한 차이를 구분하는 특징 표현이 충분히 형성되지 않을 가능성이 있습니다.

3. **Fake-only supervision 구조**  
   Method classification loss는 Fake 샘플에 대해서만 계산되므로, Task 1에 비해 Task 2의 학습 신호가 약해질 수 있습니다.

4. **압축 및 해상도 영향**  
   영상 압축이나 프레임 해상도 저하는 조작 기법별 미세한 artifact를 약화시킬 수 있습니다.

## 8. 향후 개선 방향

Task 2의 성능을 개선하기 위해 다음 방향을 고려할 수 있습니다.

- Real/Fake 탐지 모델과 조작 기법 분류 모델을 분리한 2-stage architecture
- FFT, DCT 등 주파수 영역 특징을 활용한 보조 입력
- 얼굴 랜드마크 및 기하학적 특징 기반 auxiliary feature
- Task-specific branch 또는 attention module 추가
- FaceForensics++ 외 DFDC, Celeb-DF 등 외부 데이터셋 기반 일반화 평가
- Task balancing, uncertainty weighting, gradient surgery 등 멀티태스크 최적화 기법 적용

## 9. 저장소 구조

```text
deepguard-multitask-deepfake-detection/
├─ configs/
│  └─ default.yaml
├─ data/
│  └─ manifests/
├─ docs/
│  ├─ failure_analysis.md
│  └─ model_card.md
├─ reports/
├─ scripts/
├─ src/
│  ├─ data/
│  ├─ evaluation/
│  ├─ inference/
│  ├─ models/
│  ├─ training/
│  └─ utils/
├─ tests/
├─ README.md
├─ requirements.txt
└─ pyproject.toml
```

## 10. 실행 방법

### 1) 패키지 설치

```bash
pip install -r requirements.txt
```

### 2) 데이터 manifest 생성

```bash
python -m src.data.build_manifest \
  --data_root data/processed_frames \
  --output_dir data/manifests \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --split_by video_id
```

### 3) 모델 학습

```bash
python -m src.training.train --config configs/default.yaml
```

### 4) 평가

```bash
python -m src.evaluation.evaluate \
  --config configs/default.yaml \
  --checkpoint outputs/deepguard_resnet34_multitask/checkpoints/best.pt \
  --split test
```

### 5) 영상 단위 추론

```bash
python -m src.inference.infer_video \
  --config configs/default.yaml \
  --checkpoint outputs/deepguard_resnet34_multitask/checkpoints/best.pt \
  --video_path samples/sample.mp4
```

## 11. 주요 기여 내용

본 프로젝트에서의 주요 기여는 다음과 같습니다.

- 원본 영상으로부터 학습용 프레임을 구성하는 전처리 파이프라인 설계
- Real/Fake 및 조작 기법 분류를 위한 레이블 구조 정리
- 프레임 단위 예측과 영상 단위 aggregation 방식 정리
- 멀티태스크 모델 구조 및 학습 전략 구성
- Task 1과 Task 2의 성능 차이에 대한 원인 분석
- 조작 기법 분류 성능 개선을 위한 후속 연구 방향 제안
- 수업 제출용 코드를 GitHub 포트폴리오용 재현 가능 프로젝트 구조로 리팩토링

## 12. 포트폴리오 관점의 의의

이 프로젝트는 단순히 딥페이크 탐지 모델을 학습한 사례가 아니라, 다음 역량을 보여주는 포트폴리오 프로젝트입니다.

- 컴퓨터 비전 기반 딥러닝 모델링
- 멀티태스크 학습 설계
- 영상 데이터 전처리 및 프레임 기반 추론
- 성능 지표 기반 모델 평가
- 실패 원인 분석 및 개선 방향 도출
- 연구형 프로젝트 문서화 및 재현 가능한 코드 구조화

## 13. Responsible Use

본 프로젝트는 교육 및 연구 목적의 딥페이크 탐지 실험을 위해 작성되었습니다. 무단 감시, 개인 식별, 괴롭힘, 명예훼손, 사생활 침해 목적의 사용을 금지합니다.

## 14. License

본 저장소의 코드는 별도 명시가 없는 한 MIT License를 따릅니다. 단, 외부 데이터셋 및 사전학습 모델의 사용 조건은 각 제공처의 라이선스를 따릅니다.
