# DeepGuard: 멀티태스크 기반 딥페이크 탐지 및 조작 기법 분류

FaceForensics++ 계열 데이터셋을 기반으로 딥페이크 영상의 진위 여부를 판별하고, 조작 영상으로 판단된 경우 사용된 합성 기법을 분류하는 멀티태스크 딥러닝 프로젝트입니다.

## 1. 프로젝트 개요

DeepGuard는 입력 영상이 실제 영상인지 딥페이크 영상인지 판별하는 Real/Fake detection과, 딥페이크로 판단된 영상에 대해 사용된 조작 기법을 분류하는 manipulation method classification을 함께 수행하도록 설계했습니다.

기존의 딥페이크 탐지 문제는 대체로 실제 영상과 조작 영상을 구분하는 이진 분류 문제로 접근됩니다. 그러나 실제 분석 환경에서는 단순한 진위 판별뿐만 아니라, 어떤 방식의 조작이 이루어졌는지를 함께 파악하는 것이 후속 분석에 도움이 될 수 있습니다.

본 프로젝트는 하나의 공유 백본을 기반으로 두 개의 분류 head를 구성하고, 프레임 단위 예측 결과를 영상 단위로 집계하는 구조를 사용했습니다.

## 2. 문제 정의

본 프로젝트는 다음 두 가지 task를 동시에 다룹니다.

| 구분 | 과제 | 설명 |
|---|---|---|
| Task 1 | Real/Fake Classification | 입력 프레임 또는 영상이 실제인지 딥페이크인지 판별 |
| Task 2 | Manipulation Method Classification | 딥페이크로 판단된 경우 사용된 조작 기법을 분류 |

Task 2의 분류 대상은 FaceForensics++에서 제공하는 대표적인 네 가지 조작 기법입니다.

- Deepfakes
- FaceSwap
- Face2Face
- NeuralTextures

Real 샘플에는 조작 기법 레이블이 존재하지 않기 때문에, 조작 기법 분류 손실은 Fake 샘플에 대해서만 계산되도록 구성했습니다.

## 3. 전체 파이프라인

영상 전체를 직접 입력하는 대신, 먼저 영상에서 프레임을 추출하고 각 프레임에 대해 모델 추론을 수행한 뒤, 프레임별 결과를 평균 집계하여 영상 단위 예측을 산출했습니다.

```text
Raw Video
  -> Frame Extraction
  -> Face Detection / Alignment
  -> Crop & Resize
  -> Multi-task Model Inference
  -> Video-level Aggregation
  -> Final Decision
```

이 방식은 일부 프레임에서 발생할 수 있는 예측 노이즈를 완화하고, 여러 프레임의 확률값을 활용해 더 안정적인 영상 단위 판단을 수행하기 위한 구조입니다.

## 4. 데이터 구성

본 저장소는 FaceForensics++ 계열의 전처리된 프레임 데이터 구조를 가정합니다. 원본 영상과 전체 전처리 프레임은 용량 및 데이터셋 사용 조건을 고려하여 포함하지 않았습니다.

예상되는 디렉터리 구조는 다음과 같습니다.

```text
data/processed_frames/
├─ Real/
├─ Deepfakes/
├─ FaceSwap/
├─ Face2Face/
└─ NeuralTextures/
```

각 프레임 샘플은 두 종류의 레이블을 갖습니다.

| 레이블 | 설명 |
|---|---|
| `label_rf` | Real/Fake 이진 레이블. Real=0, Fake=1 |
| `label_method` | Fake 조작 기법 레이블. Real 샘플은 -1로 처리 |

## 5. 모델 구조

모델은 하나의 공유 CNN backbone과 두 개의 task-specific classification head로 구성됩니다.

```text
Input Frame
  -> Shared ResNet Backbone
  -> Feature Vector
      -> Binary Head: Real/Fake
      -> Method Head: Manipulation Method
```

기본 백본은 ImageNet 사전학습 가중치를 사용하는 ResNet34입니다. 공유 백본은 입력 프레임의 공통 시각 특징을 추출하고, 두 개의 head는 각각 진위 판별과 조작 기법 분류를 담당합니다.

## 6. 학습 전략

전체 손실 함수는 Real/Fake 이진 분류 손실과 조작 기법 다중 분류 손실의 가중합으로 정의했습니다.

```text
L_total = L_real_fake + lambda * L_method
```

| 손실 | 설명 |
|---|---|
| `L_real_fake` | 모든 샘플에 대해 계산되는 Real/Fake 이진 분류 손실 |
| `L_method` | Fake 샘플에 대해서만 계산되는 조작 기법 다중 분류 손실 |
| `lambda` | 두 task의 상대적 중요도를 조정하는 가중치 |

Task 1은 모든 샘플에 대해 학습되지만, Task 2는 Fake 샘플에 대해서만 유효한 레이블을 갖습니다. 따라서 학습 과정에서 `label_method >= 0`인 샘플만 선택하여 method classification loss를 계산했습니다.

## 7. 실험 결과 요약

초기 실험에서 Real/Fake detection은 안정적인 성능을 보였습니다. 반면 manipulation method classification은 일부 클래스 구분 신호가 관찰되었지만, 기법 간 혼동이 남아 있어 추가적인 구조 개선이 필요한 task로 확인되었습니다.

| 과제 | 주요 지표 | 결과 해석 |
|---|---|---|
| Task 1: Real/Fake Detection | Accuracy / F1-score | 비교적 안정적인 진위 판별 성능 확인 |
| Task 2: Method Classification | Macro F1 / Confusion Matrix | 제한적인 분류 신호는 있으나, 기법 간 혼동이 존재 |
| Joint 5-Class Classification | Accuracy | Task 2 성능에 민감하게 영향을 받는 종합 지표 |

이 결과는 딥페이크 여부를 구분하는 특징과 조작 기법 간 차이를 구분하는 특징이 반드시 동일하지 않다는 점을 보여줍니다. 특히 공유 백본 기반의 멀티태스크 구조에서는 상대적으로 쉬운 Task 1이 학습을 주도하면서 Task 2에 필요한 세밀한 표현이 충분히 분리되지 않을 수 있습니다.

## 8. Task 2 한계 분석

조작 기법 분류가 상대적으로 어려웠던 원인은 다음과 같이 정리할 수 있습니다.

### 8.1 조작 기법 간 시각적 유사성

Deepfakes, FaceSwap, Face2Face, NeuralTextures는 모두 얼굴 영역을 중심으로 합성 흔적이 발생합니다. 저해상도 또는 압축된 프레임에서는 기법별 artifact가 약해져 클래스 간 차이가 뚜렷하게 드러나지 않을 수 있습니다.

### 8.2 멀티태스크 학습 간섭

공유 백본은 Real/Fake 구분에 유리한 특징을 우선적으로 학습할 가능성이 있습니다. 이 경우 모든 Fake 샘플에 공통적인 합성 흔적은 잘 포착하지만, 각 조작 기법을 구분하는 세부적인 특징은 충분히 학습되지 않을 수 있습니다.

### 8.3 Fake-only supervision 구조

조작 기법 분류 손실은 Fake 샘플에서만 계산됩니다. 따라서 method head는 binary head보다 상대적으로 적은 학습 신호를 받게 되며, 전체 손실 최적화 과정에서 Task 1의 영향이 더 크게 작용할 수 있습니다.

### 8.4 압축 및 해상도 영향

영상 압축과 프레임 해상도 저하는 조작 기법별 미세한 artifact를 약화시킬 수 있습니다. 특히 조작 기법 분류는 Real/Fake 판별보다 더 세밀한 차이를 요구하기 때문에 이러한 영향에 더 민감할 수 있습니다.

## 9. 향후 개선 방향

Task 2의 성능을 개선하기 위해 다음과 같은 후속 실험을 고려할 수 있습니다.

- Real/Fake 탐지 모델과 조작 기법 분류 모델을 분리한 2-stage architecture
- FFT, DCT 등 주파수 영역 특징을 활용한 보조 입력
- 얼굴 랜드마크 및 기하학적 특징 기반 auxiliary feature
- 공유 백본 이후 Task 2 전용 branch 또는 attention module 추가
- FaceForensics++ 외 DFDC, Celeb-DF 등 외부 데이터셋 기반 일반화 평가
- Task balancing, uncertainty weighting, gradient surgery 등 멀티태스크 최적화 기법 적용

## 10. 저장소 구조

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

## 11. 실행 방법

### 11.1 패키지 설치

```bash
pip install -r requirements.txt
```

### 11.2 데이터 manifest 생성

```bash
python -m src.data.build_manifest \
  --data_root data/processed_frames \
  --output_dir data/manifests \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --split_by video_id
```

### 11.3 모델 학습

```bash
python -m src.training.train --config configs/default.yaml
```

### 11.4 평가

```bash
python -m src.evaluation.evaluate \
  --config configs/default.yaml \
  --checkpoint outputs/deepguard_resnet34_multitask/checkpoints/best.pt \
  --split test
```

### 11.5 영상 단위 추론

```bash
python -m src.inference.infer_video \
  --config configs/default.yaml \
  --checkpoint outputs/deepguard_resnet34_multitask/checkpoints/best.pt \
  --video_path samples/sample.mp4
```

## 12. 주요 구현 내용

- FaceForensics++ 계열 프레임 데이터 구조를 기준으로 manifest 생성
- Real/Fake 및 조작 기법 레이블을 동시에 관리하는 dataset class 구현
- ResNet34 기반 shared-backbone multi-task model 구성
- Fake 샘플에 대해서만 method classification loss를 계산하는 masked loss 구조 적용
- 프레임 단위 예측 결과를 평균 집계하는 영상 단위 inference 로직 구현
- Task별 metric 및 joint 5-class accuracy 계산 코드 구성
- 실험 한계 분석과 후속 개선 방향 문서화
