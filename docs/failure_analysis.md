# Failure Analysis: 조작 기법 분류 성능 한계 분석

## 1. 관찰 결과

DeepGuard의 실험 결과 Real/Fake 탐지는 높은 수준의 성능을 보였으나, 조작 기법 분류(Task 2)는 상대적으로 낮은 성능을 보였습니다. 이는 단순 구현 오류라기보다, 데이터 특성과 멀티태스크 학습 구조에서 비롯된 문제로 해석할 수 있습니다.

## 2. 원인 1: 조작 기법 간 시각적 유사성

Deepfakes, FaceSwap, Face2Face, NeuralTextures는 모두 얼굴 영역을 중심으로 합성 흔적을 발생시킵니다. 따라서 저해상도 또는 압축된 프레임에서는 기법별 artifact가 명확히 분리되지 않을 수 있습니다.

## 3. 원인 2: 멀티태스크 학습 간섭

공유 백본은 Real/Fake 구분에 유리한 특징을 먼저 학습할 가능성이 높습니다. 이때 모든 fake 샘플에 공통적으로 나타나는 특징은 Task 1에는 유용하지만, fake 기법 간 차이를 구분하는 Task 2에는 충분하지 않을 수 있습니다.

## 4. 원인 3: Fake-only Loss 구조

Task 2의 손실은 fake 샘플에 대해서만 계산됩니다. 이로 인해 method head는 binary head에 비해 상대적으로 적은 학습 신호를 받게 되며, 전체 최적화 과정에서 Task 1이 더 큰 영향을 미칠 수 있습니다.

## 5. 향후 개선 방향

- 2-stage model: Real/Fake 탐지와 조작 기법 분류를 분리
- Frequency-domain feature: FFT, DCT 기반 artifact 표현 추가
- Facial landmark feature: 얼굴 기하학 정보를 보조 입력으로 활용
- Task-specific branch: 공유 백본 이후 Task 2 전용 branch 추가
- Multi-task optimization: uncertainty weighting 또는 gradient balancing 적용
