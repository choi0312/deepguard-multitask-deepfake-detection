# Failure Analysis: 조작 기법 분류 성능 한계 분석

## 1. 관찰 결과

Real/Fake 탐지는 비교적 안정적인 성능을 보였으나, 조작 기법 분류는 상대적으로 낮은 성능을 보였습니다. 이는 단순 구현 오류라기보다 데이터 특성과 멀티태스크 학습 구조에서 비롯된 문제로 해석할 수 있습니다.

## 2. 조작 기법 간 시각적 유사성

Deepfakes, FaceSwap, Face2Face, NeuralTextures는 모두 얼굴 영역을 중심으로 합성 흔적을 발생시킵니다. 따라서 저해상도 또는 압축된 프레임에서는 기법별 artifact가 명확히 분리되지 않을 수 있습니다.

## 3. 멀티태스크 학습 간섭

공유 백본은 Real/Fake 구분에 유리한 특징을 먼저 학습할 가능성이 높습니다. 이때 모든 Fake 샘플에 공통적으로 나타나는 특징은 Task 1에는 유용하지만, Fake 기법 간 차이를 구분하는 Task 2에는 충분하지 않을 수 있습니다.

## 4. Fake-only loss 구조

Task 2의 손실은 Fake 샘플에 대해서만 계산됩니다. 이로 인해 method head는 binary head에 비해 상대적으로 적은 학습 신호를 받게 되며, 전체 최적화 과정에서 Task 1이 더 큰 영향을 미칠 수 있습니다.

## 5. 개선 방향

- Real/Fake 탐지와 조작 기법 분류를 분리한 2-stage 구조
- FFT, DCT 기반 주파수 영역 feature 추가
- 얼굴 landmark 또는 geometry feature 추가
- Task 2 전용 branch 또는 attention module 추가
- 외부 데이터셋 기반 일반화 성능 평가
