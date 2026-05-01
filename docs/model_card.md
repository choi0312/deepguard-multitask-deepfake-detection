# Model Card: DeepGuard

## Model Summary

DeepGuard는 딥페이크 영상의 Real/Fake 여부와 조작 기법을 동시에 예측하기 위한 멀티태스크 딥러닝 모델입니다.

## Architecture

- Backbone: ResNet34
- Head 1: Binary classification head
- Head 2: Method classification head

## Input

- RGB face frame image
- Default image size: 224 x 224

## Output

- Real/Fake probability
- Manipulation method probability

## Limitations

- 데이터셋 분포에 민감함
- 조작 기법 분류는 추가 개선 필요
- 실제 환경 적용 전 외부 데이터셋 검증 필요
