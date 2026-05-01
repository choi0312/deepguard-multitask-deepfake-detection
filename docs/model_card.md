# Model Card: DeepGuard

## Model Summary

DeepGuard는 딥페이크 영상의 Real/Fake 여부와 조작 기법을 동시에 예측하기 위한 멀티태스크 딥러닝 모델입니다.

## Intended Use

- 딥페이크 탐지 연구
- 컴퓨터 비전 수업 프로젝트
- 멀티태스크 학습 실험
- 딥페이크 포렌식 모델 프로토타입

## Not Intended Use

- 무단 감시
- 개인 식별
- 사생활 침해
- 허위 고발 또는 자동 판정 시스템

## Architecture

- Backbone: ResNet34
- Head 1: Binary classification head
- Head 2: Method classification head

## Limitations

- 데이터셋 분포에 민감함
- 조작 기법 분류는 추가 개선 필요
- 실제 서비스 적용 전 외부 데이터셋 검증 필요
