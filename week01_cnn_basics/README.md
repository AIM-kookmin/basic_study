# 1주차 — CNN ① 이미지에서 패턴 찾기

## 목표
- 합성곱의 출력 크기와 파라미터 수를 계산한다.
- FashionMNIST 분류기를 구현하고 학습을 진단한다.
- 특징 맵과 오분류를 근거로 모델의 한계를 설명한다.

## 준비와 운영
Python, 텐서, nn.Module, 역전파 기초를 전제로 한다. FashionMNIST 다운로드를 수업 전에 완료한다. 실습 기본값은 일부 학습 데이터와 짧은 학습이며 목표는 최고 정확도보다 올바른 파이프라인이다.

수업은 PPT의 핵심 구간으로 90분입니다. 부록은 사전 읽기, 질문 대응, 추가 세션에 사용합니다. 실습 노트북은 위에서 아래로 실행하며, 과제는 자율입니다.

## 파일
- `lecture.pptx`: 발표자 노트 포함 강의안
- `practice.ipynb`: 설명·실행 코드·관찰 질문이 있는 독립 실행 실습
- `homework_optional.ipynb`: 독립 실행 준비 코드 + 단계별 자율 과제
- `leader_solution.ipynb`: 리더용 과제 해설 및 실행 가능한 참고 구현
- `leader_guide.md`: 슬라이드별 설명과 질문 답안

## 참고 문헌
- [PyTorch · 이미지 분류 튜토리얼](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Conv2d · 공식 API](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- [Deep Residual Learning · He et al.](https://arxiv.org/abs/1512.03385)