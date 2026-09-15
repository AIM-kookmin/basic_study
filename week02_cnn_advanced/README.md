# 2주차 — 잘 맞혔다고, 잘 본 것일까?

2026-09-15 개념 중심 개정판. **28페이지 · 90분**, 코딩 없이 진행할 수 있습니다.

## 개정 수업 자료와 목표

- [개념 웹북](../cnn_concept_web/index.html?week=2) · [2주차 PDF](lecture.pdf)
- [페이지별 리더 해설과 활동 답안](../cnn_concept_web/CNN_2주차_리더가이드.md)
- [전체 웹북 안내](../cnn_concept_web/README.md)

정답을 맞힌 분류기가 어떤 단서를 보았는지 묻습니다. 형태·질감·배경의 차이에서 출발해 데이터 수집, 증강, 모델 구조와 일반화 평가로 연결합니다.

- 형태·질감·배경 지름길이 분류에 미치는 영향을 설명한다.
- 증강의 의미와 라벨을 보존하지 못하는 변환을 구분한다.
- 등변성·불변성, 잔차 연결, 정규화의 역할을 설명한다.
- 정확도·손실·혼동 행렬을 함께 보고 실패 원인을 가설로 만든다.
- 시각화의 한계를 이해하고 단서를 바꾸는 통제 실험을 설계한다.

PDF 2페이지의 90분 진행표를 따릅니다. 23–24페이지에서는 사진 40장으로 컵/병 분류기의 수집·분할·평가 계획을 설계합니다(15분). 웹 8페이지에서는 이동·반전·가림으로 남는 단서를 비교합니다.

아래는 **선택 코딩 실습 및 이전 PPT 판본** 안내입니다. `lecture.pptx`와 `leader_guide.md`는 새 PDF의 페이지와 대응하지 않습니다. 개념 강의 출처는 새 PDF 마지막 페이지에 있습니다.

## 선택 코딩 실습 목표
- CIFAR10에서 증강과 정규화의 역할을 구분한다.
- Residual block을 구현하고 공정한 비교 실험을 한다.
- 오분류·클래스별 성능·일반화 간격으로 결과를 해석한다.

## 준비와 운영
1주차 CNN과 학습 루프를 복습한다. CIFAR10은 최초 약 170MB 다운로드가 필요하다. 학습 데이터 일부와 작은 모델로 시작하며 고성능 벤치마크 재현을 목표로 하지 않는다.

개념 수업 후 별도로 진행하는 선택 실습입니다. 실습 노트북은 위에서 아래로 실행하며, 과제는 자율입니다.

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
