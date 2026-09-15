# 1주차 — 본다는 것은, 분류한다는 것

2026-09-15 개념 중심 개정판. **28페이지 · 90분**, 코딩 없이 진행할 수 있습니다.

## 개정 수업 자료와 목표

- [개념 웹북](../cnn_concept_web/index.html?week=1) · [1주차 PDF](lecture.pdf)
- [페이지별 리더 해설과 활동 답안](../cnn_concept_web/CNN_1주차_리더가이드.md)
- [전체 웹북 안내](../cnn_concept_web/README.md)

## 개념 연계 실습 · 바로 실행

[concept_practice.ipynb](concept_practice.ipynb) · [Colab에서 열기](https://colab.research.google.com/github/AIM-kookmin/basic_study/blob/main/week01_cnn_basics/concept_practice.ipynb)

필터의 손계산 → 컵의 경계·ReLU·풀링 → 이동 관찰 → 작은 CNN 학습 → 학습 전후 특징 맵 → 가림 실험으로 진행합니다. 코드는 완성되어 있으며 변수와 관찰 기록을 직접 바꿉니다. 데이터는 내부에서 생성하므로 다운로드와 GPU가 필요 없습니다.

- 수업 15분: 준비 셀과 1–3절. PDF 24–25쪽의 종이 활동 대신 실행합니다.
- 확장 20–30분: 4–6절의 학습·특징 맵·오분류 분석.
- 7절: 관찰 기록 JSON 저장과 자율 실험. 저장된 실행 결과도 함께 읽을 수 있습니다.
- [실행 검증 보고서](../cnn_concept_web/실습_검증_보고서.md)

## 개념 강의 흐름

컵의 모양과 손잡이를 보고 범주를 알아보는 경험에서 시작합니다. 위치·크기·각도가 달라도 같은 사물을 알아보는 문제가 왜 어려운지 살펴본 뒤, 시각 연구에서 얻은 영감과 CNN의 설계 선택을 연결합니다.

- 인간의 분류에 쓰이는 부분·관계·맥락을 설명한다.
- 국소성, 가중치 공유, 특징의 계층이 필요한 이유를 설명한다.
- 합성곱·채널·비선형성·풀링을 그림과 작은 수치 예제로 이해한다.
- 분류 점수, 손실, 역전파가 학습 과정에서 맡는 역할을 구분한다.
- CNN의 생물학적 영감과 인간 시각을 그대로 복제하지 못하는 한계를 구분한다.

PDF 2페이지의 90분 진행표를 따릅니다. 24–25페이지에는 종이 합성곱 활동과 해설(15분)이 있고, 웹 14페이지에는 필터를 움직이는 관찰 도구가 있습니다.

아래는 **선택 코딩 실습 및 이전 PPT 판본** 안내입니다. `lecture.pptx`와 `leader_guide.md`는 새 PDF의 페이지와 대응하지 않습니다. 개념 강의 출처는 새 PDF 마지막 페이지에 있습니다.

## 선택 코딩 실습 목표
- 합성곱의 출력 크기와 파라미터 수를 계산한다.
- FashionMNIST 분류기를 구현하고 학습을 진단한다.
- 특징 맵과 오분류를 근거로 모델의 한계를 설명한다.

## 준비와 운영
Python, 텐서, nn.Module, 역전파 기초를 전제로 한다. FashionMNIST 다운로드를 수업 전에 완료한다. 실습 기본값은 일부 학습 데이터와 짧은 학습이며 목표는 최고 정확도보다 올바른 파이프라인이다.

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
