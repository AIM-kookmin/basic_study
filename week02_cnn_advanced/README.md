# 2주차 — AlexNet으로 배우는 CNN과 논문 리딩

2026-09-23 개정 · **22페이지 · 90분**. Python·PyTorch 기초와 1주차 CNN 개념을 배운 참가자를 대상으로 합니다. 코딩 없이 논문 읽기·개념 설명·실험 분석·개인 정리로 진행합니다.

- [발표 PDF](lecture.pdf)
- [웹 발표 자료](../cnn_concept_web/index.html?week=2)
- [페이지별 리더 가이드](../cnn_concept_web/CNN_2주차_리더가이드.md)
- [나의 논문 읽기 기록지](논문_읽기_기록지.md)
- [원문 PDF · NeurIPS](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

## 읽을 논문

Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton (2012), *ImageNet Classification with Deep Convolutional Neural Networks*.

먼저 2–7페이지에서 CNN을 6페이지로 복습합니다. 이어 초록과 Introduction에서 기존 문제·필요한 조건·해결책을 찾습니다. 이어 문제 정의를 재구성하고 CNN의 설계 원리를 개념으로 먼저 설명한 다음 필요한 수식을 읽습니다. 원문 그래프·표·시각화를 분석하고 Discussion을 읽은 뒤, 네 가지 질문으로 스스로 정리합니다.

이 논문에는 독립된 **Problem statement** 절이 없습니다. §1·§2·§3.5·§5에서 관련 정보를 모읍니다. 마지막 본문 절의 제목은 **Discussion**입니다.

## 90분 진행표

| 시간 | 페이지 | 내용 |
|---|---|---|
| 00–02 | 1 | 논문 소개 |
| 02–20 | 2–7 | CNN 기초 복습: 입력·필터·채널·ReLU/pooling·학습·일반화 |
| 20–31 | 8–9 | 초록·서론, 데이터·입출력·학습 목표·평가지표 |
| 31–59 | 10–15 | 구조·학습된 필터·ReLU·LRN/pooling·증강/dropout·학습 조건 |
| 59–80 | 16–20 | Tables 1–2, 구성 요소 비교, Figure 4의 예측·표현 분석 |
| 80–85 | 21 | Discussion: 저자의 관찰·제약·후속 방향 |
| 85–90 | 22 | 개인 기록 2분 · 공유 2분 · 정리 1분 |

자료는 원문 그림과 표에 집중하고, 설명과 확인 질문을 리더 가이드에서 보완합니다. 8페이지에서는 초록·서론을 읽고, 10페이지에서는 구조를 설명하며, 17페이지에서는 Table 2의 비교 조건을 판정합니다. 코딩 실습은 이 90분에 추가로 합산하지 않습니다.

## 원문 근거 사용

Figures 1–4, Tables 1–2, LRN 수식과 논문 제목을 공식 PDF에서 직접 캡처했습니다. 각 캡처에 원문 페이지·번호·링크가 있으며, 한국어 해설과 자체 제작 개념도는 구별합니다. [캡처 기록](../cnn_concept_web/assets/alexnet/provenance.json)에 원문 해시·좌표·이미지 해시를 보관합니다.

2010/2012, validation/test, Top-1/Top-5, 단일 모델/앙상블을 구분합니다. 특히 15.3%는 추가 사전학습 모델을 포함한 앙상블의 Top-5 시험 오류율입니다. CNN 구조 자체를 처음 발명한 논문으로 소개하지 않습니다.

## 기존 노트북과 PPT — 별도 선택 확장

- [concept_practice.ipynb](concept_practice.ipynb): 합성 데이터의 배경 지름길을 비교하는 기존 실습. **AlexNet 재현 실험이 아닙니다.** 원문을 읽은 후 통제 실험·일반화 질문을 더 탐구할 때 사용합니다. 준비 0–2절, 관찰 3–5절, 확장 6–7절이며 현재 PDF 페이지와 일대일 대응하지 않습니다.
- [Colab에서 기존 확장 실습 열기](https://colab.research.google.com/github/AIM-kookmin/basic_study/blob/main/week02_cnn_advanced/concept_practice.ipynb)
- `practice.ipynb`, `homework_optional.ipynb`: CIFAR10 증강·잔차 연결 등의 별도 실습·자율 과제. CIFAR10 최초 약170MB 다운로드가 필요합니다.
- `lecture.pptx`, `leader_guide.md`: 이전 코드 중심 판본. 새 PDF의 페이지 번호·내용과 다릅니다.
- `leader_solution.ipynb`: 기존 자율 과제 참고 구현.

원문 분석 수업에서 ImageNet 전체 학습이나 AlexNet 성능 재현을 요구하지 않습니다.
