# 2주차 — AlexNet으로 배우는 CNN과 논문 리딩

2026-09-22 개정 · **40페이지 · 90분**. Python·PyTorch 기초와 1주차 CNN 개념을 배운 참가자를 대상으로 합니다. 코딩 없이 논문 읽기·개념 설명·실험 분석·개인 정리로 진행합니다.

- [발표 PDF](lecture.pdf)
- [웹 발표 자료](../cnn_concept_web/index.html?week=2)
- [페이지별 리더 가이드](../cnn_concept_web/CNN_2주차_리더가이드.md)
- [나의 논문 읽기 기록지](논문_읽기_기록지.md)
- [원문 PDF · NeurIPS](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

## 읽을 논문

Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton (2012), *ImageNet Classification with Deep Convolutional Neural Networks*.

초록과 Introduction에서 기존 문제·필요한 조건·해결책을 찾습니다. 이어 문제 정의를 재구성하고 CNN의 설계 원리를 개념으로 먼저 설명한 다음 필요한 수식을 읽습니다. 원문 그래프·표·시각화를 분석하고 Discussion을 읽은 뒤, 네 가지 질문으로 스스로 정리합니다.

이 논문에는 독립된 **Problem statement** 절이 없습니다. §1·§2·§3.5·§5에서 관련 정보를 모읍니다. 마지막 본문 절의 제목은 **Discussion**입니다.

## 90분 진행표

| 시간 | 페이지 | 내용 |
|---|---|---|
| 00–08 | 1–4 | 주장·근거·질문 메모, 제목·목차에서 읽기 경로 찾기 |
| 08–20 | 5–9 | 초록·서론: 기존 문제 → 필요한 것 → 제안 |
| 20–30 | 10–13 | 입력·출력·목표·제약·데이터·Top-1/Top-5 |
| 30–58 | 14–25 | 필터·채널·공유·계층·AlexNet 구조·ReLU·pooling·LRN·증강·Dropout |
| 58–65 | 26–28 | 개념 이후 수식: 합성곱, softmax/손실, 원문 LRN |
| 65–80 | 29–35 | Figure 1, Tables 1–2, 구성 요소 비교, Figure 4 |
| 80–83 | 36 | Discussion: 지지된 주장·범위·남은 질문 |
| 83–90 | 37–40 | 네 가지 개인 정리, 예시, 다음 논문에 적용 |

9페이지는 짝 설명, 32페이지는 표의 주장 판별, 38페이지는 개인 기록 활동입니다. 코딩 실습 시간을 추가로 합산하지 않습니다. 웹의 32페이지 해설은 클릭해 펼칩니다. PDF에는 해설이 함께 있으므로 질문부터 읽게 한 뒤 해설합니다.

## 원문 근거 사용

Figures 1–4, Tables 1–2, LRN 수식과 제목·짧은 구절을 공식 PDF에서 직접 캡처했습니다. 각 캡처에 원문 페이지·번호·링크가 있으며, 한국어 해설과 자체 제작 개념도는 구별합니다. [캡처 기록](../cnn_concept_web/assets/alexnet/provenance.json)에 원문 해시·좌표·이미지 해시를 보관합니다.

2010/2012, validation/test, Top-1/Top-5, 단일 모델/앙상블을 구분합니다. 특히 15.3%는 추가 사전학습 모델을 포함한 앙상블의 Top-5 시험 오류율입니다. CNN 구조 자체를 처음 발명한 논문으로 소개하지 않습니다.

## 기존 노트북과 PPT — 별도 선택 확장

- [concept_practice.ipynb](concept_practice.ipynb): 합성 데이터의 배경 지름길을 비교하는 기존 실습. **AlexNet 재현 실험이 아닙니다.** 원문을 읽은 후 통제 실험·일반화 질문을 더 탐구할 때 사용합니다. 준비 0–2절, 관찰 3–5절, 확장 6–7절이며 현재 PDF 페이지와 일대일 대응하지 않습니다.
- [Colab에서 기존 확장 실습 열기](https://colab.research.google.com/github/AIM-kookmin/basic_study/blob/main/week02_cnn_advanced/concept_practice.ipynb)
- `practice.ipynb`, `homework_optional.ipynb`: CIFAR10 증강·잔차 연결 등의 별도 실습·자율 과제. CIFAR10 최초 약170MB 다운로드가 필요합니다.
- `lecture.pptx`, `leader_guide.md`: 이전 코드 중심 판본. 새 PDF의 페이지 번호·내용과 다릅니다.
- `leader_solution.ipynb`: 기존 자율 과제 참고 구현.

원문 분석 수업에서 ImageNet 전체 학습이나 AlexNet 성능 재현을 요구하지 않습니다.
