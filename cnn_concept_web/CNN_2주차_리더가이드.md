# CNN 2주차 · AlexNet 논문 리딩 리더 가이드

2026-09-23 개정 · 22페이지 · 90분. 2–7페이지는 CNN 기초 복습 6페이지입니다. 이후 원문의 문제·구조·학습·실험·Discussion을 따라 읽습니다.

진행: 소개 2분 → 복습 18분 → 문제 정의 11분 → 구조·학습 28분 → 실험 21분 → Discussion·개인 기록 10분.

슬라이드는 핵심 근거만 싣고, 그림·표를 짚으며 설명합니다. 실습 노트북은 별도 선택 활동입니다.

원문에는 독립된 Problem statement 절이 없습니다. 입력·출력·목표를 관련 절에서 재구성합니다. 마지막 절의 실제 제목은 Discussion입니다.

22페이지에서 개인 기록 2분, 공유 2분, 마무리 1분을 배정합니다.

## 01. AlexNet (2012)
CNN의 구조를 복습한 뒤 원문의 문제, 방법, 실험을 연결해 읽는다.


핵심: CNN의 구조를 복습한 뒤 원문의 문제, 방법, 실험을 연결해 읽는다.

시간: 00–02분 / 근거: [논문 제목 · pp. 1–8](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=1)

## 02. 이미지 분류의 입력과 출력
분류 모델의 입력은 픽셀 배열이고 출력은 클래스별 점수다.


핵심: 분류 모델의 입력은 픽셀 배열이고 출력은 클래스별 점수다.

시간: 02–05분 / 근거: [복습 · §§2, 3.5](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=2)

## 03. 합성곱: 같은 필터를 여러 위치에 적용
국소 연결과 가중치 공유는 이미지에 대한 가정을 모델 구조에 넣는다.


핵심: 국소 연결과 가중치 공유는 이미지에 대한 가정을 모델 구조에 넣는다.

시간: 05–08분 / 근거: [복습 · §1, §3.5](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=1)

## 04. 채널과 깊이: 여러 반응을 조합
입력 채널은 정보의 종류, 출력 채널 수는 필터 수와 연결된다.


핵심: 입력 채널은 정보의 종류, 출력 채널 수는 필터 수와 연결된다.

시간: 08–11분 / 근거: [복습 · §§1, 3.5](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=4)

## 05. ReLU와 pooling의 역할
ReLU와 pooling은 합성곱과 다른 역할을 하는 연산이다.


핵심: ReLU와 pooling은 합성곱과 다른 역할을 하는 연산이다.

시간: 11–14분 / 근거: [복습 · §§3.1, 3.4](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=4)

## 06. 분류와 학습: 정답 확률을 높이는 과정
필터는 고정된 그림 검출기가 아니라 손실을 통해 학습되는 파라미터다.


핵심: 필터는 고정된 그림 검출기가 아니라 손실을 통해 학습되는 파라미터다.

시간: 14–17분 / 근거: [복습 · §§3.5, 5](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=4)

## 07. 일반화: 처음 보는 이미지에서도 맞히기
AlexNet의 과제는 큰 모델을 학습시키는 동시에 일반화 성능을 확보하는 것이다.


핵심: AlexNet의 과제는 큰 모델을 학습시키는 동시에 일반화 성능을 확보하는 것이다.

시간: 17–20분 / 근거: [복습 · §§2, 4, 5](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=2)

## 08. Abstract · Introduction
초록과 서론을 실제 원문에서 읽을 시간을 약 1분 둔다. “데이터를 늘리면 끝인가?”에서 모델 용량·계산·과적합의 연결을 설명한다.


핵심: 기여는 CNN의 최초 발명이 아니라 대규모 이미지 분류에서 깊은 CNN의 성능을 실증한 데 있다.

시간: 20–26분 / 근거: [Abstract · §1 · pp. 1–2](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=1)

## 09. The Dataset · 문제 정의
데이터, 목표, 평가지표를 먼저 고정해야 실험 표의 숫자를 해석할 수 있다.


핵심: 데이터, 목표, 평가지표를 먼저 고정해야 실험 표의 숫자를 해석할 수 있다.

시간: 26–31분 / 근거: [§2 · §3.5 · pp. 2, 4–5](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=2)

## 10. The Architecture
연산의 역할과 정보 흐름을 먼저 보고 채널 수와 연결 방식을 확인한다.


핵심: 연산의 역할과 정보 흐름을 먼저 보고 채널 수와 연결 방식을 확인한다.

시간: 31–37분 / 근거: [§§3.2, 3.5 · Figure 2 · p. 5](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=5)

## 11. 첫 합성곱 층은 무엇을 학습했나
필터 가중치와 입력별 특징 맵을 구분한다.


핵심: 필터 가중치와 입력별 특징 맵을 구분한다.

시간: 37–41분 / 근거: [§6.1 · Figure 3 · p. 6](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=6)

## 12. ReLU: 학습 속도의 근거
그래프의 데이터셋, 가로축, 세로축, 비교 대상을 먼저 읽는다.


핵심: 그래프의 데이터셋, 가로축, 세로축, 비교 대상을 먼저 읽는다.

시간: 41–46분 / 근거: [§3.1 · Figure 1 · p. 3](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=3)

## 13. LRN과 overlapping pooling
수식에서 x,y는 고정하고 j가 채널 방향으로 움직인다는 점을 짚는다.


핵심: 수식에서 x,y는 고정하고 j가 채널 방향으로 움직인다는 점을 짚는다.

시간: 46–51분 / 근거: [§§3.3–3.5 · p. 4](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=4)

## 14. Reducing Overfitting
학습 데이터의 변형과 모델 내부의 규제를 구분하고, 시험 시 예측 절차도 확인한다.


핵심: 학습 데이터의 변형과 모델 내부의 규제를 구분하고, 시험 시 예측 절차도 확인한다.

시간: 51–55분 / 근거: [§4 · pp. 5–6](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=5)

## 15. Details of Learning
논문 결과는 구조·데이터 처리·학습 설정·계산 자원을 함께 사용한 결과다.


핵심: 논문 결과는 구조·데이터 처리·학습 설정·계산 자원을 함께 사용한 결과다.

시간: 55–59분 / 근거: [§5 · p. 6](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=6)

## 16. ILSVRC-2010: 이전 방법과의 비교
같은 데이터와 지표를 비교하고 %와 %p를 구분한다.


핵심: 같은 데이터와 지표를 비교하고 %와 %p를 구분한다.

시간: 59–63분 / 근거: [§6 · Table 1 · p. 7](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=7)

## 17. ILSVRC-2012: 단일 모델과 앙상블
15.3%는 단일 기본 AlexNet의 시험 오류율이 아니다.


핵심: 15.3%는 단일 기본 AlexNet의 시험 오류율이 아니다.

시간: 63–68분 / 근거: [§6 · Table 2 · p. 7](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=7)

## 18. 각 구성 요소는 얼마나 기여했나
본문의 개별 비교는 근거이지만 모든 요인을 완전히 분리한 통제 실험은 아니다.


핵심: 본문의 개별 비교는 근거이지만 모든 요인을 완전히 분리한 통제 실험은 아니다.

시간: 68–72분 / 근거: [§§3.2–3.4, 7 · pp. 3–4, 8](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=4)

## 19. 예측 사례: 무엇을 맞히고 틀렸나
정성적 사례는 숫자를 보완하지만 전체 성능 통계를 대신하지 않는다.


핵심: 정성적 사례는 숫자를 보완하지만 전체 성능 통계를 대신하지 않는다.

시간: 72–76분 / 근거: [§6.1 · Figure 4 왼쪽 · p. 8](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=8)

## 20. 은닉 표현: 어떤 이미지가 가까운가
유사성은 어떤 표현에서 어떤 거리로 측정했는지에 따라 달라진다.


핵심: 유사성은 어떤 표현에서 어떤 거리로 측정했는지에 따라 달라진다.

시간: 76–80분 / 근거: [§6.1 · Figure 4 오른쪽 · p. 8](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=8)

## 21. Discussion
관찰한 실험 결과가 어디까지 주장을 지지하는지 확인하고 후속 질문을 남긴다.


핵심: 관찰한 실험 결과가 어디까지 주장을 지지하는지 확인하고 후속 질문을 남긴다.

시간: 80–85분 / 근거: [§7 · p. 8](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=8)

## 22. 읽은 뒤 남길 네 가지
주장·중요 요소·미해결 질문·후속 학습을 자신의 문장으로 남긴다.


핵심: 주장·중요 요소·미해결 질문·후속 학습을 자신의 문장으로 남긴다.

시간: 85–90분 / 근거: [논문 전체 · 개인 기록](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=1)
