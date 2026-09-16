# 1주차 — 숫자 한 장에서, 분류의 원리까지

2026-09-16 흐름 개정판. **28페이지 · 90분**입니다.

## 먼저 열 자료

- [강의 PDF](lecture.pdf) · [웹 발표자료](../cnn_concept_web/index.html?week=1)
- [페이지별 리더 가이드](../cnn_concept_web/CNN_1주차_리더가이드.md)
- [MNIST 개념 실습](concept_practice.ipynb) · [Colab에서 열기](https://colab.research.google.com/github/AIM-kookmin/basic_study/blob/main/week01_cnn_basics/concept_practice.ipynb)
- [MNIST 실행 검증](../cnn_concept_web/MNIST_실습검증_보고서.md)

## 강의 흐름

| PDF | 시간 | 내용 |
|---|---:|---|
| 1–5쪽 | 15분 | 이미지 분류와 객체 인식의 용어, 검출·분할의 출력 차이 |
| 6–8쪽 | 15분 | 사람의 획·부분 관계·경험·맥락에 의한 분류 |
| 9–18쪽 | 30분 | 비유와 연산으로 익히는 픽셀·필터·특징 맵·채널·공유·수용 영역·학습 |
| 19–27쪽 | 25분 | 실제 MNIST 이미지·픽셀·분할·필터·CNN·학습·예측·오분류 |
| 28쪽 | 5분 | 출구 질문과 출처 |

객체 인식은 넓은 표현이므로 이미지 분류와 객체 검출을 구체적으로 구분합니다. 사람의 비유 뒤에는 실제 곱셈·합·배열·연결을 설명합니다.
웹12쪽은 필터 관찰 도구, PDF27쪽은 실제 오분류를 해석하는 5분 활동입니다.

## MNIST 실습

실제 손글씨 관찰 → 픽셀·라벨 → 필터 계산 → CNN의 크기 변화 → 학습·예측·오분류 → 특징 맵·이동 실험으로 진행합니다.
CPU로 실행하며 첫 실행에는 약11MB MNIST 다운로드와 인터넷 연결이 필요합니다. 이후 캐시를 검증해 재사용합니다. 다른 Python 파일이나 이전 주차의 실행 상태는 필요 없습니다.

전체 실습은 별도45–50분입니다. PDF 수업에서는 실제 실행 결과 그림을 읽습니다. 짧은 Colab 시연은 준비·학습을 미리 실행한 후6–7절의 예측·오류를 읽는 방식으로 진행합니다.
9절에서 관찰 기록을 `outputs/week01_mnist/observations.json`으로 저장합니다.

발표의 실측값은 seed17·6epoch·학습6,000/검증1,000/test1,000 기준입니다. 공식 test10,000장 전체 성능이나 여러 seed 통계가 아닙니다.

## 이전 판본의 선택 확장

- `practice.ipynb`: FashionMNIST MLP/CNN 비교·특징 맵·오분류
- `homework_optional.ipynb`: shape·파라미터·폭·이동 강건성 자율 과제
- `leader_solution.ipynb`: 위 과제의 참고 구현
- `lecture.pptx`, `leader_guide.md`: 이전 코드 중심 판본. 현재 PDF와 페이지가 대응하지 않습니다.

출처는 개정 PDF28쪽에 있습니다. 대본은 리더 개인용 Git 제외 폴더에서 별도 보관합니다.
