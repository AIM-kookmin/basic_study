# AIM · 2026 가을학기 딥러닝 스터디

Python·PyTorch 기초를 이수한 동아리원을 위한 **6주 × 90분 = 9시간** 과정입니다.
CNN 2주, 어텐션·언어 모델 2주, VAE 2주로 구성했습니다. Google Colab을 기준으로 작성했으며 CPU에서도 작은 설정으로 실행할 수 있습니다.

## 먼저 열 파일

1. 리더는 [리더 운영 안내](리더_운영안내.md)와 [환경설정](환경설정.md)을 확인합니다.
2. 해당 주차의 `lecture.pptx`로 발표합니다. **발표자 노트에 설명·정답·오개념 교정**이 들어 있습니다.
3. 참가자는 `practice.ipynb`를 Colab에 업로드합니다. 독립 실행 파일이라 이전 주차의 실행 상태가 필요 없습니다.
4. 자율 학습자는 `homework_optional.ipynb`를 사용합니다. `leader_solution.ipynb`는 참고 구현과 해설입니다.

## 전체 구성

| 주차 | 강의·폴더 | 핵심 개념 | 실습 결과물 | 자율 과제 |
|---|---|---|---|---|
| 1 | [CNN ①](week01_cnn_basics/README.md) | 합성곱, 채널, shape, 수용 영역, CE | FashionMNIST MLP/CNN 비교, 오류·특징 맵 | shape·파라미터 구현, 폭 비교, 이동 강건성 |
| 2 | [CNN ②](week02_cnn_advanced/README.md) | 증강, BN, residual, 공정한 평가 | CIFAR10 base/augment 비교, confusion matrix | shortcut, recall, residual 비교·통제 설계 |
| 3 | [Attention ①](week03_attention_basics/README.md) | Q/K/V, scaling, mask, next-token | 손계산 heatmap, bigram/attention LM, 생성 | attention 구현, shift, 누출 탐지, 문맥 실험 |
| 4 | [Attention ②](week04_transformer_lm/README.md) | MHA, pre-norm, FFN, sampling | TinyTransformer, top-k 생성, checkpoint 복원 | top-k, block, head 비교, top-p |
| 5 | [VAE ①](week05_vae_basics/README.md) | posterior, prior, 재매개화, ELBO | MNIST VAE, 재구성·prior 샘플·보간 | 재매개화, loss, latent 차원, AE 비교 |
| 6 | [VAE ②](week06_vae_advanced/README.md) | β, collapse 진단, CVAE | β 3조건 비교, traversal, 조건부 생성 | warm-up, 조건 격자, 스케줄 비교, 차원 진단 |

**분량:** PPT 6개 × 40장 = **240장**, 실습 6개, 자율 과제 6개, 리더 해설 노트북 6개.
각 PPT에는 90분 핵심 28장과 심화·시각 보충·참고 12장이 있습니다. 40장을 모두 강의식으로 읽는 방식보다
핵심 28장으로 수업하고 부록은 질문 대응과 자율 복습에 사용하는 구성을 권합니다.
강의 PDF 6개도 함께 제공하여 PowerPoint 없이 읽을 수 있습니다.

## 매주 90분 진행표

| 시간 | PPT | 활동 | 참가자 산출물 |
|---|---|---|---|
| 00–05 | 1–2 | 이전 개념 회상, 목표, 환경 실행 | 실행 환경 확인 |
| 05–20 | 3–9 | 개념과 데이터 흐름 | 용어·shape 메모 |
| 20–35 | 10–15 | 수식, 손계산, 짝 설명 | 계산 답과 근거 |
| 35–50 | 16–20 | 구현 설계, 디버깅·평가 규칙 | 코드 실행 전 예측 |
| 50–75 | 21–23 | 실습 노트북 실행 | 학습 곡선·표·이미지 |
| 75–85 | 24–26 | 결과 비교와 실패 분석 | 해석 2–5문장 |
| 85–90 | 27–28 | 출구 퀴즈, 다음 주 연결 | 핵심 질문 답 |
| 수업 외 | 29–40 | 심화, 시각 자료, 참고 읽기 | 자율 과제·추가 실험 |

실습 정의 셀은 수업 앞부분에서 함께 읽고, 다운로드는 도입 때 미리 실행합니다. 25분 실습 구간에 설치까지 포함시키지 않습니다.
자율 과제를 모두 수행하면 주차별 약 80–130분과 추가 학습 시간이 필요합니다. A·B만 선택하거나 C·D 중 하나만 선택해도 됩니다.

## 파일 사용과 결과 보관

- `.pptx`: 편집 가능한 텍스트·도식, 발표자 노트 포함.
- `.pdf`: 강의 열람·인쇄용. 발표자 노트는 주차별 `leader_guide.md`에서 읽습니다.
- `practice.ipynb`: 설명, 완성 코드, 수치 검사, 관찰 질문, 실험 기록표.
- `homework_optional.ipynb`: 준비 코드, TODO 4문제, 검사 셀, 실험·해석 틀. 미완성 셀은 안내 후 넘어갑니다.
- `leader_solution.ipynb`: 실제 실행 가능한 참고 구현. 성능 숫자는 정답이 아닙니다.
- `outputs/`: 실행으로 생긴 모델·실험 로그. 로컬 CPU 기본 실행 결과를 포함합니다.
- `_검증/`: 실행·PPT 검사 보고서와 이미지 미리보기.
- `_제작소스/`: 강의·노트북 재생성용 원고와 Python 스크립트. 참가자는 실행할 필요가 없습니다.

노트북에 저장된 실행 결과는 제작 환경에서 관측한 예시입니다. 재실행하면 출력이 달라질 수 있습니다.
`QUICK=True`의 이미지 평가는 고정된 test 부분집합 성능이며 전체 공식 test 성능을 의미하지 않습니다.
Colab GPU에서의 실제 소요 시간은 할당된 장비와 혼잡도에 따라 달라집니다. 검증 범위와 관측 시간은 [검증 보고서](검증_보고서.md)에 기록합니다.

## 데이터와 원문

FashionMNIST·CIFAR10·MNIST는 torchvision의 공식 Dataset 클래스로 다운로드합니다.
영문 문자 코퍼스는 외부 저작물 대신 이 수업을 위해 만든 조합 문장이며 문장 단위로 train/val/test를 분리합니다.
원문 논문·PyTorch 공식 문서는 각 주차의 README와 PPT 노트에 연결했습니다. 설명과 코드·도식은 새로 작성했습니다.

기존 `D:\AIM\pytorch_2026_1` 자료를 덮어쓰지 않고, 이번 요청의 저장 위치인 `D:\AIM\활동\_2026\_2`에 독립적으로 구성했습니다.
