# AIM · 2026 가을학기 딥러닝 스터디

Python·PyTorch 기초를 이수한 동아리원을 위한 **6주 × 90분 = 9시간** 과정입니다.
CNN 2주, 어텐션·언어 모델 2주, VAE 2주로 구성했습니다. Google Colab을 기준으로 작성했으며 CPU에서도 작은 설정으로 실행할 수 있습니다.

## 먼저 열 파일

**CNN 개정판 (1주차 2026-09-16 / 2주차 2026-09-23):** [개념 웹북](cnn_concept_web/index.html) · [1주차 PDF](week01_cnn_basics/lecture.pdf) · [2주차 PDF](week02_cnn_advanced/lecture.pdf) · [통합 PDF](cnn_concept_web/CNN_개념강의_통합.pdf).
1주차28페이지·2주차22페이지, 각각90분입니다. 1주차는 사람의 분류와 MNIST, 2주차는 초반 CNN 복습 6페이지 뒤에 AlexNet 원문의 문제·구조·학습·실험·Discussion을 읽습니다.

**개념 연계 실습:** [1주차 노트북](week01_cnn_basics/concept_practice.ipynb) · [2주차 노트북](week02_cnn_advanced/concept_practice.ipynb).
1주차는 과제 구분 → 사람의 분류 → CNN 기초 용어 → 실제 MNIST로 진행합니다. 개정 실습은 MNIST를 처음 약11MB 다운로드해 CPU로 실행합니다. 2주차 기존 노트북은 내부 합성 데이터의 배경 지름길을 관찰하는 선택 확장입니다. AlexNet 논문 리딩 수업과 별도이며 논문 재현 실험이 아닙니다. 전체 코딩 실습은 별도 시간을 배정하며, 각 주차 README에서 짧은 시연 구간을 확인하세요.

1. 리더는 [리더 운영 안내](리더_운영안내.md)와 [환경설정](환경설정.md)을 확인합니다.
2. CNN 1·2주차는 개념 웹북 또는 새 `lecture.pdf`로 발표합니다. 해설은 [웹북의 리더 가이드](cnn_concept_web/README.md)를 사용합니다. 3–6주차는 `lecture.pptx`와 발표자 노트를 사용합니다.
3. CNN 개념 수업에서는 `concept_practice.ipynb`, 실제 이미지 확장 및 3–6주차에서는 `practice.ipynb`를 Colab에 업로드합니다. 모든 파일은 독립 실행합니다.
4. 자율 학습자는 `homework_optional.ipynb`를 사용합니다. `leader_solution.ipynb`는 참고 구현과 해설입니다.

## 전체 구성

| 주차 | 강의·폴더 | 핵심 개념 | 실습 결과물 | 자율 과제 |
|---|---|---|---|---|
| 1 | [CNN ①](week01_cnn_basics/README.md) | 분류·인식·검출 구분, 사람의 단서, CNN 용어 | 실제 MNIST 입력·학습·예측·오분류 | FashionMNIST MLP/CNN, 특징 맵, 이동 강건성 |
| 2 | [CNN ②](week02_cnn_advanced/README.md) | AlexNet 논문 리딩, CNN 구조·학습·실험 근거 | 네 가지 질문으로 작성한 논문 읽기 기록 | 기존 CIFAR10·배경 지름길 실습은 선택 확장 |
| 3 | [Attention ①](week03_attention_basics/README.md) | Q/K/V, scaling, mask, next-token | 손계산 heatmap, bigram/attention LM, 생성 | attention 구현, shift, 누출 탐지, 문맥 실험 |
| 4 | [Attention ②](week04_transformer_lm/README.md) | MHA, pre-norm, FFN, sampling | TinyTransformer, top-k 생성, checkpoint 복원 | top-k, block, head 비교, top-p |
| 5 | [VAE ①](week05_vae_basics/README.md) | posterior, prior, 재매개화, ELBO | MNIST VAE, 재구성·prior 샘플·보간 | 재매개화, loss, latent 차원, AE 비교 |
| 6 | [VAE ②](week06_vae_advanced/README.md) | β, collapse 진단, CVAE | β 3조건 비교, traversal, 조건부 생성 | warm-up, 조건 격자, 스케줄 비교, 차원 진단 |

**개정 CNN 분량:** 웹북·PDF 총50페이지, 주차별 리더 가이드, MNIST 관찰과 AlexNet 논문 읽기 활동, 개인 읽기 기록지.
기존 PPT 6개 × 40장 = **240장**, 실습 6개, 자율 과제 6개, 리더 해설 노트북 6개도 보관합니다. CNN PPT는 이전 코드 중심 판본이며 새 PDF와 내용·페이지가 다릅니다.
각 PPT에는 90분 핵심 28장과 심화·시각 보충·참고 12장이 있습니다. 40장을 모두 강의식으로 읽는 방식보다
핵심 28장으로 수업하고 부록은 질문 대응과 자율 복습에 사용하는 구성을 권합니다.
강의 PDF 6개도 함께 제공하여 PowerPoint 없이 읽을 수 있습니다.

## 3–6주차 및 기존 PPT의 90분 진행표

CNN 개정 수업의 90분 진행표는 주차별 README를 사용합니다. 아래 표의 코딩 실습 시간을 추가로 합치지 않습니다.

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
- `.pdf`: 강의 열람·인쇄용. CNN 해설은 `cnn_concept_web/CNN_1주차_리더가이드.md`와 `CNN_2주차_리더가이드.md`, 나머지 주차는 `leader_guide.md`에서 읽습니다.
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

자료의 현재 저장 위치는 `D:\AIM\활동_2026_2`입니다. CNN 웹북 수정·PDF 재생성 방법은 [웹북 안내](cnn_concept_web/README.md)에 있습니다.
