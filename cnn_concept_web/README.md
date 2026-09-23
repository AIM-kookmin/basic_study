# CNN · 보는 것에서, 배우는 것으로

1주차는 2026-09-16 흐름 개정판, 2주차는 2026-09-22 AlexNet 논문 리딩판입니다. CNN 1·2주차를 각각 90분 수업으로 새로 구성했습니다.

## 바로 열기

- [웹북 열기](index.html): `index.html`을 Chrome 또는 Edge에서 엽니다. 인터넷 연결 없이 본문·폰트·도식·위젯이 동작합니다.
- [1주차 PDF · 28페이지](CNN_1주차_개념강의.pdf)
- [2주차 PDF · 40페이지](CNN_2주차_개념강의.pdf)
- [통합 PDF · 68페이지](CNN_개념강의_통합.pdf)
- [1주차 리더 가이드](CNN_1주차_리더가이드.md) · [2주차 리더 가이드](CNN_2주차_리더가이드.md)
- [1주차 개념 실습](../week01_cnn_basics/concept_practice.ipynb) · [2주차 개념 실습](../week02_cnn_advanced/concept_practice.ipynb) · [현재 실습 검증](현재_실습검증_보고서.md)

PDF는 웹의 인쇄 전용 레이아웃을 Chromium으로 렌더링한 **A4 가로형**입니다. 텍스트는 검색·선택 가능하며 개념도는 SVG, 실제 MNIST 표본·계산·학습 그림은 고해상도 PNG로 제공합니다. 원문 링크도 PDF에서 클릭할 수 있습니다.
기존 `week01_cnn_basics/lecture.pdf`, `week02_cnn_advanced/lecture.pdf`에도 이 개정판을 반영합니다.

## 설명의 흐름

**1주차 — 숫자 한 장에서, 분류의 원리까지.**

이미지 분류·객체 인식·검출·분할의 출력 구분 → 사람의 획·관계·경험·맥락 → 사람의 비유와 실제 CNN 기초 연산 → 실제 MNIST 입력·학습·예측·오분류. MNIST 그림은 실제 데이터와 실행 결과이며 공식 test의 고정1,000장 부분집합을 평가했습니다.

**2주차 — CNN을 안다는 것에서, 논문을 읽는 것으로.**

AlexNet(2012)을 중심으로 초록·서론 → 문제 정의 재구성 → CNN의 핵심 개념 → 수식 → 원문 실험 표·그림 → Discussion → 네 가지 개인 정리의 순서로 읽습니다. 개념28분·수식7분을 포함한 총90분입니다. [논문 읽기 기록지](../week02_cnn_advanced/논문_읽기_기록지.md)를 함께 사용합니다.

인간 시각에서 얻은 영감을 설명하되 CNN이 인간의 뇌를 그대로 복제한 것처럼 설명하지 않습니다. 원 논문의 대상·조건과 수업용 비유의 한계를 구분합니다.

## 수업 운영

각 PDF 2페이지에 합계 90분의 진행표가 있습니다. 코딩이 없어도 수업을 진행할 수 있습니다.

- 1주차 활동: 11–12쪽에서 작은 필터 손계산과 반응을 확인하고, 27쪽에서 실제 MNIST 오분류를 읽으며 다음 실험을 설계합니다(5분).
- 2주차 활동: 9쪽 문제·제안 짝 설명, 32쪽 Table 2 주장 판별, 38쪽 네 가지 개인 정리. 원문에 없는 인과 관계나 범위를 추가하지 않는 읽기를 연습합니다.
- 본문 중 질문을 짝 토론에 활용하고, 마지막 출구 질문으로 개념을 확인합니다.
- 기존 `practice.ipynb`·자율 과제는 **별도 선택 실습**으로 연결합니다. 90분에 기존 코드 실습까지 모두 추가하도록 요구하지 않습니다.
- 개정 1주차 `concept_practice.ipynb`는 MNIST 전체 실습45–50분입니다. 준비·학습을 미리 실행하면6–7절의 예측·오류를 짧게 시연할 수 있습니다. 2주차 기존 노트북은 배경 지름길을 다루는 별도 선택 확장으로, 새 PDF의 페이지에 대응하거나 AlexNet을 재현하지 않습니다.
- 두 파일은 독립 실행하며 GPU가 필요 없습니다. 1주차는 MNIST 약11MB 최초 다운로드, 2주차는 내부 합성 데이터를 사용합니다. 실행 결과·예상 질문·변수 변경·자율 실험·JSON 기록을 포함합니다.

## 웹 조작

- 상단 `01 숫자 분류` / `02 AlexNet 리딩` / `전체`로 전환합니다.
- 목차에서 원하는 페이지로 이동합니다. ← → 키와 하단 버튼으로도 넘길 수 있습니다.
- 1주차 12페이지: 슬라이더로 합성곱 창을 이동하며 실제 예제 반응값을 확인합니다.
- 2주차 32페이지: 토론 후 해설을 펼쳐 표가 지지하는 주장과 그렇지 않은 주장을 구별합니다. PDF에서는 해설도 함께 표시됩니다.
- 모바일에서는 본문과 그림이 세로로 재배치됩니다.

위젯은 개념을 관찰하는 도구입니다. 실제 CNN을 학습하거나 분류 성능을 측정하지 않습니다. PDF에는 조작 없이 읽을 수 있는 초기 상태 또는 여러 변환 상태를 정적으로 담았습니다.

## 출처와 디자인 자산

생물학적 배경·학습 기법·모델 한계에 관한 원문은 본문 하단과 각 주차 마지막 페이지에 링크했습니다.
1주차와 2주차의 설명용 개념도는 자체 제작했습니다. 2주차는 AlexNet 공식 PDF의 Figures 1–4, Tables 1–2, LRN 수식 등을 직접 캡처하여 사용합니다. 캡처마다 원문 페이지·번호·링크가 있으며 [출처 및 추출 기록](assets/alexnet/provenance.json)을 보관합니다. 원문 수치·사진·곡선은 수정하지 않았고, 수업용 설명·계산과 구분했습니다.

Noto Sans KR은 Google Fonts 배포본을 사용했습니다. 폰트 라이선스는 [assets/OFL.txt](assets/OFL.txt)에 있습니다.

## 수정·재생성

- `src/content.py`·`src/week1_mnist_content.py`·`src/week2_alexnet_content.py`: 68페이지 원고와 리더 해설
- `src/paper_components.py`·`paper.css`: 논문 리딩 전용 레이아웃·개념도
- `src/prepare_alexnet.py`: 공식 원문 다운로드와 캡처·출처 기록 생성
- `src/mnist_lab.py`·`src/prepare_mnist.py`: 실제 MNIST 계산·학습·발표 그림 생성
- `src/build_mnist_notebook.py`·`src/validate_mnist_notebook.py`: 1주차 실습 생성·검증
- `src/diagrams.py`: SVG 도식과 계산 예제
- `styles.css`: 웹·모바일·인쇄 디자인
- `app.js`: 목차·페이지 이동·관찰 도구
- `src/build.py`: HTML과 리더 가이드 생성
- `src/render_and_check.py`: PDF 렌더링, 페이지 수·레이아웃·모바일·위젯 검사
- `src/publish.py`: 검증된 PDF를 주차 폴더에 반영하고 배포본 생성
- `src/build_notebooks.py`: 개념 연계 실습 노트북 두 개 생성
- `src/validate_notebooks.py`: 새 커널·빈 폴더에서 전체 실행, 결과 저장, 실습 검증 보고서 생성

```powershell
python -m pip install playwright pymupdf pillow torch numpy matplotlib
python -m playwright install chromium
python src/prepare_mnist.py
python src/prepare_alexnet.py
python src/build.py
python src/render_and_check.py
python src/publish.py
```

노트북 수정 후에는 `python -m pip install torch numpy matplotlib nbformat nbclient ipykernel`로 도구를 준비하고 `python src/build_notebooks.py`, `python src/validate_notebooks.py`, `python src/publish.py` 순서로 실행합니다. 생성 스크립트는 출력이 없는 노트북을 만들며, 검증 스크립트가 실제 실행 출력을 저장합니다. 웹북 ZIP에서는 노트북이 `notebooks/`에 포함됩니다.

위 명령은 `cnn_concept_web` 폴더에서 실행합니다. `build.py`는 웹과 가이드를, 렌더 스크립트는 PDF를 다시 생성합니다. 완료 후 `_preview/validation.json`에 검증 결과가 남습니다. `_preview/`와 배포 ZIP은 Git에서 무시합니다.
