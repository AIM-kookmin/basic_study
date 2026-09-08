# 6주차 — VAE ② 생성 제어와 잠재 공간 실험

## 목표
- β-VAE 실험을 같은 손실 척도로 비교한다.
- posterior collapse의 징후와 한계를 설명한다.
- 조건부 VAE로 숫자 라벨을 제어하며 생성한다.

## 준비와 운영
5주차의 손실과 재매개화 코드를 독립적으로 다시 포함한다. 실습은 β 비교와 작은 CVAE 학습을 수행한다. GPU가 없으면 각 학습 epoch와 TRAIN_N을 줄이고 기본 분석을 우선한다.

수업은 PPT의 핵심 구간으로 90분입니다. 부록은 사전 읽기, 질문 대응, 추가 세션에 사용합니다. 실습 노트북은 위에서 아래로 실행하며, 과제는 자율입니다.

## 파일
- `lecture.pptx`: 발표자 노트 포함 강의안
- `practice.ipynb`: 설명·실행 코드·관찰 질문이 있는 독립 실행 실습
- `homework_optional.ipynb`: 독립 실행 준비 코드 + 단계별 자율 과제
- `leader_solution.ipynb`: 리더용 과제 해설 및 실행 가능한 참고 구현
- `leader_guide.md`: 슬라이드별 설명과 질문 답안

## 참고 문헌
- [Auto-Encoding Variational Bayes · Kingma & Welling](https://arxiv.org/abs/1312.6114)
- [β-VAE · Higgins et al.](https://openreview.net/forum?id=Sy2fzU9gl)
- [BCE with logits · 공식 API](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html)