# 5주차 — VAE ① 잠재변수로 이미지 생성하기

## 목표
- AE와 VAE의 차이를 확률 분포 관점에서 설명한다.
- 재매개화와 ELBO 손실을 올바르게 구현한다.
- MNIST 재구성·prior 샘플·잠재 보간을 구분한다.

## 준비와 운영
로그, 평균과 분산, 정규분포를 짧게 복습한다. MNIST 픽셀은 0~1 범위로 유지한다. 이미지 분류에서 사용한 [-1,1] Normalize를 그대로 적용하지 않는다.

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