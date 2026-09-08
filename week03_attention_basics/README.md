# 3주차 — Attention ① 관계를 계산하는 방법

## 목표
- Q·K·V와 scaled dot-product attention을 손으로 계산한다.
- causal mask와 텐서 shape를 검증한다.
- 단일 헤드 문자 언어 모델을 학습하고 생성한다.

## 준비와 운영
행렬 곱, softmax, cross entropy를 복습한다. 외부 텍스트 다운로드 없이 교육용으로 직접 만든 문장 코퍼스를 사용한다. 결과는 언어 모델 원리 관찰용이며 자연어 능력 벤치마크가 아니다.

수업은 PPT의 핵심 구간으로 90분입니다. 부록은 사전 읽기, 질문 대응, 추가 세션에 사용합니다. 실습 노트북은 위에서 아래로 실행하며, 과제는 자율입니다.

## 파일
- `lecture.pptx`: 발표자 노트 포함 강의안
- `practice.ipynb`: 설명·실행 코드·관찰 질문이 있는 독립 실행 실습
- `homework_optional.ipynb`: 독립 실행 준비 코드 + 단계별 자율 과제
- `leader_solution.ipynb`: 리더용 과제 해설 및 실행 가능한 참고 구현
- `leader_guide.md`: 슬라이드별 설명과 질문 답안

## 참고 문헌
- [Attention Is All You Need · Vaswani et al.](https://arxiv.org/abs/1706.03762)
- [Scaled Dot Product Attention · 공식 API](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [CrossEntropyLoss · 공식 API](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)