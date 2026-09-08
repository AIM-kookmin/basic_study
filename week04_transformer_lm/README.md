# 4주차 — Attention ② 작은 Transformer 만들기

## 목표
- multi-head attention과 pre-norm block을 구현한다.
- 소형 decoder 언어 모델을 학습하고 sampling을 비교한다.
- 데이터 누출·인과성·평가 조건을 검사한다.

## 준비와 운영
3주차 attention과 문자 코퍼스를 독립적으로 다시 정의해 이전 실행 상태 없이 시작할 수 있다. 기본 실습은 짧은 문맥과 작은 모델을 사용한다. 대형 모델 학습이나 챗봇 품질을 목표로 하지 않는다.

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