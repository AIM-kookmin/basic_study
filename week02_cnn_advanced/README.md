# 2주차 — CNN ② 성능을 개선하는 실험

## 목표
- CIFAR10에서 증강과 정규화의 역할을 구분한다.
- Residual block을 구현하고 공정한 비교 실험을 한다.
- 오분류·클래스별 성능·일반화 간격으로 결과를 해석한다.

## 준비와 운영
1주차 CNN과 학습 루프를 복습한다. CIFAR10은 최초 약 170MB 다운로드가 필요하다. 학습 데이터 일부와 작은 모델로 시작하며 고성능 벤치마크 재현을 목표로 하지 않는다.

수업은 PPT의 핵심 구간으로 90분입니다. 부록은 사전 읽기, 질문 대응, 추가 세션에 사용합니다. 실습 노트북은 위에서 아래로 실행하며, 과제는 자율입니다.

## 파일
- `lecture.pptx`: 발표자 노트 포함 강의안
- `practice.ipynb`: 설명·실행 코드·관찰 질문이 있는 독립 실행 실습
- `homework_optional.ipynb`: 독립 실행 준비 코드 + 단계별 자율 과제
- `leader_solution.ipynb`: 리더용 과제 해설 및 실행 가능한 참고 구현
- `leader_guide.md`: 슬라이드별 설명과 질문 답안

## 참고 문헌
- [PyTorch · 이미지 분류 튜토리얼](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Conv2d · 공식 API](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- [Deep Residual Learning · He et al.](https://arxiv.org/abs/1512.03385)