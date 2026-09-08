"""수업용 원고: | 로 제목/핵심 설명/도식 또는 수식/리더 해설을 구분한다."""
CNN_REFS=[('PyTorch · 이미지 분류 튜토리얼','https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html'),('Conv2d · 공식 API','https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html'),('Deep Residual Learning · He et al.','https://arxiv.org/abs/1512.03385')]
ATT_REFS=[('Attention Is All You Need · Vaswani et al.','https://arxiv.org/abs/1706.03762'),('Scaled Dot Product Attention · 공식 API','https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html'),('CrossEntropyLoss · 공식 API','https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html')]
VAE_REFS=[('Auto-Encoding Variational Bayes · Kingma & Welling','https://arxiv.org/abs/1312.6114'),('β-VAE · Higgins et al.','https://openreview.net/forum?id=Sy2fzU9gl'),('BCE with logits · 공식 API','https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html')]

def lines(s): return [x.strip() for x in s.strip().splitlines() if x.strip()]

WEEKS=[dict(folder='week01_cnn_basics', title='CNN ① 이미지에서 패턴 찾기',tag='IMAGE\n→ CLASS',goals=['합성곱의 출력 크기와 파라미터 수를 계산한다.','FashionMNIST 분류기를 구현하고 학습을 진단한다.','특징 맵과 오분류를 근거로 모델의 한계를 설명한다.'],intro='Python, 텐서, nn.Module, 역전파 기초를 전제로 한다. FashionMNIST 다운로드를 수업 전에 완료한다. 실습 기본값은 일부 학습 데이터와 짧은 학습이며 목표는 최고 정확도보다 올바른 파이프라인이다.',refs=CNN_REFS,
개념=lines('''
이미지 분류를 함수로 쓰기 | 입력은 픽셀 텐서, 출력은 클래스별 점수다.~학습은 이미지와 정답의 대응을 반복해서 조정한다.~추론에서는 가장 큰 점수의 클래스를 선택한다. | FLOW:이미지;B×1×28×28>특징 추출>C개 점수>argmax | FashionMNIST는 28×28 회색조 의류 이미지 10종이다. 점수 자체는 확률이 아니며 logits이라고 부른다. 분류 정확도가 높아도 모든 종류의 이미지에 안전하게 일반화한다는 뜻은 아니다.
MLP에서 CNN으로 | Flatten은 정보를 삭제하지 않지만 2차원 이웃 구조를 직접 활용하지 않는다.~전결합층은 위치마다 별도 가중치를 가진다.~CNN은 지역 연결과 가중치 공유를 구조에 넣는다. | 784 → 128: 100,480개 / Conv 1→16, 3×3: 160개 | Flatten이 공간 정보를 완전히 잃는다는 표현은 부정확하다. 고정된 순서라면 원래 위치를 복구할 수 있다. 핵심은 MLP 구조가 인접성과 이동을 활용하도록 강제하지 않는다는 점이다. 파라미터 비교는 첫 층만의 비교다.
필터 하나가 하는 일 | 작은 창과 필터를 원소별로 곱한 뒤 합한다.~같은 필터를 이미지 여러 위치에 적용한다.~출력 특징 맵은 특정 패턴에 대한 반응 지도다. | FLOW:지역 패치;3×3>곱하고 합하기>스칼라 반응>전체 특징 맵 | 세로 경계 필터를 예로 들어 밝고 어두운 부분의 차이를 계산한다. 초기 필터는 무작위이며 학습으로 바뀐다. 모든 필터를 사람이 이해할 수 있는 의미로 이름 붙일 수 있는 것은 아니다.
채널은 무엇인가 | RGB는 입력 채널이 3개다.~출력 채널 수는 서로 다른 필터 수다.~각 출력 채널은 모든 입력 채널의 반응을 합한다. | W: [C_out, C_in, K_h, K_w] | 출력 채널이 16이면 필터는 16개지만 각 필터의 깊이는 입력 채널 수와 같다. RGB 채널별 출력이 독립적으로 남는다는 오해를 바로잡는다. bias는 출력 채널마다 하나다.
이동에 대한 올바른 직관 | 같은 패턴이 다른 위치에 나타나도 같은 필터가 반응한다.~stride 1 합성곱은 경계 등을 제외하면 이동 등변성을 갖는다.~풀링·경계·stride 때문에 완전한 이동 불변성은 보장되지 않는다. | 등변성: 입력 이동 → 특징 맵도 이동 | 불변성과 등변성을 구분한다. 이미지가 이동해도 출력 특징 맵이 동일한 것은 아니다. 최종 분류의 강건함은 풀링, 학습 데이터, 증강 등과 함께 만들어진다.
비선형성과 다운샘플링 | ReLU는 음수 반응을 0으로 바꾼다.~MaxPool은 작은 영역의 최댓값을 남긴다.~공간 해상도를 줄이면 계산량과 세부 정보가 함께 줄어든다. | FLOW:Conv>ReLU>MaxPool>Conv>ReLU | 선형층만 쌓으면 전체가 하나의 선형 변환이다. MaxPool은 학습 파라미터가 없다. 풀링을 지나치게 반복하면 작은 물체의 특징이 사라질 수 있다.
데이터 분할부터 시작 | train은 가중치 학습에 사용한다.~validation은 구조와 학습 설정 선택에 사용한다.~test는 선택을 끝낸 모델의 마지막 평가에만 사용한다. | train ≠ validation ≠ test | 실습에서는 공식 학습 세트를 고정 시드로 train/validation으로 나눈다. 공식 test를 매 epoch 보며 하이퍼파라미터를 조정하지 않는다. 결과를 비교할 때 분할도 같아야 한다.
'''),
수식=lines('''
합성곱 한 칸 손계산 | 패치 [[1,2],[3,4]]와 필터 [[1,0],[0,-1]]을 사용한다.~원소별 곱은 1, 0, 0, -4다.~합은 -3이며 bias가 1이면 출력은 -2다. | y = Σ patch × weight + bias = −2 | 직접 계산 후 노트북의 F.conv2d 출력과 비교한다. PyTorch Conv2d는 엄밀하게는 커널을 뒤집지 않는 cross-correlation을 수행한다. 딥러닝 문맥에서는 관례적으로 합성곱이라 부른다.
출력 크기 공식 | 입력 H, 커널 K, 패딩 P, stride S를 확인한다.~dilation D까지 포함하면 유효 커널은 D(K−1)+1이다.~가로와 세로에 공식을 각각 적용한다. | H_out = floor((H + 2P − D(K−1) − 1) / S + 1) | dilation은 이번 핵심 구현에서 1이다. 나눗셈 후 내림을 놓치지 않도록 한다. 출력 크기가 음수나 0이 되면 해당 구성이 유효하지 않다.
형상 계산 연습 | 28×28, K=3, P=1, S=1이면 28×28이다.~2×2 MaxPool, S=2를 거치면 14×14다.~두 블록 후 32×7×7을 펼치면 1,568차원이다. | [B,1,28,28] → [B,16,14,14] → [B,32,7,7] | 참가자에게 Linear의 in_features를 먼저 말하게 한다. B는 배치 크기이며 특징 차원에 곱하지 않는다. 입력 해상도가 달라지면 이 고정 Linear의 크기도 달라진다.
파라미터 수를 계산하기 | Conv의 bias는 출력 채널마다 하나다.~공간 크기가 커져도 같은 필터를 공유한다.~풀링과 ReLU에는 학습 파라미터가 없다. | Conv params = C_out × (C_in × K_h × K_w + 1) | 16→32, 3×3 Conv는 32×(16×9+1)=4,640개다. bias=False이면 마지막 +1을 빼야 한다. 출력 텐서 원소 수와 파라미터 수를 구분한다.
수용 영역을 넓히기 | 3×3 합성곱 두 층은 stride 1에서 5×5 영역을 본다.~중간 ReLU가 추가적인 비선형성을 만든다.~stride와 pooling이 있으면 누적 간격도 추적한다. | r_new = r_old + (K−1) × jump_old | 3×3 두 층이 5×5 한 층과 같은 함수는 아니다. r=1, jump=1에서 시작해 각 층의 수용 영역을 계산한다. stride를 지나면 jump_new=jump_old×stride다.
분류 손실과 확률 | CrossEntropyLoss는 logits와 정수 라벨을 받는다.~내부적으로 log-softmax와 정답 항의 음의 로그를 계산한다.~학습 직전 softmax를 따로 적용하지 않는다. | L = −log softmax(logits)[y] | 확률 예측 [0.1,0.7,0.2]에서 정답이 두 번째면 손실은 약 0.357이다. 정답 확률이 0.1이면 약 2.303으로 더 크다. 라벨 dtype은 torch.long이다.
'''),
설계=lines('''
오늘의 SmallCNN | 첫 블록은 1→16 채널, 둘째 블록은 16→32다.~각 블록은 3×3 Conv, ReLU, 2×2 Pool로 구성한다.~분류기는 1,568→64→10으로 연결한다. | FLOW:28×28×1>14×14×16>7×7×32>64>10 logits | 노트북의 features와 classifier를 연결해 읽는다. 서로 다른 모듈 이름을 따라가기보다 각 단계의 shape를 주석으로 써 보게 한다. dropout은 2주차 비교를 위해 이번에는 넣지 않는다.
학습 루프의 다섯 단계 | 이전 기울기를 비우고 forward를 계산한다.~손실을 계산하고 backward로 기울기를 만든다.~optimizer.step으로 파라미터를 갱신한다. | CODE:optimizer.zero_grad();;logits = model(x);;loss = F.cross_entropy(logits, y);;loss.backward();;optimizer.step() | step 전에 zero_grad를 하지 않으면 기울기가 누적된다. loss.item()은 기록용 Python 숫자이며 그 값으로 backward할 수 없다. inputs와 model은 같은 device에 있어야 한다.
train과 eval을 구분하기 | model.train()은 학습 모드로 설정한다.~model.eval()은 Dropout·BatchNorm의 동작을 바꾼다.~no_grad는 기울기 기록을 끄는 별도 기능이다. | eval() ≠ no_grad() | 이번 모델에서 두 모드 차이가 작더라도 습관을 미리 잡는다. 평가 함수는 eval과 no_grad를 모두 사용한다. validation 뒤 다음 epoch 시작에 train으로 복귀해야 한다.
평균 손실을 제대로 집계 | 배치 평균에 배치 크기를 곱해 누적한다.~마지막 작은 배치까지 실제 표본 수로 나눈다.~정확도는 맞힌 개수 / 전체 개수다. | epoch_loss = Σ(batch_loss × batch_size) / N | 배치 평균들의 단순 평균은 마지막 배치 크기가 다르면 가중치가 왜곡된다. 손실 로그는 optimizer 업데이트 전 예측에서 나온 값이라는 점도 설명한다.
디버깅의 순서 | 입력의 shape·dtype·범위를 먼저 출력한다.~한 배치 forward, loss, backward가 되는지 확인한다.~소량 데이터에 과적합할 수 있는지 검사한다. | 입력 → 한 배치 → 소량 학습 → 전체 학습 | loss가 전혀 줄지 않으면 정답과 데이터 대응, lr, optimizer 대상, train 상태부터 확인한다. 처음부터 모델을 크게 바꾸면 원인을 찾기 어렵다.
'''),
실습=lines('''
LAB 1 · 데이터와 연산 확인 | practice §1–3: 환경, 분할, 이미지 격자를 실행한다.~손계산 결과와 conv2d가 같은지 확인한다.~레이어별 shape와 파라미터 수를 출력한다. | 50–57분 / 산출물: 이미지 격자 + shape 표 | 환경 셀은 수업 시작에 미리 실행한다. FashionMNIST 라벨과 그림이 맞는지 먼저 본다. 계산 정답은 -2다. shape 오류가 나면 reshape 대신 바로 전 층 출력부터 확인한다.
LAB 2 · 기준 모델 학습 | practice §4–5: MLP와 CNN을 같은 분할에서 학습한다.~각 epoch의 train/validation 손실을 기록한다.~validation 손실이 가장 낮은 가중치를 보관한다. | 57–68분 / 산출물: 학습 곡선 + 비교 표 | 학습 중 나머지 참가자와 다음 loss 방향을 예측한다. 시간이 부족하면 epochs를 줄이고 전체 셀을 끝까지 실행한다. 두 모델 파라미터 수가 다르므로 구조 효과만 엄밀히 분리한 실험은 아니다.
LAB 3 · 틀린 예측 들여다보기 | practice §6–7: confusion matrix와 오분류를 시각화한다.~첫 Conv의 활성화를 원본과 비교한다.~모델 선택을 끝낸 뒤 test를 한 번 평가한다. | 68–75분 / 산출물: 오분류 8장 + 해석 2문장 | Shirt와 T-shirt/top 같은 혼동을 관찰할 수 있으나 실제 결과를 보고 말한다. 특징 맵의 밝기는 해당 채널의 반응이지 중요한 픽셀의 증명이 아니다. test 숫자를 다시 개선하기 위한 반복 튜닝은 하지 않는다.
'''),
분석=lines('''
학습 곡선으로 원인 좁히기 | train·validation 손실이 모두 높으면 학습 부족을 의심한다.~train만 계속 좋아지면 과적합 가능성을 본다.~validation 진동은 작은 표본·큰 lr 등도 원인이 될 수 있다. | 관찰 → 가설 → 한 변수 변경 → 재실험 | 한 곡선 모양만으로 원인을 확정하지 않는다. train loss는 업데이트 중, validation loss는 epoch 종료 모델로 측정되어 조건이 다르다. 증강과 dropout도 비교에 영향을 준다.
정확도 하나로 충분할까 | 클래스별 표본 수와 recall을 함께 본다.~confusion matrix는 행이 정답, 열이 예측이다.~정답률이 같아도 실패하는 클래스가 다를 수 있다. | recall_c = TP_c / 실제 클래스 c의 개수 | 행 합으로 정규화하면 각 클래스가 어디로 혼동되는지 비교하기 좋다. 표본 수가 0인 클래스의 recall은 정의되지 않는다. 노트북은 0 나눗셈을 막고 클래스 수를 함께 보여 준다.
실험 기록을 남기는 최소 단위 | 데이터 분할·seed·모델·lr·epoch를 남긴다.~최고 validation과 마지막 epoch를 구분한다.~결과 차이가 작으면 여러 seed로 다시 비교한다. | 한 줄 결론 + 결과 표 + 재현 설정 | 한 번의 실험으로 CNN이 항상 우월하다고 결론내리지 않는다. 자율 과제에서는 커널과 채널을 바꾸되 나머지 조건을 고정한다. 계산량과 성능을 같이 비교하게 한다.
'''),
회고=lines('''
출구 퀴즈 · 근거까지 말하기 | Q1. 32×32, K=3, P=0, S=2의 출력 크기는?~Q2. Conv 3→8, 3×3, bias=True의 파라미터 수는?~Q3. CrossEntropyLoss 전에 softmax를 넣어야 할까? | 1분 개인 답 → 1분 짝 설명 | 정답: 15×15, 224개, 넣지 않는다. 첫 문제에서 (32−3)/2+1=15.5를 내림한다. 둘째는 8×(3×9+1)이다. softmax는 확률 시각화 때만 따로 쓴다.
다음 주로 가져갈 질문 | 모델을 더 깊게 하면 언제 도움이 될까?~증강이 validation 정확도를 반드시 높일까?~같은 예산으로 개선 여부를 어떻게 확인할까? | 자율 과제: 커널·채널 비교 + 오류 분석 | 과제는 기본, 실험, 도전으로 나뉜다. 수업 결과를 저장하고 다음 주에는 CIFAR10의 컬러 이미지로 확장한다. 진행자는 가장 많이 나온 오류 한 가지를 기록한다.
'''),
부록=lines('''
부록 · padding의 경계 효과 | zero padding은 바깥을 0으로 가정한다.~reflection padding은 주변을 반사한다.~경계의 가정은 출력 특징에 영향을 준다. | 같은 크기 유지 ≠ 경계 정보 보존 | padding=1이라고 모든 픽셀이 같은 이웃 정보를 받는 것은 아니다. 이미지 중심과 가장자리의 유효 입력 개수 차이를 손으로 그려 본다.
부록 · 1×1 합성곱 | 공간 이웃을 섞지 않고 채널을 조합한다.~각 위치에 같은 선형 변환을 적용한다.~채널을 줄이면 뒤의 3×3 계산량을 줄일 수 있다. | [B,C_in,H,W] → [B,C_out,H,W] | 1×1에도 ReLU를 붙이면 비선형 채널 변환이 된다. 공간 정보가 없다는 표현 대신 공간적으로 이웃을 직접 합치지 않는다고 설명한다.
부록 · 계산량 추산 | 출력 한 원소는 C_in×K²개의 곱셈을 요구한다.~전체 출력 원소는 B×C_out×H_out×W_out이다.~파라미터 수가 작아도 큰 해상도에서는 계산량이 커진다. | MACs ≈ B H_out W_out C_out C_in K² | MAC과 FLOP의 집계 관례가 다르므로 보고 기준을 적는다. 이 식은 bias와 활성화 등의 연산을 제외한 근사다.
부록 · Global Average Pooling | 채널별 공간 평균을 구해 하나의 값으로 만든다.~입력 해상도 변화에 대응하기 쉬워진다.~정밀한 위치 정보는 평균 과정에서 줄어든다. | [B,C,H,W] → [B,C,1,1] | GAP는 작은 분류 헤드를 만들 때 유용하다. 2주차 모델에서 AdaptiveAvgPool2d(1)을 사용해 Flatten 크기 오류를 줄인다.
부록 · 특징 맵 읽기의 한계 | 한 채널의 활성화는 표현의 일부다.~밝은 위치가 예측의 유일한 원인은 아니다.~모델·입력 변형에 따른 출력 변화도 함께 관찰한다. | 활성화 시각화는 진단 도구다 | 시각적으로 그럴듯한 그림을 인과 설명으로 단정하지 않는다. 특정 부분을 가렸을 때 예측이 변하는지도 추가로 실험할 수 있다.
부록 · tiny overfit 점검 | 32개 표본만 반복해서 학습한다.~규제를 잠시 줄여 학습 가능한지 본다.~거의 외우지 못하면 파이프라인 오류를 먼저 찾는다. | 디버깅용 과적합 ≠ 일반화 평가 | 작은 표본 암기는 충분조건이 아니다. 잘 외워도 test 일반화는 별개의 문제다. 이 과정에서 얻은 숫자를 최종 성능으로 보고하지 않는다.
부록 · 자율 과제 설계 | 기본: 출력 크기와 파라미터 함수 구현.~실험: 채널 폭을 바꾸고 같은 예산으로 비교.~도전: 이동 후 예측 변화와 오분류 사례를 분석. | 근거 없는 최고 정확도 경쟁은 피한다 | 리더용 해설 노트북의 코드는 참고 구현이다. 개선 실패도 설정과 결과, 가능한 이유를 기록하면 유효한 과제다.
''')),

dict(folder='week02_cnn_advanced',title='CNN ② 성능을 개선하는 실험',tag='BASELINE\n→ EVIDENCE',goals=['CIFAR10에서 증강과 정규화의 역할을 구분한다.','Residual block을 구현하고 공정한 비교 실험을 한다.','오분류·클래스별 성능·일반화 간격으로 결과를 해석한다.'],intro='1주차 CNN과 학습 루프를 복습한다. CIFAR10은 최초 약 170MB 다운로드가 필요하다. 학습 데이터 일부와 작은 모델로 시작하며 고성능 벤치마크 재현을 목표로 하지 않는다.',refs=CNN_REFS,
개념=lines('''
FashionMNIST에서 CIFAR10으로 | 흑백 28×28에서 컬러 32×32로 바뀐다.~배경·시점·색상 변화가 커진다.~같은 10개 클래스라도 분류 난도는 다르다. | [B,1,28,28] → [B,3,32,32] | 첫 Conv의 in_channels를 3으로 바꾼다. CIFAR10 클래스 순서는 airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck이다. augmentation은 의미를 보존해야 한다.
실험의 출발점은 baseline | 단순한 기준 모델을 먼저 고정한다.~변경할 요인은 한 번에 하나를 우선한다.~데이터 분할과 평가 절차를 모든 조건에서 공유한다. | 가설: 좌우 반전·crop이 일반화에 도움을 줄까? | 성능이 낮다고 곧바로 많은 기법을 동시에 넣으면 어느 변화가 도움이 되었는지 알 수 없다. 기본 실습은 같은 네트워크에서 증강 유무를 비교하고 residual은 별도 확장한다.
증강은 학습 데이터에만 | RandomCrop은 위치 변화에 대응하도록 한다.~HorizontalFlip은 좌우 변화에 대응하도록 한다.~validation/test는 고정된 전처리를 사용한다. | FLOW:train>무작위 증강>정규화>모델 | validation에 무작위 crop이 있으면 평가 자체가 흔들린다. 같은 Dataset 객체의 transform을 바꾸면 두 Subset에 동시에 영향을 줄 수 있어 별도 Dataset 객체를 사용한다.
정규화와 규제는 다르다 | 입력 Normalize는 데이터의 수치 범위를 조정한다.~BatchNorm은 중간 활성화의 통계를 사용한다.~weight decay·Dropout은 과적합 완화를 위한 수단이다. | 입력 처리 / 활성화 처리 / 파라미터·표현 규제 | 같은 정규화라는 한국어 때문에 개념을 혼동하기 쉽다. 실습 입력은 채널별 (x−0.5)/0.5로 변환한다. 이는 데이터 평균을 추정한 값이 아닌 고정적인 범위 변환이다.
BatchNorm의 두 가지 모드 | 학습 시에는 미니배치 통계를 사용한다.~평가 시에는 누적된 running 통계를 사용한다.~작은 배치나 분포 변화에서는 성능이 흔들릴 수 있다. | Conv → BatchNorm → ReLU | BatchNorm의 affine 파라미터와 running 통계를 구분한다. running_mean은 optimizer로 학습되는 파라미터가 아니다. eval을 놓치면 시험 데이터 배치 구성에 예측이 영향을 받는다.
더 깊으면 항상 더 좋을까 | 깊이는 표현력을 늘리지만 최적화를 어렵게 만들 수 있다.~학습 데이터가 적으면 과적합도 커질 수 있다.~연결 구조와 학습 예산을 함께 고려해야 한다. | 모델 크기 × 데이터 × 최적화 × 예산 | train 성능까지 나빠지는 문제와 train은 좋고 val만 나쁜 문제를 구분한다. residual은 깊은 네트워크 최적화를 돕는 구조이지 정확도 상승 보장 장치가 아니다.
잔차 연결의 핵심 | 블록은 x 전체 대신 수정량 F(x)를 학습한다.~입력 x가 덧셈 경로로 전달된다.~덧셈하려면 두 경로의 shape가 같아야 한다. | FLOW:x>Conv-BN-ReLU>Conv-BN>F(x)+x>ReLU | 실습은 두 개 3×3 Conv를 갖는 작은 블록이다. 엄밀한 전체 ResNet 아키텍처 재현이라고 부르지 않는다. shortcut은 학습 경로를 추가한다.
'''),
수식=lines('''
입력 Normalize 손계산 | ToTensor 뒤 픽셀은 보통 0~1 범위다.~mean=0.5, std=0.5이면 0은 −1, 1은 1이다.~시각화에는 역변환을 적용한다. | x_norm = (x−0.5)/0.5; x = 0.5 x_norm + 0.5 | 정규화된 텐서를 그대로 imshow하면 색이 이상할 수 있다. 색상을 자를 때 학습 입력을 바꾸지 말고 시각화 복사본에만 clamp를 적용한다.
BatchNorm의 식 | 채널별 평균과 분산으로 활성화를 표준화한다.~학습 가능한 γ와 β가 스케일과 이동을 조정한다.~ε는 분모가 0에 가까울 때 수치 불안정을 줄인다. | y = γ (x−μ) / sqrt(σ²+ε) + β | Conv2d의 BatchNorm은 채널별로 배치와 공간 축의 통계를 집계한다. 실습에서 내부 분산 추정 세부를 직접 구현하기보다 모드별 동작 차이에 집중한다.
Residual의 기울기 경로 | y=x+F(x)이면 직접 경로의 미분이 1을 포함한다.~F가 작아도 입력이 전달될 수 있다.~모든 gradient 문제가 자동으로 사라지는 것은 아니다. | dy/dx = I + dF/dx | 벡터에서는 1이 아니라 항등행렬 I다. 마지막 ReLU 같은 추가 연산을 제외한 덧셈 지점 기준 식임을 말한다. 안정성이 개선될 수 있다는 정도로 설명한다.
projection shortcut | stride 2나 채널 증가로 shape가 달라질 수 있다.~1×1 Conv와 같은 stride로 shortcut도 맞춘다.~덧셈은 브로드캐스팅에 기대지 않고 동일 shape로 한다. | F(x): [B,64,16,16] / P(x): [B,64,16,16] | 입력 [B,32,32,32]를 예로 든다. projection에는 추가 파라미터가 생긴다. 비교 실험에서는 파라미터 수 변화를 보고해야 한다.
Dropout과 weight decay | Dropout은 학습 시 일부 활성화를 무작위로 비운다.~AdamW는 파라미터 감쇠를 optimizer update와 분리한다.~각 기법의 강도는 validation으로 선택한다. | p_dropout ↑ / weight_decay ↑ → 항상 성능 ↑ 는 아님 | Dropout의 출력 스케일은 PyTorch가 조정한다. AdamW의 weight_decay를 손실에 L2 항을 더한 것과 일반적으로 완전히 같다고 설명하지 않는다.
학습률 곡선 읽기 | 너무 큰 lr은 손실을 진동·발산시킬 수 있다.~너무 작은 lr은 같은 예산에서 충분히 학습하지 못한다.~scheduler를 썼다면 실제 lr 기록을 함께 남긴다. | 학습률 × 업데이트 수를 함께 비교한다 | epoch 수가 같아도 데이터 수나 batch size가 다르면 업데이트 수가 다르다. 오늘 필수 비교는 동일 lr과 epoch로 고정하며 scheduler는 심화 항목이다.
'''),
설계=lines('''
Dataset 공유 함정 피하기 | 분할 인덱스를 먼저 한 번만 만든다.~train용 Dataset에는 증강 transform을 준다.~validation용 Dataset에는 고정 transform을 준다. | FLOW:고정 인덱스>train Dataset / val Dataset>Subset>DataLoader | random_split의 두 결과가 하나의 Dataset을 공유하는 경우 transform을 별도로 설정할 수 없다. 실습은 동일 원본을 읽는 Dataset 인스턴스를 분리한다.
작은 CIFAR 분류기 | Conv-BN-ReLU 두 블록으로 특징을 추출한다.~AdaptiveAvgPool로 공간 평균을 구한다.~Linear는 채널 수에서 클래스 수로 연결한다. | FLOW:32×32×3>16×16×32>8×8×64>GAP>10 | GAP를 사용하면 입력 공간 크기를 Linear 상수에 직접 쓰지 않아도 된다. 다운샘플 횟수와 작은 이미지 해상도의 균형을 설명한다.
최고 모델을 제대로 복사 | validation loss가 개선될 때 state_dict를 복사한다.~참조만 저장하면 뒤의 학습으로 값이 바뀔 수 있다.~학습 후 최고 가중치를 복원해 분석한다. | CODE:best = copy.deepcopy(model.state_dict());;model.load_state_dict(best) | best_state=model.state_dict()만 저장하는 실수를 설명한다. CPU 텐서 clone으로 저장해 GPU 메모리를 아낄 수도 있다. 노트북은 독립적인 복사본을 보관한다.
비교 조건을 표로 고정 | baseline과 augmentation 모델은 같은 초기 seed를 쓴다.~분할, batch, lr, epoch와 평가 표본을 맞춘다.~벽시계 시간과 파라미터 수를 같이 기록한다. | 비교: base / augment / residual(확장) | seed를 맞춰도 모든 랜덤 연산 순서까지 완전히 같아지는 것은 아니다. 엄밀한 결론은 여러 seed와 반복이 필요하다. 이번 실습은 통제 실험 설계의 첫 단계다.
실패 시 복구 경로 | 다운로드는 수업 전에 끝낸다.~CPU에서는 TRAIN_N과 EPOCHS를 줄인다.~학습이 길면 첫 비교까지만 하고 분석 셀로 이동한다. | 먼저 끝까지 실행 → 그다음 예산 확대 | 가짜 데이터를 실제 CIFAR 성능처럼 보고하지 않는다. 데이터 접근이 실패하면 환경 점검만 가능하다는 것을 명시한다. 리더는 실행 결과가 저장된 실습 노트북을 백업으로 사용한다.
'''),
실습=lines('''
LAB 1 · 증강을 눈으로 검증 | practice §1–3: 같은 이미지의 증강 8장을 그린다.~validation 이미지가 반복 호출에도 같은지 검사한다.~작은 모델과 residual block의 shape를 확인한다. | 50–57분 / 산출물: 증강 격자 + shape assert | 좌우 반전이 클래스 의미를 유지하는지 말하게 한다. 숫자 데이터에서는 같은 증강이 적절하지 않을 수 있다. fixed normalization의 역변환도 확인한다.
LAB 2 · 두 조건 비교 | practice §4: base와 augment를 같은 예산으로 학습한다.~validation 손실로 각 조건의 최고 모델을 저장한다.~곡선과 파라미터·시간을 표에 정리한다. | 57–69분 / 산출물: 조건별 곡선과 결과 표 | 기본 설정의 적은 데이터와 epoch에서는 증강 조건이 더 낮을 수도 있다. 관측을 기대에 맞춰 해석하지 말고 왜 그런지 가설을 적는다. residual 비교는 자율 과제로 확장한다.
LAB 3 · 선택과 최종 평가 | practice §5–6: validation으로 최종 조건을 선택한다.~confusion matrix와 오분류 이미지를 확인한다.~선택한 모델만 test에서 한 번 평가한다. | 69–75분 / 산출물: 최종 평가 + 실패 유형 | test로 두 모델 중 승자를 고르지 않는다. 오분류는 해상도, 배경, 객체 유사성 등 관찰 가능한 근거로 설명한다. 모델의 생각을 추측해 단정하지 않는다.
'''),
분석=lines('''
증강이 손해처럼 보일 때 | 같은 epoch에서 학습 문제가 더 어려워진다.~훈련 표본이나 업데이트 수가 부족할 수 있다.~변환이 과하거나 클래스 의미를 훼손했을 수 있다. | 한 번의 하락 ≠ 증강은 쓸모없다 | 두 조건의 train loss는 입력 분포가 다르다. train loss만 직접 비교하지 말고 동일한 validation 입력에서 비교한다. 증강 강도와 학습 시간을 분리해서 후속 실험한다.
ablation을 읽는 법 | 하나의 구성 요소를 제거하거나 추가한다.~나머지 조건을 고정해 차이를 해석한다.~상호작용이 있으면 한 기법의 효과가 조건에 따라 달라진다. | 기준 → +증강 → +잔차 → 반복 seed | 모든 조합을 수업 시간에 돌릴 필요는 없다. 과제 팀끼리 조건을 나누더라도 분할과 설정을 공유해야 결과를 합칠 수 있다.
좋은 결과 보고서 | 주장: 어떤 조건이 어떤 지표에서 달랐는가.~증거: 표·곡선·실패 예시를 제시한다.~한계: 예산·표본·seed·데이터 범위를 적는다. | 주장 / 증거 / 한계 / 다음 실험 | 1% 상승만 쓰지 말고 몇 표본에서 몇 개 더 맞혔는지 연결한다. 짧은 실험의 불확실성을 인정하는 것이 좋은 실험 보고다.
'''),
회고=lines('''
출구 퀴즈 | Q1. validation에도 RandomCrop을 넣어도 될까?~Q2. eval()은 gradient 계산을 자동으로 끌까?~Q3. 채널 수가 다르면 residual을 어떻게 더할까? | 답보다 이유가 중요하다 | 정답: 기본 단일 평가에서는 고정 전처리 사용, eval은 gradient를 끄지 않음, projection으로 채널과 공간 크기를 맞춤. test-time augmentation은 별도 평가 기법이며 오늘 범위와 구분한다.
이미지에서 문장으로 | CNN은 지역 패턴을 공유 가중치로 찾는다.~문장에서는 먼 토큰의 관계도 직접 연결하고 싶다.~다음 주에는 관계의 가중치를 입력마다 계산한다. | 다음: Q·K·V를 직접 구현한다 | 합성곱과 어텐션을 완전히 대립되는 개념으로 소개하지 않는다. 둘 다 표현을 섞지만 이웃 선택과 가중치 결정 방식이 다르다는 연결점을 제시한다.
'''),
부록=lines('''
부록 · 전이 학습의 절차 | 사전학습 가중치와 해당 전처리를 함께 확인한다.~분류 헤드를 교체하고 먼저 헤드만 학습할 수 있다.~이후 작은 lr로 일부 또는 전체를 미세조정한다. | pretrained backbone → new head → fine-tune | 핵심 실습은 다운로드와 입력 크기 비용을 줄이기 위해 작은 모델을 처음부터 학습한다. 사전학습 ResNet을 추가한다면 weights.transforms()와 클래스 수 변경을 확인한다.
부록 · freeze와 eval | requires_grad=False는 파라미터 gradient를 막는다.~eval은 BatchNorm·Dropout의 모드를 바꾼다.~동결된 backbone의 통계 갱신 여부도 결정해야 한다. | freeze ≠ eval | model.train() 호출이 모든 하위 모듈을 학습 모드로 되돌릴 수 있다. backbone.eval()을 다시 설정해야 하는 상황을 예로 든다.
부록 · label smoothing | 정답 확률 목표를 한 점에서 조금 분산한다.~과도한 확신을 완화할 수 있다.~강도가 크면 학습 신호를 약하게 만들 수 있다. | nn.CrossEntropyLoss(label_smoothing=0.1) | 정확도와 calibration은 서로 다른 지표다. label smoothing이 모든 데이터에서 반드시 개선된다고 말하지 않는다.
부록 · class imbalance | 전체 정확도가 다수 클래스에 좌우될 수 있다.~클래스별 recall과 macro 평균을 함께 본다.~샘플링·가중 손실은 분포와 목적에 따라 선택한다. | macro recall = 클래스별 recall의 평균 | CIFAR10은 기본적으로 클래스 균형 데이터지만 부분집합은 약간 달라질 수 있다. 실제 업무 데이터에서 분포가 달라질 경우를 토론한다.
부록 · mixed precision | 일부 연산을 낮은 정밀도로 수행한다.~GPU에서 메모리와 속도 이점을 얻을 수 있다.~수치 안정성과 지원 dtype을 확인해야 한다. | 속도 향상은 하드웨어와 모델에 의존한다 | 이 강의의 필수 코드는 복잡도를 줄여 float32로 실행한다. 최신 환경의 torch.amp API를 확인하고 적용한다. CPU에서 GPU와 같은 이득을 기대하지 않는다.
부록 · 재현성의 경계 | seed는 난수 흐름을 고정하는 출발점이다.~device와 라이브러리 버전에 따라 결과가 달라질 수 있다.~완전한 bitwise 재현과 통계적 재현을 구분한다. | 동일 설정 + 여러 seed + 분포 보고 | 평균과 표준편차는 반복 수와 함께 적는다. seed 하나에서 이긴 기법을 보편적 승자로 부르지 않는다.
부록 · 다음 실험 제안 | residual on/off를 같은 조건으로 비교한다.~class recall을 구현하고 최약 클래스를 찾는다.~실패 사례를 한 가지 증강 가설로 연결한다. | 자율 과제: residual ablation + 보고서 | 리더용 해설은 projected shortcut과 클래스 지표 계산을 포함한다. 실험 결과는 실행마다 달라서 정해진 정확도 기준으로 통과 여부를 판정하지 않는다.
''')),

dict(folder='week03_attention_basics',title='Attention ① 관계를 계산하는 방법',tag='TOKENS\n→ RELATIONS',goals=['Q·K·V와 scaled dot-product attention을 손으로 계산한다.','causal mask와 텐서 shape를 검증한다.','단일 헤드 문자 언어 모델을 학습하고 생성한다.'],intro='행렬 곱, softmax, cross entropy를 복습한다. 외부 텍스트 다운로드 없이 교육용으로 직접 만든 문장 코퍼스를 사용한다. 결과는 언어 모델 원리 관찰용이며 자연어 능력 벤치마크가 아니다.',refs=ATT_REFS,
개념=lines('''
언어 모델의 과제 | 앞의 토큰을 보고 다음 토큰의 분포를 예측한다.~학습 입력과 정답은 한 칸 어긋난 시퀀스다.~생성은 예측한 토큰을 문맥 뒤에 붙이며 반복한다. | FLOW:the ca>다음 토큰 분포>t 샘플링>the cat | 토큰은 단어와 같지 않다. 오늘은 구현이 단순한 문자 단위를 사용한다. 실제 대형 언어 모델의 subword 토큰화와는 다르다.
문자를 정수로 바꾸기 | train 문자 집합으로 vocabulary를 만든다.~각 문자를 ID로 바꾸고 embedding에서 벡터를 찾는다.~모르는 문자를 위한 UNK를 둔다. | 문자 → token ID → 학습 가능한 벡터 | 정수 ID의 크기는 의미적 순서가 아니다. ID 7이 ID 3보다 어떤 의미로 크다는 뜻이 아니다. embedding은 학습되는 lookup table이다.
고정 문맥의 한계 | bigram은 바로 이전 문자에만 조건부로 의존한다.~문장 의미는 더 먼 위치에도 의존할 수 있다.~어텐션은 문맥 안의 여러 위치를 가중합한다. | p(x_t ∣ x_<t) 와 p(x_t ∣ x_(t−1)) | 문맥 창 밖 정보는 오늘 모델에서 접근할 수 없다. 어텐션이 모든 기억을 무한히 보관하는 구조가 아님을 짚는다.
Q·K·V의 역할 | Query는 현재 위치가 찾는 관계의 기준이다.~Key는 각 위치가 비교에 내놓는 표현이다.~Value는 가중합으로 전달할 내용이다. | FLOW:X>Q=XW_Q / K=XW_K / V=XW_V>유사도>가중합 | 도서관 검색 비유는 직관일 뿐 각 차원이 실제 검색어에 대응하는 것은 아니다. 세 투영은 서로 다른 학습 파라미터다. self-attention에서는 같은 X에서 세 가지를 만든다.
self와 cross attention | self-attention은 Q·K·V가 같은 시퀀스에서 나온다.~cross-attention은 Q와 K·V의 출처가 다르다.~오늘은 decoder 방식의 causal self-attention을 만든다. | self: X→Q,K,V / cross: X→Q, Y→K,V | 원래 Transformer 논문은 encoder-decoder 구조다. 이 수업의 소형 문자 모델은 decoder-only 교육용 모델이며 논문 전체를 그대로 재현하지 않는다.
미래를 보지 않는 규칙 | 위치 t는 위치 t 이하만 볼 수 있다.~학습에서 모든 위치를 병렬로 계산해도 이 규칙을 지킨다.~미래 점수에 −∞를 넣고 softmax한다. | FLOW:점수 행렬>상삼각 차단>행별 softmax>과거 정보 혼합 | mask 없이 한 칸 밀린 정답을 학습하면 뒤 위치의 정답 정보가 들어와 손실이 부당하게 낮아진다. 입력에 현재 문자는 있고 목표는 그 다음 문자이므로 대각선은 허용한다.
위치 정보가 필요한 이유 | 내용만 같으면 순서 차이를 직접 표시하기 어렵다.~토큰 embedding에 위치 embedding을 더한다.~문맥 길이는 위치 embedding 범위를 넘지 않아야 한다. | H_t = token_embedding(x_t) + position_embedding(t) | causal mask도 순서 구조를 제공하지만 위치 embedding은 위치를 명시적으로 표현한다. 순서를 전혀 모른다는 단정 대신 명시적 위치 표현의 목적을 설명한다.
'''),
수식=lines('''
전체 식을 세 단계로 읽기 | QKᵀ로 query와 key의 내적을 계산한다.~sqrt(d_k)로 나누고 각 행을 softmax한다.~행의 가중치로 V를 합한다. | Attention(Q,K,V) = softmax(QKᵀ / sqrt(d_k) + M)V | softmax는 key 축인 마지막 차원에 적용한다. query마다 별도의 확률 분포가 나온다. 값의 차원 d_v는 일반적으로 d_k와 같을 필요가 없지만 오늘은 같게 둔다.
shape 추적표 | Q,K,V는 [B,T,D]다.~Q @ K.transpose(−2,−1)은 [B,T,T]다.~가중치 @ V의 결과는 [B,T,D]다. | [B,T,D] × [B,D,T] → [B,T,T] | transpose를 배치 축에 적용하는 실수를 경계한다. B는 독립된 시퀀스 묶음이다. 다른 배치의 문장끼리 attention하지 않는다.
손계산 · 두 key와 한 query | q=[1,0], k₁=[1,0], k₂=[0,1]로 둔다.~scaled score는 [0.707,0]이다.~softmax는 약 [0.670,0.330]이다. | v₁=[2,0], v₂=[0,2] → output≈[1.34,0.66] | 내적, 나눗셈, 지수와 정규화, value 가중합을 순서대로 적는다. output은 key 벡터의 평균이 아니라 value 벡터의 가중합이라는 점을 다시 확인한다.
왜 sqrt(d_k)로 나눌까 | 독립·평균 0·분산 1인 성분을 가정한다.~d_k개 곱의 합은 분산이 d_k 규모가 된다.~스케일을 줄여 softmax가 너무 뾰족해지는 것을 완화한다. | Var(q·k)≈d_k → Var((q·k)/sqrt(d_k))≈1 | 이는 초기 분포에 대한 직관이지 모든 학습 시점의 정확한 분산 보장이 아니다. 큰 logits에서는 softmax가 포화해 gradient가 작아질 수 있다.
causal mask 손으로 채우기 | 길이 3이면 첫 행은 첫 key만 허용한다.~둘째 행은 첫째·둘째 key를 허용한다.~마지막 행은 세 key를 모두 허용한다. | M = [[0,−∞,−∞],[0,0,−∞],[0,0,0]] | softmax 전에 mask를 더하거나 masked_fill한다. 행 전체를 −∞로 만들면 NaN이 될 수 있다. 오늘의 대각선 허용 mask는 각 행에 최소 한 개 허용 위치가 있다.
언어 모델 손실 | logits는 [B,T,Vocab]이고 target은 [B,T]다.~B와 T를 묶어 위치별 cross entropy를 계산한다.~평균 손실의 지수값은 이 설정의 perplexity다. | CE(logits.reshape(−1,V), target.reshape(−1)) | perplexity는 같은 tokenizer, 데이터, 평가 조건에서 비교한다. 문자 모델과 subword 모델의 perplexity를 그대로 비교하면 의미가 다르다. 자연로그 CE를 사용하므로 exp를 취한다.
'''),
설계=lines('''
한 헤드부터 직접 구현 | Linear 세 개로 Q·K·V를 만든다.~scaled score와 causal mask를 직접 계산한다.~attention 결과를 vocabulary logits로 변환한다. | FLOW:token+position>single-head attention>Linear>next token | 이 모델은 residual과 FFN을 갖춘 완전한 Transformer block이 아니다. 4주차에서 그 요소를 추가한다. 간단한 모델로 attention 핵심을 분리해 배운다.
softmax 축을 검증하기 | 각 query 행의 가중치 합이 1인지 검사한다.~미래 key의 가중치가 0인지 검사한다.~출력 shape와 유한값 여부를 검사한다. | CODE:assert torch.allclose(A.sum(-1),;;    torch.ones_like(A.sum(-1))) | softmax(dim=1)처럼 query 축을 정규화해도 shape는 맞아 오류가 숨을 수 있다. shape 검사만으로 충분하지 않아 의미적 불변량을 함께 확인한다.
코퍼스를 먼저 분할하기 | 문장 단위 train/validation/test를 먼저 나눈다.~각 split 안에서만 문자 시퀀스와 window를 만든다.~겹치는 window를 무작위로 나눈 뒤 평가하지 않는다. | 문장 분할 → 인코딩 → split별 window | 교육용 코퍼스는 동일 문법의 조합 문장이다. exact duplicate 문장은 split 간 중복되지 않지만 문법과 단어는 공유하므로 일반 자연어 일반화 증거가 아니다. UNK 비율을 확인한다.
입력과 target 확인 | block_size=32면 33개 토큰이 필요하다.~앞 32개는 입력, 뒤 32개는 정답이다.~decode로 한 칸 shift를 눈으로 확인한다. | x=tokens[i:i+T], y=tokens[i+1:i+T+1] | train과 val 토큰을 이어 붙여 window를 만들면 경계를 넘어 데이터가 섞인다. 노트북 함수는 주어진 split만 사용한다. 생성 prompt도 vocabulary 범위 안으로 인코딩한다.
샘플링은 선택 규칙이다 | argmax는 가장 높은 확률을 항상 고른다.~multinomial은 분포에서 무작위로 뽑는다.~temperature는 logits를 나눠 분포의 집중도를 바꾼다. | p = softmax(logits / temperature), temperature > 0 | temperature가 낮아지면 더 집중된다. 생성에는 eval과 no_grad를 사용하고 문맥을 마지막 block_size만 남긴다. 같은 prompt라도 seed와 설정에 따라 결과가 달라진다.
'''),
실습=lines('''
LAB 1 · attention 연산 구현 | practice §1–3: Q·K·V 손계산과 코드를 비교한다.~mask 전후 heatmap을 나란히 그린다.~행 합·미래 차단·shape assert를 통과한다. | 50–58분 / 산출물: 점수와 attention heatmap | heatmap의 행은 query, 열은 key다. 숫자를 읽어 첫 행이 자기 자신에 1을 주는지 확인한다. 이 손계산 Q,K,V는 학습된 값이 아닌 설명용 값이다.
LAB 2 · 문자 언어 모델 학습 | practice §4–5: 코퍼스 분할과 shift를 확인한다.~bigram과 단일 헤드 모델을 짧게 학습한다.~동일 validation 토큰에서 손실을 비교한다. | 58–70분 / 산출물: CE·perplexity 비교 | 두 모델은 구조와 파라미터 수가 달라 엄밀한 attention 단독 효과 실험은 아니다. 유의미한 방향을 관찰하는 기준 비교다. 최종 선택은 val을 사용한다.
LAB 3 · 생성과 누출 검사 | practice §6–7: temperature를 바꾸어 생성한다.~미래 토큰만 바꿔 이전 logits가 동일한지 검사한다.~학습된 attention 한 행을 읽고 한계를 적는다. | 70–75분 / 산출물: 생성 3개 + causal 검사 | 미래 변경에 앞부분 logits가 변하면 mask나 attention 계산 축을 점검한다. eval 모드로 dropout 등을 끈 뒤 비교해야 한다. 어텐션 가중치를 문법 이해의 직접 증거로 해석하지 않는다.
'''),
분석=lines('''
좋은 문자열과 좋은 모델 | 그럴듯한 샘플 한 개는 충분한 평가가 아니다.~held-out loss와 여러 prompt 출력을 함께 본다.~반복·철자·문장 구조를 분리해 기록한다. | 숫자 평가 + 표본 관찰 + 데이터 한계 | 작은 합성 코퍼스는 생성 모델의 흐름을 보여 주기 위한 장치다. 작은 perplexity가 실제 글쓰기 능력을 의미하지 않는다. 선택한 좋은 샘플만 발표하지 않는다.
mask를 빼면 왜 손실이 낮을까 | 학습 target은 입력의 다음 위치에 존재한다.~미래 참조를 허용하면 정답을 훔쳐볼 수 있다.~생성 시 미래가 없으므로 평가와 사용 조건이 달라진다. | 낮은 손실이 항상 좋은 모델은 아니다 | 양방향 attention 자체가 잘못된 것은 아니다. causal 다음 토큰 학습이라는 목적과 맞지 않는 것이 문제다. masked language modeling과 구분한다.
attention 그림의 한계 | 가중치는 value를 섞는 계수다.~뒤의 Linear와 다른 표현 경로도 예측에 영향을 준다.~높은 가중치를 인과적 중요도로 단정하지 않는다. | 관계를 관찰하되 설명을 과장하지 않는다 | 특정 key의 value를 제거하거나 토큰을 바꾸는 실험도 가능하지만 그 역시 분포 변화를 만든다. 해석의 한계를 명시하는 습관을 배운다.
'''),
회고=lines('''
출구 퀴즈 | Q1. QKᵀ가 만드는 두 T축은 각각 무엇인가?~Q2. mask는 softmax 전과 후 중 언제 적용할까?~Q3. value 대신 key를 가중합하면 같은 연산일까? | query 위치 / key 위치 / 전달할 내용 | 정답: 행=query, 열=key; softmax 전; 일반적으로 다르다. 후에 0으로만 지우면 행 합이 1이 아니며 재정규화 등 별도 처리가 필요하다.
다음: 여러 관점과 깊이 | 단일 헤드로 한 종류의 혼합을 만들었다.~여러 헤드와 FFN으로 표현을 확장한다.~residual과 LayerNorm으로 블록을 연결한다. | 자율 과제: mask·scaled attention·문맥 길이 | 자율 과제에서는 SDPA 공식 API와 직접 구현을 비교한다. API마다 boolean mask의 의미가 다를 수 있음을 다음 부록과 연결한다.
'''),
부록=lines('''
부록 · 안정적인 softmax | 각 행에서 최댓값을 빼도 확률은 같다.~지수 계산의 overflow 위험을 줄인다.~PyTorch softmax는 직접 exp를 나누는 것보다 적절하다. | softmax(s) = softmax(s−max(s)) | softmax 함수의 이동 불변성을 손으로 유도한다. temperature로 나누는 스케일 변환과 최댓값을 빼는 이동은 다른 연산이다.
부록 · padding mask | 길이가 다른 문장을 배치로 묶을 때 padding을 넣는다.~padding key를 보지 않도록 차단할 수 있다.~padding target은 손실에서 ignore_index로 제외한다. | causal mask + padding key mask + loss mask | 오늘 고정 길이 window에는 padding이 필요 없다. attention mask와 loss mask는 서로 대체되지 않는다. 모두 masked인 query 행의 NaN도 고려한다.
부록 · API의 boolean mask | 직접 masked_fill은 True인 곳을 차단하도록 작성할 수 있다.~SDPA의 bool attn_mask는 True인 곳을 허용한다.~MultiheadAttention의 일부 mask 규칙과 혼동하지 않는다. | 사용 중인 API 문서를 먼저 확인한다 | 수업 구현은 변수명을 blocked로 두어 True=차단을 명시한다. SDPA와 비교할 때는 ~blocked를 전달한다. 규칙을 외우기보다 작은 행렬로 검증한다.
부록 · 문맥 길이와 비용 | attention 행렬은 T×T다.~문맥 길이를 두 배로 하면 원소 수는 네 배다.~projection 비용과 batch 크기도 총비용에 영향을 준다. | attention matrix memory ∝ B T² | 효율적 구현은 전체 행렬 저장을 줄일 수 있으나 수업의 명시적 구현은 실제 T² 행렬을 만든다. 계산 복잡도와 메모리 구현을 구분한다.
부록 · tokenizer의 선택 | 문자 단위는 vocabulary가 작고 시퀀스가 길다.~단어 단위는 희귀 단어와 UNK 문제가 커진다.~subword는 두 방식 사이의 절충을 제공한다. | 토큰화가 바뀌면 T와 V가 함께 바뀐다 | 오늘 영문 합성 코퍼스를 쓰는 이유는 원리를 가볍게 실행하기 위해서다. 한국어로 바꿀 때는 자모와 완성형 문자, 띄어쓰기 정책도 고려한다.
부록 · 온도 손계산 | logits=[2,1,0]에서 τ를 비교한다.~τ=0.5는 더 뾰족하고 τ=2는 더 완만하다.~τ가 작아도 확률 샘플링은 완전한 argmax와 다를 수 있다. | τ→0⁺일 때 최대 점수에 확률 집중 | temperature=0을 나눗셈에 넣지 않는다. greedy는 별도 argmax 분기로 구현한다. ties가 있으면 극한의 설명에도 주의한다.
부록 · baseline의 가치 | bigram은 장거리 정보를 볼 수 없다.~구현이 단순해 데이터·loss 오류를 찾기 쉽다.~복잡한 모델이 이기지 못하면 설정과 예산을 점검한다. | baseline은 비교 기준이자 디버깅 도구다 | 짧은 코퍼스에서 bigram도 일부 철자 패턴을 학습한다. 더 복잡한 모델이 제한된 시간 안에 항상 이기는 것은 아니다.
''')),

dict(folder='week04_transformer_lm',title='Attention ② 작은 Transformer 만들기',tag='BLOCKS\n→ LANGUAGE',goals=['multi-head attention과 pre-norm block을 구현한다.','소형 decoder 언어 모델을 학습하고 sampling을 비교한다.','데이터 누출·인과성·평가 조건을 검사한다.'],intro='3주차 attention과 문자 코퍼스를 독립적으로 다시 정의해 이전 실행 상태 없이 시작할 수 있다. 기본 실습은 짧은 문맥과 작은 모델을 사용한다. 대형 모델 학습이나 챗봇 품질을 목표로 하지 않는다.',refs=ATT_REFS,
개념=lines('''
단일 헤드에서 multi-head로 | 같은 입력을 여러 부분 공간으로 투영한다.~각 헤드가 별도의 attention 분포를 계산한다.~헤드 출력을 합치고 출력 투영을 적용한다. | FLOW:X>head 1 / head 2 / head 3 / head 4>concat>W_O | 헤드마다 문법적 역할이 자동으로 하나씩 배정되는 것은 아니다. 서로 다른 혼합을 학습할 수 있는 구조적 여지를 준다고 설명한다.
블록의 두 하위 연산 | attention은 토큰 사이 정보를 섞는다.~FFN은 각 위치의 채널 표현을 비선형으로 바꾼다.~각 하위 연산에 residual 경로를 둔다. | FLOW:LayerNorm>Attention>+residual>LayerNorm>FFN + residual | FFN은 위치별 같은 가중치를 사용하며 직접 다른 위치를 참조하지 않는다. CNN의 1×1 Conv와 채널 변환 관점에서 연결할 수 있다.
LayerNorm과 BatchNorm | LayerNorm은 보통 각 토큰의 마지막 특징 축을 정규화한다.~배치의 다른 문장 통계를 모으지 않는다.~train/eval에서 BatchNorm처럼 running 통계를 전환하지 않는다. | LayerNorm(D): 각 [D] 벡터를 정규화 | LayerNorm에도 학습 가능한 scale과 bias가 있다. 정규화 차원을 지정하는 normalized_shape가 중요하다. B,T까지 함께 정규화하지 않도록 한다.
pre-norm으로 쌓기 | 하위 연산 전에 LayerNorm을 적용한다.~residual 경로는 정규화 바깥으로 통과한다.~최종 블록 뒤에도 LayerNorm을 적용한다. | x ← x + Attn(LN(x)); x ← x + FFN(LN(x)) | 원래 Transformer 논문의 post-norm과 수업의 pre-norm이 다르다는 점을 명시한다. 교육용 구현을 원문 완전 복제라고 소개하지 않는다.
학습과 생성의 계산 차이 | 학습은 정답 문맥으로 모든 위치를 병렬 계산한다.~생성은 새 토큰 하나를 뽑은 뒤 다시 실행한다.~이전 예측의 실수가 다음 문맥에 들어간다. | teacher forcing / autoregressive generation | 오늘 생성 코드는 전체 문맥을 반복 계산해 단순하지만 비효율적이다. KV cache는 부록에서 개념만 다룬다.
decoder-only 모델의 범위 | token·position embedding과 여러 block을 연결한다.~마지막 Linear가 다음 문자 logits를 만든다.~지시 따르기나 대화 정렬 학습은 포함하지 않는다. | 작은 next-token 모델 ≠ 완성된 대화 서비스 | 구조를 배운 것과 대규모 데이터에서 능력이 생기는 것을 구분한다. 작은 모델의 철자 생성 결과를 과장하지 않는다.
좋은 평가 단위 | 같은 held-out 토큰에 CE와 perplexity를 계산한다.~최고 validation checkpoint를 선택한다.~test는 선택 완료 후 한 번 사용한다. | 토큰 수로 가중 평균한 CE → exp(CE) | 랜덤 배치 몇 개만 뽑으면 평가가 흔들린다. 노트북은 평가 시 고정된 연속 window를 사용하고 같은 표본에 대해 비교한다.
'''),
수식=lines('''
헤드 shape 변환 | d_model=64, heads=4이면 d_head=16이다.~[B,T,64]를 [B,T,4,16]으로 나눈다.~transpose 후 [B,4,T,16]으로 attention한다. | [B,H,T,d] @ [B,H,d,T] → [B,H,T,T] | d_model이 heads로 나누어 떨어져야 한다. head 수를 늘린다고 d_model 고정에서 projection 파라미터가 단순히 head 수 배로 늘지 않는다.
헤드 결합하기 | 결과 [B,H,T,d]를 [B,T,H,d]로 돌린다.~reshape로 [B,T,D]를 만든다.~W_O로 헤드 정보를 섞는다. | CODE:y = y.transpose(1, 2);;y = y.contiguous().view(B,T,D);;y = self.out(y) | transpose 후 메모리 배치가 비연속일 수 있어 view 전에 contiguous가 필요하다. reshape는 필요한 경우 복사할 수 있다. 버그가 나면 shape뿐 아니라 축 의미를 읽는다.
FFN의 폭과 파라미터 | D→4D→D의 두 선형층을 사용한다.~중간에 GELU와 dropout을 넣을 수 있다.~attention 외에 FFN도 큰 파라미터 비중을 가진다. | FFN weight count ≈ 8D² (bias 제외) | D×4D와 4D×D를 더해 8D²다. Q,K,V,O 투영은 약 4D²이므로 작은 블록에서도 FFN을 무시할 수 없다.
LayerNorm 계산 | 한 토큰의 D개 값에서 평균과 분산을 구한다.~표준화 후 학습 가능한 affine 변환을 한다.~배치 크기 B에 직접 의존하지 않는다. | LN(h) = γ ⊙ (h−mean(h))/sqrt(var(h)+ε) + β | 노트북의 nn.LayerNorm(D)는 마지막 D축에 적용한다. 정규화로 모든 값이 같은 것이 아니라 평균과 스케일을 조정하는 것이다.
top-k sampling | vocabulary 중 점수가 높은 k개만 남긴다.~나머지는 −∞로 바꾼 뒤 softmax한다.~k=1이면 최대 점수 토큰만 선택한다. | logits < kth_value → −∞ | ties 처리 때문에 임계값 방식은 k개보다 많이 남길 수 있다. 실습은 topk 인덱스를 scatter해 정확히 k개를 남긴다. k는 1부터 vocabulary 크기까지 제한한다.
perplexity 해석 | CE가 ln(10)이면 perplexity는 10이다.~확률적으로 유효한 선택지 수라는 직관으로 읽을 수 있다.~정확한 후보 개수를 세는 지표는 아니다. | PPL = exp(평균 token NLL) | 배치별 perplexity의 평균과 전체 CE의 exp는 다르다. loss를 먼저 표본 수로 집계한 뒤 exp한다. 같은 vocabulary의 uniform baseline도 참고할 수 있다.
'''),
설계=lines('''
모듈을 네 부분으로 나누기 | CausalMHA는 헤드 분할·mask·결합을 맡는다.~Block은 LN·residual·FFN을 묶는다.~TinyTransformer는 embedding과 block stack·head를 연결한다. | FLOW:Embedding>Block × 2>LayerNorm>LM head | 한 클래스에 모든 연산을 넣기보다 기능별로 나누면 shape와 causal 테스트가 쉽다. notebook은 외부 모듈 import 없이 모든 정의를 포함한다.
dropout이 평가를 흔들지 않게 | 학습에서는 model.train()으로 dropout을 활성화한다.~평가·생성·causal 검사에서는 model.eval()을 사용한다.~평가를 마친 후 학습 모드를 복원한다. | 반복 forward가 같은지 eval에서 확인 | no_grad만 사용하면 dropout은 계속 작동한다. 직접 구현은 nn.Dropout을 사용하며 SDPA의 dropout_p 설정과 규칙이 다르다.
학습 안정성 점검 | loss가 유한한지 확인하고 gradient norm을 기록한다.~필요하면 clip_grad_norm_으로 큰 gradient를 제한한다.~optimizer는 모든 학습 파라미터를 받아야 한다. | zero_grad → backward → clip → step | clip을 backward 전에 호출하면 현재 gradient가 없다. clipping은 데이터 누출이나 잘못된 loss를 고쳐 주지 않는다. max_norm=1.0은 수업 기본값이지 보편적 최적값이 아니다.
실험 예산을 명시하기 | steps·batch·context 길이로 처리 토큰 수를 계산한다.~비교할 때 parameter 수와 학습 시간을 함께 기록한다.~하나의 seed 결과는 탐색 결과로 표현한다. | processed tokens = steps × B × T | 모델이 보는 window는 겹칠 수 있어 처리 토큰 수가 고유 데이터 양과 같지 않다. GPU를 바꾸면 같은 시간 예산의 업데이트 수가 달라진다.
checkpoint에 넣을 것 | model state와 구조 설정을 저장한다.~vocabulary와 토큰화 규칙도 함께 저장한다.~생성을 재현하려면 prompt·sampling 설정·seed를 남긴다. | weights + config + vocabulary + run settings | state_dict만으로 다른 모델 구조를 자동 복원할 수 없다. 외부 출처 checkpoint를 무심코 실행하지 말고 본인이 만든 파일을 읽는다. 실습은 로컬 생성 파일만 사용한다.
'''),
실습=lines('''
LAB 1 · multi-head 구현 검사 | practice §1–3: head shape와 causal mask를 확인한다.~미래 토큰 변경에 과거 logits가 불변인지 검사한다.~모델 파라미터 수를 출력한다. | 50–59분 / 산출물: shape·causality assert | mask는 모든 head와 batch에 브로드캐스트된다. 미래 변경은 중간 위치 이후만 바꾸고 그 이전 출력들을 비교한다. padding 없는 고정 window를 사용한다.
LAB 2 · 학습과 평가 | practice §4: TinyTransformer를 짧게 학습한다.~train/validation CE 곡선을 저장한다.~최고 val state를 복원하고 test CE를 구한다. | 59–69분 / 산출물: CE 곡선 + 최종 PPL | 시간이 부족하면 steps를 줄이되 평가 셀을 생략하지 않는다. 기본 학습으로 유창한 문장이 나오지 않을 수 있다. 검증 보고서의 실제 출력과 함께 설명한다.
LAB 3 · 생성 제어 | practice §5–6: greedy, temperature, top-k를 비교한다.~동일 prompt와 seed를 사용하고 반복성을 관찰한다.~checkpoint를 저장·복원해 logits가 같은지 확인한다. | 69–75분 / 산출물: 4개 생성 결과 + 복원 검사 | 각 조건을 여러 번 봐야 안정적으로 평가할 수 있다. 과도하게 높은 온도는 다양성뿐 아니라 오류도 늘릴 수 있다. 저장 파일은 실행 위치의 outputs 폴더에 생성된다.
'''),
분석=lines('''
어떤 샘플링이 좋은가 | greedy는 반복되고 단조로울 수 있다.~온도를 높이면 낮은 확률 문자도 더 자주 나온다.~top-k는 후보를 제한하지만 사실성을 보장하지 않는다. | 유창성 / 반복 / 다양성 / 목적 | 샘플링 변화는 가중치를 다시 학습하는 것이 아니다. 생성 알고리즘과 모델 학습을 구분해서 보고한다.
헤드 수 실험 해석 | D를 고정하면 head당 차원이 달라진다.~head 수가 많아도 성능이 항상 좋아지지 않는다.~같은 예산과 여러 seed로 비교한다. | H↑ ⇒ d_head=D/H↓ | 더 많은 헤드가 더 많은 독립 지식을 저장한다는 설명은 피한다. 다른 분할 방식이 최적화와 표현에 영향을 주는 것을 실험한다.
학습 데이터와 생성 결과 | 훈련 문장의 반복을 생성 능력으로 과장하지 않는다.~새 조합에서도 규칙을 유지하는지 관찰한다.~합성 코퍼스 성능을 일반 자연어로 외삽하지 않는다. | 암기 / 조합 일반화 / 실제 언어 능력 구분 | 데이터의 문법 자체가 단순하므로 평가 난도가 낮다. 모델 능력을 논하려면 더 다양한 실제 held-out 데이터와 적합한 지표가 필요하다.
'''),
회고=lines('''
출구 퀴즈 | Q1. D=96, H=6이면 head 차원은?~Q2. FFN은 서로 다른 위치를 직접 섞을까?~Q3. 생성 시 dropout을 끄는 호출은? | shape / 연산 역할 / 실행 모드 | 정답: 16, 직접 섞지 않음, model.eval(). no_grad는 메모리 절약과 gradient 비활성화를 위해 함께 사용한다.
다음: 확률 분포에서 이미지 생성 | 언어 모델은 다음 토큰의 분포를 학습했다.~VAE는 잠재변수와 이미지의 확률적 관계를 학습한다.~우도·샘플링·표현이라는 공통 주제로 이어진다. | 다음: z를 뽑아 새로운 이미지를 만든다 | autoregressive 모델과 VAE는 생성 분해 방식이 다르다. VAE가 언어 모델을 대체하는 후속 단계라기보다 다른 생성 모델 계열이라고 설명한다.
'''),
부록=lines('''
부록 · sinusoidal position | 위치마다 사인·코사인 주파수를 조합한다.~학습형 위치 embedding과 다른 설계 선택이다.~더 긴 길이의 성능이 자동 보장되는 것은 아니다. | PE(pos,2i)=sin(pos / 10000^(2i/D)) | 수업 구현은 learned positional embedding을 쓴다. 원 논문의 위치 표현과 차이를 비교한다. 모양을 그려 주파수가 다른 축들을 관찰한다.
부록 · KV cache | 생성 중 과거 key와 value를 저장해 재사용한다.~새 토큰의 query 중심으로 추가 계산한다.~cache 길이와 위치 인덱스 관리가 필요하다. | 반복 계산 감소 ↔ cache 메모리 증가 | 오늘 sliding window와 learned position을 그대로 캐시화하면 위치 재설정 문제가 생긴다. 단순히 과거 K,V를 붙이는 것만으로 정확한 구현이 되지 않을 수 있다.
부록 · weight tying | token embedding과 출력 projection 가중치를 공유할 수 있다.~vocabulary 관련 파라미터 수를 줄인다.~shape와 초기화 전략을 확인한다. | lm_head.weight = token_embedding.weight | 수업 핵심은 독립 가중치를 사용한다. tying을 추가할 때 optimizer가 중복 업데이트하지 않는지 파라미터 집계를 확인한다.
부록 · top-p sampling | 정렬된 확률의 누적합이 p에 닿는 후보 집합을 쓴다.~확신 수준에 따라 후보 수가 달라진다.~최소 한 개의 후보를 유지해야 한다. | 고정 k / 가변 nucleus | threshold를 넘기는 첫 토큰을 포함할지의 구현을 주의한다. 자율 과제의 도전 구현으로 제안하며 핵심 실습은 top-k까지만 한다.
부록 · SDPA 사용 시 주의 | fused attention API가 속도·메모리를 개선할 수 있다.~dropout_p는 eval에서 자동 0이 된다고 가정하지 않는다.~직접 구현과 dropout=0에서 수치 비교한다. | dropout_p = p if training else 0.0 | 공식 문서를 확인한 뒤 사용한다. bool mask는 True=허용이다. 수치 허용 오차는 dtype과 device에 맞춰 설정한다.
부록 · 긴 문맥과 학습 예산 | T 증가로 한 step의 토큰 수와 attention 비용이 늘어난다.~같은 steps 비교는 처리 토큰 예산이 달라진다.~동일 토큰 예산과 동일 시간 예산을 구분한다. | 공정성의 기준을 먼저 정한다 | 정확히 같은 모든 조건을 동시에 고정할 수 없는 경우가 있다. 무엇을 고정하고 무엇이 바뀌었는지 표에 명시하는 것이 중요하다.
부록 · 작은 프로젝트 제안 | head 수 또는 FFN 폭을 한 가지 바꾼다.~학습 곡선과 생성 오류 유형을 함께 비교한다.~세 번 반복하고 평균·변동을 보고한다. | 자율 과제: top-k + block + head ablation | 좋은 실험에는 실패 사례와 개선되지 않은 조건도 포함된다. 리더 해설의 참고 구현으로 단위 검사를 확인하고 결과 해석은 직접 작성한다.
''')),

dict(folder='week05_vae_basics',title='VAE ① 잠재변수로 이미지 생성하기',tag='IMAGE\n↔ LATENT',goals=['AE와 VAE의 차이를 확률 분포 관점에서 설명한다.','재매개화와 ELBO 손실을 올바르게 구현한다.','MNIST 재구성·prior 샘플·잠재 보간을 구분한다.'],intro='로그, 평균과 분산, 정규분포를 짧게 복습한다. MNIST 픽셀은 0~1 범위로 유지한다. 이미지 분류에서 사용한 [-1,1] Normalize를 그대로 적용하지 않는다.',refs=VAE_REFS,
개념=lines('''
분류에서 생성으로 | 분류기는 이미지가 어떤 클래스인지 예측한다.~생성 모델은 데이터의 분포를 학습하려 한다.~VAE는 관측 x와 숨은 변수 z의 관계를 모델링한다. | FLOW:z ~ prior>decoder>이미지 분포>샘플 x | 생성 이미지에서 픽셀 평균을 그리는 것과 likelihood에서 실제 픽셀을 샘플링하는 것을 구분한다. 실습 그림은 보통 sigmoid 확률을 표시한다.
Autoencoder 복습 | encoder가 입력을 작은 벡터로 압축한다.~decoder가 그 벡터에서 입력을 복원한다.~재구성만 잘해도 임의 z가 좋은 이미지를 만든다는 보장은 없다. | FLOW:x>encoder>z>decoder>x̂ | 잠재 공간에 학습 입력이 없는 빈 영역이 생길 수 있다. AE의 임의 Gaussian 샘플이 자연스럽지 않을 수 있다는 동기를 제시하되 모든 AE가 생성 불가능하다고 단정하지 않는다.
VAE의 encoder는 분포를 만든다 | 입력마다 μ와 log σ²를 출력한다.~qφ(z∣x)를 diagonal Gaussian으로 둔다.~z 한 점 대신 가능한 잠재 표현의 분포를 학습한다. | qφ(z∣x) = N(μ(x), diag(σ²(x))) | encoder가 실제 posterior를 정확히 계산하는 것은 아니다. 복잡한 p(z∣x)를 근사하기 위해 학습하는 분포다. 대각 공분산은 표현상의 가정이다.
prior와 posterior | prior p(z)는 입력을 보기 전의 가정이다.~posterior p(z∣x)는 관측 후의 분포다.~qφ(z∣x)는 posterior를 대신하는 근사 분포다. | p(z)=N(0,I), qφ(z∣x)≈pθ(z∣x) | 생성 시에는 입력 이미지가 없으므로 prior에서 z를 뽑는다. reconstruction은 q에서 얻은 z 또는 μ를 사용한다. 두 경로를 구분하는 것이 오늘 핵심이다.
decoder도 확률 모델 | decoder는 pθ(x∣z)의 파라미터를 출력한다.~Bernoulli 모델이라면 픽셀별 logits를 출력한다.~sigmoid 후 값은 각 픽셀의 1 확률이다. | decoder(z) → logits → sigmoid → pixel probability | MNIST 회색조를 [0,1] target으로 BCE에 넣는 실용적 관례를 사용한다. 엄밀한 Bernoulli 관측은 이진값이며 필요하면 binarization을 사용할 수 있음을 밝힌다.
두 요구의 균형 | reconstruction은 입력 정보를 z에 담도록 한다.~KL은 q가 prior에서 지나치게 멀어지는 것을 억제한다.~둘의 균형이 재구성과 생성에 영향을 준다. | 잘 복원하기 + prior와 연결되는 잠재 공간 | KL이 모든 입력의 μ를 정확히 0으로 만들도록 목표한다고만 설명하지 않는다. 분포 전체와 정보량에 대한 제약이며 재구성 항과 함께 최적화한다.
샘플링과 미분의 문제 | z를 무작정 무작위 추출하면 gradient 경로를 설명하기 어렵다.~외부 잡음 ε를 뽑고 μ·σ로 변환한다.~랜덤성은 ε, 학습 가능한 경로는 μ·σ에 둔다. | z = μ + σ ⊙ ε, ε ~ N(0,I) | 이 재매개화가 연속 Gaussian 잠재변수에서 쓰이는 pathwise gradient 방식이다. 모든 종류의 이산 샘플링에 같은 식을 적용할 수는 없다.
'''),
수식=lines('''
주변 우도가 어려운 이유 | pθ(x)는 가능한 모든 z의 기여를 적분한다.~복잡한 decoder에서는 이 적분을 직접 계산하기 어렵다.~계산 가능한 하한을 최적화한다. | log pθ(x) = log ∫ pθ(x∣z)p(z) dz | 적분을 latent 차원 전체의 합산으로 직관화한다. likelihood와 posterior를 서로 바꾸어 쓰지 않도록 기호를 정리한다.
ELBO의 두 항 | 재구성 기대 로그우도는 데이터를 설명하도록 한다.~KL(q∥p)는 근사 분포와 prior 차이를 측정한다.~ELBO를 최대화하면 log p(x)의 하한을 높인다. | ELBO = E_q[log pθ(x∣z)] − KL(qφ(z∣x) ∥ p(z)) | ELBO와 log likelihood가 같아지는 것은 q가 실제 posterior와 같을 때다. negative ELBO를 loss로 최소화하므로 KL 앞 부호는 플러스가 된다.
하한과 정확한 차이 | log pθ(x)는 ELBO에 posterior KL을 더한 값이다.~KL은 0 이상이라 ELBO가 하한이 된다.~prior KL과 posterior KL을 혼동하지 않는다. | log pθ(x) = ELBO + KL(qφ(z∣x) ∥ pθ(z∣x)) | 이 식 오른쪽의 KL은 실제 posterior에 대한 KL이다. 학습 loss의 closed form KL은 prior N(0,I)에 대한 것으로 서로 대상이 다르다.
Gaussian KL 닫힌식 | prior는 표준정규, q는 대각 Gaussian이다.~μ², exp(logvar), logvar로 계산한다.~잠재 차원별 값을 합하고 배치 평균한다. | KL = ½ Σ_j [μ_j² + exp(logvar_j) − 1 − logvar_j] | μ=0, logvar=0이면 0이다. μ=[1,0], logvar=[0,0]이면 0.5다. 부호 오류를 이 작은 사례로 검증한다.
log variance에서 표준편차로 | logvar는 log σ²이다.~variance는 exp(logvar)이다.~standard deviation은 exp(0.5×logvar)이다. | z = μ + exp(0.5 × logvar) ⊙ randn_like(μ) | exp(logvar)를 바로 std로 쓰면 분산이 잘못된다. 노트북에서 큰 표본으로 평균과 분산을 경험적으로 검사한다. logvar를 출력하는 이유는 분산 양수 조건을 편하게 처리하기 위해서다.
손실 reduction을 맞추기 | reconstruction은 픽셀 합, KL은 잠재 차원 합이다.~두 항을 샘플별로 만든 뒤 배치 평균한다.~픽셀 평균으로 바꾸면 KL의 상대적 세기가 달라진다. | loss = mean_B(recon_sum_pixels + KL_sum_latents) | 28×28이면 픽셀이 784개다. recon을 평균하고 KL을 합한 뒤 β를 그대로 두면 기존과 다른 목적함수 비율이 된다. 구현에서 reduction을 명시한다.
'''),
설계=lines('''
오늘의 MLP VAE | encoder는 784→256 은닉 표현을 만든다.~μ head와 logvar head가 각각 latent_dim을 출력한다.~decoder는 latent_dim→256→784 logits를 출력한다. | FLOW:x>encoder>μ / logvar>reparameterize>decoder logits | latent_dim=2를 기본으로 하여 시각화가 쉽다. 고품질 생성에는 제한적인 설정이다. 마지막 decoder에 sigmoid를 넣지 않고 BCEWithLogits와 조합한다.
BCEWithLogits로 수치 안정성 | logits를 그대로 손실 함수에 넣는다.~시각화 시에만 sigmoid를 적용한다.~target은 0~1 범위여야 한다. | CODE:recon = F.binary_cross_entropy_with_logits(;;    logits, x, reduction='none');;recon = recon.flatten(1).sum(1) | sigmoid 후 BCE를 직접 조합하는 것보다 안정적인 구현이다. -1~1 normalized 이미지 target을 넣으면 관측 모델과 입력 범위가 맞지 않는다.
AE와 VAE를 비교할 때 | 같은 데이터와 유사한 은닉 폭을 사용한다.~AE는 deterministic z, VAE는 μ·logvar와 샘플링을 쓴다.~재구성만으로 생성 모델 전체를 평가하지 않는다. | reconstruction quality ≠ prior sample quality | 실습은 VAE에 집중하고 AE는 자율 확장한다. 샘플과 재구성 경로를 그림으로 나란히 비교한다. 모델 파라미터 수가 완전히 같지 않은 점을 보고한다.
평가의 랜덤성 관리 | μ를 넣은 재구성은 반복 비교하기 쉽다.~ELBO 평가의 reconstruction 기대값은 샘플로 근사한다.~같은 seed·같은 표본으로 조건 간 변동을 줄인다. | μ reconstruction / stochastic ELBO estimate 구분 | μ reconstruction BCE에 KL을 더한 값은 일반적으로 정확한 MC ELBO estimator가 아니다. 노트북은 평가 지표에 z 샘플을 쓰고 시각화에 μ를 사용한다.
검증할 불변량 | μ=0, logvar=0에서 KL이 0이어야 한다.~reparameterization 출력 shape가 μ와 같아야 한다.~backward 후 μ와 logvar 경로에 gradient가 있어야 한다. | KL≥0 / finite loss / gradient path | 부동소수점 오차로 매우 작은 음수가 나올 가능성을 허용 오차와 함께 생각한다. 빈 배치나 과도한 exp overflow도 진단 대상으로 본다.
'''),
실습=lines('''
LAB 1 · 손실과 재매개화 | practice §1–3: MNIST 범위와 shape를 확인한다.~Gaussian KL의 두 손계산 사례를 검사한다.~ε 샘플의 평균·분산과 gradient를 확인한다. | 50–58분 / 산출물: KL 검사 + 표본 통계 | 표본 평균은 정확히 0이 아니며 표본 수에 따라 근사한다. 경험적 분산 테스트는 충분한 표본과 여유 있는 tolerance를 사용한다.
LAB 2 · VAE 학습 | practice §4: recon·KL·total을 따로 기록한다.~최고 validation negative ELBO를 보관한다.~학습 곡선의 세 항을 분리해서 읽는다. | 58–69분 / 산출물: 3개 손실 곡선 | KL이 낮아지는 것만으로 좋아졌다고 말하지 않는다. recon 변화와 함께 읽는다. 최소 실행에서는 blurry한 결과가 정상일 수 있으며 성능을 보장하지 않는다.
LAB 3 · 세 종류의 이미지 | practice §5–7: 원본·μ 재구성을 나란히 그린다.~prior N(0,I)에서 z를 뽑아 이미지를 생성한다.~두 입력의 μ 사이를 선형 보간한다. | 69–75분 / 산출물: 재구성·prior 샘플·보간 | reconstruction에 학습 입력 정보가 들어간다는 점을 재확인한다. prior 샘플에는 특정 입력이 없다. 보간이 부드럽다고 disentanglement가 증명되는 것은 아니다.
'''),
분석=lines('''
VAE 그림이 흐릿한 이유 | 제한된 모델 용량과 학습 예산이 영향을 준다.~픽셀별 조건부 모델이 평균적인 출력을 선호할 수 있다.~KL과 재구성의 균형도 영향을 준다. | 흐릿함의 원인은 하나가 아니다 | VAE는 무조건 흐리다는 식의 과도한 일반화는 피한다. 지금은 작은 MLP와 2차원 latent라는 제약이 크다. decoder 확률을 그렸다는 점도 구분한다.
좋은 재구성과 좋은 생성 | 재구성은 특정 입력에 조건화한 z를 사용한다.~prior 샘플은 잠재 공간의 다른 영역에 놓일 수 있다.~두 결과를 별도 격자로 평가한다. | q(z∣x)에서 복원 / p(z)에서 생성 | 학습 데이터 주변만 잘 복원하는 모델과 prior 전반에서 잘 생성하는 모델을 구분한다. 샘플 격자는 seed를 고정해 조건 사이 비교가 쉽도록 한다.
잠재 공간 scatter 읽기 | 입력별 μ를 2차원 평면에 찍는다.~라벨 색은 시각화용이며 학습 loss에 사용하지 않는다.~군집의 분리는 분류 성능이나 독립 요인을 보장하지 않는다. | μ(x)의 지도는 q 전체와 다르다 | μ만 그리면 각 입력 분포의 σ가 보이지 않는다. 필요하면 일부 점에 분산 타원을 추가한다. 숫자 클래스가 완전히 분리되지 않아도 VAE 학습 실패로 단정하지 않는다.
'''),
회고=lines('''
출구 퀴즈 | Q1. logvar=0이면 표준편차는?~Q2. 생성 시 z는 encoder와 prior 중 어디서 뽑을까?~Q3. negative ELBO에서 KL의 부호는? | 분포 / 생성 경로 / 목적함수 | 정답: 1, 입력 없는 생성은 prior, 플러스. reconstruction 시에는 encoder의 q 또는 μ를 사용할 수 있다. 경로에 따라 용어를 구분한다.
다음: 잠재 공간을 조절하기 | β로 재구성과 KL의 상대 가중치를 바꾼다.~조건 라벨을 주어 원하는 숫자를 생성한다.~같은 손실 척도와 같은 z로 비교한다. | 자율 과제: KL·재매개화·latent 차원 | 다음 주에는 β가 커졌을 때의 현상을 직접 관찰한다. 더 큰 β가 언제나 더 좋은 생성이나 disentanglement를 만든다는 보장은 없다.
'''),
부록=lines('''
부록 · Jensen으로 하한 만들기 | q(z∣x)를 곱하고 나눠 기대값 형태로 바꾼다.~log E_q[p(x,z)/q]에 Jensen 부등식을 적용한다.~E_q[log p(x,z)−log q]를 얻는다. | log E[f] ≥ E[log f] | q의 support 등 수식이 성립하기 위한 조건을 전제한다. 이 기대값을 likelihood 항과 prior KL로 정리하면 핵심 ELBO 식이 된다.
부록 · KL은 거리일까 | KL은 비음수지만 일반적으로 대칭이 아니다.~KL(q∥p)와 KL(p∥q)는 다르다.~삼각부등식을 만족하는 일반적인 거리로 취급하지 않는다. | KL(q∥p) ≠ KL(p∥q) | 수업에서는 분포 차이를 측정하는 방향성 있는 양이라고 소개한다. 학습 목적에서 어느 방향을 쓰는지 반드시 기호를 확인한다.
부록 · 관측 모델 바꾸기 | Gaussian decoder는 실수값 관측에 쓸 수 있다.~고정 분산 Gaussian의 NLL은 MSE와 연결된다.~분산 선택에 따라 reconstruction 항의 스케일이 바뀐다. | likelihood 선택 → loss 형태와 스케일 | BCE와 MSE의 숫자 크기를 직접 비교해 어느 모델이 좋다고 결론내리지 않는다. 데이터 범위와 확률 가정을 함께 바꿔야 한다.
부록 · latent 차원의 영향 | 너무 작으면 정보를 충분히 담기 어렵다.~크게 해도 일부 차원을 사용하지 않을 수 있다.~차원별 KL과 μ의 변동을 함께 관찰한다. | nominal dimension ≠ used dimension | dimension을 키우면 파라미터와 KL 합의 규모도 바뀐다. 공정한 비교에서는 동일 loss reduction과 학습 예산을 유지한다.
부록 · stochastic reconstruction | 같은 x에서도 z가 달라져 복원이 달라질 수 있다.~μ를 쓰면 대표적인 복원을 확인할 수 있다.~μ decoding은 x의 정확한 조건부 평균과 일반적으로 다르다. | decoder(E[z]) ≠ E[decoder(z)] | decoder가 비선형이므로 평균을 통과시킬 수 없다. 여러 샘플 평균을 구해 차이를 관찰하는 추가 실험을 제안한다.
부록 · latent grid | 2차원 z의 좌표를 격자로 골라 decode한다.~prior 밀도가 낮은 바깥 영역은 낯선 결과를 낼 수 있다.~정규분포 분위수 격자도 사용할 수 있다. | z₁·z₂ 좌표 → 이미지 지도 | 격자 탐색과 prior random sample은 샘플링 분포가 다르다. 그림이 다양해 보인다고 실제 prior에서 그 이미지들이 자주 나오는 것은 아니다.
부록 · 자율 과제 방향 | reparameterize와 KL을 빈칸부터 구현한다.~latent_dim 2와 8을 같은 예산으로 비교한다.~재구성과 prior 샘플을 각각 설명한다. | 수치 검사 + 시각화 + 해석 | 리더 해설은 손실 reduction을 명시한다. 학생에게 단지 격자를 제출하게 하지 말고 각 그림이 어떤 z 경로에서 나왔는지 적게 한다.
''')),

dict(folder='week06_vae_advanced',title='VAE ② 생성 제어와 잠재 공간 실험',tag='LATENT\n→ CONTROL',goals=['β-VAE 실험을 같은 손실 척도로 비교한다.','posterior collapse의 징후와 한계를 설명한다.','조건부 VAE로 숫자 라벨을 제어하며 생성한다.'],intro='5주차의 손실과 재매개화 코드를 독립적으로 다시 포함한다. 실습은 β 비교와 작은 CVAE 학습을 수행한다. GPU가 없으면 각 학습 epoch와 TRAIN_N을 줄이고 기본 분석을 우선한다.',refs=VAE_REFS,
개념=lines('''
β가 바꾸는 것 | reconstruction과 KL의 상대 중요도를 조절한다.~β=1은 표준 VAE의 negative ELBO다.~β≠1 목적값을 표준 ELBO와 동일시하지 않는다. | L_β = reconstruction + β × KL | β가 다른 모델의 total loss 숫자를 그대로 비교하면 목적함수 자체가 달라 공정하지 않다. 공통 β=1 평가와 recon, KL을 따로 보고한다.
압축과 복원의 trade-off | KL을 강하게 제한하면 z가 담는 정보가 줄 수 있다.~재구성 품질이 떨어질 수 있다.~잠재 공간 구조가 나아지는지는 관찰과 평가가 필요하다. | 작은 β / 큰 β / 실제 관측 결과 | β가 크면 반드시 disentanglement가 생긴다는 주장은 하지 않는다. 데이터, 아키텍처, 최적화 등 여러 조건에 의존한다.
posterior collapse | q(z∣x)가 prior에 가까워지고 decoder가 z를 덜 쓸 수 있다.~전체 KL뿐 아니라 차원별 KL과 z 변화 효과를 본다.~KL이 작다는 사실 하나로 collapse를 확정하지 않는다. | KL↓ + latent 반응↓ + reconstruction 확인 | 강한 decoder나 학습 역학 등이 관련될 수 있다. 간단한 MLP MNIST에서 항상 collapse가 나타나는 것은 아니다. 나타나지 않아도 실험이 실패한 것이 아니다.
KL warm-up | 초기에 KL 가중치를 작게 두고 점차 올린다.~먼저 재구성 경로가 학습되도록 유도할 수 있다.~최종 목적과 스케줄을 모두 기록한다. | β(step) = β_target × min(1, step / warmup_steps) | warm-up이 모든 collapse를 해결하지는 않는다. total loss 곡선은 β가 변하므로 단순히 숫자가 증가했다고 악화로 판단하지 않는다.
조건부 생성의 아이디어 | 숫자 라벨 y를 encoder와 decoder에 함께 준다.~latent는 라벨 외의 세부 변동을 표현할 수 있다.~생성 때 원하는 y와 prior z를 조합한다. | FLOW:x + y>q(z∣x,y)>z + y>p(x∣z,y) | 기본 CVAE prior는 y와 무관한 N(0,I)로 둔다. 조건부 prior를 따로 학습하는 설계도 있지만 핵심 범위 밖이다. 라벨은 분류 target이 아닌 조건 입력이다.
라벨과 스타일 분리의 한계 | 같은 z에서 y만 바꾸어 생성 결과를 본다.~같은 y에서 z를 바꾸어 다양성을 본다.~z가 순수한 스타일만 담는다는 보장은 없다. | 행: y 고정 / 열: z 고정 | CVAE 구조만으로 인과적 disentanglement가 증명되지 않는다. 숫자가 바뀌지 않거나 흐려지면 학습 예산과 조건 사용 여부를 점검한다.
생성 모델 평가의 여러 축 | reconstruction은 입력 보존을 본다.~prior 샘플은 생성 분포를 본다.~다양성·조건 일치·반복·실패 사례를 함께 본다. | 한 장의 좋은 샘플보다 체계적인 격자 | 단순한 격자 관찰은 정량 평가를 대체하지 않는다. 이 수업에서는 계산 예산과 데이터 규모에 맞는 기본 진단에 집중한다.
'''),
수식=lines('''
β 비교의 공통 척도 | 훈련은 β별 목적함수를 사용한다.~평가 표에는 recon, KL, recon+KL을 모두 쓴다.~같은 표본과 같은 평가 난수를 사용한다. | training: R+βK / common evaluation: R+K | 공통 β=1 지표는 MC negative ELBO estimate라고 명시한다. 모델별 beta objective는 서로 다른 최적화 대상이며 작은 값 순으로 순위를 매기지 않는다.
차원별 KL | latent 축 j마다 KL_j를 구한다.~배치 평균 후 막대그래프로 시각화한다.~합하면 평균 전체 KL과 일치해야 한다. | KL_j = ½ E_x[μ_j² + exp(logvar_j) − 1 − logvar_j] | latent_dim=8에서 어떤 차원들이 쓰이는지 비교한다. 임의 threshold의 active units는 진단용이며 절대적인 품질 기준이 아니다.
조건부 ELBO | qφ(z∣x,y)에서 z를 뽑는다.~decoder likelihood는 pθ(x∣z,y)다.~단순한 경우 prior는 여전히 p(z)=N(0,I)다. | ELBO(x,y)=E_q[log pθ(x∣z,y)]−KL(qφ(z∣x,y)∥p(z)) | label one-hot을 입력에 concat해 구현한다. label을 손실 target으로 넣는 cross entropy를 추가하지 않아도 조건부 VAE를 학습할 수 있다.
CVAE shape 계산 | x를 펼치면 784차원이다.~10종 라벨의 one-hot을 붙이면 794차원이다.~decoder 입력은 latent_dim+10차원이다. | encoder input: [B,794] / decoder input: [B,Z+10] | one_hot의 dtype을 이미지 dtype과 맞춘다. 라벨은 0~9의 long tensor로 유지하고 내부에서 float one-hot으로 바꾼다. batch 크기도 일치해야 한다.
보간과 traversal은 다르다 | 보간은 두 z 사이의 경로를 따른다.~traversal은 한 z의 특정 좌표만 변화시킨다.~둘 다 prior sampling과 다른 탐색 절차다. | interpolate: (1−α)z_a+αz_b / traverse: z_j←v | 두 점 사이 선분이 항상 높은 prior 밀도 영역을 지나거나 의미적으로 선형이라는 보장은 없다. 고차원 정규분포에서 원점과 typical set의 차이는 심화 토론 주제다.
통계적 비교의 최소 조건 | 같은 train/val 분할과 학습 예산을 사용한다.~가능하면 여러 seed로 학습을 반복한다.~지표의 평균과 분산, 실패 예시를 보고한다. | 평균 ± 표준편차 + 반복 횟수 | 수업 기본 실행은 시간이 제한돼 seed 하나다. 이를 통계적 우월성 증거로 과장하지 않는다. 자율 과제에서 seed 반복을 확장한다.
'''),
설계=lines('''
β 실험 구성 | latent_dim=8의 동일한 VAE를 만든다.~β=0.25, 1, 4로 각각 학습한다.~같은 seed 초기화와 고정 평가 표본을 사용한다. | 조건 3개 × 동일 epoch × 동일 분할 | 3개 모델은 처음부터 새로 학습한다. β를 바꾸면서 이전 가중치를 이어 학습하면 독립된 β 효과 비교가 아니다. optimizer도 새로 만든다.
공통 z를 재사용하기 | 모든 β 모델에 같은 prior z를 넣는다.~같은 validation 이미지의 μ 재구성을 나란히 둔다.~훈련 목표와 공통 평가 척도를 구분해 저장한다. | 비교 격자: 행=조건 / 열=같은 입력 또는 z | 같은 z의 의미가 서로 다른 모델에서 정확히 대응하는 것은 아니다. 난수 차이를 줄이는 시각화이지 스타일 정렬 보장은 아니다.
CVAE 구현의 변경점 | encoder 입력에 one-hot label을 concat한다.~decoder 입력에도 같은 label을 concat한다.~학습·평가·생성에서 label 전달을 빠뜨리지 않는다. | CODE:y1 = F.one_hot(y,10).float();;h = encoder(torch.cat([x_flat,y1],1));;logits = decoder(torch.cat([z,y1],1)) | encoder에만 라벨을 넣고 decoder에 주지 않으면 생성 시 라벨 제어 경로가 없다. 양쪽 경로의 입력 차원을 각각 검증한다.
생성 격자의 축 고정 | 행마다 원하는 숫자 라벨을 고정한다.~열마다 동일 z를 공유한다.~추가 격자에서는 라벨 하나에 여러 z를 샘플링한다. | 10개 라벨 × 8개 latent 샘플 | label grid는 y.repeat_interleave(n_cols), z는 같은 표본을 라벨 수만큼 반복하면 된다. 텐서 배치 순서와 그림 순서가 일치하는지 확인한다.
해석 가능한 기록 저장 | config, seed, β, latent_dim을 결과 표에 남긴다.~재구성·샘플·차원별 KL 그림을 보관한다.~정량 결론과 시각적 관찰을 구분해 쓴다. | 결과물 = 표 + 그림 + 5문장 해석 | 학기 마지막 발표에서는 세 모델 계열의 데이터, 출력, loss, 생성 경로를 연결해 설명하도록 한다. 파일 저장 경로를 확인한다.
'''),
실습=lines('''
LAB 1 · β sweep | practice §1–4: β=0.25,1,4 모델을 학습한다.~공통 β=1 평가로 recon·KL·NLL bound를 정리한다.~같은 입력과 prior z로 격자를 비교한다. | 50–63분 / 산출물: β 비교 표 + 격자 | 느린 CPU에서는 epoch와 표본 수를 줄인다. 이 경우 품질 비교는 예비 관찰로 적는다. 각 조건의 train total 크기를 단순 순위로 읽지 않도록 확인한다.
LAB 2 · latent 진단 | practice §5: 차원별 KL과 traversal을 그린다.~z 변경이 생성 이미지에 영향을 주는지 본다.~collapse라는 결론에 필요한 추가 증거를 적는다. | 63–68분 / 산출물: KL 막대 + traversal | 어떤 좌표는 변화가 작을 수 있다. 그 하나만으로 모델 전체가 붕괴했다고 하지 않는다. KL과 재구성, 여러 z 반응을 함께 본다.
LAB 3 · 조건부 생성 | practice §6–7: 작은 CVAE를 학습한다.~행=라벨, 열=같은 z의 생성 격자를 만든다.~조건 일치와 다양성의 실패 예시를 고른다. | 68–75분 / 산출물: 10×8 조건부 격자 | 짧은 학습에서 숫자 라벨을 충분히 따르지 못할 수 있다. 결과를 삭제하지 말고 학습이 진행되며 개선되는지 추가 예산 실험으로 연결한다.
'''),
분석=lines('''
β 실험의 결론 쓰기 | β 증가에 따라 recon과 KL이 어떻게 변했는지 적는다.~샘플 격자에서 관찰한 변화를 구체화한다.~한 seed·작은 데이터라는 한계를 적는다. | 현상 → 가능한 설명 → 후속 검증 | 예상 경향과 다르면 측정과 초기화, reduction부터 점검한다. 그 뒤 짧은 최적화 예산이나 모델 용량을 원인 후보로 논의한다.
CVAE 조건이 먹히지 않을 때 | decoder에 라벨이 실제로 전달되는지 확인한다.~라벨 순서와 그림 격자 순서가 맞는지 본다.~같은 z에서 y를 바꿨을 때 logits가 달라지는지 검사한다. | shape 검사 + 입력 변화 검사 + 실제 샘플 | logits가 달라진다고 정확한 숫자가 생성된다는 보장은 없다. 조건 사용의 최소 검사와 의미적 정확도 평가를 분리한다.
세 모델 계열을 연결하기 | CNN은 이미지에서 클래스 점수를 만든다.~LM은 문맥에서 다음 토큰 분포를 만든다.~VAE는 latent에서 이미지 조건부 분포를 만든다. | 입력 / 출력 / loss / sampling을 한 표로 | CNN도 확률 출력으로 해석할 수 있지만 학습 대상이 생성 분포와 다르다. 학기 전체를 모델 이름보다 목적함수와 데이터 흐름으로 기억하게 한다.
'''),
회고=lines('''
최종 출구 퀴즈 | Q1. β가 다른 total loss를 바로 비교해도 될까?~Q2. CVAE 생성에 실제 입력 이미지가 필요할까?~Q3. KL≈0 하나로 collapse를 확정할까? | 공통 척도 / prior+label / 여러 진단 | 정답: 그대로 비교하지 않음, 필요 없음, 확정하지 않음. prior z와 원하는 label로 생성한다. 모델이 실제 z를 사용하는지 추가로 검사한다.
학기 마무리 · 3분 발표 | 해결하려는 문제와 모델 선택을 설명한다.~한 가지 통제 실험과 실제 결과를 보여 준다.~실패 사례와 다음 실험을 제안한다. | 자율 최종 과제: 재현 가능한 작은 연구 노트 | 과제는 선택이다. CNN, LM, VAE 중 관심 주제 하나를 선택해 확장할 수 있다. 결과가 좋지 않아도 실험 설계와 근거가 명확하면 충분한 학습 성과다.
'''),
부록=lines('''
부록 · active units | 데이터에 따른 μ_j의 분산을 계산한다.~일정 임계값 이상인 차원을 세는 진단이 가능하다.~임계값에 따라 수치가 달라짐을 보고한다. | AU_j = Var_x[μ_j(x)] > threshold | 차원별 KL과 active units는 같은 지표가 아니다. 표본 수와 scale에 영향을 받으므로 절대 기준으로 평가하지 않는다.
부록 · free bits의 직관 | KL의 작은 값 구간에 완충 구간을 둘 수 있다.~너무 이른 정보 압축을 줄이려는 방법이다.~차원별·그룹별 정의에 따라 구현이 달라진다. | KL 조절법은 목적함수를 바꾼다 | clamp를 어디에 적용하는지에 따라 gradient가 달라진다. 핵심 실습은 β와 warm-up만 다루고 free bits는 추가 읽기로 연결한다.
부록 · disentanglement의 한계 | 독립된 의미 요인을 좌표별로 분리하고 싶다.~관찰 데이터만으로 식별이 어려운 경우가 있다.~구조·감독·가정과 평가 기준을 명시해야 한다. | 시각적으로 예쁜 traversal ≠ 증명 | 한 좌표가 굵기를 바꾸는 것처럼 보여도 다른 의미 요인도 함께 바뀔 수 있다. 여러 입력과 정량 평가가 필요하다.
부록 · convolutional VAE | encoder에 CNN을 사용해 공간 구조를 활용한다.~decoder에는 upsampling과 convolution 등을 사용한다.~출력 해상도와 logits shape를 끝까지 추적한다. | CNN encoder → μ,logvar → image decoder | 1주차의 합성곱 shape 계산과 연결한다. ConvTranspose2d의 출력 크기는 일반 Conv와 다르므로 공식 확인이 필요하다.
부록 · 생성 평가와 FID | 특징 분포를 비교하는 지표가 쓰이기도 한다.~특징 추출기·표본 수·전처리에 민감하다.~소수 MNIST 샘플의 FID를 만능 기준으로 쓰지 않는다. | 평가 지표는 데이터와 목적에 맞춰 선택한다 | 이 수업에서는 FID를 구현하지 않는다. 부정확한 소규모 수치보다 재구성, prior 샘플, 조건 일치, 다양성의 한계를 투명하게 보고한다.
부록 · 실험 재현 체크 | 코드·환경·분할·seed·config를 남긴다.~모델 선택에 사용한 지표를 명시한다.~실행한 결과와 예상 결과를 구분한다. | 관측값에만 숫자 붙이기 | 실행하지 않은 정확도나 ELBO 값을 예시 결과처럼 쓰지 않는다. 재실행 불가능한 스크린샷만 제출하기보다 notebook과 설정을 함께 보관한다.
부록 · 최종 프로젝트 선택지 | CNN: 증강 한 종류를 통제 실험한다.~LM: 문맥 길이와 생성 반복을 비교한다.~VAE: β·latent 크기·조건부 생성 중 하나를 확장한다. | 질문 1개 / 비교 2~3개 / 근거 있는 결론 | 각 팀은 가설, 통제 변수, 실행 예산, 평가 기준을 먼저 적는다. 결과를 보고 가설을 바꿨다면 탐색과 검증을 구분해 서술하도록 한다.
'''))]
