# CNN 2주차 · AlexNet 논문 리딩 리더 가이드

2026-09-22 개정 · 40페이지 · 90분. 1주차 개념을 AlexNet의 문제·방법·근거와 연결합니다. 원 논문 캡처와 수업용 재구성을 구별합니다. 기존 PPT·노트북은 이 논문의 재현 실험이 아닙니다.

진행: 읽기 준비8분 → 초록·서론12분 → 문제 정의10분 → 개념28분 → 수식7분 → 실험15분 → 결론3분 → 나의 정리7분.

논문에는 별도 Problem statement 절이 없고 마지막 절은 Discussion입니다. 출처의 페이지는 원문 PDF의 인쇄 페이지와 같습니다.

웹에서 32페이지 해설은 클릭해 펼칩니다. PDF에서는 해설이 함께 보이므로 활동 때 먼저 질문만 읽고 답을 설명합니다.

## 01. CNN을 안다는 것에서, 논문을 읽는 것으로.
1분. 오늘은 ImageNet 전체를 직접 학습하지 않는다. CNN 개념을 실제 논문의 주장·설계·근거와 연결하는 수업임을 안내한다.


핵심: 초록·서론 → 문제 정의 → 개념 → 수식 → 실험 → 결론 → 나의 정리

시간: 00–01분 / 근거: [paper overview](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=1)

## 02. 오늘의 읽기 순서가, 발표의 순서다.
2분. 40페이지, 총90분. 개념 설명28분·수식7분으로 중심을 잡고, 질문·표 읽기·마지막 개인 정리를 시간 안에 포함한다.


핵심: 모든 문장을 처음부터 완벽히 이해하려 하지 말고, 다음에 확인할 질문을 남깁니다.

시간: 01–03분 / 근거: [reading plan](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=1)

## 03. 읽으면서 세 종류의 메모를 남긴다.
2분. 답: 어떤 데이터·분할·지표·모델 수·비교 대상인가. C/E/Q는 논문 용어가 아니라 수업의 읽기 도구다.

질문: “성능이 좋아졌다”는 문장 옆에 반드시 적어야 할 조건은 무엇일까요?
핵심: 모르는 단어만 모으는 대신, 주장과 근거 사이에 남는 질문을 적습니다.

시간: 03–05분 / 근거: [reader-created method](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=1)

## 04. 제목에서 약속을 찾고, 목차에서 경로를 찾는다.
3분. 배경·색·크기·가림·자세·클래스 수를 받는다. object recognition이라는 표현이 있어도 본 실험의 출력은 이미지 분류임을 상기한다.

질문: 1주차의 MNIST에서 자연 이미지로 가면, 입력에서 어떤 변화가 더 커질까요?
핵심: 이 논문은 CNN을 처음 발명한 논문이 아닙니다. 제목만으로 기여를 단정하지 않습니다.

시간: 05–08분 / 근거: [title / section map](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=1)

## 05. 초록은 번역문보다, 질문의 지도다.
3분. 원문 링크로 초록 위치를 보여준다. 이 슬라이드는 초록 전체 번역이 아니라 독자의 정보 추출 틀이다. 2010과2012 결과가 함께 등장한다는 점만 예고한다.

질문: 초록의 큰 성과가 “모든 CNN” 또는 “모든 이미지”에 대한 약속일까요?
핵심: 초록은 논문의 주장 지도입니다. 숫자의 정확한 조건은 본문에서 다시 확인합니다.

시간: 08–11분 / 근거: [Abstract](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=1)

## 06. Introduction: 쉬운 데이터의 성공이 끝은 아니다.
3분. 사람의 변형된7 인식에서 배운 관점을 자연 이미지에 확장한다. MNIST가 쓸모없다는 결론이 아니라 과제 난도가 달라짐을 말한다.


핵심: 배경 설명을 읽을 때는 “그래서 무엇이 부족한가?”를 한 문장으로 적습니다.

시간: 11–14분 / 근거: [§1 Introduction](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=1)

## 07. 필요한 것은 하나가 아니라, 함께 맞는 조건들.
2분. 세 요구를 이후 구조·ReLU/GPU·증강/Dropout과 연결할 예고다. 도식은 수업용 문제 구조화다.

질문: 모델만 키웠는데 학습 시간이 너무 길어진다면, 표현력만의 문제일까요?
핵심: “무엇이 필요하다”와 “이 논문이 그것을 어떻게 마련했다”를 연결합니다.

시간: 14–16분 / 근거: [§1, pp.1–2](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=2)

## 08. 기여를, 문제와 해결책의 짝으로 바꾼다.
2분. 각 기법의 발명 우선권을 이 논문으로 돌리지 않는다. 여기서는 저자가 채택한 선택들의 역할을 분석한다.


핵심: 한 가지 마법의 기법보다, 문제를 풀 수 있도록 구성한 시스템을 읽습니다.

시간: 16–18분 / 근거: [§1, §§3–6](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=2)

## 09. 잠깐 멈춤: 아직 구조도는 보지 않는다.
2분:30초개인생각,60초짝설명,30초한팀공유. 예: 자연 이미지의 다양성을 처리할 큰CNN을 실제로 학습·일반화시키는 것이 문제. 기술 명칭만 나열하지 않도록 유도한다.


핵심: 설명하지 못한 빈칸은 실패가 아니라, 다음 절에서 확인할 질문입니다.

시간: 18–20분 / 근거: [Abstract / §1](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=2)

## 10. Problem statement라는 제목이 없어도 읽을 수 있다.
2분. problem statement는 우리가 붙인 분석 틀이며 원문에 있는 절인 것처럼 인용하지 않는다. 객체 위치를 출력하는 검출과 구별한다.


핵심: 저자의 문장을 찾아서 입력·출력·목표·제약을 채우는 것이 문제 정의 분석입니다.

시간: 20–22분 / 근거: [§§1, 2, 3.5, 5](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=2)

## 11. 데이터셋 이름 뒤에 숨은 조건을 펼친다.
3분. 직접순위비교불가:연도·평가집합다름. §2의150,000test는원문설명을2012규모로일괄전용하지않는다. 본자료는혼동방지위주.

질문: 2010 시험 오류율과 2012 검증 오류율을 같은 표의 순위처럼 비교해도 될까요?
핵심: 비교의 단위는 모델 이름 하나가 아니라, 모델·데이터·평가 절차의 묶음입니다.

시간: 22–25분 / 근거: [§2 / §6](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=2)

## 12. Top-1과 Top-5: 질문이 달라지면 오류도 달라진다.
2분. 표는수업용가상예제. top5가정답5개를요구하는지표가아님을질문으로확인. 단일정답이상위5개에들어가는지평가.


핵심: Top-5 오류율 15.3%를 “정확도 15.3%” 또는 “Top-1 오류율”로 읽지 않습니다.

시간: 25–27분 / 근거: [§2 metrics](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=2)

## 13. 우리 말로 다시 쓴, 이 연구의 문제.
3분.훈련데이터암기만으로성공이라고판단할수있다.최적화와일반화구분.다음부터개념28분진행.

질문: 이 질문에서 “새 이미지”를 빼면 무엇을 놓치게 될까요?
핵심: 이제부터 모든 기법을 이 문제의 어느 부분에 답하는지 연결해 읽습니다.

시간: 27–30분 / 근거: [§§1–2, reader synthesis](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=2)

## 14. 핵심 theory를, 설계 원리로 먼저 읽는다.
3분. theory를수식목록으로오해하지않게한다. 이론적직관과실험으로입증한주장을구분. 사람비유한계를1주차와연결.


핵심: 원리 → 무엇을 계산하는지 → 왜 필요한지 → 근거의 순서로 설명합니다.

시간: 30–33분 / 근거: [§1 / §3](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=2)

## 15. 필터는 “이 근처에 이런 패턴이 있는가?”라는 질문.
2분.도식반응3은손계산값.원논문성능이나학습필터아님.특징맵은확률지도와다른중간표현.


핵심: 학습 후 필터를 관찰하는 것과, 사람이 의미 있는 필터를 미리 넣는 것은 다릅니다.

시간: 33–35분 / 근거: [§3.5; teaching example](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=4)

## 16. RGB 세 장을 따로 분류하는 것이 아니다.
3분.필터1개와이미지마다바뀌는특징맵을구별.원문96필터는11x11x3이며이슬라이드의채널도식4장은개념축약.

질문: 이 그림의 한 칸을 “고양이 확률”이라고 부를 수 있을까요?
핵심: 필터·특징 맵·채널은 같은 말이 아닙니다. “무엇의 그림인가?”를 먼저 확인합니다.

시간: 35–38분 / 근거: [§3.5 / Figure 3 / §6.1](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=6)

## 17. 가중치 공유는, 같은 질문을 재사용하는 약속.
2분.합성곱bias포함80,완전연결4923520.동일출력크기비교일뿐정확도동등성주장아님.계산이길면80만확인.


핵심: 적은 가중치와 같은 성능은 같은 말이 아닙니다. 구조의 가정이 과제에 맞는지 묻습니다.

시간: 38–40분 / 근거: [§1; teaching calculation](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=1)

## 18. 깊이는 “새 필터를 더한다”보다 많은 일을 한다.
2분.수용영역3/5/7은AlexNet의실제층크기가아닌단순예제.깊어지면자동성능상승주장하지않는다.


핵심: 더 넓은 영역을 조합할 수 있다는 설명과, 실제 의미를 배웠다는 증거를 구분합니다.

시간: 40–42분 / 근거: [§3.5 / §7; teaching diagram](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=4)

## 19. Figure 2는, 읽는 순서를 정하면 덜 복잡하다.
3분. 원문은 가중치가 있는 층 8개를 셉니다. 두 줄은 서로 독립된 CNN 2개의 앙상블이 아닙니다. GPU별 채널 수와 총 채널 수를 구별하고, 메모리·연결 제약이 구조에도 영향을 주었음을 짚습니다.

질문: pooling과 ReLU가 있는데 왜 8개 층이라고 할까요?
핵심: 8개 “학습 가능한 층”이라는 표현을 모든 연산의 개수로 읽지 않습니다.

시간: 42–45분 / 근거: [§3.5 / Figure 2](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=5)

## 20. 같은 구조를, 역할 중심으로 다시 그린다.
2분. FC는학습필터와분리된사람규칙이아니라같이학습하는가중치다. 기존표현을집계하는역할설명.


핵심: 이 재구성은 이해를 위한 요약입니다. 정확한 재현에는 원문과 구현의 세부 조건이 더 필요합니다.

시간: 45–47분 / 근거: [§3.5 / Figure 2](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=5)

## 21. ReLU: 신호를 바꾸는 작은 규칙의 큰 차이.
3분.오른쪽개념곡선은원문그래프가아님.선형행렬곱설명은한문장으로하고미분증명은생략.원문속도근거는뒤29페이지로보류.


핵심: ReLU가 모든 학습 문제를 해결한다는 말 대신, 어떤 조건에서 어떤 개선을 보였는지 읽습니다.

시간: 47–50분 / 근거: [§3.1 / Figure 1](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=3)

## 22. Pooling: 주변의 강한 반응을 남긴다.
2분.강한반응이정답을보장하는것아님.같은출력크기의2x2stride2와비교한원문조건은실험검토때연결.


핵심: 논문의 선택: 창 3 × 3, stride 2. 개념을 이해한 뒤 비교 대상과 출력 크기를 확인합니다.

시간: 50–52분 / 근거: [§3.4](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=4)

## 23. LRN: 같은 위치의 다른 채널을 함께 본다.
2분. 채널 순서는 임의라는 원문 조건. 주변이 공간 이웃이 아니라 같은 xy의 채널이라는 점 강조. 수식은28페이지에서 돌아온다.


핵심: 수식의 복잡한 첨자를 보기 전에 “어떤 값을 무엇으로 나누는가?”를 이해합니다.

시간: 52–54분 / 근거: [§3.3](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=4)

## 24. 증강: 라벨을 보존하는 변화부터 생각한다.
2분.훈련은가중치학습,평가다중crop은하나이미지예측집계.원문256이미지에서224crop사용.구현크기질문은38페이지에서정리.

질문: 훈련에서 여러 crop을 보이는 것과 시험에서 여러 crop을 평균하는 것은 같은 역할일까요?
핵심: 훈련 증강과 평가 때 여러 crop의 예측을 평균하는 절차를 따로 기록합니다.

시간: 54–56분 / 근거: [§4.1](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=5)

## 25. Dropout: 특정 조합에만 기대지 않도록.
2분.원문0.5 dropout. 현대구현inverted방식의스케일링은개념부연이며원문이라고하지않는다.두독립앙상블과구별.


핵심: 앙상블과 연결되는 직관은 유용하지만, 독립 모델 여러 개를 실제로 학습한 것과 같지는 않습니다.

시간: 56–58분 / 근거: [§4.2](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=6)

## 26. 수식 ①: 필터와 특징 맵에 기호를 붙인다.
3분.식의W에i,j가없음을찾게한다.이는위치별가중치공유를보여준다.padding/stride정확구현은별도이며범위명시.


핵심: 식을 외우기 전에 입력·출력·합산 대상·학습 변수를 찾습니다.

시간: 58–61분 / 근거: [§3.5; teaching notation](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=4)

## 27. 수식 ②: 정답 확률을 높인다는 말의 의미.
2분.정답확률큰쪽손실작다.교차엔트로피와오류율은같지않고,표의오류율을손실값처럼해석하지않는다.


핵심: 확률을 출력한다는 것만으로 그 값이 잘 보정된 신뢰도라는 결론은 나오지 않습니다.

시간: 61–63분 / 근거: [§3.5 objective; teaching notation](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=4)

## 28. 수식 ③: LRN의 긴 첨자를, 질문으로 분해한다.
2분.양수상수의조건에서분모커져현재출력작아진다.이수식이정확도향상을수학적으로증명하는것은아님.

질문: 분자를 고정하고 주변 반응 제곱합만 키우면 출력은 어떻게 될까요?
핵심: 모든 첨자를 외우는 대신, “주변 반응이 커지면 현재 출력은 어떻게 되는가?”에 답합니다.

시간: 63–65분 / 근거: [§3.3 equation](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=4)

## 29. Figure 1: 축을 읽기 전에 결론을 읽지 않는다.
2분.원문25%trainerror도달6배설명은해당조건.학습률은각각빠르게학습되도록별도선택.벽시계시간을6배로단정하지않는다.


핵심: 데이터 → 축 → 범례 → 비교 조건 → 결론 순서로 그래프를 읽습니다.

시간: 65–67분 / 근거: [§3.1 / Figure 1](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=3)

## 30. Table 1: 비교는 같은 열에서 시작한다.
2분.8.7/25.7≈33.9%상대감소이며8.7%감소와구별.각방법이여러기술을포함한전체시스템이라는점.


핵심: 같은 표의 개선을 읽되, 그 표가 분리하지 않은 원인까지 단정하지 않습니다.

시간: 67–69분 / 근거: [§6 / Table 1](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=7)

## 31. Table 2: 15.3%의 주어를 정확히 읽는다.
3분.5개일반CNN+2개추가데이터사전학습CNN의결합.추가모델은6번째conv도포함.표의*표기와본문의구성설명까지읽어야한다.

질문: 별표·빈칸·val/test를 지우면, 같은 표에서 어떤 잘못된 이야기를 만들 수 있을까요?
핵심: “AlexNet 단일 모델의 시험 오류율은 15.3%”라는 문장은 이 표를 정확히 설명하지 못합니다.

시간: 69–72분 / 근거: [§6 / Table 2](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=7)

## 32. 짧은 활동: 표에 없는 결론을 찾아낸다.
2분.45초개인판정,45초짝비교,30초해설.웹해설은펼침으로,PDF해설은인쇄시에보이도록제공.먼저해설부분을가리거나구두질문을제시.


핵심: 실험 표는 강한 근거이지만, 그 안에 없는 비교까지 대신해 주지는 않습니다.

시간: 72–74분 / 근거: [§6 / Table 2](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=7)

## 33. 구성 요소 비교: 좋아졌다는 말 뒤의 통제를 본다.
2분.여기ablation은독자분석용표현.기여도총합을빼거나더하는것은상호작용을무시함.한요인의완벽한통제를무조건요구하기보다한계를명시.


핵심: 논문이 보고한 관찰을 인정하면서, 인과적으로 말할 수 있는 범위를 따로 적습니다.

시간: 74–76분 / 근거: [§§3.1–3.4 / §7](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=3)

## 34. Figure 4 왼쪽: 숫자 뒤에 어떤 실패가 있는가?
2분.정답이top5에있으면빨간막대라는캡션조건을해설.모든빨간막대가top1정답은아님.오류라벨의모호성도질문.


핵심: 정성 분석은 질문을 풍부하게 하고, 정량 평가는 그 질문의 빈도와 크기를 확인합니다.

시간: 76–78분 / 근거: [§6.1 / Figure 4 left](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=8)

## 35. Figure 4 오른쪽: 모델은 무엇을 비슷하다고 보나?
2분.마지막4096차원은닉층의유클리드거리.이미지수준확률이나픽셀L2로선택한것과구분.첫열test나머지train.


핵심: 시각화는 표현을 조사하는 도구입니다. 모델이 왜 판단했는지 완전히 설명하는 증명은 아닙니다.

시간: 78–80분 / 근거: [§6.1 / Figure 4 right](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=8)

## 36. 결론은, 처음의 질문으로 돌아가 읽는다.
3분.논문이기술한방향과독자의보편화구분.비지도사전학습을안썼다는논문결론과Table2의지도사전학습은모순이아님을질문나오면설명.

질문: “더 큰 모델이 가능하다”와 “더 크게 만들면 항상 낫다”는 왜 다른 말일까요?
핵심: 초록에서 만든 주장 목록에 돌아가, 확인한 것과 남은 것을 표시합니다.

시간: 80–83분 / 근거: [§7 Discussion](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=8)

## 37. 마지막 한 페이지는, 저자의 요약이 아닌 나의 정리.
1분.다음페이지에서직접작성.기준은모든빈칸정답이아니라자기언어와근거,질문.


핵심: 이 네 칸이 채워지면, 완벽히 이해하지 못했어도 다음 읽기를 시작할 수 있습니다.

시간: 83–84분 / 근거: [reader-created worksheet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=8)

## 38. 3분, AlexNet을 내 언어로 적어 본다.
3분.먼저모범답안을보여주지않는다.가능하면4칸전부작성,시간부족시핵심메시지와질문우선.별도논문_읽기_기록지.md연결.


핵심: 2분 개인 작성 + 1분 짝에게 핵심 주장과 아직 남은 질문 설명.

시간: 84–87분 / 근거: [reader-created worksheet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=8)

## 39. 좋은 정리의 예: 확신과 질문을 함께 남긴다.
2분.224,k11,s4,p0이면floor((224-11)/4)+1=54,그림55와차이.227/p0또는224/p2는55가능하지만원문그조건확정은불가.특정구현을원논문과동일하다고단정하지않는다.


핵심: 계산이 안 맞으면 몰래 숫자를 고치지 말고, 원문·도식·구현의 차이를 질문으로 남깁니다.

시간: 87–89분 / 근거: [§3.5 / Figure 2; reader questions](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=5)

## 40. 다음 논문에서도, 같은 순서로 묻는다.
1분.원문링크및출처좌표안내.기존2주차합성데이터노트북은추가학습용이며이논문재현실험이아니다.숙제강제없음.


핵심: 출구 질문: AlexNet의 주장 하나와 근거 하나, 아직 모르는 질문 하나를 말할 수 있나요?

시간: 89–90분 / 근거: [primary source / reading recap](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=1)
