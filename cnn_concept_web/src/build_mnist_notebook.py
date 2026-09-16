"""Build the independent MNIST lab aligned with the September 16 lecture."""
from pathlib import Path
from textwrap import dedent
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[2]

def cells():
    def md(s):return nbf.v4.new_markdown_cell(dedent(s).strip())
    def code(s,hidden=False):
        c=nbf.v4.new_code_cell(dedent(s).strip())
        if hidden:c.metadata['jupyter']={'source_hidden':True};c.metadata['tags']=['helper']
        return c
    return [md('''
# 1주차 · 사람의 단서에서 MNIST 분류까지
2026-09-16 개정. **이미지 분류/객체 인식 구분 → 사람의 단서 → CNN 용어 → 실제 MNIST** 흐름입니다.

이 노트북은 숫자 한 장을 0–9 중 하나로 고릅니다. 사진 안의 여러 숫자의 위치 상자를 찾는 검출기는 아닙니다.
객체 인식은 넓은 표현이므로 분류·검출·분할 중 어떤 출력을 요구하는지 함께 말하세요.

| 구간 | PDF | 할 일 | 권장 시간 |
|---|---|---|---|
| 0–2 | 3–10, 19–22쪽 | 단서 관찰·픽셀·라벨 | 10분 |
| 3–4 | 11–17, 23–24쪽 | 필터 계산·모델 크기 | 10분 |
| 5–7 | 25–27쪽 | 학습·예측·오류 관찰 | 15분 |
| 8–9 | 확장 | 특징 맵·변환·보고서 | 10–15분 |

전체 실습은 45–50분을 따로 배정합니다. 90분 PDF 수업에서 전부 추가하지 않습니다.
짧게 시연하려면 준비·학습을 미리 실행하고 6–7의 예측과 오류를 읽으세요.

Colab에서 위부터 실행합니다. torch·numpy·matplotlib를 사용하고 **CPU**로 계산합니다.
처음에는 실제 MNIST 압축 파일 약11MB를 다운로드하므로 인터넷 연결이 필요합니다. 이미 있으면 무결성을 확인하고 재사용합니다.
노트북 하나만으로 실행되며 추가 Python 파일·이전 주차 실행 상태는 필요 없습니다.
로컬에 패키지가 없으면 `%pip install torch numpy matplotlib`를 별도 셀에서 실행하세요.

실제 관측 출력과 그림이 저장되어 있습니다. 서로 다른 환경에서 성능·시간이 같다는 보장은 아닙니다.
'''),md('''
## 0. 도구 준비
다음 보조 함수 셀은 실행만 합니다. 데이터 로더·작은 CNN·그림 함수가 들어 있습니다.
사람의 “질문/지도/피드백”이라는 비유가 실제로는 가중치·배열·손실 계산이라는 점을 기억하세요.
'''),code((Path(__file__).parent/'mnist_lab.py').read_text(encoding='utf-8'),True),
code(r'''
SEED = 17
EPOCHS = 6  # test를 보기 전에 결정합니다.
style_figures()
torch.set_num_threads(min(4,torch.get_num_threads()))
print('PyTorch:',torch.__version__,'/ device: CPU')
'''),md('''
## 1. 서로 다른 글씨를 같은 숫자로 묶기
**먼저 예상:** 7을 알아볼 때 어떤 획과 관계를 보나요? 두께·기울기가 달라도 무시할 수 있는 차이는 무엇인가요?
공식 MNIST는 train60,000/test10,000입니다. 여기서는 클래스마다 train600·val100·test100장씩 선택합니다.
train/val은 공식 train 안에서 서로 겹치지 않습니다. 마지막 평가용 test1,000장은 학습에 사용하지 않습니다.
'''),code(r'''
data = prepare_data(SEED)
print('수업용 train/val/test:',len(data['y_train']),len(data['y_val']),len(data['y_test']))
sample_figure(data); plt.show()
'''),md('''
**관찰:** 한 열은 같은 라벨의 실제 표본입니다. 모양이 같은 픽셀 배열인지, 다른 모습에서 비슷한 구조를 읽는지 말해 보세요.
“사람은 항상 정답”이라는 가정 없이 애매한 글씨도 찾아 봅니다.

## 2. 그림과 숫자 배열의 연결
**픽셀**은 한 위치의 밝기, **라벨**은 맞혀야 할 범주입니다. [1,28,28]의 1은 흑백 채널 하나이며 클래스 수는10입니다.
'''),code(r'''
DIGIT = 7  # 0~9. 변경 후 이 셀부터 관련 그림 셀을 다시 실행합니다.
assert DIGIT in range(10)
SAMPLE = int(torch.where(data['y_test']==DIGIT)[0][0])
pixels_figure(data,SAMPLE); plt.show()
print('입력 shape:',tuple(data['x_test'][SAMPLE].shape),'정답 label:',int(data['y_test'][SAMPLE]))
print('원래 픽셀0~255를 /255하여 입력0~1로 변환했습니다.')
'''),md('''
## 3. 필터라는 작은 질문을 실제로 계산
먼저 단순한 5×5 입력으로 확인합니다. 각 행이 [−1,0,1]인 필터는 오른쪽에서 왼쪽을 뺍니다.
**예상:** 경계가 있는 창과 모두 밝은 창의 반응은 어떻게 다를까요? 반응은 클래스 확률이 아닙니다.
'''),code(r'''
toy=torch.tensor([[0,0,1,1,1]]*5,dtype=torch.float32)
kernel=torch.tensor([[-1,0,1]]*3,dtype=torch.float32)
COL=0  # 0→2로 변경하고 이 셀만 실행해 보세요.
assert COL in (0,1,2)
patch=toy[:3,COL:COL+3]
response=F.conv2d(toy[None,None],kernel[None,None])[0,0]
print('패치:',patch.tolist(),'\n원소별 곱의 합:',float((patch*kernel).sum()))
print('특징 맵:',response.tolist())
assert torch.isclose((patch*kernel).sum(),response[0,COL])
assert torch.equal(response,torch.tensor([[3.,3.,0.]]*3))
'''),md('''
이제 실제 MNIST 숫자에 두 필터를 적용합니다. 좌우/상하 차이에 따라 반응 위치와 부호가 다릅니다.
**관찰:** 입력의 어느 획이 각 지도에서 보이나요? 이 그림은 수작업 필터의 결과이며 학습된 CNN을 설명한 heatmap은 아닙니다.
'''),code(r'''
manual_filter_figure(data,SAMPLE); plt.show()
'''),md('''
## 4. 단어를 모델 구조와 연결하기
같은 필터를 여러 위치에 쓰는 **공유**, 반응 지도의 축인 **채널**, 음수 반응을0으로 만드는 **ReLU**, 작은 영역의 대표값을 남기는 **pooling**을 연결합니다.
아래에서는 이미지 크기와 채널 수가 어떻게 변하는지 관찰하세요. 출력10개는 숫자 후보0–9의 점수입니다.
'''),code(r'''
shape_model=DigitCNN()
x=data['x_train'][:1]
print('input',tuple(x.shape))
with torch.no_grad():
    for layer in shape_model.features:
        x=layer(x); print(type(layer).__name__,tuple(x.shape))
    print('class scores',tuple(shape_model.classifier(x.flatten(1)).shape))
print('파라미터 수:',sum(p.numel() for p in shape_model.parameters()))
assert sum(p.numel() for p in shape_model.parameters())==9098
print('ReLU 예제:',F.relu(torch.tensor([-2.,0.,3.])).tolist())
print('2x2 max:',float(F.max_pool2d(torch.tensor([[[[1.,4.],[2.,3.]]]]),2).item()))
'''),md('''
## 5. 예측 → 손실 → 변화율 → 수정
**예상:** 학습 전에 10개 후보의 정답을 잘 고를 수 있을까요? 같은 정확도라도 정답 점수가 달라지면 손실이 달라질까요?
매 호출마다 새 모델을 같은 seed로 초기화합니다. validation은 관찰용이며 미리 정한 최종epoch를 사용합니다.
'''),code(r'''
model,history,initial_weights=fit_digit(data,EPOCHS,SEED)
history_figure(history); plt.show()
'''),md('''
## 6. 한 장의 점수와 전체 정확도 구분
**logits**는 클래스별 점수, **softmax**는 합이1인 상대적 출력입니다. 그 값이 실제 정답률과 자동으로 같지는 않습니다.
test는 고정1,000장 부분집합입니다. 공식test10,000장 전체 벤치마크 성능이라고 쓰지 마세요.
'''),code(r'''
test_result=evaluate_digit(model,data['x_test'],data['y_test'])
print(f"고정 test {len(data['y_test'])}장: accuracy={test_result['accuracy']:.3f}, CE={test_result['ce']:.3f}")
prediction_figure(data,test_result,SAMPLE); plt.show()
'''),md('''
## 7. 사람의 단서로 오류를 관찰하기
아래는 실제 오분류 중 앞6장입니다. 전체test의 대표 표본이 아닙니다.
**활동:** 한 장을 골라 애매한 획을 말하고, 필터·특징 맵·수용 영역 중 두 용어로 확인할 가설을 적으세요.
그림만으로 “이 뉴런 때문에 틀렸다”고 원인을 확정하지 않습니다.
'''),code(r'''
fig,error_ids,are_errors=errors_figure(data,test_result); plt.show()
print('실제 오분류 예시' if are_errors else '이번 실행에는 오분류가 없어 정상 예시를 표시했습니다.')
'''),md('''
## 8. 특징 맵과 입력 변환 · 선택 확장
학습된 첫 층의 반응8장을 봅니다. 채널 이름은 번호이며 자동으로 “고리 뉴런” 같은 의미를 붙이지 않습니다.
같은 모델에 위치 이동이나 가림을 적용하고 원본과 비교합니다. 이동에는 반대편으로 감기는 wrap-around를 쓰지 않습니다.
'''),code(r'''
feature_figure(model,data,SAMPLE); plt.show()
SHIFT=3  # 0~6, 오른쪽 이동
assert SHIFT in range(7)
shifted=torch.zeros_like(data['x_test'])
if SHIFT:shifted[...,SHIFT:]=data['x_test'][...,:-SHIFT]
else:shifted=data['x_test'].clone()
shift_result=evaluate_digit(model,shifted,data['y_test'])
print('원본 정확도',round(test_result['accuracy'],3),'이동 후',round(shift_result['accuracy'],3))
print('공유 필터를 사용해도 최종 분류의 완전한 이동 불변성이 보장되지는 않습니다.')
'''),md('''
## 9. 관찰 네 문장 저장
설정과 측정값은 자동으로 기록됩니다. 아래 네 문장을 본인 관찰로 바꾸고 실행하세요.
Colab 파일 목록에서 JSON이나 출력이 담긴 ipynb를 내려받아 보관합니다.
'''),code(r'''
reflection={'과제':'이 노트북의 출력이 분류/검출/분할 중 무엇인지 근거와 함께 작성하세요.',
    '비유와 연산':'필터 또는 특징 맵을 사람의 비유와 실제 연산으로 각각 설명하세요.',
    '관찰':'실제 정확도 또는 오류 이미지의 구체적인 획을 인용하세요.',
    '한계':'다음에 시험할 입력 변화와 이 결과로 아직 말할 수 없는 것을 적으세요.'}
metrics=experiment_metrics(data,model,history,test_result,EPOCHS,SEED)
metrics.update(shift=SHIFT,shift_accuracy=shift_result['accuracy'])
out=Path('outputs/week01_mnist');out.mkdir(parents=True,exist_ok=True)
(out/'observations.json').write_text(json.dumps({'metrics':metrics,'reflection':reflection},ensure_ascii=False,indent=2),encoding='utf-8')
print('저장:',out/'observations.json')
for k,v in reflection.items():print(k+':',v)
'''),md('''
### 자율 실험과 해설 방향
1. **필터:** COL을0과2로 비교합니다. 모두 밝아도 좌우 차이가 없으면 0이 되는 이유를 설명하세요.
2. **입력 변화:** SHIFT를0·2·4로 비교합니다. 0은 원본과 같은 결과여야 합니다. 8절만 재실행합니다.
3. **학습:** EPOCHS를2와6으로 바꾸고5절부터 실행합니다. 비교 설정은 test로 선택하지 않고 validation으로 검토하세요.

정확도는 실제 관측값을 적습니다. test를 본 뒤 바꾼 설정은 탐색으로 표시하고 최종 주장은 별도 보류 데이터로 확인합니다.
사람과 같은 숫자를 맞혔다고 같은 내부 처리·경험·맥락을 사용했다고 결론 내리지 않습니다.

출처: [MNIST 구성](https://www.tensorflow.org/datasets/catalog/mnist),
[Torchvision 공식 MNIST 미러·체크섬](https://docs.pytorch.org/vision/stable/_modules/torchvision/datasets/mnist.html),
[분류·검출·분할 과제](https://docs.pytorch.org/vision/stable/models.html).
원 데이터는 LeCun·Cortes·Burges의 MNIST입니다. FashionMNIST·인쇄 글꼴·수업용 합성 도형과 구분합니다.
''')]

def main():
    nb=nbf.v4.new_notebook(cells=cells(),metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'},'colab':{'name':'concept_practice.ipynb','provenance':[]}})
    for i,c in enumerate(nb.cells):
        c.id=f'mnist-week01-{i:02d}'
        if c.cell_type=='code':compile(c.source,f'cell{i}','exec')
    nbf.validate(nb);nbf.write(nb,ROOT/'week01_cnn_basics/concept_practice.ipynb')
    print('Built MNIST notebook:',len(nb.cells),'cells')

if __name__=='__main__':main()
