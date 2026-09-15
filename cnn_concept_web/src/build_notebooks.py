"""Build two independent, concept-first CNN labs. Run validate_notebooks.py next."""
from pathlib import Path
from textwrap import dedent
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]

def md(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip())

def code(text, helper=False):
    cell = nbf.v4.new_code_cell(dedent(text).strip())
    if helper:
        cell.metadata['jupyter'] = {'source_hidden': True}
        cell.metadata['tags'] = ['helper']
    return cell

SETUP = r'''
import copy, json, random, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from IPython.display import display, Markdown

# 이 작은 실험은 GPU 없이 실행합니다. 그림 속 영문은 폰트 설치를 피하기 위한 표기입니다.
torch.set_num_threads(min(4, torch.get_num_threads()))
DEVICE = torch.device('cpu')
SEED = 17
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.facecolor': '#f4f2e9',
    'figure.facecolor': '#faf9f5', 'axes.prop_cycle': plt.cycler(color=['#187a70','#dd7046','#596ba5'])})
NAMES = ['circle', 'square']
print('PyTorch', torch.__version__, '| device:', DEVICE)
'''

HELPERS = r'''
def make_objects(n, seed):
    """서로 다른 seed로 연속적인 위치·크기·잡음을 생성합니다. 라벨 0=원, 1=정사각형."""
    assert n % 2 == 0
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], n // 2); rng.shuffle(y)
    yy, xx = np.mgrid[:28, :28].astype('float32')
    masks = []
    for label in y:
        cx, cy = rng.uniform(11, 17, 2)
        radius = rng.uniform(5, 8)
        dx, dy = xx-cx, yy-cy
        # 넓이만으로 분류하지 못하도록 사각형의 반폭을 조절합니다.
        half = radius * np.sqrt(np.pi) / 2
        mask = dx*dx + dy*dy <= radius*radius if label == 0 else (abs(dx) <= half) & (abs(dy) <= half)
        masks.append(mask)
    masks = torch.tensor(np.stack(masks), dtype=torch.float32).unsqueeze(1)
    noise = torch.tensor(rng.normal(0, .025, (n, 1, 28, 28)), dtype=torch.float32)
    return masks, torch.tensor(y, dtype=torch.long), noise

def grayscale(objects):
    mask, _, noise = objects
    return (.12 + .76*mask + noise).clamp(0, 1)

def gallery(images, labels=None, titles=None, n=8):
    n = min(n, len(images))
    fig, axes = plt.subplots(1, n, figsize=(2*n, 2.4), squeeze=False)
    for i, ax in enumerate(axes[0]):
        img = images[i].detach().cpu()
        if img.shape[0] == 1:
            ax.imshow(img[0], cmap='gray', vmin=0, vmax=1)
        else:
            ax.imshow(img.permute(1,2,0).clamp(0,1))
        if titles is not None: ax.set_title(titles[i], fontsize=9)
        elif labels is not None: ax.set_title(NAMES[int(labels[i])])
        ax.axis('off')
    plt.tight_layout(); plt.show()

class SmallCNN(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Linear(16*7*7, 2)
    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))

@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    logits = model(x.to(DEVICE)).cpu()
    pred = logits.argmax(1)
    cm = torch.bincount(2*y + pred, minlength=4).reshape(2,2)
    return {'accuracy': float((pred == y).float().mean()),
            'loss': float(F.cross_entropy(logits,y)), 'cm': cm, 'logits': logits}

def train_model(x, y, vx, vy, epochs, seed=None):
    # 호출할 때마다 같은 초기값, 같은 배치 순서로 새 모델을 만듭니다.
    seed = SEED if seed is None else seed
    torch.manual_seed(seed)
    model = SmallCNN(x.shape[1]).to(DEVICE)
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    order_rng = torch.Generator().manual_seed(seed + 100)
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train(); total = 0.
        for ix in torch.randperm(len(y), generator=order_rng).split(64):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x[ix].to(DEVICE)), y[ix].to(DEVICE))
            assert torch.isfinite(loss), '손실이 유한한지 확인하세요.'
            loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(ix)
        val = evaluate(model, vx, vy)
        history['train_loss'].append(total/len(y))
        history['val_loss'].append(val['loss'])
        history['val_accuracy'].append(val['accuracy'])
        if epoch == 0 or epoch == epochs-1:
            print(f"epoch {epoch+1:2d}/{epochs} | train CE {total/len(y):.3f} | val acc {val['accuracy']:.3f}")
    assert not torch.equal(before['features.0.weight'], model.features[0].weight.detach().cpu())
    print(f'elapsed: {time.perf_counter()-start:.1f}s')
    # epoch 수는 test를 보기 전에 고정합니다. 최종 epoch 모델로 평가합니다.
    return model, history, before

def show_history(history):
    fig, axes = plt.subplots(1,2,figsize=(10,3))
    epochs = np.arange(1, len(history['train_loss'])+1)
    axes[0].plot(epochs, history['train_loss'], label='train')
    axes[0].plot(epochs, history['val_loss'], label='validation')
    axes[0].set(xlabel='epoch', ylabel='cross entropy'); axes[0].legend()
    axes[1].plot(epochs, history['val_accuracy'])
    axes[1].set(xlabel='epoch', ylabel='validation accuracy', ylim=(0,1.05))
    plt.tight_layout(); plt.show()
'''

W1 = [
md('''
# 1주차 실습 · 작은 단서가 분류가 되기까지
**관찰 → 예상 → 한 값 변경 → 실행 → 근거 설명**

사람은 컵의 테두리와 손잡이 같은 단서로 물체를 알아봅니다. 컴퓨터는 픽셀에서 어떤 차이를 계산할 수 있을까요?
이 노트북에서는 사람이 정한 필터를 먼저 관찰하고, 작은 CNN이 예시로부터 필터를 학습하는 과정까지 연결합니다.

| 진행 | 강의 연결 | 활동 | 시간 |
|---|---|---|---|
| 핵심 0–3 | PDF 13–18, 24–25쪽 | 경계 손계산·필터·이동·풀링 | 15분 |
| 확장 4–6 | PDF 19–23쪽 | CNN 학습·특징 맵·오분류 | 20–30분 |
| 정리 7 | PDF 26쪽 | 관찰 기록 저장·자율 실험 | 5분 이상 |

**90분 수업에서는 종이 활동 15분을 핵심 0–3으로 교체**합니다. 확장은 별도 시간에 진행합니다.
Colab에서 `런타임 → 모두 실행`하거나 위에서 아래로 실행하세요. 필요한 패키지는 torch, numpy, matplotlib이며
Colab 기본 환경의 설치본을 사용합니다. 로컬에서 없으면 `%pip install torch numpy matplotlib`를 별도 셀에서 실행하세요.
GPU·계정·외부 데이터·다른 노트북은 필요 없습니다. 실행 출력이 저장되어 있어 실행 전에도 예시를 읽을 수 있습니다.

도형은 수업용 합성 데이터이며 실제 사물 인식이나 인간 시각을 재현한 데이터가 아닙니다.
'''),
md('## 0. 준비 · 1분\n아래 두 셀은 실행만 합니다. 보조 함수의 코드를 모두 읽을 필요는 없습니다.'),
code(SETUP), code(HELPERS, True),
md('''
## 1. “오른쪽이 밝다”를 숫자로 말하기 · 4분
**실행 전 예상:** 3×3 필터의 각 행이 `[-1, 0, 1]`이면 왼쪽과 오른쪽이 같은 곳, 밝아지는 경계, 어두워지는 경계에서 각각 어떤 부호가 나올까요?
세로 경계는 좌우 밝기 차이로 찾습니다. 여기서 사용하는 연산은 PyTorch Conv2d의 교차상관 방식입니다.
'''),
code(r'''
toy = torch.tensor([[0,0,1,1,1]]*5, dtype=torch.float32)
kernel = torch.tensor([[-1,0,1]]*3, dtype=torch.float32)
# 바꿔 볼 곳: ROW, COL을 각각 0, 1, 2 중 선택합니다.
ROW, COL = 0, 0
assert 0 <= ROW <= 2 and 0 <= COL <= 2
patch = toy[ROW:ROW+3, COL:COL+3]
manual = (patch*kernel).sum()
response = F.conv2d(toy[None,None], kernel[None,None])[0,0]
print('입력 조각:\n', patch.numpy(), '\n원소별 곱:\n', (patch*kernel).numpy())
print('손계산 합:', float(manual), '| 전체 특징 맵:\n', response.numpy())
assert torch.isclose(manual, response[ROW,COL])
assert torch.equal(response, torch.tensor([[3.,3.,0.]]*3))
'''),
md('''
**직접 해 보기:** `COL=2`로 바꿔 실행하고, 값이 0인 이유를 말해 보세요. 다음 셀에서는 입력을 좌우 반전합니다.
0은 언제나 “아무것도 없음”일까요, 아니면 이 필터가 측정한 좌우 차이가 없다는 뜻일까요?
'''),
code(r'''
flipped = toy.flip(-1)
flipped_response = F.conv2d(flipped[None,None], kernel[None,None])[0,0]
print('반전 후 특징 맵:\n', flipped_response.numpy())
assert torch.equal(flipped_response, torch.tensor([[0.,-3.,-3.]]*3))
fig, axes = plt.subplots(1,3,figsize=(9,3))
for ax, arr, title in zip(axes, [toy,kernel,response], ['pixels','filter','response']):
    ax.imshow(arr, cmap='RdBu_r', vmin=-3, vmax=3)
    for (r,c), value in np.ndenumerate(arr.numpy()): ax.text(c,r,f'{value:g}',ha='center',va='center')
    ax.set_title(title); ax.axis('off')
plt.tight_layout(); plt.show()
'''),
md('''
## 2. 같은 컵, 다른 단서 · 5분
수직 경계·수평 경계·평균 필터를 같은 이미지에 적용합니다. **예상:** 손잡이의 좌우 가장자리와 몸체의 윗변은 어느 필터에서 잘 보일까요?
우리가 넣은 필터는 수작업 규칙입니다. 뒤의 CNN은 같은 역할의 숫자를 학습으로 바꿉니다.
'''),
code(r'''
cup = torch.full((1,1,28,28), .12)
cup[:,:,7:23,5:18] = .85
cup[:,:,9:18,18:24] = .85
cup[:,:,11:16,18:22] = .12
filters = torch.stack([kernel/3, kernel.T/3, torch.ones(3,3)/9])[:,None]
maps = F.conv2d(cup, filters, padding=1)
fig, axes = plt.subplots(1,4,figsize=(12,3))
axes[0].imshow(cup[0,0],cmap='gray',vmin=0,vmax=1); axes[0].set_title('input cup')
for ax, fmap, name in zip(axes[1:], maps[0], ['left-right difference','top-bottom difference','local average']):
    ax.imshow(fmap, cmap='RdBu_r',vmin=-1,vmax=1); ax.set_title(name,fontsize=10)
for ax in axes: ax.axis('off')
plt.tight_layout(); plt.show()
print('입력:', tuple(cup.shape), '출력:', tuple(maps.shape), '(batch, channel, height, width)')
'''),
md('''
**한 값 변경:** 아래 `CHANNEL`을 0, 1, 2로 바꿔 ReLU와 풀링 전후를 비교합니다.
음수 반응을 없애면 어떤 방향의 경계 정보가 사라질까요? 풀링은 세부 위치를 얼마나 남길까요?
'''),
code(r'''
CHANNEL = 0
assert CHANNEL in (0,1,2)
chosen = maps[:,CHANNEL:CHANNEL+1]
activated = F.relu(chosen)
pooled = F.max_pool2d(activated, 2)
fig, axes = plt.subplots(1,3,figsize=(9,3))
for ax, value, title in zip(axes,[chosen,activated,pooled],['signed response','ReLU','2x2 max pool']):
    ax.imshow(value[0,0], cmap='RdBu_r',vmin=-1,vmax=1)
    ax.set_title(f'{title}\n{tuple(value.shape[-2:])}'); ax.axis('off')
plt.tight_layout(); plt.show()
'''),
md('''
## 3. 물체가 움직이면 특징도 움직일까? · 5분
CNN의 공유 필터는 같은 패턴을 다른 위치에서도 계산합니다. 이것은 **특징 맵이 함께 이동하는 등변성**의 직관입니다.
**분류 결과가 같다는 불변성**과 구분합니다. 유한 이미지의 경계·stride·pooling은 이 관계를 깨뜨릴 수 있습니다.
'''),
code(r'''
# 오른쪽으로 1칸 또는 2칸 이동합니다. torch.roll의 반대편 감김을 쓰지 않습니다.
SHIFT = 1
assert SHIFT in (1,2)
def shift_right(x, amount, fill=0.):
    out = torch.full_like(x, fill)
    out[...,amount:] = x[...,:-amount]
    return out

moved = shift_right(cup, SHIFT, .12)
a = F.relu(F.conv2d(cup, filters[:1], padding=1))
b = F.relu(F.conv2d(moved, filters[:1], padding=1))
expected = shift_right(a, SHIFT)
# 이미지 경계를 제외하고, 두 연산 순서의 일치 여부를 수치로 확인합니다.
inner = (..., slice(3,-3), slice(4,-3))
error = float((b[inner]-expected[inner]).abs().max())
assert error < 1e-6
print('경계 제외: 필터→이동 vs 이동→필터 최대 차이 =', error)
gallery(torch.cat([cup,moved,a,b]), titles=['input','shifted input','feature','shifted feature'], n=4)
pa,pb = F.max_pool2d(a,2), F.max_pool2d(b,2)
print('풀링 맵이 완전히 같은가?', bool(torch.allclose(pa,pb)))
'''),
md('''
### 핵심 활동 기록 · 여기까지가 15분 수업 범위
| 질문 | 내 관찰과 근거 |
|---|---|
| `COL=0`과 `COL=2`에서 값이 다른 이유 | 여기에 작성 |
| 반전했을 때 부호가 바뀌는 이유 | 여기에 작성 |
| ReLU/풀링이 버린 정보 | 여기에 작성 |
| 등변성과 분류 불변성의 차이 | 여기에 작성 |

**자가 확인:** 0인 반응은 물체가 없다는 뜻이 아닙니다. 특징 맵이 이동했다고 최종 분류도 항상 같다고 결론 내릴 수 없습니다.
'''),
md('''
## 4. 필터를 사람이 정하지 않으면? · 확장 10분
원/정사각형의 경계를 구분하는 CNN을 학습합니다. 위치·크기·잡음은 바뀌고 면적은 비슷하도록 만듭니다.
train은 가중치 조정, validation은 학습 관찰, test는 고정한 실험의 마지막 평가에 사용합니다.
서로 다른 seed로 생성한 세 집합이며 같은 이미지 파일을 나누는 방식이 아닙니다.

**실행 전 예상:** 랜덤 필터가 어느 방향으로 바뀔지 구체적인 이름까지 예측할 수 있을까요?
'''),
code(r'''
# 변경 후 이 셀부터 다시 실행합니다. epoch는 test 결과를 보기 전에 정합니다.
EPOCHS = 10
train_objects = make_objects(768, 101)
val_objects = make_objects(192, 202)
test_objects = make_objects(192, 303)
x_train, y_train = grayscale(train_objects), train_objects[1]
x_val, y_val = grayscale(val_objects), val_objects[1]
x_test, y_test = grayscale(test_objects), test_objects[1]
gallery(x_train, y_train)
print('train / val / test:', len(y_train), len(y_val), len(y_test))
'''),
md('''
다음 셀은 모델 생성과 학습을 한 번에 실행합니다. 손실은 정답 점수를 더 높이도록 가중치를 조정하는 기준입니다.
**할 일:** 첫 epoch와 마지막 epoch의 손실·검증 정확도를 비교하세요. test 결과에 맞춰 epoch를 골라서는 안 됩니다.
'''),
code(r'''
model, history, initial_weights = train_model(x_train, y_train, x_val, y_val, EPOCHS)
show_history(history)
test_result = evaluate(model,x_test,y_test)
print(f"고정 test {len(y_test)}장: accuracy={test_result['accuracy']:.3f}, CE={test_result['loss']:.3f}")
'''),
md('''
## 5. 학습된 특징을 관찰하기 · 확장 5분
같은 이미지에 대한 첫 층의 학습 전/후 반응을 비교합니다. **밝은 영역은 이 채널의 반응이 크다는 뜻**입니다.
이 그림만으로 “모델이 사람처럼 테두리를 이해했다”거나 “이 채널은 원 뉴런이다”라고 단정하지 않습니다.
'''),
code(r'''
SAMPLE = 0  # 0~191 중 선택
assert 0 <= SAMPLE < len(x_test)
before_model = SmallCNN(); before_model.load_state_dict(initial_weights); before_model.eval()
with torch.no_grad():
    old_maps = F.relu(before_model.features[0](x_test[SAMPLE:SAMPLE+1]))[0]
    new_maps = F.relu(model.features[0](x_test[SAMPLE:SAMPLE+1]))[0]
fig, axes = plt.subplots(2,8,figsize=(14,4))
limit = max(float(old_maps.max()),float(new_maps.max()),1e-6)
for row, values in enumerate([old_maps,new_maps]):
    for i, ax in enumerate(axes[row]):
        ax.imshow(values[i],cmap='magma',vmin=0,vmax=limit)
        ax.set_title(('before' if row==0 else 'after')+f' / {i}',fontsize=9); ax.axis('off')
plt.tight_layout(); plt.show()
gallery(x_test[SAMPLE:SAMPLE+1],y_test[SAMPLE:SAMPLE+1],n=1)
print('첫 필터 평균 절대 변화:', float((model.features[0].weight.detach()-initial_weights['features.0.weight']).abs().mean()))
'''),
md('''
## 6. 단서를 가리면 어떻게 될까? · 확장 5분
같은 test 물체의 가운데 일부를 배경색으로 가립니다. **예상:** 원과 정사각형의 경계를 모두 지우나요, 일부만 지우나요?
가림은 낯선 입력을 만들기도 하므로 정확도 변화만으로 유일한 원인을 확정하지 않습니다.
'''),
code(r'''
OCCLUSION_WIDTH = 10  # 0~20. 모델은 다시 학습하지 않고 입력만 바꿉니다.
assert 0 <= OCCLUSION_WIDTH <= 20
occluded = x_test.clone()
left = 14-OCCLUSION_WIDTH//2
occluded[:,:,:,left:left+OCCLUSION_WIDTH] = .12
occluded_result = evaluate(model,occluded,y_test)
print('원본 정확도:', round(test_result['accuracy'],3), '| 가림 정확도:', round(occluded_result['accuracy'],3))
pred = occluded_result['logits'].argmax(1)
wrong = torch.where(pred != y_test)[0][:6]
chosen = wrong if len(wrong) else torch.arange(6)
print('오분류를 먼저 표시합니다.' if len(wrong) else '이번 조건에는 오분류가 없어 앞의 6장을 표시합니다.')
gallery(occluded[chosen], titles=[f'true {NAMES[y_test[i]]}\npred {NAMES[pred[i]]}' for i in chosen], n=6)
'''),
md('''
## 7. 결과를 내 말로 남기기
아래 문자열을 본인의 관찰로 바꿉니다. 출력 파일은 `outputs/week01_concept/`에 저장됩니다.
Colab에서는 왼쪽 파일 패널에서 JSON을 다운로드하거나, 실행 결과를 포함한 ipynb를 다운로드하세요.
'''),
code(r'''
reflection = {
    '예상': '실행 전에 예상한 변화를 작성하세요.',
    '관찰': '필터·특징 맵 또는 가림 전후에서 실제로 본 결과를 작성하세요.',
    '설명': '관찰이 내 예상과 같거나 달랐던 이유를 작성하세요.',
    '한계': '이 합성 데이터의 결과로 실제 컵 인식에 대해 말할 수 없는 것을 작성하세요.'}
metrics = {'seed': SEED, 'epochs': EPOCHS, 'test_n': len(y_test),
    'test_accuracy': test_result['accuracy'], 'test_ce': test_result['loss'],
    'occlusion_width': OCCLUSION_WIDTH, 'occluded_accuracy': occluded_result['accuracy']}
out = Path('outputs/week01_concept'); out.mkdir(parents=True,exist_ok=True)
(out/'observations.json').write_text(json.dumps({'metrics':metrics,'reflection':reflection},ensure_ascii=False,indent=2),encoding='utf-8')
display(Markdown('\n'.join(f'- **{k}**: {v}' for k,v in reflection.items())))
print('저장:', out/'observations.json')
'''),
md('''
### 자율 실험 · 하나만 골라도 좋습니다
1. **필터:** 수평/수직 필터의 부호를 바꾸고 ReLU 전후 차이를 설명하세요. 핵심 2의 셀부터 실행합니다.
2. **가림:** 폭 0·6·12를 비교하고 정확도와 남은 경계를 기록하세요. 6의 셀만 다시 실행합니다.
3. **학습:** epoch 2와 10을 validation으로 비교하세요. 4부터 새 모델로 실행하고, 비교를 정한 뒤 test를 확인합니다.

| 바꾼 변수 | 고정한 조건 | 예상 | 관찰 | 다음 질문 |
|---|---|---|---|---|
| 작성 | 작성 | 작성 | 작성 | 작성 |

**해석 점검:** 수작업 필터와 학습된 필터는 둘 다 숫자로 된 가중치입니다. 밝은 특징 맵은 정답의 증명이 아닙니다.
가림 후 정확도가 유지되더라도 모든 가림에 강하다는 뜻은 아닙니다.

### 출처와 다음 실습
- [PyTorch Conv2d: 채널과 교차상관 연산](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- [PyTorch: 손실·역전파·optimizer](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- 다음 주 `concept_practice.ipynb`에서 배경색만 바꾸는 통제 실험을 합니다.
- 실제 이미지로 확장하려면 기존 `practice.ipynb`의 FashionMNIST 실습과 `homework_optional.ipynb`를 사용합니다.
''')]

W2 = [
md('''
# 2주차 실습 · 잘 맞혔다고, 모양을 본 것일까?
**같은 물체를 두고 배경만 바꾸는 실험**

사람은 원/정사각형을 구분할 때 모양을 볼 수 있습니다. CNN도 그 단서를 배웠을까요?
원은 붉은 배경, 정사각형은 푸른 배경에 자주 나타나는 데이터를 만들고 배경과 라벨의 관계가 바뀔 때 평가합니다.

| 진행 | 강의 연결 | 활동 | 시간 |
|---|---|---|---|
| 준비 0–2 | PDF 4–8쪽 | 데이터·분할 관찰, 모델 두 개 학습 | 수업 전 또는 5–10분 |
| 핵심 3–5 | PDF 17–24쪽 | 배경 교체·가림·혼동 행렬·해석 | 15분 |
| 확장 6–7 | PDF 22–27쪽 | 조건 변경·보고서·자율 실험 | 15–25분 |

**90분 수업에서는 리더가 준비 셀을 미리 실행하고**, 종이 설계 활동 15분을 핵심 3–5로 교체할 수 있습니다.
처음 배우는 참가자가 전체 노트북을 읽고 실행할 때는 별도 확장 시간이 필요합니다.
Colab에서 위부터 실행합니다. torch·numpy·matplotlib만 필요하며 GPU·데이터 다운로드·1주차 실행 상태는 필요 없습니다.
로컬에서 패키지가 없다면 `%pip install torch numpy matplotlib`를 별도 셀에서 실행하세요.

이것은 지름길 학습을 관찰하기 위한 합성 실험입니다. 아래 성능은 실행으로 측정되며 실제 사물 데이터의 성능이 아닙니다.
'''),
md('## 0. 준비\n보조 함수 두 셀은 실행만 합니다. 1주차와 독립된 파일입니다.'),
code(SETUP), code(HELPERS, True),
md('''
## 1. 모양과 배경 중 무엇이 쉬운 단서일까?
**예상부터 적기:** 학습 이미지에서 배경색만으로 거의 정답을 알 수 있다면 CNN이 꼭 모양을 배워야 할까요?
두 학습 집합의 **물체·라벨·잡음·표본 수는 같고**, 배경-라벨의 상관만 다르게 만듭니다.

- `biased`: 원→붉은색, 정사각형→푸른색이 기본값 98% 확률로 맞습니다.
- `balanced`: 물체마다 붉은색/푸른색 배경을 각각 한 번 넣습니다. 두 집합 모두 물체를 두 번씩 사용하므로 학습 표본 수가 같습니다.
- 같은 물체의 두 배경 사진은 같은 split 안에 둡니다. 다른 split은 다른 seed로 새 물체와 잡음을 생성합니다.
'''),
code(r'''
def paint(objects, background_ids, hide_object=False):
    mask, _, noise = objects
    palette = torch.tensor([[.58,.10,.10],[.10,.10,.58]],dtype=torch.float32)
    bg = palette[background_ids][:,:,None,None].expand(-1,-1,28,28)
    visible = torch.zeros_like(mask) if hide_object else mask
    return (bg*(1-visible) + .92*visible + noise).clamp(0,1)

def repeat_objects(objects):
    return tuple(torch.cat([t,t]) for t in objects)

def training_set(objects, correlation, seed, balanced=False):
    twice = repeat_objects(objects)
    labels = twice[1]
    if balanced:
        bg = torch.cat([objects[1],1-objects[1]])
    else:
        match = torch.rand(len(labels),generator=torch.Generator().manual_seed(seed)) < correlation
        bg = torch.where(match,labels,1-labels)
    return paint(twice,bg), labels, bg

CORRELATION = .98  # 자율 실험: .5, .8, .98. 변경 시 이 셀부터 다시 실행합니다.
EPOCHS = 10        # test를 보기 전에 고정합니다.
assert .0 <= CORRELATION <= 1.0
train_objects = make_objects(384,101)
val_objects = make_objects(96,202)
test_objects = make_objects(192,303)
xb,yb,bgb = training_set(train_objects,CORRELATION,501)
xc,yc,bgc = training_set(train_objects,CORRELATION,501,balanced=True)
vb,vy,_ = training_set(val_objects,CORRELATION,502)
vc,_,_ = training_set(val_objects,CORRELATION,502,balanced=True)
assert torch.equal(yb,yc) and len(yb) == len(yc)
mask = repeat_objects(train_objects)[0].expand(-1,3,-1,-1).bool()
assert torch.equal(xb[mask],xc[mask]), '전경 픽셀은 같아야 합니다.'
print('biased 실제 배경-라벨 일치율:', float((bgb == yb).float().mean()))
print('balanced 실제 배경-라벨 일치율:', float((bgc == yc).float().mean()))
print('고유 물체 train/val/test:',384,96,192,'| 학습 이미지 수:',len(yb))
gallery(xb,yb)
gallery(xc,yc)
'''),
md('''
### 데이터 검토
처음 몇 장만 보는 것으로 전체 상관을 판단하지 않습니다. 아래 표에서 클래스마다 두 배경의 개수를 확인합니다.
`balanced`는 같은 물체의 배경만 바꾸는 수집/증강을 흉내 냅니다. 실제 사진의 배경 교체는 조명·그림자도 바꿀 수 있어 이 실험처럼 완벽히 통제하기 어렵습니다.
'''),
code(r'''
rows = ['| train set | shape | red | blue |','|---|---|---:|---:|']
for name,y,bg in [('biased',yb,bgb),('balanced',yc,bgc)]:
    for label in [0,1]:
        counts = [int(((y==label)&(bg==color)).sum()) for color in [0,1]]
        rows.append(f'| {name} | {NAMES[label]} | {counts[0]} | {counts[1]} |')
display(Markdown('\n'.join(rows)))
'''),
md('''
## 2. 같은 구조와 학습 예산으로 모델 두 개 만들기
두 모델은 같은 초기 가중치·배치 순서·optimizer·epoch 수를 사용합니다. 변경한 것은 학습 배경입니다.
validation은 각 학습 환경과 같은 배경 규칙으로 구성하며 학습 관찰에만 씁니다. **두 모델 모두 미리 정한 마지막 epoch**로 평가합니다.
test 결과가 좋은 epoch를 고르는 코드는 없습니다.
'''),
code(r'''
print('A: biased training')
biased_model,biased_history,initial_a = train_model(xb,yb,vb,vy,EPOCHS)
print('\nB: balanced backgrounds')
balanced_model,balanced_history,initial_b = train_model(xc,yc,vc,vy,EPOCHS)
assert all(torch.equal(initial_a[k],initial_b[k]) for k in initial_a)
models = {'biased': biased_model, 'balanced': balanced_model}
show_history(biased_history)
show_history(balanced_history)
'''),
md('''
## 3. 시험지의 배경 규칙만 바꾸기 · 핵심 5분
**실행 전 예상:** 각 모델에 대해 아래 세 조건의 정확도 순서를 적으세요.

1. `matched`: 모든 원은 붉은색, 모든 정사각형은 푸른색. 강한 상관이 유지됩니다.
2. `swapped`: 같은 물체의 배경만 반대로 바꿉니다.
3. `balanced`: 같은 test 물체를 양쪽 배경에서 모두 평가합니다.

matched는 학습과 **같은 상관 방향**이며 학습의 98%를 정확히 재현한 분포는 아닙니다.
조건 사이에 test 물체를 공유하는 것은 배경만 통제하려는 의도입니다. 이를 train/test 누출과 혼동하지 마세요.
'''),
code(r'''
ty = test_objects[1]
matched = paint(test_objects,ty)
swapped = paint(test_objects,1-ty)
both_objects = repeat_objects(test_objects)
balanced_test = paint(both_objects,torch.cat([ty,1-ty]))
test_conditions = {'matched':(matched,ty),'swapped':(swapped,ty),
                   'balanced':(balanced_test,both_objects[1])}
foreground = test_objects[0].expand(-1,3,-1,-1).bool()
assert torch.equal(matched[foreground],swapped[foreground])
gallery(torch.stack([matched[0],swapped[0],matched[1],swapped[1]]),
    titles=['object 0 / matched','object 0 / swapped','object 1 / matched','object 1 / swapped'],n=4)
results = {}
table = ['| model | condition | images | accuracy | CE |','|---|---|---:|---:|---:|']
for name,net in models.items():
    results[name] = {}
    for condition,(x,y) in test_conditions.items():
        result = evaluate(net,x,y); results[name][condition] = result
        table.append(f"| {name} | {condition} | {len(y)} | {result['accuracy']:.3f} | {result['loss']:.3f} |")
display(Markdown('\n'.join(table)))
'''),
md('''
### 같은 물체의 예측이 얼마나 뒤집혔나?
정확도 차이 외에 **배경 교체 전후 예측이 달라진 물체 비율**도 봅니다. 이것은 이 실험에서 배경 변화에 민감하다는 증거입니다.
정확도가 낮거나 예측이 바뀐 것만으로 모델 내부의 모든 계산을 설명할 수는 없습니다.
'''),
code(r'''
flip_rates = {}
fig, ax = plt.subplots(figsize=(8,3))
positions = np.arange(3)
for j,(name,net) in enumerate(models.items()):
    values = [results[name][c]['accuracy'] for c in test_conditions]
    ax.bar(positions+(j-.5)*.32,values,width=.32,label=name)
    p = results[name]['matched']['logits'].argmax(1)
    q = results[name]['swapped']['logits'].argmax(1)
    flip_rates[name] = float((p!=q).float().mean())
    print(name, '| 배경 교체 시 예측 변경 비율:', round(flip_rates[name],3))
ax.set_xticks(positions,list(test_conditions)); ax.set(ylabel='accuracy',ylim=(0,1.08))
ax.legend(); plt.tight_layout(); plt.show()
'''),
md('''
## 4. 어떤 클래스를 틀렸나? · 핵심 4분
`MODEL`과 `CONDITION`을 바꿔 혼동 행렬을 비교합니다. 행은 실제 라벨, 열은 예측입니다.
**질문:** 정확도 한 숫자에서 감춰졌던 클래스별 오류가 있나요? 모든 원을 정사각형으로, 모든 정사각형을 원으로 예측하는 것과 한 클래스만 예측하는 것은 어떻게 다른가요?
'''),
code(r'''
MODEL = 'biased'       # 'biased' 또는 'balanced'
CONDITION = 'swapped'  # 'matched', 'swapped', 'balanced'
r = results[MODEL][CONDITION]
cm = r['cm'].numpy()
fig, ax = plt.subplots(figsize=(4,3))
ax.imshow(cm,cmap='Blues',vmin=0,vmax=max(cm.max(),1))
for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha='center',va='center',fontsize=16)
ax.set_xticks([0,1],NAMES); ax.set_yticks([0,1],NAMES)
ax.set(xlabel='predicted',ylabel='true',title=f'{MODEL} / {CONDITION}')
plt.tight_layout(); plt.show()
recall = cm.diagonal()/cm.sum(1).clip(min=1)
print('클래스별 recall:', dict(zip(NAMES,recall.round(3))))
x,y = test_conditions[CONDITION]
pred = r['logits'].argmax(1)
wrong = torch.where(pred!=y)[0][:6]
chosen = wrong if len(wrong) else torch.arange(6)
print('오분류 예시' if len(wrong) else '오분류 없음: 정상 예시')
gallery(x[chosen],titles=[f'true {NAMES[y[i]]}\npred {NAMES[pred[i]]}' for i in chosen],n=6)
'''),
md('''
## 5. 모양을 지워도 정답을 맞힐까? · 핵심 6분
이번에는 배경을 그대로 두고 **도형 전체를 제거**합니다. 평가 라벨은 원래 물체의 라벨을 그대로 사용합니다.
따라서 이 점수는 “없는 물체를 인식한 정확도”가 아니라 **배경만으로 원래 라벨을 얼마나 예측하는가**입니다.
**예상:** biased와 balanced 중 어느 모델이 배경만 보고도 높은 점수를 얻을까요?
'''),
code(r'''
background_only = paint(test_objects,ty,hide_object=True)
gallery(background_only[:4],titles=['background only']*4,n=4)
background_results = {}
for name,net in models.items():
    r = evaluate(net,background_only,ty)
    background_results[name] = r
    print(f"{name}: 원래 라벨에 대한 배경-only 일치율 {r['accuracy']:.3f}")
'''),
md('''
### 15분 활동의 결론 쓰기
| 질문 | 관찰한 수치/이미지 | 그 증거가 뒷받침하는 주장 |
|---|---|---|
| 배경을 바꾸면 예측이 얼마나 달라졌나? | 작성 | 작성 |
| 배경-only의 점수는 무엇을 뜻하나? | 작성 | 작성 |
| 배경을 균형 있게 수집한 효과는? | 작성 | 작성 |

**자가 확인:** 맞힌 이유를 알려면 평가 조건을 바꿔야 합니다. 배경 균형화가 이 장난감 문제에서 도움이 되어도 실제 환경의 모든 지름길을 제거했다는 뜻은 아닙니다.
차이가 예상보다 작으면 배경 상관율·학습 정도·표본 수를 점검하세요. 원하는 결론이 나올 때까지 test에 맞춰 설정을 고르지 않습니다.
'''),
md('''
## 6. 비교의 조건을 점검하기 · 확장
두 모델이 같은 초기값과 표본 수로 학습했는지 다시 확인합니다. 한 seed의 결과는 여러 번 반복한 평균과 다릅니다.
`balanced` test의 384장은 192개 물체의 쌍이며 독립 물체 384개로 취급해서는 안 됩니다.
'''),
code(r'''
assert all(torch.equal(initial_a[k],initial_b[k]) for k in initial_a)
assert len(xb)==len(xc) and torch.equal(yb,yc)
assert len(biased_history['train_loss']) == len(balanced_history['train_loss']) == EPOCHS
assert float((bgc==yc).float().mean()) == .5
assert len(ty) == 192 and len(balanced_test) == 2*len(ty)
print('확인: 같은 초기값·학습 예산·전경·라벨, 배경 균형 50%, paired test 192개 물체')
'''),
md('''
## 7. 작은 실험 보고서 저장
네 문장을 자신의 관찰로 바꾸세요. 실제 측정값과 글은 `outputs/week02_concept/report.json`에 저장합니다.
Colab 파일 패널에서 다운로드하거나 출력이 포함된 ipynb를 저장하세요.
'''),
code(r'''
reflection = {
    '질문': '배경과 라벨의 상관을 줄이면 배경 교체에 덜 민감해지는가?',
    '방법': '고정한 조건과 바꾼 변수를 내 말로 작성하세요.',
    '증거': '표에서 실제 관측한 정확도·예측 변경 비율을 인용하세요.',
    '한계와 다음 실험': '다른 seed 또는 다른 배경·도형에서 확인할 내용을 작성하세요.'}
metrics = {'seed': SEED, 'epochs': EPOCHS, 'correlation': CORRELATION,
    'train_images':len(yb), 'test_unique_objects':len(ty), 'flip_rates':flip_rates,
    'scores':{name:{cond:{'accuracy':r['accuracy'],'ce':r['loss'],'confusion':r['cm'].tolist()}
       for cond,r in group.items()} for name,group in results.items()},
    'background_only_label_agreement':{name:r['accuracy'] for name,r in background_results.items()}}
out = Path('outputs/week02_concept'); out.mkdir(parents=True,exist_ok=True)
(out/'report.json').write_text(json.dumps({'metrics':metrics,'reflection':reflection},ensure_ascii=False,indent=2),encoding='utf-8')
display(Markdown('\n'.join(f'- **{k}**: {v}' for k,v in reflection.items())))
print('저장:',out/'report.json')
'''),
md('''
### 자율 실험 · 하나를 선택하세요
1. **상관율:** `.5`, `.8`, `.98`을 비교합니다. 1부터 새 모델로 실행하고 matched/swapped 결과를 함께 기록하세요.
2. **반복:** 준비 셀 아래 `SEED`를 17·18·19로 바꾸고 두 모델을 다시 학습하세요. 데이터 seed를 유지하면 초기값/배치 순서의 변동을 비교합니다. 평균과 최솟값·최댓값을 쓰세요.
3. **새 시험지:** 다른 배경색이나 원/정사각형의 새로운 위치를 정합니다. 테스트 전에 평가 가설을 적고 `paint` 또는 `make_objects`를 복사해 별도 조건을 만드세요.

| 변경한 변수 | 고정한 변수 | matched | swapped | 예측 변경 비율 | 해석 |
|---|---|---|---|---|---|
| 작성 | 작성 | 측정값 | 측정값 | 측정값 | 작성 |

**해설 방향:** 배경-only에서도 원래 라벨을 맞히면 모양이 없는 단서만으로도 답을 얻었다는 뜻입니다.
배경 교체와 물체 제거는 서로 다른 개입이므로 함께 해석합니다. balanced 모델도 충분히 학습하지 못하거나 다른 단서에 의존할 수 있습니다.
실험 결과를 본 뒤 바꾼 설정은 탐색으로 표시하고, 최종 주장은 별도 보류 데이터로 확인하세요.

### 출처와 실제 이미지 확장
- [Geirhos et al.: Shortcut Learning](https://arxiv.org/abs/2004.07780) — 문제 설정의 배경. 이 노트북은 논문 실험의 재현이 아닙니다.
- [PyTorch: 손실과 최적화](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- 기존 `practice.ipynb`의 CIFAR10 증강 비교와 `homework_optional.ipynb`로 실제 이미지 실험을 확장할 수 있습니다.
''')]

def write_notebook(folder, cells):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'}, 'colab':{'name':'concept_practice.ipynb','provenance':[]}}
    for i,cell in enumerate(nb.cells):
        cell.id = f'concept-{folder[:6]}-{i:02d}'
        if cell.cell_type == 'code': compile(cell.source, f'{folder}/cell-{i}', 'exec')
    nbf.validate(nb)
    path = ROOT/folder/'concept_practice.ipynb'
    nbf.write(nb,path)
    print(f'Built {path.relative_to(ROOT)}: {len(cells)} cells')

if __name__ == '__main__':
    write_notebook('week01_cnn_basics',W1)
    write_notebook('week02_cnn_advanced',W2)
