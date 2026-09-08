from pathlib import Path
import textwrap, nbformat as nbf
from curriculum import WEEKS
import notebook_code as C

ROOT=Path(__file__).resolve().parents[1]
def md(s): return nbf.v4.new_markdown_cell(textwrap.dedent(s).strip())
def code(s): return nbf.v4.new_code_cell(textwrap.dedent(s).strip())
def save(cells,path):
    nb=nbf.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.13'},
        'colab':{'provenance':[]},'aim':{'course':'2026-2','standalone':True}})
    nbf.validate(nb); nbf.write(nb,path)

def section(cells,title,description,source,question=None):
    cells += [md('## '+title+'\n\n'+description),code(source)]
    if question: cells += [md('**관찰·토론**\n\n'+question)]

def base(wi,kind='실습'):
    w=WEEKS[wi-1]
    cells=[md(f'''# AIM · {wi}주차 {kind}\n## {w['title']}

**대상:** Python·PyTorch 기초 이수자 / **권장:** Google Colab Python 3, GPU 선택 가능.

{w['intro']}

### 학습 목표
'''+ '\n'.join('- '+g for g in w['goals'])+'''

### 실행 방법
1. Colab에서 `파일 → 노트북 업로드`로 이 파일을 엽니다.
2. GPU를 사용할 경우 `런타임 → 런타임 유형 변경`에서 GPU를 선택합니다.
3. 위에서 아래로 셀을 실행합니다. 이전 주차 파일이나 별도 Python 모듈은 필요 없습니다.
4. Colab 기본 torch·torchvision을 우선 사용합니다. import가 실패하면 새 런타임에서 시작하세요.
   로컬 환경은 루트 `환경설정.md`를 참고하세요. 데이터 최초 다운로드에는 인터넷이 필요합니다.
5. 학습이 길면 `TRAIN_N`, `EPOCHS` 또는 `STEPS`를 줄입니다. 작은 실험의 품질은 보장되지 않습니다.

**시간 운영:** 실습 셀 번호가 아니라 아래 § 구간으로 진행합니다. 준비·다운로드는 수업 시작에,
핵심 실행은 50–75분에, 관찰 답변은 75–85분에 수행합니다. 심화와 자율 과제는 수업 밖에서 진행합니다.

**재현성:** 고정 seed도 다른 GPU·라이브러리 버전의 완전히 같은 수치를 보장하지 않습니다.
실제 출력과 예측을 구분해 기록하세요. `outputs/`는 현재 실행 폴더에 생성됩니다.
'''),md('## §1 · 환경 확인\n\n첫 실행에서 버전과 device를 확인합니다. CPU도 지원하며 설치가 필요한 경우 환경설정 문서를 먼저 읽습니다.'),code(C.SETUP)]
    if wi in (1,2,5,6):
        name={1:'fashion',2:'cifar',5:'mnist',6:'mnist'}[wi]
        quick_epochs={1:3,2:3,5:10,6:8}[wi]
        cells += [code(f"DATASET = '{name}'\nTRAIN_N = 256 if SMOKE else (4096 if QUICK else 40000)\nVAL_N = 128 if SMOKE else (1024 if QUICK else 5000)\nTEST_N = 128 if SMOKE else (1024 if QUICK else 10000)\nBATCH = 64 if SMOKE else 128\nEPOCHS = 1 if SMOKE else ({quick_epochs} if QUICK else 20)"),md('## §2 · 데이터와 분할\n\n공식 train을 고정 인덱스로 나누고 test는 마지막에만 사용합니다. QUICK의 평가 숫자는 전체 공식 test의 성능이 아닌 고정 부분집합 성능입니다. MNIST 계열은 0~1, CIFAR10만 -1~1로 변환합니다.'),code(C.VISION_DATA)]
    else:
        cells += [code('BATCH = 8 if SMOKE else 32\nCONTEXT = 16 if SMOKE else 32\nSTEPS = 6 if SMOKE else (240 if QUICK else 1500)'),md('## §2 · 코퍼스, vocabulary, next-token target\n\n수업용으로 새로 작성한 조합 문장을 사용합니다. 문장 단위로 먼저 분할하고 split 내부에서만 window를 만듭니다. exact 문장은 분리되지만 문법과 어휘는 공유됩니다. 실제 자연어 능력을 평가하는 자료는 아닙니다.'),code(C.CORPUS)]
    return cells

def foundations(wi,cells):
    if wi in (1,2):
        section(cells,'§3 · 분류 모델 정의','모델 코드의 각 층 옆에 입출력 shape를 손으로 적어 보세요. MLP/SmallCNN은 FashionMNIST용, CifarCNN은 컬러 이미지용입니다.',C.CLASS_MODELS)
        section(cells,'§4 · 공통 학습·평가 함수','샘플 수 가중 평균, eval/no_grad, 최고 validation checkpoint 복사를 확인합니다.',C.CLASS_TRAIN)
    elif wi in (3,4):
        section(cells,'§3 · attention 직접 구현','`scaled_attention`은 True인 위치를 차단하는 `blocked`를 사용합니다. softmax는 key 축에 적용합니다.',C.ATTENTION)
        if wi==4: section(cells,'§3-2 · multi-head와 Transformer block','pre-norm, residual, FFN과 헤드 reshape를 추적합니다. 이 모델은 원 논문의 encoder-decoder 전체 재현이 아닙니다.',C.TRANSFORMER)
        section(cells,'§4 · 언어 모델 학습과 생성 함수','평가는 고정 window의 token CE를 집계합니다. 마지막 불완전 window는 제외합니다. 생성은 매번 마지막 CONTEXT 문자를 다시 계산합니다.',C.LM_TRAIN)
    else:
        section(cells,'§3 · VAE와 ELBO 구현','decoder는 logits를 출력합니다. recon은 픽셀 합, KL은 latent 합, 최종 손실은 배치 평균입니다. 평가는 1-sample MC negative ELBO이며 그림의 μ 재구성과 구분합니다.',C.VAE)
    return cells

def practice(wi):
    cells=foundations(wi,base(wi))
    if wi==1:
        section(cells,'§3-검사 · 합성곱 손계산','출력을 실행하기 전에 계산하세요. bias=1을 포함해야 합니다.',r'''
patch=torch.tensor([[[[1.,2.],[3.,4.]]]])
kernel=torch.tensor([[[[1.,0.],[0.,-1.]]]])
result=F.conv2d(patch,kernel,bias=torch.tensor([1.]))
assert result.item()==-2
print('manual convolution:',result.item())
model=SmallCNN(); x=torch.zeros(2,1,28,28)
for layer in model.features:
    x=layer(x); print(type(layer).__name__,tuple(x.shape))
assert x.flatten(1).shape[1]==1568
assert model(torch.zeros(2,1,28,28)).shape==(2,10)
print('MLP params:',num_params(MLP()),'CNN params:',num_params(model))
''','Flatten을 하면 왜 1,568차원인가요? MLP와 CNN의 파라미터 수가 다르다는 점은 비교 해석에 어떤 영향을 주나요?')
        section(cells,'§5 · MLP와 CNN 학습','학습 시작 전에 val loss가 어떻게 바뀔지 예측합니다. 두 모델 모두 같은 분할과 epoch 예산을 사용합니다.',r'''
models={}; histories={}; rows=[]
for name,factory in [('MLP',MLP),('CNN',SmallCNN)]:
    train_loader.generator.manual_seed(SEED)
    m,h,seconds=fit_classifier(factory,train_loader)
    models[name]=m; histories[name]=h
    r=classification_epoch(m,val_loader)
    rows.append(dict(model=name,params=num_params(m),seconds=round(seconds,2),val_loss=r['loss'],val_acc=r['accuracy']))
display(rows); plot_classification(histories)
(OUT/'cnn_comparison.json').write_text(json.dumps(rows,indent=2))
''','CNN이 더 좋았나요? 아니라면 학습 예산, 모델 크기, 초기화 중 무엇을 먼저 추가 확인하겠나요?')
        section(cells,'§6 · CNN의 실패 사례와 특징 맵','분석은 validation에서 수행합니다. 특징 맵은 활성화이며 인과적 중요도 지도가 아닙니다.',r'''
cnn=models['CNN']; cm=inspect_classifier(cnn)
cnn.eval()
with torch.no_grad():
    sample=next(iter(val_loader))[0][:1].to(DEVICE)
    activation=F.relu(cnn.features[0](sample))[0,:8].unsqueeze(1)
image_grid(activation / activation.amax((2,3),keepdim=True).clamp_min(1e-6),title='First Conv activations (each channel scaled)')
''','가장 많이 혼동된 클래스 쌍을 찾고 실제 이미지 2개로 가능한 이유를 설명하세요. 채널별 정규화 때문에 밝기를 채널 간 직접 비교할 수 있을까요?')
        section(cells,'§7 · 모델 선택 후 test 한 번','val loss로 조건을 선택합니다. test를 본 뒤 설정을 바꾸면 새로운 검증 절차가 필요합니다.',r'''
chosen=min(rows,key=lambda r:r['val_loss'])['model']
final=classification_epoch(models[chosen],test_loader)
print('selected:',chosen,'| test subset loss:',final['loss'],'| accuracy:',final['accuracy'])
torch.save({'state_dict':cpu_state(models[chosen]),'model_name':chosen,'seed':SEED},OUT/'classifier.pt')
''')
    elif wi==2:
        section(cells,'§3-검사 · 증강과 residual','같은 원본의 여러 변형과 validation의 결정성을 확인합니다.',r'''
aug_train,_,_=vision_loaders('cifar',augment=True)
versions=torch.stack([aug_train.dataset[0][0] for _ in range(8)])
image_grid(versions,title='Same training image, random augmentation',normalized=True)
torch.testing.assert_close(val_loader.dataset[0][0],val_loader.dataset[0][0])
block=ResidualBlock(32,64,stride=2)
assert block(torch.randn(2,32,16,16)).shape==(2,64,8,8)
assert CifarCNN()(torch.randn(2,3,32,32)).shape==(2,10)
print('PASS: deterministic validation / projected shortcut shapes')
''','반전·crop 중 클래스 의미를 바꾸는 변환이 있는지 확인하세요. 다른 데이터에서도 같은 증강이 적절할까요?')
        section(cells,'§4-실행 · base 대 augment','네트워크는 동일하며 train 변환만 바꿉니다. residual 비교는 자율 과제에 있습니다.',r'''
models={}; histories={}; rows=[]
for name,loader in [('base',train_loader),('augment',aug_train)]:
    loader.generator.manual_seed(SEED)
    m,h,seconds=fit_classifier(CifarCNN,loader)
    models[name]=m; histories[name]=h
    result=classification_epoch(m,val_loader)
    rows.append(dict(condition=name,params=num_params(m),seconds=round(seconds,2),val_loss=result['loss'],val_acc=result['accuracy']))
display(rows); plot_classification(histories)
(OUT/'ablation.json').write_text(json.dumps(rows,indent=2))
''','증강 모델의 train loss가 더 높다면 반드시 나쁜 모델인가요? 두 모델의 train loss 측정 입력 조건을 비교하세요.')
        section(cells,'§5 · validation으로 선택·오류 분석','선택 기준은 val loss입니다. 클래스별 recall의 분모와 confusion matrix 축을 설명해 보세요.',r'''
chosen=min(rows,key=lambda r:r['val_loss'])['condition']; selected=models[chosen]
print('selected by validation loss:',chosen)
cm=inspect_classifier(selected)
torch.save({'state_dict':cpu_state(selected),'condition':chosen,'seed':SEED},OUT/'cifar_classifier.pt')
''','최약 클래스에 대한 후속 실험 한 가지를 제안하세요. 제안이 데이터 관찰에 근거하는지 확인하세요.')
        section(cells,'§6 · 선택한 모델의 최종 test','부분집합 평가이며 전체 공식 test의 결과와 다릅니다.',r'''
result=classification_epoch(selected,test_loader)
print('test subset size:',len(test_loader.dataset),'loss:',result['loss'],'accuracy:',result['accuracy'])
''')
    elif wi==3:
        section(cells,'§3-검사 · 숫자로 attention 이해','Q,K,V 예제는 설명용이며 학습된 값이 아닙니다. 행 합과 mask를 함께 검사합니다.',r'''
q=torch.tensor([[[1.,0.]]]); k=torch.tensor([[[1.,0.],[0.,1.]]]); v=torch.tensor([[[2.,0.],[0.,2.]]])
out,a=scaled_attention(q,k,v,causal=False)
print('weights:',a,'output:',out)
torch.testing.assert_close(out,torch.tensor([[[1.3395,.6605]]]),atol=1e-3,rtol=1e-3)
q=torch.randn(1,5,8); k=torch.randn(1,5,8); v=torch.randn(1,5,8)
_,full=scaled_attention(q,k,v,False); out,masked=scaled_attention(q,k,v,True)
torch.testing.assert_close(masked.sum(-1),torch.ones(1,5))
assert masked.triu(1).abs().max()==0 and out.shape==(1,5,8)
fig,axes=plt.subplots(1,2,figsize=(8,3))
for ax,a,title in zip(axes,[full,masked],['No mask','Causal mask']):
    ax.imshow(a[0],vmin=0,vmax=1); ax.set(xlabel='key',ylabel='query',title=title)
plt.show()
''','첫 query의 가중치가 [1,0,0,0,0]인 이유를 설명하세요. 모든 행을 이렇게 만들면 문맥을 사용할 수 있을까요?')
        section(cells,'§5 · 기준 모델과 attention LM 학습','기본 예산에서는 철자나 간단한 문법 일부만 배울 수 있습니다. 평균 CE를 uniform baseline ln(V)와도 비교하세요.',r'''
models={}; histories={}; rows=[]
for name,factory in [('bigram',BigramLM),('attention',SingleHeadLM)]:
    m,h,seconds=fit_lm(factory)
    models[name]=m; histories[name]=h
    val=lm_eval(m)
    rows.append(dict(model=name,params=num_params(m),val_ce=val,val_ppl=math.exp(val),seconds=round(seconds,2)))
display(rows); print('uniform CE:',math.log(VOCAB)); plot_lm(histories)
''','두 모델의 파라미터 수와 val CE를 함께 읽으세요. 복잡한 모델이 더 좋다는 보편적 결론을 낼 수 있나요?')
        section(cells,'§6 · 생성과 learned attention','같은 prompt·seed에서 temperature만 바꿉니다.',r'''
lm=models['attention']; assert_causal(lm)
for tau in [.5,1.,1.5]: print(f'\ntemperature={tau}\n'+generate(lm,temperature=tau))
prompt='the cat reads a book'[:CONTEXT]
ids=torch.tensor([encode(prompt)],device=DEVICE)
with torch.no_grad(): logits,a=lm(ids,return_attention=True)
fig,ax=plt.subplots(figsize=(7,5)); ax.imshow(a[0].cpu(),vmin=0,vmax=1)
ax.set(xticks=range(len(prompt)),xticklabels=list(prompt),yticks=range(len(prompt)),yticklabels=list(prompt),xlabel='key',ylabel='query')
plt.show()
''','반복, 철자 오류, 문장 구조를 따로 기록하세요. 높은 attention이 곧 예측의 유일한 원인이라고 말할 수 있나요?')
        section(cells,'§7 · 선택 완료 후 test','같은 tokenizer와 평가 window 조건으로 비교합니다.',r'''
chosen=min(rows,key=lambda r:r['val_ce'])['model']; ce=lm_eval(models[chosen],'test')
print('selected:',chosen,'test CE:',ce,'test PPL:',math.exp(ce))
(OUT/'lm_comparison.json').write_text(json.dumps(rows,indent=2))
''')
    elif wi==4:
        section(cells,'§3-검사 · shape와 causal 불변성','eval 상태에서 미래 토큰만 변경해 과거 logits가 같은지 확인합니다.',r'''
m=TinyTransformer().to(DEVICE); x,y=lm_batch(batch=2)
assert m(x).shape==(2,CONTEXT,VOCAB)
assert_causal(m)
print('params:',num_params(m),'D=64 / heads=4 / head_dim=16')
''','헤드 수를 8로 바꾸면 head 차원과 QKV projection 파라미터 수는 어떻게 되나요?')
        section(cells,'§4-실행 · Transformer 학습','train CE는 기록 시점의 한 배치, val CE는 고정 평가 window 전체 평균입니다.',r'''
lm,history,seconds=fit_lm(TinyTransformer)
plot_lm({'transformer':history})
val_ce=lm_eval(lm); test_ce=lm_eval(lm,'test')
print('seconds:',seconds,'val CE:',val_ce,'test CE:',test_ce,'test PPL:',math.exp(test_ce))
assert_causal(lm)
''','곡선의 train 값이 val 값보다 높거나 낮은 경우를 무조건 과적합이라고 부를 수 있나요? dropout과 측정 표본 차이도 고려하세요.')
        section(cells,'§5 · 네 가지 생성 방식','모든 조건은 같은 학습된 가중치와 prompt를 사용합니다. sampling은 재학습이 아닙니다.',r'''
settings=[('greedy',dict(greedy=True)),('temperature .7',dict(temperature=.7)),
          ('temperature 1.3',dict(temperature=1.3)),('top-k 5',dict(top_k=5))]
samples={name:generate(lm,**kwargs) for name,kwargs in settings}
for name,text in samples.items(): print('\n'+name+'\n'+text)
(OUT/'generated_samples.json').write_text(json.dumps(samples,indent=2))
''','반복되는 문자열 길이, 잘못된 철자, 문장 마침표 사용을 비교하세요. 한 샘플만 보고 설정을 고르지 마세요.')
        section(cells,'§6 · 저장·복원 검증','자신이 만든 checkpoint만 불러옵니다. vocabulary와 구조 설정을 함께 보존합니다.',r'''
checkpoint={'state_dict':cpu_state(lm),'config':lm.config,'itos':itos,'context':CONTEXT,'seed':SEED}
torch.save(checkpoint,OUT/'tiny_transformer.pt')
loaded=torch.load(OUT/'tiny_transformer.pt',map_location='cpu',weights_only=True)
assert loaded['itos']==itos and loaded['context']==CONTEXT
restored=TinyTransformer(**loaded['config']).to(DEVICE)
restored.load_state_dict(loaded['state_dict']); restored.eval(); lm.eval()
x,_=lm_batch(batch=2)
with torch.no_grad(): torch.testing.assert_close(restored(x),lm(x))
print('PASS: checkpoint round-trip logits')
''')
    elif wi==5:
        section(cells,'§3-검사 · KL와 재매개화','정답을 먼저 손으로 계산하세요. 큰 표본 통계는 근사이므로 허용 오차가 필요합니다.',r'''
zero=torch.zeros(2,3)
torch.testing.assert_close(gaussian_kl(zero,zero),zero)
mu=torch.tensor([[1.,0.]]); lv=torch.zeros_like(mu)
assert gaussian_kl(mu,lv).sum().item()==.5
seed_all(); z=reparameterize(torch.full((20000,),2.),torch.full((20000,),math.log(4.)))
print('sample mean:',z.mean().item(),'variance:',z.var().item())
assert abs(z.mean().item()-2)<.1 and abs(z.var().item()-4)<.2
mu=torch.zeros(4,2,requires_grad=True); lv=torch.zeros(4,2,requires_grad=True)
reparameterize(mu,lv).square().mean().backward()
assert mu.grad is not None and lv.grad is not None
print('PASS: KL closed form and gradient paths')
''','표준편차 계산에 0.5를 빼먹으면 표본 분산이 어떻게 바뀌나요?')
        section(cells,'§4 · 2차원 VAE 학습','기본 β=1. train과 val은 stochastic reconstruction을 사용하며, val 평가 난수는 고정합니다.',r'''
vae,history,seconds=fit_vae(zdim=2)
plot_vae(history)
print('seconds:',seconds,'validation:',eval_vae(vae,val_loader))
torch.save({'state_dict':cpu_state(vae),'zdim':2,'seed':SEED},OUT/'vae.pt')
''','recon만 줄고 KL이 증가하면 학습 실패인가요? 두 항의 역할로 설명하세요.')
        section(cells,'§5 · 재구성과 prior 샘플','그림은 decoder의 픽셀 확률입니다. 특정 입력의 재구성과 입력 없는 생성을 구분합니다.',r'''
recon_and_samples(vae)
''','어느 격자가 입력 이미지 정보를 사용했나요? prior 샘플의 품질을 재구성 품질로 대신 설명할 수 있을까요?')
        section(cells,'§6 · 2D μ 지도와 보간','라벨은 학습에 쓰지 않고 색으로만 표시합니다.',r'''
mus=[]; labels=[]; vae.eval()
with torch.no_grad():
    for x,y in val_loader:
        mu,lv=vae.encode(x.to(DEVICE)); mus.append(mu.cpu()); labels.append(y)
mus=torch.cat(mus); labels=torch.cat(labels)
plt.figure(figsize=(6,4)); plt.scatter(mus[:,0],mus[:,1],c=labels,cmap='tab10',s=8,alpha=.65)
plt.colorbar(ticks=range(10)); plt.xlabel('mu_1'); plt.ylabel('mu_2'); plt.show()
alpha=torch.linspace(0,1,8,device=DEVICE).unsqueeze(1)
za,zb=mus[0].to(DEVICE),mus[1].to(DEVICE)
with torch.no_grad(): interpolated=vae.decode((1-alpha)*za+alpha*zb).sigmoid()
image_grid(interpolated,title=f'Interpolation: labels {labels[0]} to {labels[1]}')
''','μ scatter에 분산 σ²는 나타나 있나요? 보간이 부드럽다는 사실만으로 각 축이 독립적인 의미라고 말할 수 있나요?')
        section(cells,'§7 · 최종 평가와 로그 저장','최종 표에는 reconstruction, KL, 둘의 합을 함께 씁니다.',r'''
test_metrics=eval_vae(vae,test_loader)
print('test subset 1-sample MC metrics:',test_metrics)
(OUT/'vae_metrics.json').write_text(json.dumps({'history':history,'test':test_metrics},indent=2))
''')
    else:
        section(cells,'§4 · β 3조건 비교','같은 초기화·분할·shuffle 예산으로 각 모델을 새로 학습합니다. 조건별 선택은 R+βK, 비교 보고는 공통 R+K입니다.',r'''
betas=[.25,1.,4.]; models={}; histories={}; rows=[]
for beta in betas:
    m,h,seconds=fit_vae(beta=beta,zdim=8)
    models[beta]=m; histories[beta]=h; metrics=eval_vae(m,val_loader)
    rows.append(dict(beta=beta,seconds=round(seconds,2),**metrics))
display(rows)
fig,axes=plt.subplots(1,3,figsize=(10,3))
for ax,key in zip(axes,['recon','kl','nelbo']):
    ax.bar([str(r['beta']) for r in rows],[r[key] for r in rows]); ax.set(xlabel='training beta',title='common val '+key)
plt.tight_layout(); plt.show()
(OUT/'beta_comparison.json').write_text(json.dumps(rows,indent=2))
''','objective는 eval β=1이므로 nelbo와 같습니다. 훈련 β를 곱한 total들을 직접 비교하면 왜 잘못된 결론이 나오나요?')
        section(cells,'§4-시각화 · 같은 z와 같은 입력','같은 z의 의미가 서로 다른 모델에서 정렬된다는 보장은 없습니다. 난수 차이를 줄이는 비교입니다.',r'''
seed_all(123); fixed_z=torch.randn(8,8,device=DEVICE)
x,y=next(iter(val_loader)); x=x[:8].to(DEVICE)
samples=[]; reconstructions=[]
with torch.no_grad():
    for beta in betas:
        m=models[beta]; m.eval()
        samples.append(m.decode(fixed_z).sigmoid())
        reconstructions.append(m(x,sample=False)[0].sigmoid())
image_grid(torch.cat(samples),title='Prior samples: rows beta=.25 / 1 / 4')
image_grid(torch.cat([x]+reconstructions),title='Original then reconstructions: beta=.25 / 1 / 4')
''')
        section(cells,'§5 · 차원별 KL과 traversal','collapse 진단은 작은 KL 하나만으로 끝내지 않습니다.',r'''
kl_rows=[]
with torch.no_grad():
    for beta in betas:
        parts=[]
        for x,y in val_loader:
            mu,lv=models[beta].encode(x.to(DEVICE)); parts.append(gaussian_kl(mu,lv).cpu())
        kl_rows.append(torch.cat(parts).mean(0))
for beta,k in zip(betas,kl_rows): plt.plot(range(8),k,marker='o',label=f'beta={beta}')
plt.xlabel('latent dimension'); plt.ylabel('mean KL per dimension'); plt.legend(); plt.show()
base_z=torch.zeros(8,8,device=DEVICE); base_z[:,0]=torch.linspace(-3,3,8,device=DEVICE)
with torch.no_grad(): traversal=models[1.].decode(base_z).sigmoid()
image_grid(traversal,title='Traversal of z[0], all other coordinates zero')
''','변화가 거의 없는 차원이 있나요? latent를 바꾸어도 그림이 같다면 어떤 추가 실험을 하겠나요?')
        section(cells,'§6 · 조건부 VAE 학습','one-hot label을 encoder와 decoder에 모두 넣습니다. 조건부 prior는 N(0,I)로 고정합니다.',r'''
CVAE_EPOCHS = 1 if SMOKE else max(20,EPOCHS)
# 조건 y의 역할을 작은 예산에서도 관찰하기 쉽게 CVAE latent는 2차원으로 제한합니다.
# 앞의 무조건부 beta 실험(8차원)과는 별도 과제이며 성능을 직접 순위 비교하지 않습니다.
cvae,c_history,seconds=fit_vae(beta=1.,zdim=2,conditional=True,epochs=CVAE_EPOCHS)
plot_vae(c_history)
print('CVAE val metrics:',eval_vae(cvae,val_loader))
''')
        section(cells,'§7 · 10×8 조건부 생성','행마다 라벨을 고정하고 열마다 같은 z를 공유합니다.',r'''
seed_all(123); z=torch.randn(8,cvae.zdim,device=DEVICE).repeat(10,1)
y=torch.arange(10,device=DEVICE).repeat_interleave(8)
with torch.no_grad(): probs=cvae.decode(z,y).sigmoid()
image_grid(probs,[f'y={i}' for i in y.tolist()],title='CVAE: rows=labels, columns=shared z')
assert probs.shape==(80,1,28,28) and torch.isfinite(probs).all()
with torch.no_grad():
    a=cvae.decode(z[:8],torch.zeros(8,dtype=torch.long,device=DEVICE))
    b=cvae.decode(z[:8],torch.ones(8,dtype=torch.long,device=DEVICE))
print('same z / different label mean logit difference:',(a-b).abs().mean().item())
chosen=min(rows,key=lambda r:r['nelbo'])['beta']
print('VAE beta chosen by common val NELBO:',chosen,'test:',eval_vae(models[chosen],test_loader))
print('CVAE test (separate conditional task):',eval_vae(cvae,test_loader))
torch.save({'state_dict':cpu_state(cvae),'zdim':cvae.zdim,'conditional':True},OUT/'cvae.pt')
''','조건이 맞는 행과 실패하는 행을 각각 고르세요. logits가 달라진다는 수치 검사와 올바른 숫자 생성은 같은 검증인가요?')
    cells += [md('''## 실험 기록표 · 직접 작성

| 항목 | 기록 |
|---|---|
| 가설 / 바꿀 변수 | |
| 고정한 분할·seed·학습 예산 | |
| 실제 관측 지표와 그림 | |
| 예상과 다른 점 | |
| 가능한 이유 2가지 | |
| 한계와 다음 실험 | |

## 출구 확인

PPT 출구 퀴즈에 답한 뒤, 오늘 모델의 **입력 → 중간 표현 → 출력 → loss → 평가**를 다섯 줄로 적으세요.
자율 과제는 `homework_optional.ipynb`, 리더 참고 구현은 `leader_solution.ipynb`입니다.
'''),md('## 출처 및 추가 읽기\n\n'+ '\n'.join(f'- [{n}]({u})' for n,u in WEEKS[wi-1]['refs'])+'\n\n모든 설명용 코드와 도식은 수업용으로 새로 작성했습니다. 외부 원문 코드를 그대로 복제하지 않았습니다.')]
    save(cells,ROOT/WEEKS[wi-1]['folder']/'practice.ipynb')

if __name__=='__main__':
    for wi in range(1,7): practice(wi)
    print('6 practice notebooks written')
