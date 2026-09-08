from pathlib import Path
from build_notebooks import base,foundations,md,code,save
from curriculum import WEEKS
ROOT=Path(__file__).resolve().parents[1]

# title, task statement, starter, reference implementation, contract tests, explanation
TASKS={
1:[
('A · 출력 크기 계산 / 10분',
 '일반 Conv2d의 한 축 출력 크기를 반환하세요. dilation과 내림을 포함하고, 크기는 양수라고 가정합니다.',
 'def conv_out(h,k,p=0,s=1,d=1):\n    # TODO: 정수 출력 크기 반환\n    return None',
 'def conv_out(h,k,p=0,s=1,d=1):\n    return (h+2*p-d*(k-1)-1)//s+1',
 "if conv_out(28,3,1) is None:\n    print('미완성: A를 구현한 뒤 다시 검사하세요.')\nelse:\n    assert conv_out(28,3,1)==28\n    assert conv_out(32,3,0,2)==15\n    assert conv_out(9,3,0,1,2)==5\n    print('PASS A')",
 '유효 커널은 d(k−1)+1입니다. 정수 나눗셈으로 내림합니다. padding=1이 항상 같은 크기를 만드는 것은 아니며 k와 stride도 확인해야 합니다.'),
('B · Conv 파라미터 수 / 10분',
 '입출력 채널과 커널 크기로 파라미터 수를 계산하세요. groups와 bias=False를 지원하세요. 채널 수는 groups로 나누어 떨어진다고 가정합니다.',
 'def conv_params(cin,cout,k,bias=True,groups=1):\n    # TODO\n    return None',
 'def conv_params(cin,cout,k,bias=True,groups=1):\n    return cout*(cin//groups*k*k+int(bias))',
 "if conv_params(3,8,3) is None:\n    print('미완성: B를 구현하세요.')\nelse:\n    assert conv_params(3,8,3)==224\n    assert conv_params(4,8,3,False,2)==144\n    for args in [(1,16,3,True,1),(4,8,3,False,2)]:\n        ci,co,k,b,g=args\n        assert conv_params(*args)==num_params(nn.Conv2d(ci,co,k,bias=b,groups=g))\n    print('PASS B')",
 'group마다 입력 채널을 나누어 사용하므로 cin/groups가 됩니다. 출력 채널 전체에 대해 계산하고 bias는 cout개입니다.'),
('C · 채널 폭 ablation / 30–45분',
 'SmallCNN width=8과 width=16을 같은 초기 seed·분할·epoch로 학습하세요. 결과를 `width, params, val_loss, val_acc` 열의 리스트로 반환하세요. test는 사용하지 마세요.',
 'def width_experiment():\n    # TODO: fit_classifier와 classification_epoch 사용\n    return None',
 '''def width_experiment():
    rows=[]
    for width in [8,16]:
        train_loader.generator.manual_seed(SEED)
        m,h,seconds=fit_classifier(lambda:SmallCNN(width=width),train_loader)
        r=classification_epoch(m,val_loader)
        rows.append(dict(width=width,params=num_params(m),val_loss=r['loss'],val_acc=r['accuracy']))
    return rows''',
 "width_rows=width_experiment()\nif width_rows is None:\n    print('미완성: C 실험을 작성하세요.')\nelse:\n    assert len(width_rows)==2 and {r['width'] for r in width_rows}=={8,16}\n    assert all(np.isfinite(r['val_loss']) and 0<=r['val_acc']<=1 for r in width_rows)\n    assert width_rows[0]['params']<width_rows[1]['params']\n    display(width_rows)",
 '폭이 커지면 파라미터와 계산량이 늘지만 성능 상승은 보장되지 않습니다. 여러 seed 실험 없이 작은 차이에 대해 우월성을 확정하지 않습니다.'),
('D · 이동 강건성 / 20–30분',
 '학습된 SmallCNN의 입력을 오른쪽으로 1픽셀 이동시키고 왼쪽을 0으로 채우세요. torch.roll의 wrap-around는 사용하지 마세요. 원본 대비 예측 변경 비율을 반환하세요.',
 'def shift_rate():\n    # TODO: 새 CNN을 학습하고 validation에서 비교\n    return None',
 '''def shift_rate():
    train_loader.generator.manual_seed(SEED)
    m,_,_=fit_classifier(SmallCNN,train_loader); m.eval(); changed=count=0
    with torch.no_grad():
        for x,y in val_loader:
            x=x.to(DEVICE); shifted=torch.zeros_like(x); shifted[:,:,:,1:]=x[:,:,:,:-1]
            changed+=(m(x).argmax(1)!=m(shifted).argmax(1)).sum().item(); count+=len(x)
    return changed/count''',
 "rate=shift_rate()\nif rate is None: print('미완성: D는 도전 과제입니다.')\nelse:\n    assert 0<=rate<=1\n    print('prediction change rate:',rate)",
 '변경률이 0이 아니어도 CNN 구현이 잘못된 것은 아닙니다. padding, stride, pooling, 경계에서 정보가 달라집니다. 변경률은 정확도 하락과도 다른 지표입니다.')],
2:[
('A · projection shortcut / 15분',
 'cin=32, cout=64, stride=2인 입력 경로를 만들 수 있는 shortcut 함수를 작성하세요. shape가 같으면 Identity를 반환하세요.',
 'def make_shortcut(cin,cout,stride=1):\n    # TODO: nn.Module 반환\n    return None',
 '''def make_shortcut(cin,cout,stride=1):
    if cin==cout and stride==1: return nn.Identity()
    return nn.Sequential(nn.Conv2d(cin,cout,1,stride,bias=False),nn.BatchNorm2d(cout))''',
 "shortcut=make_shortcut(32,64,2)\nif shortcut is None: print('미완성: A를 구현하세요.')\nelse:\n    assert shortcut(torch.randn(2,32,16,16)).shape==(2,64,8,8)\n    assert isinstance(make_shortcut(32,32),nn.Identity)\n    print('PASS A')",
 'projection은 채널과 공간 stride를 함께 맞춥니다. 1×1 Conv에도 stride=2를 적용해야 합니다. Identity에는 학습 파라미터가 없습니다.'),
('B · 클래스별 recall / 15분',
 '행=true, 열=pred인 confusion matrix에서 recall과 macro recall을 반환하세요. 실제 표본이 없는 클래스는 NaN으로 두고 macro 평균에서 제외하세요.',
 'def class_recall(cm):\n    # TODO: (recall tensor, macro float) 반환\n    return None',
 '''def class_recall(cm):
    support=cm.sum(1); recall=cm.diag().float()/support.clamp_min(1)
    recall[support==0]=float('nan')
    return recall,torch.nanmean(recall).item()''',
 "r=class_recall(torch.tensor([[3,1,0],[2,2,0],[0,0,0]]))\nif r is None: print('미완성: B를 구현하세요.')\nelse:\n    recalls,macro=r\n    torch.testing.assert_close(recalls[:2],torch.tensor([.75,.5]))\n    assert torch.isnan(recalls[2]) and abs(macro-.625)<1e-6\n    print('PASS B:',r)",
 '정답 클래스의 전체 개수는 행 합입니다. 빈 클래스를 0으로 넣으면 macro recall을 부당하게 낮춥니다. precision은 열 합을 분모로 쓰는 다른 지표입니다.'),
('C · residual on/off / 35–50분',
 '증강을 켠 동일한 loader로 CifarCNN(residual=False/True)를 새로 학습하세요. 조건·파라미터 수·시간·val loss·val accuracy를 기록하세요.',
 'def residual_experiment():\n    # TODO\n    return None',
 '''def residual_experiment():
    rows=[]
    for residual in [False,True]:
        loader,_,_=vision_loaders('cifar',augment=True)
        m,h,seconds=fit_classifier(lambda:CifarCNN(residual=residual),loader)
        r=classification_epoch(m,val_loader)
        rows.append(dict(residual=residual,params=num_params(m),seconds=seconds,val_loss=r['loss'],val_acc=r['accuracy']))
    return rows''',
 "rows=residual_experiment()\nif rows is None: print('미완성: C를 작성하세요.')\nelse:\n    assert {r['residual'] for r in rows}=={False,True}\n    assert all(np.isfinite(r['val_loss']) and r['seconds']>0 for r in rows)\n    display(rows)",
 '이 비교는 skip 경로뿐 아니라 블록의 Conv 수와 파라미터 수도 바뀝니다. 따라서 residual 연결만의 인과 효과를 분리한 실험은 아닙니다. 순수 연결 효과를 보려면 동일 body에서 덧셈만 on/off하는 후속 실험을 설계하세요.'),
('D · 순수 skip ablation 설계 / 30–45분',
 '입출력 shape가 같은 두 Conv body를 두고 skip=True/False만 바뀌는 블록을 작성하세요. 같은 state_dict에서 F(x)와 F(x)+x를 비교할 수 있어야 합니다.',
 'def make_ablation_block(skip=True):\n    # TODO: 32→32, stride=1 블록 반환\n    return None',
 '''def make_ablation_block(skip=True):
    class AblationBlock(nn.Module):
        def __init__(self):
            super().__init__(); self.skip=skip
            self.body=nn.Sequential(nn.Conv2d(32,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,32,3,padding=1))
        def forward(self,x): return self.body(x)+(x if self.skip else 0)
    return AblationBlock()''',
 "a=make_ablation_block(True)\nif a is None: print('미완성: D는 도전 과제입니다.')\nelse:\n    b=make_ablation_block(False); b.load_state_dict(a.state_dict())\n    x=torch.randn(2,32,8,8)\n    torch.testing.assert_close(a(x)-b(x),x)\n    assert num_params(a)==num_params(b)\n    print('PASS D: same body and parameter count')",
 '이 블록은 비교를 선명하게 하기 위해 덧셈 뒤 ReLU를 생략합니다. 파라미터 수와 body가 같으므로 skip만 바뀝니다. 실제 학습 비교를 확장할 때 분할·초기화·예산도 고정합니다.')],
3:[
('A · scaled dot-product / 15분',
 'Q,K,V의 마지막 두 축에서 attention을 계산하고 output과 weights를 반환하세요. causal=True이면 미래 key를 차단하세요.',
 'def my_attention(q,k,v,causal=True):\n    # TODO\n    return None',
 '''def my_attention(q,k,v,causal=True):
    score=q@k.transpose(-2,-1)/math.sqrt(q.size(-1))
    if causal:
        mask=torch.ones(q.size(-2),k.size(-2),dtype=torch.bool,device=q.device).triu(1)
        score=score.masked_fill(mask,float('-inf'))
    a=score.softmax(-1)
    return a@v,a''',
 "q,k,v=[torch.randn(2,4,8) for _ in range(3)]\nr=my_attention(q,k,v)\nif r is None: print('미완성: A를 구현하세요.')\nelse:\n    out,a=r\n    torch.testing.assert_close(a.sum(-1),torch.ones(2,4))\n    assert a.triu(1).abs().max()==0\n    reference=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.)\n    torch.testing.assert_close(out,reference,atol=1e-5,rtol=1e-5)\n    print('PASS A: reference SDPA match')",
 'softmax는 마지막 key 축입니다. 직접 mask는 True=차단이지만 SDPA bool mask는 True=허용입니다. 여기서는 is_causal=True로 API 차이를 피합니다.'),
('B · next-token window / 10분',
 '1차원 tokens, 시작 i, 길이 t에서 x와 y를 반환하세요. 범위 밖 요청은 AssertionError로 거절하세요.',
 'def next_window(data,i,t):\n    # TODO\n    return None',
 '''def next_window(data,i,t):
    assert i>=0 and t>0 and i+t<len(data)
    return data[i:i+t],data[i+1:i+t+1]''',
 "r=next_window(torch.arange(10),2,4)\nif r is None: print('미완성: B를 구현하세요.')\nelse:\n    x,y=r\n    assert x.tolist()==[2,3,4,5] and y.tolist()==[3,4,5,6]\n    try: next_window(torch.arange(10),8,3)\n    except AssertionError: print('PASS B')\n    else: raise AssertionError('out-of-range request must fail')",
 '정답은 입력의 오른쪽 한 칸입니다. 마지막 입력 토큰의 다음 토큰까지 필요하므로 i+t<len(data)입니다. split 경계를 넘기는 window를 만들지 않습니다.'),
('C · causal 누출 탐지 / 15분',
 '주어진 모델과 ids에서 cut 이후 토큰을 바꾸고 cut 이전 logits 차이의 최댓값을 반환하세요. eval, no_grad를 사용하세요.',
 'def future_difference(model,ids,cut):\n    # TODO\n    return None',
 '''def future_difference(model,ids,cut):
    model.eval(); changed=ids.clone(); changed[:,cut:]=(changed[:,cut:]+1)%VOCAB
    with torch.no_grad(): return (model(ids)[:,:cut]-model(changed)[:,:cut]).abs().max().item()''',
 "m=SingleHeadLM().to(DEVICE); x,_=lm_batch(batch=2)\nr=future_difference(m,x,CONTEXT//2)\nif r is None: print('미완성: C를 구현하세요.')\nelse:\n    assert r<1e-5\n    class Leaky(nn.Module):\n        def forward(self,x): return F.one_hot(x,VOCAB).float().mean(1,keepdim=True).expand(-1,x.size(1),-1)\n    leak=future_difference(Leaky().to(DEVICE),torch.zeros_like(x),CONTEXT//2)\n    assert leak>0\n    print('PASS C:',r,'leaky model difference:',leak)",
 '좋은 테스트는 정상 모델뿐 아니라 의도적으로 잘못된 모델도 검출해야 합니다. Leaky는 전체 문맥의 평균을 모든 위치에 사용하여 미래 정보를 누출합니다.'),
('D · 문맥 길이 실험 / 30–45분',
 '같은 SingleHeadLM으로 context=8,32를 비교하세요. 처리 토큰 예산을 맞추기 위해 steps를 반비례 조절하세요. 비교 후 전역 CONTEXT를 원래 값으로 복원하세요. 결과에 context,steps,tokens,val_ce를 기록하세요.',
 'def context_experiment():\n    # TODO: 전역 CONTEXT와 lm_batch 기본 인자 주의\n    return None',
 '''def context_experiment():
    global CONTEXT,lm_batch
    old_context,old_batch=CONTEXT,lm_batch; rows=[]
    token_budget=STEPS*BATCH*32
    try:
        for context in [8,32]:
            CONTEXT=context
            # 원래 함수의 기본 context는 정의 시 고정되므로 명시적으로 전달
            lm_batch=lambda split='train',batch=BATCH,context=context: old_batch(split,batch,context)
            steps=max(1,token_budget//(BATCH*context))
            m,h,seconds=fit_lm(SingleHeadLM,steps=steps)
            rows.append(dict(context=context,steps=steps,tokens=steps*BATCH*context,val_ce=lm_eval(m),seconds=seconds))
    finally: CONTEXT=old_context; lm_batch=old_batch
    return rows''',
 "rows=context_experiment()\nif rows is None: print('미완성: D는 도전 과제입니다.')\nelse:\n    assert rows[0]['tokens']==rows[1]['tokens']\n    assert all(np.isfinite(r['val_ce']) for r in rows)\n    display(rows)",
 '처리 토큰 예산을 같게 해도 고유 데이터 양, 문맥 조건, 위치 embedding 수와 시간은 다릅니다. 평가 문맥 길이도 바뀌므로 순수 구조 효과가 아닌 문맥 설정의 비교로 보고합니다.')],
4:[
('A · top-k 필터 / 15분',
 '2차원 logits에서 각 행의 상위 k개 인덱스만 유지하고 나머지는 -inf로 바꾸세요. k는 vocabulary 범위로 제한하세요.',
 'def top_k_filter(logits,k):\n    # TODO\n    return None',
 '''def top_k_filter(logits,k):
    k=max(1,min(int(k),logits.size(-1))); values,ids=logits.topk(k,dim=-1)
    return torch.full_like(logits,float('-inf')).scatter(-1,ids,values)''',
 "x=torch.tensor([[1.,3.,2.,0.],[2.,2.,1.,0.]])\nr=top_k_filter(x,2)\nif r is None: print('미완성: A를 구현하세요.')\nelse:\n    assert torch.isfinite(r).sum(-1).tolist()==[2,2]\n    assert torch.isfinite(top_k_filter(x,99)).all()\n    assert top_k_filter(x,1).softmax(-1).max(-1).values.eq(1).all()\n    print('PASS A')",
 'threshold 방식은 동점일 때 k보다 많은 후보를 남길 수 있습니다. topk 인덱스를 scatter하면 정확히 k개를 유지합니다. -inf로 지운 후 softmax가 다시 정규화합니다.'),
('B · pre-norm block / 20분',
 'LayerNorm 두 개, CausalMHA, D→4D→D FFN, residual 두 개로 block을 작성하세요. 함수는 nn.Module을 반환합니다.',
 'def make_block(d=32,heads=4):\n    # TODO\n    return None',
 '''def make_block(d=32,heads=4):
    class Block(nn.Module):
        def __init__(self):
            super().__init__(); self.n1=nn.LayerNorm(d); self.n2=nn.LayerNorm(d)
            self.a=CausalMHA(d,heads,dropout=0.)
            self.f=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
        def forward(self,x):
            x=x+self.a(self.n1(x)); return x+self.f(self.n2(x))
    return Block()''',
 "block=make_block()\nif block is None: print('미완성: B를 구현하세요.')\nelse:\n    x=torch.randn(2,8,32,requires_grad=True); y=block(x)\n    assert y.shape==x.shape\n    y.square().mean().backward(); assert torch.isfinite(x.grad).all()\n    # attentionとFFNをゼロにするとresidualだけが残る\n    for p in block.parameters(): p.data.zero_()\n    torch.testing.assert_close(block(x),x)\n    print('PASS B')".replace('# attentionとFFNをゼロにするとresidualだけが残る','# 하위 연산 가중치를 0으로 만들면 residual 입력만 남음'),
 'pre-norm에서는 정규화 결과를 원래 x에 덮어쓰지 않습니다. residual 바깥 경로의 x를 보존합니다. 모든 파라미터가 0이면 두 하위 연산 출력도 0이므로 항등 함수가 됩니다.'),
('C · head 수 비교 / 35–50분',
 'D=64, layers=2를 고정하고 heads=2,4를 새로 학습하세요. steps와 seed를 고정하고 params, val_ce, time을 보고하세요.',
 'def head_experiment():\n    # TODO\n    return None',
 '''def head_experiment():
    rows=[]
    for heads in [2,4]:
        m,h,seconds=fit_lm(lambda:TinyTransformer(d=64,heads=heads,layers=2))
        rows.append(dict(heads=heads,params=num_params(m),val_ce=lm_eval(m),seconds=seconds))
    return rows''',
 "rows=head_experiment()\nif rows is None: print('미완성: C를 작성하세요.')\nelse:\n    assert rows[0]['params']==rows[1]['params']\n    assert all(np.isfinite(r['val_ce']) for r in rows)\n    display(rows)",
 'D가 고정되어 QKV와 출력 투영의 파라미터 수는 같습니다. head 차원은 32와 16으로 바뀝니다. 한 번의 결과 차이는 seed 변동일 수 있습니다.'),
('D · top-p 구현 / 20–30분',
 '정렬한 확률 누적합이 p를 처음 넘기는 토큰까지 유지하세요. 각 행에 최소 하나를 남기고 결과를 원래 vocabulary 순서로 돌리세요.',
 'def top_p_filter(logits,p=.9):\n    # TODO: 0<p<=1\n    return None',
 '''def top_p_filter(logits,p=.9):
    assert 0<p<=1
    values,ids=logits.sort(dim=-1,descending=True)
    cumulative=values.softmax(-1).cumsum(-1)
    remove=cumulative>p
    remove[...,1:]=remove[...,:-1].clone(); remove[...,0]=False
    values=values.masked_fill(remove,float('-inf'))
    return torch.full_like(logits,float('-inf')).scatter(-1,ids,values)''',
 "logits=torch.log(torch.tensor([[.6,.3,.1]])); r=top_p_filter(logits,.8)\nif r is None: print('미완성: D는 도전 과제입니다.')\nelse:\n    assert torch.isfinite(r).tolist()==[[True,True,False]]\n    assert torch.isfinite(top_p_filter(logits,.1)).sum()==1\n    assert torch.isfinite(top_p_filter(logits,1.)).all()\n    print('PASS D')",
 '누적합이 p를 넘긴 첫 토큰을 제외하면 후보 질량이 p보다 작아집니다. remove mask를 한 칸 이동해 그 토큰까지 포함합니다. 매우 뾰족한 분포에서도 최소 하나를 남깁니다.')],
5:[
('A · 재매개화 / 15분',
 'μ, logvar, 선택적 eps를 받아 z를 반환하세요. eps가 없으면 randn_like를 사용하고, 있을 때는 결정적인 값으로 검사 가능하게 하세요.',
 'def my_reparam(mu,logvar,eps=None):\n    # TODO\n    return None',
 '''def my_reparam(mu,logvar,eps=None):
    if eps is None: eps=torch.randn_like(mu)
    return mu+torch.exp(.5*logvar)*eps''',
 "mu=torch.tensor([[2.,-1.]],requires_grad=True); lv=torch.full_like(mu,math.log(4.),requires_grad=True)\nr=my_reparam(mu,lv,torch.ones_like(mu))\nif r is None: print('미완성: A를 구현하세요.')\nelse:\n    torch.testing.assert_close(r,torch.tensor([[4.,1.]]))\n    r.sum().backward(); torch.testing.assert_close(mu.grad,torch.ones_like(mu))\n    assert lv.grad is not None\n    print('PASS A')",
 'variance=4이면 std=2입니다. eps=1일 때 μ에 2를 더합니다. detach나 numpy 변환을 넣으면 gradient 경로가 끊깁니다.'),
('B · 배치별 negative ELBO / 20분',
 'logits,x,mu,logvar를 받아 loss scalar, recon [B], KL [B]를 반환하세요. β를 지원하고 픽셀·잠재 축은 합, 배치는 평균을 쓰세요.',
 'def my_vae_loss(logits,x,mu,logvar,beta=1.):\n    # TODO\n    return None',
 '''def my_vae_loss(logits,x,mu,logvar,beta=1.):
    r=F.binary_cross_entropy_with_logits(logits,x,reduction='none').flatten(1).sum(1)
    k=.5*(mu.square()+logvar.exp()-1-logvar).sum(1)
    return (r+beta*k).mean(),r,k''',
 "logits=torch.zeros(2,1,2,2); x=torch.ones_like(logits); mu=torch.tensor([[1.,0.],[0.,0.]]); lv=torch.zeros_like(mu)\nr=my_vae_loss(logits,x,mu,lv)\nif r is None: print('미완성: B를 구현하세요.')\nelse:\n    loss,recon,kl=r\n    torch.testing.assert_close(recon,torch.full((2,),4*math.log(2)))\n    torch.testing.assert_close(kl,torch.tensor([.5,0.]))\n    assert abs(loss.item()-(4*math.log(2)+.25))<1e-5\n    doubled=my_vae_loss(logits.repeat(2,1,1,1),x.repeat(2,1,1,1),mu.repeat(2,1),lv.repeat(2,1))[0]\n    torch.testing.assert_close(loss,doubled)\n    print('PASS B: reduction invariant to duplicated batch')",
 '배치를 복제해도 평균 loss는 같아야 합니다. 모든 픽셀의 logits=0이면 확률=.5라 BCE는 각 픽셀 ln2입니다. 이 검사는 부호와 reduction 오류를 함께 찾습니다.'),
('C · latent 차원 비교 / 35–45분',
 'zdim=2와 8을 β=1, 같은 예산으로 학습하고 recon·KL·nelbo를 보고하세요. 각 모델에서 prior 샘플을 그리세요.',
 'def latent_experiment():\n    # TODO\n    return None',
 '''def latent_experiment():
    rows=[]
    for zdim in [2,8]:
        m,h,seconds=fit_vae(zdim=zdim)
        rows.append(dict(zdim=zdim,params=num_params(m),**eval_vae(m,val_loader)))
        recon_and_samples(m)
    return rows''',
 "rows=latent_experiment()\nif rows is None: print('미완성: C를 작성하세요.')\nelse:\n    assert {r['zdim'] for r in rows}=={2,8}\n    assert all(abs(r['nelbo']-r['recon']-r['kl'])<1e-4 for r in rows)\n    display(rows)",
 '잠재 차원이 커져도 모두 사용한다고 보장되지 않습니다. 파라미터 수와 KL 합의 규모도 바뀝니다. 재구성과 prior sample의 관찰을 별도로 작성합니다.'),
('D · deterministic AE 비교 / 30–45분',
 'VAE(zdim=2)를 μ만 사용하는 AE처럼 학습하세요. loss는 reconstruction만 사용합니다. AE식 재구성과 임의 prior 샘플을 시각화하세요. val reconstruction 수치를 반환하세요.',
 'def ae_experiment():\n    # TODO\n    return None',
 '''def ae_experiment():
    seed_all(); ae=VAE(zdim=2).to(DEVICE); opt=torch.optim.Adam(ae.parameters(),lr=1e-3)
    train_loader.generator.manual_seed(SEED)
    for _ in range(EPOCHS):
        ae.train()
        for x,y in train_loader:
            x=x.to(DEVICE); opt.zero_grad(set_to_none=True)
            logits,mu,lv=ae(x,sample=False)
            loss=F.binary_cross_entropy_with_logits(logits,x,reduction='none').flatten(1).sum(1).mean()
            loss.backward(); opt.step()
    ae.eval(); total=count=0
    with torch.no_grad():
        for x,y in val_loader:
            x=x.to(DEVICE); logits=ae(x,sample=False)[0]
            total+=F.binary_cross_entropy_with_logits(logits,x,reduction='sum').item(); count+=len(x)
    recon_and_samples(ae)
    return total/count''',
 "r=ae_experiment()\nif r is None: print('미완성: D는 도전 과제입니다.')\nelse:\n    assert np.isfinite(r) and r>=0\n    print('deterministic AE val reconstruction:',r)",
 '이는 VAE 클래스의 μ 경로를 재사용한 AE 실험입니다. logvar head는 loss에 연결되지 않아 학습되지 않습니다. 이 모델의 Gaussian prior 샘플은 학습 목적이 보장하는 생성 경로가 아닙니다.')],
6:[
('A · KL warm-up 스케줄 / 10분',
 'step=0에서 0, warmup_steps에서 target_beta에 도달하는 선형 스케줄을 작성하세요. warmup_steps=0이면 즉시 목표 β를 반환하세요.',
 'def beta_schedule(step,warmup_steps,target_beta=1.):\n    # TODO\n    return None',
 '''def beta_schedule(step,warmup_steps,target_beta=1.):
    assert step>=0 and warmup_steps>=0
    return target_beta if warmup_steps==0 else target_beta*min(1.,step/warmup_steps)''',
 "r=beta_schedule(0,100,4.)\nif r is None: print('미완성: A를 구현하세요.')\nelse:\n    assert r==0 and beta_schedule(50,100,4.)==2\n    assert beta_schedule(200,100,4.)==4 and beta_schedule(0,0,4.)==4\n    print('PASS A')",
 'step과 epoch 단위를 혼동하지 않습니다. warm-up 중 total loss는 목적함수 자체가 변하므로 recon, KL, effective beta를 별도로 기록해야 합니다.'),
('B · 조건부 생성 격자 / 15분',
 'CVAE 모델과 n_cols를 받아 [10*n_cols,1,28,28] 확률 이미지를 반환하세요. 행은 라벨, 열은 동일한 z입니다.',
 'def conditional_grid(model,n_cols=8):\n    # TODO: model.zdim, model.decode 사용\n    return None',
 '''@torch.no_grad()
def conditional_grid(model,n_cols=8):
    model.eval(); dev=next(model.parameters()).device
    z=torch.randn(n_cols,model.zdim,device=dev).repeat(10,1)
    y=torch.arange(10,device=dev).repeat_interleave(n_cols)
    return model.decode(z,y).sigmoid()''',
 "m=VAE(zdim=8,conditional=True).to(DEVICE); r=conditional_grid(m,3)\nif r is None: print('미완성: B를 구현하세요.')\nelse:\n    assert r.shape==(30,1,28,28) and r.min()>=0 and r.max()<=1\n    # 未学習モデルなので数字品質は検査しない\n    print('PASS B: shape/range, quality requires training')".replace('# 未学習モデルなので数字品質は検査しない','# 미학습 모델이므로 숫자 생성 품질은 검사하지 않음'),
 'repeat와 repeat_interleave를 구분합니다. z는 전체 열 샘플 묶음을 10회 반복하고 y는 각 라벨을 열 개수만큼 반복합니다. 미학습 모델 shape 검사는 조건 충실도 평가를 대신하지 못합니다.'),
('C · β 고정 대 warm-up / 35–50분',
 'β_target=4, zdim=8에서 constant와 warm-up을 비교하세요. fit_vae의 기본 warm-up은 epoch 단위입니다. 더 정밀하게 하려면 A의 step 함수를 학습 루프에 넣으세요.',
 'def warmup_experiment():\n    # TODO\n    return None',
 '''def warmup_experiment():
    rows=[]
    for warmup in [False,True]:
        m,h,seconds=fit_vae(beta=4.,zdim=8,warmup=warmup,epochs=max(4,EPOCHS) if not SMOKE else 2)
        rows.append(dict(warmup=warmup,**eval_vae(m,val_loader)))
        plot_vae(h)
    return rows''',
 "rows=warmup_experiment()\nif rows is None: print('미완성: C를 작성하세요.')\nelse:\n    assert len(rows)==2 and all(np.isfinite(r['nelbo']) for r in rows)\n    display(rows)",
 '짧은 학습에서는 warm-up의 차이가 작거나 없을 수 있습니다. 여기서는 최소 4 epoch를 사용해 초기에 β가 달라지는 구간을 확보합니다. SMOKE는 실행 점검만 하므로 현상 비교용이 아닙니다.'),
('D · 차원별 진단 / 20–30분',
 'μ, logvar [N,Z]로부터 평균 차원별 KL과 Var_x(μ_j)를 반환하세요. 비편향 보정 없이 population variance를 사용하세요.',
 'def latent_diagnostics(mu,logvar):\n    # TODO: dict with kl_per_dim, mu_variance\n    return None',
 '''def latent_diagnostics(mu,logvar):
    return dict(kl_per_dim=gaussian_kl(mu,logvar).mean(0),mu_variance=mu.var(0,unbiased=False))''',
 "mu=torch.tensor([[0.,1.],[0.,-1.]]); r=latent_diagnostics(mu,torch.zeros_like(mu))\nif r is None: print('미완성: D는 도전 과제입니다.')\nelse:\n    torch.testing.assert_close(r['kl_per_dim'],torch.tensor([0.,.5]))\n    torch.testing.assert_close(r['mu_variance'],torch.tensor([0.,1.]))\n    print('PASS D:',r)",
 'KL이 작고 μ 분산이 작으면 해당 차원이 정보를 거의 쓰지 않을 가능성이 있습니다. 그러나 decoder 반응과 reconstruction을 함께 확인해야 하며 threshold에 따른 active unit 개수는 진단용입니다.')]
}

def make(wi,solution=False):
    cells=base(wi,'리더용 자율 과제 해설' if solution else '자율 과제')
    cells.insert(1,md('''### 자율 과제 안내

**A·B 기본 / C 비교 실험 / D 도전**입니다. 모두 선택이며 전부 하면 약 80–130분과 학습 시간이 필요합니다.
한 문제만 골라도 괜찮습니다. 이 노트북은 실습 파일 실행 없이 독립적으로 시작할 수 있습니다.

준비 코드는 제공하고, 아래 TODO 함수에 직접 구현합니다. 미완성 상태의 `return None`은 실행을 중단하지 않고 안내만 출력합니다.
코드를 작성하면 바로 다음 검사 셀을 실행하세요. 검사는 최소 계약만 확인하며 좋은 실험 해석을 자동 채점하지 않습니다.

**제출을 선택한다면:** 실행 결과가 남은 ipynb, 비교 표·그림, 아래 해석 5문장을 저장하세요.
test로 과제 조건을 반복 선택하지 말고 validation을 사용하세요.
'''))
    foundations(wi,cells)
    for title,statement,starter,answer,checks,explain in TASKS[wi]:
        cells += [md('## '+title+'\n\n'+statement),code(answer if solution else starter),code(checks)]
        cells += [md(('**참고 해설**\n\n'+explain) if solution else '**기록:** 구현한 식 또는 shape, 검사 결과, 아직 불확실한 점을 적으세요.\n\n_여기에 작성_')]
    cells += [md('''## 비교 실험 해석 · 5문장 필수 틀

1. 나의 질문과 바꾼 변수는 …
2. 고정한 조건(분할, seed, 예산, 평가)은 …
3. 실제 관측 결과는 … (숫자·그림 번호를 근거로)
4. 이 결과만으로 말할 수 없는 것은 …
5. 다음에 한 가지만 더 검증한다면 …

| 조건 | seed | 학습 예산 | params | validation 지표 | 시간 | 관찰 |
|---|---|---|---|---|---|---|
| 기준 | | | | | | |
| 변경 | | | | | | |

### 자기 점검 기준 (100점 환산, 성적 부여 목적 아님)
- 구현의 shape·수식·gradient·mask 계약 30
- 통제 조건과 재현 가능한 코드 25
- 실제 결과 표·그림과 실패 사례 25
- 과장 없는 해석과 한계·후속 실험 20

고정 정확도나 생성 품질 기준은 없습니다. 개선되지 않은 실험도 근거를 잘 남기면 좋은 결과물입니다.
''')]
    if solution: cells += [md('## 리더 진행 팁\n\nA·B의 검사 통과 후 식과 축을 말로 설명하게 합니다. C·D의 성능 숫자는 정답이 아닙니다. 비교 조건이 바뀐 부분을 먼저 찾고, 실제 출력과 관찰을 근거로 해석했는지 확인하세요.')]
    save(cells,ROOT/WEEKS[wi-1]['folder']/('leader_solution.ipynb' if solution else 'homework_optional.ipynb'))

if __name__=='__main__':
    for wi in range(1,7):
        make(wi); make(wi,True)
    print('6 homework + 6 solution notebooks written')
