"""Standalone code cells shared at generation time, never imported by students."""
SETUP = r'''
import os, sys, math, random, copy, time, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import datasets, transforms
from IPython.display import display

SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# QUICK=True: 수업용 일부 데이터. False: 데이터/학습 예산 확대.
QUICK = True
# 검증 제작자용 축소 모드. 수강생은 설정할 필요가 없습니다.
SMOKE = os.environ.get('AIM_SMOKE', '0') == '1'
DATA_ROOT = Path(os.environ.get('AIM_DATA_ROOT', './data'))
OUT = Path('outputs'); OUT.mkdir(exist_ok=True)
torch.set_num_threads(min(4, os.cpu_count() or 1))

def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_all()
print('Python', sys.version.split()[0], '| torch', torch.__version__,
      '| torchvision', torchvision.__version__, '| device', DEVICE)
print('QUICK:', QUICK, '| SMOKE:', SMOKE, '| data:', DATA_ROOT.resolve())

def image_grid(images, titles=None, cols=8, title='', normalized=False):
    images = images.detach().cpu()
    if normalized: images = (images * .5 + .5).clamp(0, 1)
    n = len(images); rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.35, rows*1.55), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis('off')
        if i < n:
            im = images[i]
            ax.imshow(im.squeeze(0) if im.shape[0]==1 else im.permute(1,2,0),
                      cmap='gray', vmin=0, vmax=1)
            if titles is not None: ax.set_title(str(titles[i]), fontsize=8)
    fig.suptitle(title); fig.tight_layout(); plt.show()
    return fig

def num_params(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
'''

VISION_DATA = r'''
# DATASET は下の設定セルで指定します。train/valは公式trainから分割。
def vision_loaders(name, augment=False):
    cls = {'fashion':datasets.FashionMNIST, 'cifar':datasets.CIFAR10, 'mnist':datasets.MNIST}[name]
    base = [transforms.ToTensor()]
    if name=='cifar': base += [transforms.Normalize((.5,)*3, (.5,)*3)]
    eval_tf = transforms.Compose(base)
    train_tf = transforms.Compose(([transforms.RandomCrop(32,padding=4),
                                    transforms.RandomHorizontalFlip()] if augment else []) + base)
    train_ds = cls(DATA_ROOT, train=True, download=True, transform=train_tf)
    val_ds = cls(DATA_ROOT, train=True, download=True, transform=eval_tf)
    test_ds = cls(DATA_ROOT, train=False, download=True, transform=eval_tf)
    ids = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(SEED))
    n_val = 5000 if name=='cifar' else 6000
    train_ids = ids[n_val:][:TRAIN_N].tolist()
    val_ids = ids[:n_val][:VAL_N].tolist()
    test_ids = torch.randperm(len(test_ds), generator=torch.Generator().manual_seed(SEED+1))[:TEST_N].tolist()
    assert not (set(train_ids) & set(val_ids))
    train = DataLoader(Subset(train_ds,train_ids), batch_size=BATCH, shuffle=True,
                       generator=torch.Generator().manual_seed(SEED), num_workers=0)
    val = DataLoader(Subset(val_ds,val_ids), batch_size=BATCH, shuffle=False, num_workers=0)
    test = DataLoader(Subset(test_ds,test_ids), batch_size=BATCH, shuffle=False, num_workers=0)
    return train,val,test

train_loader, val_loader, test_loader = vision_loaders(DATASET)
xb,yb = next(iter(val_loader))
print('split sizes:', *(len(dl.dataset) for dl in [train_loader,val_loader,test_loader]))
print('shape:', xb.shape, 'dtype:', xb.dtype, 'range:', (xb.min().item(),xb.max().item()))
image_grid(xb[:16], yb[:16].tolist(), title='Validation examples', normalized=DATASET=='cifar')
'''.replace('DATASET は下の設定セルで指定します。','DATASET은 앞의 설정 셀에서 지정합니다.')

CLASS_MODELS = r'''
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(),nn.Linear(784,128),nn.ReLU(),nn.Linear(128,10))
    def forward(self,x): return self.net(x)

class SmallCNN(nn.Module):
    def __init__(self,width=16,kernel=3):
        super().__init__()
        assert kernel % 2 == 1
        self.features=nn.Sequential(
            nn.Conv2d(1,width,kernel,padding=kernel//2),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(width,width*2,kernel,padding=kernel//2),nn.ReLU(),nn.MaxPool2d(2))
        self.classifier=nn.Sequential(nn.Flatten(),nn.Linear(width*2*7*7,64),nn.ReLU(),nn.Linear(64,10))
    def forward(self,x): return self.classifier(self.features(x))

class ResidualBlock(nn.Module):
    def __init__(self,cin,cout,stride=1):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(cin,cout,3,stride,1,bias=False),nn.BatchNorm2d(cout),nn.ReLU(),
                                nn.Conv2d(cout,cout,3,1,1,bias=False),nn.BatchNorm2d(cout))
        self.skip=nn.Identity() if cin==cout and stride==1 else nn.Sequential(
            nn.Conv2d(cin,cout,1,stride,bias=False),nn.BatchNorm2d(cout))
    def forward(self,x): return F.relu(self.body(x)+self.skip(x))

class CifarCNN(nn.Module):
    def __init__(self,residual=False):
        super().__init__()
        layers=[nn.Conv2d(3,32,3,padding=1,bias=False),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2)]
        if residual: layers += [ResidualBlock(32,64,stride=2)]
        else: layers += [nn.Conv2d(32,64,3,padding=1,bias=False),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2)]
        self.features=nn.Sequential(*layers,nn.AdaptiveAvgPool2d(1))
        self.head=nn.Linear(64,10)
    def forward(self,x): return self.head(self.features(x).flatten(1))
'''

CLASS_TRAIN = r'''
def classification_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training); total=correct=count=0
    all_y=[]; all_pred=[]
    with torch.set_grad_enabled(training):
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            if training: optimizer.zero_grad(set_to_none=True)
            logits=model(x); loss=F.cross_entropy(logits,y)
            assert torch.isfinite(loss), 'loss is not finite'
            if training: loss.backward(); optimizer.step()
            n=len(y); total+=loss.item()*n; count+=n
            pred=logits.argmax(1); correct+=(pred==y).sum().item()
            all_y.append(y.detach().cpu()); all_pred.append(pred.detach().cpu())
    return dict(loss=total/count,accuracy=correct/count,
                y=torch.cat(all_y),pred=torch.cat(all_pred))

def fit_classifier(factory,loader,epochs=EPOCHS):
    seed_all(); model=factory().to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    history=[]; best_loss=float('inf'); best=None; start=time.perf_counter()
    for ep in range(epochs):
        tr=classification_epoch(model,loader,opt); va=classification_epoch(model,val_loader)
        history.append(dict(epoch=ep+1,train_loss=tr['loss'],val_loss=va['loss'],val_acc=va['accuracy']))
        if va['loss']<best_loss: best_loss=va['loss']; best=cpu_state(model)
        print(f"epoch {ep+1}: train={tr['loss']:.3f} val={va['loss']:.3f} acc={va['accuracy']:.3f}")
    model.load_state_dict(best)
    return model,history,time.perf_counter()-start

def plot_classification(histories):
    fig,axes=plt.subplots(1,2,figsize=(10,3))
    for name,h in histories.items():
        axes[0].plot([r['epoch'] for r in h],[r['train_loss'] for r in h],label=name+' train')
        axes[0].plot([r['epoch'] for r in h],[r['val_loss'] for r in h],'--',label=name+' val')
        axes[1].plot([r['epoch'] for r in h],[r['val_acc'] for r in h],label=name)
    axes[0].set_title('Cross entropy'); axes[1].set_title('Validation accuracy')
    for ax in axes: ax.set_xlabel('epoch'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    plt.tight_layout(); plt.show()

def confusion(y,pred,nclass=10):
    return torch.bincount(y*nclass+pred,minlength=nclass*nclass).reshape(nclass,nclass)

def inspect_classifier(model):
    result=classification_epoch(model,val_loader)
    cm=confusion(result['y'],result['pred'])
    fig,ax=plt.subplots(figsize=(5,4)); im=ax.imshow(cm,cmap='Blues')
    ax.set(xlabel='predicted',ylabel='true',xticks=range(10),yticks=range(10))
    fig.colorbar(im,ax=ax); plt.show()
    recall=cm.diag().float()/cm.sum(1).clamp_min(1)
    print('class counts:',cm.sum(1).tolist()); print('recall:',recall.tolist())
    wrong=(result['y']!=result['pred']).nonzero().flatten()[:8]
    if len(wrong):
        imgs=torch.stack([val_loader.dataset[i.item()][0] for i in wrong])
        labels=[f"true={result['y'][i]} pred={result['pred'][i]}" for i in wrong]
        image_grid(imgs,labels,title='Validation errors',normalized=DATASET=='cifar')
    else: print('No error in this validation subset; inspect a larger subset.')
    return cm
'''

CORPUS = r'''
# 외부 저작물을 쓰지 않은 교육용 조합 문장. exact 문장 중복 없이 먼저 분할.
subjects=['the cat','the dog','a student','the teacher','a robot','my friend','the artist','a scientist']
verbs=['reads','finds','draws','builds','likes','studies']
objects=['a book','a small map','the new model','a red house','a quiet room','the blue box']
places=['in the lab','after class','near the river','every morning']
sentences=[f'{s} {v} {o} {p}.\n' for s in subjects for v in verbs for o in objects for p in places]
rng=random.Random(SEED); rng.shuffle(sentences)
n=len(sentences); a=int(.8*n); b=int(.9*n)
parts={'train':sentences[:a],'val':sentences[a:b],'test':sentences[b:]}
assert not (set(parts['train']) & set(parts['val']))
assert not (set(parts['train']) & set(parts['test']))
assert not (set(parts['val']) & set(parts['test']))
itos=['<UNK>']+sorted(set(''.join(parts['train'])))
stoi={c:i for i,c in enumerate(itos)}; VOCAB=len(itos)
def encode(text): return [stoi.get(c,0) for c in text]
def decode(ids): return ''.join(itos[int(i)] if int(i)!=0 else '�' for i in ids)
tokens={k:torch.tensor(encode(''.join(v)),dtype=torch.long) for k,v in parts.items()}
print('sentences:',{k:len(v) for k,v in parts.items()},'| vocab:',VOCAB)
print('tokens:',{k:len(v) for k,v in tokens.items()})
print('UNK fractions:',{k:(v==0).float().mean().item() for k,v in tokens.items()})
print('train example:',parts['train'][0])

def lm_batch(split='train',batch=BATCH,context=CONTEXT):
    data=tokens[split]; starts=torch.randint(len(data)-context,(batch,))
    x=torch.stack([data[i:i+context] for i in starts])
    y=torch.stack([data[i+1:i+context+1] for i in starts])
    return x.to(DEVICE),y.to(DEVICE)

x_example,y_example=lm_batch(batch=1)
print('input :',repr(decode(x_example[0])))
print('target:',repr(decode(y_example[0])))
assert torch.equal(x_example[:,1:],y_example[:,:-1])
'''

ATTENTION = r'''
def scaled_attention(q,k,v,causal=True):
    scores=q @ k.transpose(-2,-1) / math.sqrt(q.size(-1))
    if causal:
        blocked=torch.ones(q.size(-2),k.size(-2),device=q.device,dtype=torch.bool).triu(1)
        scores=scores.masked_fill(blocked,float('-inf'))
    weights=scores.softmax(dim=-1)
    return weights @ v,weights

class BigramLM(nn.Module):
    def __init__(self):
        super().__init__(); self.table=nn.Embedding(VOCAB,VOCAB)
    def forward(self,ids): return self.table(ids)

class SingleHeadLM(nn.Module):
    def __init__(self,d=64):
        super().__init__()
        self.token=nn.Embedding(VOCAB,d); self.position=nn.Embedding(CONTEXT,d)
        self.q=nn.Linear(d,d,bias=False); self.k=nn.Linear(d,d,bias=False); self.v=nn.Linear(d,d,bias=False)
        self.head=nn.Linear(d,VOCAB)
    def forward(self,ids,return_attention=False):
        h=self.token(ids)+self.position(torch.arange(ids.size(1),device=ids.device))
        out,a=scaled_attention(self.q(h),self.k(h),self.v(h))
        logits=self.head(out)
        return (logits,a) if return_attention else logits
'''

TRANSFORMER = r'''
class CausalMHA(nn.Module):
    def __init__(self,d=64,heads=4,dropout=.1):
        super().__init__(); assert d % heads == 0
        self.d=d; self.heads=heads
        self.qkv=nn.Linear(d,3*d); self.out=nn.Linear(d,d)
        self.attn_drop=nn.Dropout(dropout); self.out_drop=nn.Dropout(dropout)
    def forward(self,x):
        b,t,d=x.shape
        q,k,v=self.qkv(x).chunk(3,dim=-1)
        q,k,v=[u.reshape(b,t,self.heads,d//self.heads).transpose(1,2) for u in (q,k,v)]
        scores=q @ k.transpose(-2,-1)/math.sqrt(d//self.heads)
        blocked=torch.ones(t,t,device=x.device,dtype=torch.bool).triu(1)
        a=scores.masked_fill(blocked,float('-inf')).softmax(-1)
        out=self.attn_drop(a) @ v
        out=out.transpose(1,2).contiguous().view(b,t,d)
        return self.out_drop(self.out(out))

class TransformerBlock(nn.Module):
    def __init__(self,d=64,heads=4,dropout=.1):
        super().__init__()
        self.ln1=nn.LayerNorm(d); self.ln2=nn.LayerNorm(d)
        self.attn=CausalMHA(d,heads,dropout)
        self.ffn=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d),nn.Dropout(dropout))
    def forward(self,x):
        x=x+self.attn(self.ln1(x))
        return x+self.ffn(self.ln2(x))

class TinyTransformer(nn.Module):
    def __init__(self,d=64,heads=4,layers=2,dropout=.1):
        super().__init__()
        self.config=dict(d=d,heads=heads,layers=layers,dropout=dropout)
        self.token=nn.Embedding(VOCAB,d); self.position=nn.Embedding(CONTEXT,d)
        self.blocks=nn.Sequential(*[TransformerBlock(d,heads,dropout) for _ in range(layers)])
        self.norm=nn.LayerNorm(d); self.head=nn.Linear(d,VOCAB)
    def forward(self,ids):
        assert ids.size(1)<=CONTEXT
        x=self.token(ids)+self.position(torch.arange(ids.size(1),device=ids.device))
        return self.head(self.norm(self.blocks(x)))
'''

LM_TRAIN = r'''
@torch.no_grad()
def lm_eval(model,split='val'):
    was_training=model.training; model.eval(); total=count=0
    data=tokens[split]
    # 固定連続 window: 各 target を一度だけ評価。先頭 token は context に使う。
    for start in range(0,len(data)-CONTEXT,CONTEXT*BATCH):
        starts=range(start,min(start+CONTEXT*BATCH,len(data)-CONTEXT),CONTEXT)
        x=torch.stack([data[i:i+CONTEXT] for i in starts]).to(DEVICE)
        y=torch.stack([data[i+1:i+CONTEXT+1] for i in starts]).to(DEVICE)
        logits=model(x)
        total+=F.cross_entropy(logits.reshape(-1,VOCAB),y.reshape(-1),reduction='sum').item()
        count+=y.numel()
    model.train(was_training)
    return total/count

def fit_lm(factory,steps=STEPS):
    seed_all(); model=factory().to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=.01)
    history=[]; best=float('inf'); state=None; start=time.perf_counter()
    for step in range(1,steps+1):
        model.train(); x,y=lm_batch(); opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(x).reshape(-1,VOCAB),y.reshape(-1))
        assert torch.isfinite(loss)
        loss.backward(); norm=nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step==1 or step%max(1,steps//5)==0 or step==steps:
            val=lm_eval(model)
            history.append(dict(step=step,train_ce=loss.item(),val_ce=val,grad_norm=float(norm)))
            if val<best: best=val; state=cpu_state(model)
            print(f'step {step}: train={loss.item():.3f} val={val:.3f}')
    model.load_state_dict(state); model.eval()
    return model,history,time.perf_counter()-start

@torch.no_grad()
def generate(model,prompt='the ',new_tokens=120,temperature=1.0,top_k=None,greedy=False,seed=123):
    assert temperature>0 and len(prompt)>0
    seed_all(seed); model.eval()
    ids=torch.tensor([encode(prompt)],device=DEVICE)
    for _ in range(new_tokens):
        logits=model(ids[:,-CONTEXT:])[:,-1,:]/temperature
        if top_k is not None:
            k=max(1,min(int(top_k),VOCAB)); values,index=logits.topk(k)
            logits=torch.full_like(logits,float('-inf')).scatter(1,index,values)
        nxt=logits.argmax(-1,keepdim=True) if greedy else torch.multinomial(logits.softmax(-1),1)
        ids=torch.cat([ids,nxt],dim=1)
    return decode(ids[0])

def assert_causal(model):
    model.eval(); x,_=lm_batch(batch=2); changed=x.clone(); cut=CONTEXT//2
    changed[:,cut:]=(changed[:,cut:]+1)%VOCAB
    with torch.no_grad():
        a=model(x)[:,:cut]; b=model(changed)[:,:cut]
    torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
    print('PASS: changing future tokens does not change earlier logits')

def plot_lm(histories):
    for name,h in histories.items():
        plt.plot([r['step'] for r in h],[r['val_ce'] for r in h],label=name+' val')
        plt.plot([r['step'] for r in h],[r['train_ce'] for r in h],'--',label=name+' train batch')
    plt.xlabel('step'); plt.ylabel('cross entropy'); plt.legend(); plt.grid(alpha=.2); plt.show()
'''.replace('# 固定連続 window: 各 target を一度だけ評価。先頭 token は context に使う。','# 고정 연속 window: 평가 target을 중복 계산하지 않으며 남는 마지막 불완전 window는 제외.')

VAE = r'''
def reparameterize(mu,logvar):
    return mu + torch.exp(.5*logvar)*torch.randn_like(mu)

def gaussian_kl(mu,logvar):
    # shape [B,Z] 유지: 차원별 진단에 사용
    return .5*(mu.pow(2)+logvar.exp()-1-logvar)

class VAE(nn.Module):
    def __init__(self,zdim=2,conditional=False):
        super().__init__(); self.zdim=zdim; self.conditional=conditional
        cond=10 if conditional else 0
        self.encoder=nn.Sequential(nn.Linear(784+cond,256),nn.ReLU())
        self.mu_head=nn.Linear(256,zdim); self.lv_head=nn.Linear(256,zdim)
        self.decoder=nn.Sequential(nn.Linear(zdim+cond,256),nn.ReLU(),nn.Linear(256,784))
    def condition(self,y):
        return F.one_hot(y,num_classes=10).float()
    def encode(self,x,y=None):
        h=x.flatten(1)
        if self.conditional:
            assert y is not None; h=torch.cat([h,self.condition(y)],1)
        h=self.encoder(h)
        return self.mu_head(h),self.lv_head(h)
    def decode(self,z,y=None):
        if self.conditional:
            assert y is not None; z=torch.cat([z,self.condition(y)],1)
        return self.decoder(z).view(-1,1,28,28)
    def forward(self,x,y=None,sample=True):
        mu,lv=self.encode(x,y)
        z=reparameterize(mu,lv) if sample else mu
        return self.decode(z,y),mu,lv

def vae_terms(logits,x,mu,logvar):
    recon=F.binary_cross_entropy_with_logits(logits,x,reduction='none').flatten(1).sum(1)
    kl=gaussian_kl(mu,logvar).sum(1)
    return recon,kl

def vae_epoch(model,loader,optimizer=None,beta=1.0):
    training=optimizer is not None; model.train(training)
    total_r=total_k=count=0
    with torch.set_grad_enabled(training):
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            if training: optimizer.zero_grad(set_to_none=True)
            logits,mu,lv=model(x,y,sample=True)
            recon,kl=vae_terms(logits,x,mu,lv); loss=(recon+beta*kl).mean()
            assert torch.isfinite(loss), 'Nonfinite VAE loss: inspect logvar and learning rate'
            if training: loss.backward(); optimizer.step()
            total_r+=recon.detach().sum().item(); total_k+=kl.detach().sum().item(); count+=len(x)
    r,k=total_r/count,total_k/count
    return dict(recon=r,kl=k,nelbo=r+k,objective=r+beta*k)

def eval_vae(model,loader):
    # 평가 난수를 고정하고 훈련 RNG 흐름은 복원. 1-sample MC estimate.
    was_training=model.training
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(12345)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(12345)
        result=vae_epoch(model,loader)
    model.train(was_training)
    return result

def fit_vae(beta=1.,zdim=2,conditional=False,epochs=EPOCHS,warmup=False):
    seed_all(); model=VAE(zdim,conditional).to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    history=[]; best=float('inf'); state=None; start=time.perf_counter()
    # 매 조건 같은 shuffle 난수열에서 시작하도록 generator도 초기화
    train_loader.generator.manual_seed(SEED)
    for ep in range(epochs):
        effective=beta*min(1.,(ep+1)/max(1,epochs//2)) if warmup else beta
        tr=vae_epoch(model,train_loader,opt,effective); va=eval_vae(model,val_loader)
        history.append(dict(epoch=ep+1,beta=effective,train_recon=tr['recon'],train_kl=tr['kl'],
                            val_recon=va['recon'],val_kl=va['kl'],val_nelbo=va['nelbo']))
        # 各 beta の訓練目的に合わせて checkpoint 選択。共通評価は R+K。
        selection=va['recon']+beta*va['kl']
        if selection<best: best=selection; state=cpu_state(model)
        print(f"epoch {ep+1} beta={effective:.2f}: recon={va['recon']:.2f} KL={va['kl']:.2f} NELBO={va['nelbo']:.2f}")
    model.load_state_dict(state); model.eval()
    return model,history,time.perf_counter()-start

def plot_vae(history):
    fig,axes=plt.subplots(1,3,figsize=(11,3))
    for ax,key in zip(axes,['recon','kl','nelbo']):
        ax.plot([r['epoch'] for r in history],[r['val_'+key] for r in history],label='validation')
        if key!='nelbo': ax.plot([r['epoch'] for r in history],[r['train_'+key] for r in history],label='train')
        ax.set_title(key); ax.set_xlabel('epoch'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()

@torch.no_grad()
def recon_and_samples(model):
    model.eval(); x,y=next(iter(val_loader)); x,y=x[:8].to(DEVICE),y[:8].to(DEVICE)
    logits,mu,lv=model(x,y,sample=False)
    image_grid(torch.cat([x,logits.sigmoid()]),title='Top: input / Bottom: decode(mu)')
    seed_all(123); z=torch.randn(32,model.zdim,device=DEVICE)
    labels=torch.arange(32,device=DEVICE)%10
    image_grid(model.decode(z,labels).sigmoid(),title='Prior sample: displayed pixel probabilities')
    return mu
'''.replace('# 各 beta の訓練目的に合わせて checkpoint 選択。共通評価は R+K。','# 각 beta의 훈련 목적에 맞춰 checkpoint 선택. 조건 간 공통 평가 척도는 R+K.')
