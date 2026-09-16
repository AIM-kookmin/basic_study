"""MNIST teaching experiment; this module is embedded in the standalone notebook."""
from pathlib import Path
import os, gzip, hashlib, struct, urllib.request, json, time, copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from torch import nn
import torch.nn.functional as F

FILES = {
 'train-images-idx3-ubyte.gz': 'f68b3c2dcbeaaa9fbdd348bbdeb94873',
 'train-labels-idx1-ubyte.gz': 'd53e105ee54ea40749a09fcbcd1e9432',
 't10k-images-idx3-ubyte.gz': '9fb629c4189551a2d022fa330f9573f3',
 't10k-labels-idx1-ubyte.gz': 'ec29112dd5afa0611ce80d1b7f02629c'}

def read_mnist():
    # Same MNIST mirror/checksums published in torchvision.datasets.MNIST.
    cache = Path(os.environ.get('AIM_MNIST_CACHE','data/MNIST/raw'))
    cache.mkdir(parents=True,exist_ok=True)
    arrays=[]
    for name,digest in FILES.items():
        path=cache/name
        if not path.exists():
            print('Downloading',name,flush=True)
            url='https://ossci-datasets.s3.amazonaws.com/mnist/'+name
            with urllib.request.urlopen(url,timeout=60) as source:
                payload=source.read()
            assert hashlib.md5(payload).hexdigest()==digest, 'MNIST download checksum mismatch'
            path.write_bytes(payload)
        packed=path.read_bytes()
        assert hashlib.md5(packed).hexdigest()==digest, f'Corrupt cache: {path}; remove only this file and retry.'
        raw=gzip.decompress(packed)
        magic,count=struct.unpack('>II',raw[:8])
        if magic==2051:
            rows,cols=struct.unpack('>II',raw[8:16]);assert (rows,cols)==(28,28)
            array=np.frombuffer(raw, dtype=np.uint8,offset=16).copy().reshape(count,28,28)
        else:
            assert magic==2049
            array=np.frombuffer(raw,dtype=np.uint8,offset=8).copy()
            assert len(array)==count
        arrays.append(torch.from_numpy(array))
    assert len(arrays[0])==60000 and len(arrays[2])==10000
    return arrays

def prepare_data(seed=17):
    train_images,train_labels,test_images,test_labels=read_mnist()
    rng=torch.Generator().manual_seed(seed)
    ti=[];vi=[];si=[]
    for label in range(10):
        ix=torch.where(train_labels==label)[0]
        ix=ix[torch.randperm(len(ix),generator=rng)]
        ti.append(ix[:600]);vi.append(ix[600:700])
        ix=torch.where(test_labels==label)[0]
        si.append(ix[torch.randperm(len(ix),generator=rng)[:100]])
    ti,vi,si=map(torch.cat,(ti,vi,si))
    assert not set(ti.tolist()) & set(vi.tolist())
    def x(image,index):return image[index].unsqueeze(1).float()/255
    return dict(x_train=x(train_images,ti),y_train=train_labels[ti].long(),
        x_val=x(train_images,vi),y_val=train_labels[vi].long(),
        x_test=x(test_images,si),y_test=test_labels[si].long(),
        train_indices=ti,val_indices=vi,test_indices=si)

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(8,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2))
        self.classifier=nn.Linear(16*7*7,10)
    def forward(self,x):return self.classifier(self.features(x).flatten(1))

@torch.no_grad()
def evaluate_digit(model,x,y):
    model.eval()
    logits=torch.cat([model(batch) for batch in x.split(256)])
    pred=logits.argmax(1)
    cm=torch.bincount(y*10+pred,minlength=100).reshape(10,10)
    return dict(accuracy=float((pred==y).float().mean()),ce=float(F.cross_entropy(logits,y)),
        logits=logits,pred=pred,confusion=cm)

def fit_digit(data,epochs=6,seed=17):
    torch.set_num_threads(min(4,torch.get_num_threads()))
    torch.manual_seed(seed)
    model=DigitCNN();initial=copy.deepcopy(model.state_dict())
    optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    order=torch.Generator().manual_seed(seed+1)
    history={'train_ce':[],'val_ce':[],'val_accuracy':[]}
    start=time.perf_counter()
    for epoch in range(epochs):
        model.train();total=0.
        for ix in torch.randperm(len(data['y_train']),generator=order).split(128):
            optimizer.zero_grad()
            loss=F.cross_entropy(model(data['x_train'][ix]),data['y_train'][ix])
            assert torch.isfinite(loss)
            loss.backward();optimizer.step()
            total+=float(loss.detach())*len(ix)
        val=evaluate_digit(model,data['x_val'],data['y_val'])
        history['train_ce'].append(total/len(data['y_train']))
        history['val_ce'].append(val['ce']);history['val_accuracy'].append(val['accuracy'])
        print(f"epoch {epoch+1}/{epochs}: train CE {history['train_ce'][-1]:.3f}, val accuracy {val['accuracy']:.3f}")
    print(f'Training: {time.perf_counter()-start:.1f}s / CPU')
    # The final epoch is fixed in advance; test does not select a checkpoint.
    return model,history,initial

def style_figures():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'figure.dpi':130,
        'figure.facecolor':'#f4f2e9','axes.facecolor':'#f4f2e9','axes.spines.top':False,
        'axes.spines.right':False,'axes.prop_cycle':plt.cycler(color=['#087e76','#e9643d'])})

def sample_figure(data):
    fig,axs=plt.subplots(3,10,figsize=(10,3.5))
    indices=[]
    for label in range(10):
        ids=torch.where(data['y_test']==label)[0][:3];indices.extend(ids.tolist())
        for row,i in enumerate(ids):
            axs[row,label].imshow(data['x_test'][i,0],cmap='gray',vmin=0,vmax=1)
            axs[row,label].axis('off')
            if row==0:axs[row,label].set_title(str(label),fontsize=16)
    fig.tight_layout(pad=.5)
    return fig

def pixels_figure(data,index=700):
    img=data['x_test'][index,0]
    fig,axs=plt.subplots(1,2,figsize=(7,3.7))
    axs[0].imshow(img,cmap='gray',vmin=0,vmax=1)
    axs[0].add_patch(Rectangle((9.5,9.5),6,6,fill=False,edgecolor='#e9643d',linewidth=2))
    axs[0].set_title('28 x 28 / 1 channel');axs[0].axis('off')
    patch=(img[10:16,10:16]*255).round().int()
    axs[1].imshow(patch,cmap='gray',vmin=0,vmax=255)
    for (r,c),v in np.ndenumerate(patch.numpy()):
        axs[1].text(c,r,str(v),ha='center',va='center',fontsize=8,color='black' if v>120 else 'white')
    axs[1].set_title('A 6 x 6 patch / 0...255');axs[1].axis('off')
    fig.tight_layout();return fig

def manual_filter_figure(data,index=700):
    x=data['x_test'][index:index+1]
    k=torch.tensor([[-1.,0.,1.]]*3)
    maps=F.conv2d(x,torch.stack([k,k.T])[:,None],padding=1)
    fig,axs=plt.subplots(1,3,figsize=(9,3.3))
    axs[0].imshow(x[0,0],cmap='gray',vmin=0,vmax=1);axs[0].set_title('Actual MNIST input')
    for ax,m,title in zip(axs[1:],maps[0],['Left-right difference','Top-bottom difference']):
        ax.imshow(m,cmap='RdBu_r',vmin=-3,vmax=3);ax.set_title(title,fontsize=11)
    for ax in axs:ax.axis('off')
    fig.tight_layout();return fig

def history_figure(history):
    fig,axs=plt.subplots(1,2,figsize=(8,3.6));ep=np.arange(1,len(history['train_ce'])+1)
    axs[0].plot(ep,history['train_ce'],'o-',label='train');axs[0].plot(ep,history['val_ce'],'o-',label='validation')
    axs[0].set(xlabel='epoch',ylabel='cross entropy');axs[0].legend(fontsize=9)
    axs[1].plot(ep,history['val_accuracy'],'o-');axs[1].set(xlabel='epoch',ylabel='validation accuracy',ylim=(.8,1))
    fig.tight_layout();return fig

def feature_figure(model,data,index=700):
    with torch.no_grad():maps=F.relu(model.features[0](data['x_test'][index:index+1]))[0]
    fig,axs=plt.subplots(2,4,figsize=(7,4));limit=max(float(maps.max()),1e-6)
    for i,ax in enumerate(axs.flat):
        ax.imshow(maps[i],cmap='magma',vmin=0,vmax=limit);ax.set_title(f'channel {i}');ax.axis('off')
    fig.tight_layout();return fig

def errors_figure(data,result):
    ids=torch.where(result['pred']!=data['y_test'])[0][:6]
    wrong=True
    if not len(ids):ids=torch.arange(6);wrong=False
    fig,axs=plt.subplots(2,3,figsize=(6,4.5))
    for ax in axs.flat:ax.axis('off')
    for ax,i in zip(axs.flat,ids):
        ax.imshow(data['x_test'][i,0],cmap='gray',vmin=0,vmax=1)
        ax.set_title(f"label {int(data['y_test'][i])} / pred {int(result['pred'][i])}",fontsize=11,color='#b34429')
    fig.tight_layout();return fig,ids,wrong

def prediction_figure(data,result,index=700):
    fig,axs=plt.subplots(1,2,figsize=(7,3.5))
    axs[0].imshow(data['x_test'][index,0],cmap='gray',vmin=0,vmax=1);axs[0].axis('off')
    axs[0].set_title(f"label {int(data['y_test'][index])} / pred {int(result['pred'][index])}")
    probs=result['logits'][index].softmax(0).numpy()
    axs[1].bar(np.arange(10),probs,color='#087e76');axs[1].set(xticks=range(10),ylim=(0,1),xlabel='digit class',ylabel='softmax')
    fig.tight_layout();return fig

def experiment_metrics(data,model,history,result,epochs,seed):
    return dict(dataset='MNIST',seed=seed,epochs=epochs,device='cpu',torch=torch.__version__,
        train_n=len(data['y_train']),val_n=len(data['y_val']),test_n=len(data['y_test']),
        parameters=sum(p.numel() for p in model.parameters()),accuracy=result['accuracy'],ce=result['ce'],
        confusion=result['confusion'].tolist(),history=history,
        test_indices=data['test_indices'].tolist())

def export_assets(target,data,model,history,result,epochs=6,seed=17):
    target=Path(target);target.mkdir(parents=True,exist_ok=True);style_figures()
    error_fig,ids,wrong=errors_figure(data,result)
    figures={'samples':sample_figure(data),'pixels':pixels_figure(data),
        'filters':manual_filter_figure(data),'learning':history_figure(history),
        'features':feature_figure(model,data),'prediction':prediction_figure(data,result),'errors':error_fig}
    for name,fig in figures.items():
        fig.savefig(target/f'mnist_{name}.png',dpi=190,bbox_inches='tight');plt.close(fig)
    metrics=experiment_metrics(data,model,history,result,epochs,seed)
    metrics['error_example_indices']=data['test_indices'][ids].tolist()
    metrics['error_examples_are_errors']=wrong
    (target/'mnist_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    return metrics
