"""Validate the revised MNIST notebook, without rerunning unchanged week 2."""
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime
import json,sys,time,os
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT=Path(__file__).resolve().parents[2]
reportdir=ROOT/'_검증/concept_notebooks';reportdir.mkdir(parents=True,exist_ok=True)
nbpath=ROOT/'week01_cnn_basics/concept_practice.ipynb'
nb=nbformat.read(nbpath,4);count=len(nb.cells)
for c in nb.cells:
    if c.cell_type=='code':c.outputs=[];c.execution_count=None
nb.cells.append(nbformat.v4.new_code_cell('''
assert tuple(data['x_train'].shape)==(6000,1,28,28)
assert not set(data['train_indices'].tolist()) & set(data['val_indices'].tolist())
assert all(torch.equal(torch.bincount(data[k]),torch.full((10,),n)) for k,n in [('y_train',600),('y_val',100),('y_test',100)])
assert int(test_result['confusion'].sum())==1000
assert test_result['accuracy']>.9
assert history['train_ce'][-1]<history['train_ce'][0]
assert not torch.equal(initial_weights['features.0.weight'],model.features[0].weight.detach())
assert sum(p.numel() for p in model.parameters())==9098
assert torch.equal(response,torch.tensor([[3.,3.,0.]]*3))
assert all(int(test_result['pred'][i])!=int(data['y_test'][i]) for i in error_ids) if are_errors else True
assert Path('outputs/week01_mnist/observations.json').exists()
for col in [0,1,2]:assert (toy[:3,col:col+3]*kernel).sum()==response[0,col]
import platform
Path('validated.json').write_text(json.dumps({'metrics':metrics,'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__},ensure_ascii=False),encoding='utf-8')
'''))
start=time.perf_counter()
with TemporaryDirectory(prefix='mnist-lesson-') as tmp:
    temp=Path(tmp);spec=temp/'kernels/mnist';spec.mkdir(parents=True)
    # Cache path is a validator setting, not a path embedded in the notebook.
    cache=os.environ.get('AIM_MNIST_CACHE',str(ROOT/'data/MNIST/raw'))
    (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'MNIST check','language':'python','env':{'AIM_MNIST_CACHE':cache}}),encoding='utf-8')
    km=KernelManager(kernel_name='mnist',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(temp/'kernels')]))
    try:
        NotebookClient(nb,km=km,timeout=240,resources={'metadata':{'path':str(temp)}}).execute()
        result=json.loads((temp/'validated.json').read_text(encoding='utf-8'))
    finally:
        if km.has_kernel:km.shutdown_kernel(now=True)
nb.cells=nb.cells[:count];nbformat.validate(nb);nbformat.write(nb,nbpath)
downloads=sum('Downloading' in line for c in nb.cells for o in c.get('outputs',[]) if o.get('output_type')=='stream' for line in o.get('text','').splitlines())
result.update(seconds=round(time.perf_counter()-start,2),cells=count,status='PASS',downloaded_files=downloads)
(reportdir/'mnist_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
m=result['metrics']
lines=['# 1주차 MNIST 개정 실습 검증','',f'검증: {datetime.now():%Y-%m-%d %H:%M}.','',
f"새 커널·빈 작업 폴더에서 실제 배포 노트북 {count}개 셀을 모두 실행했습니다. 실행 출력과 그림을 저장했습니다. 소요 시간 {result['seconds']}초는 이 PC에서의 관측값입니다.",'',
f"환경: Python {result['python']} / PyTorch {result['torch']} / NumPy {result['numpy']} / CPU.",'',
'## 데이터와 확인 내용','',
f'- 실제 MNIST gzip 4개의 공식 체크섬·IDX 헤더 확인. 이번 실행에서 {downloads}개를 새로 다운로드하고 나머지는 캐시를 검증해 사용했습니다.',
'- 공식 train에서 6,000장 학습·1,000장 검증을 겹침 없이 선택. 공식 test에서 1,000장 고정 평가. 클래스별 균형과 인덱스 분리 확인.',
'- 28×28 입력·10개 클래스·9,098개 파라미터·손계산과 conv2d 일치 확인.',
'- 실제 가중치 변경·손실 감소·혼동 행렬 집계·오분류 예시의 실제 오류 여부·JSON 저장 확인.','',
'## 실제 결과','',f"seed17·6epoch: test accuracy {m['accuracy']:.3f}, CE {m['ce']:.3f}. 오른쪽 {m['shift']}픽셀 이동 후 accuracy {m['shift_accuracy']:.3f}.",'',
'공식 test10,000장 전체 성능 또는 여러 seed 통계가 아닙니다. 실제 Colab 원격 실행은 하지 않았습니다. Colab에서는 첫 실행에 약11MB 다운로드와 인터넷 연결이 필요합니다.','',
'발표용 그림은 `prepare_mnist.py`에서 같은 데이터 선택·모델·설정으로 별도로 실행해 생성합니다. 실제 측정값과 선택한 test 인덱스는 `assets/mnist_metrics.json`에 있습니다.','',
'재생성: `build_mnist_notebook.py` → `validate_mnist_notebook.py`. 2주차 노트북은 이번 검증에서 변경·재실행하지 않았습니다.']
(ROOT/'cnn_concept_web/MNIST_실습검증_보고서.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='metrics'},ensure_ascii=False))
print('accuracy',m['accuracy'],'shift accuracy',m['shift_accuracy'])
