"""Validate current week 1 MNIST plus unchanged week 2 shortcut lab."""
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime
import json,sys,time,subprocess
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT=Path(__file__).resolve().parents[2]
subprocess.run([sys.executable,str(Path(__file__).with_name('validate_mnist_notebook.py'))],check=True)
path=ROOT/'week02_cnn_advanced/concept_practice.ipynb'
nb=nbformat.read(path,4);count=len(nb.cells)
for c in nb.cells:
    if c.cell_type=='code':c.outputs=[];c.execution_count=None
nb.cells.append(nbformat.v4.new_code_cell('''
assert all(torch.equal(initial_a[k],initial_b[k]) for k in initial_a)
assert torch.equal(matched[foreground],swapped[foreground])
assert not torch.equal(matched[~foreground],swapped[~foreground])
assert results['biased']['matched']['accuracy']>.9
assert results['balanced']['swapped']['accuracy']>.8
assert results['biased']['matched']['accuracy']>results['biased']['swapped']['accuracy']+.3
assert len(xb)==len(xc) and torch.equal(yb,yc)
assert float((bgc==yc).float().mean())==.5
for group in results.values():
    for condition,r in group.items():assert int(r['cm'].sum())==len(test_conditions[condition][1])
assert Path('outputs/week02_concept/report.json').exists()
Path('verified.json').write_text(json.dumps(metrics),encoding='utf-8')
'''))
start=time.perf_counter()
with TemporaryDirectory(prefix='cnn-week2-') as tmp:
    temp=Path(tmp);spec=temp/'kernels/check';spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'CNN check','language':'python'}),encoding='utf-8')
    km=KernelManager(kernel_name='check',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(temp/'kernels')]))
    try:
        NotebookClient(nb,km=km,timeout=180,resources={'metadata':{'path':str(temp)}}).execute()
        metrics=json.loads((temp/'verified.json').read_text(encoding='utf-8'))
    finally:
        if km.has_kernel:km.shutdown_kernel(now=True)
seconds=round(time.perf_counter()-start,2)
nb.cells=nb.cells[:count];nbformat.validate(nb);nbformat.write(nb,path)
report=ROOT/'_검증/concept_notebooks';report.mkdir(parents=True,exist_ok=True)
(report/'week2_current.json').write_text(json.dumps({'status':'PASS','seconds':seconds,'metrics':metrics},indent=2),encoding='utf-8')
lines=['# CNN 현재 실습 통합 검증','',f'검증: {datetime.now():%Y-%m-%d %H:%M}.','',
'1주차 MNIST와 2주차 배경 지름길 노트북을 각각 새 커널·빈 작업 폴더에서 전체 실행했습니다. 실행 출력과 그림을 저장했습니다.','',
'1주차의 데이터·환경·결과는 [MNIST 검증 보고서](MNIST_실습검증_보고서.md)에 기록했습니다. 2주차는 내부 합성 데이터로 실행했습니다.','',
f'2주차 {count}개 셀: PASS, {seconds}초(CPU, 이 PC 관측). 동일 초기값·전경·예산, 배경 균형, paired test·혼동 행렬·JSON 출력을 확인했습니다.','',
'| 모델 | matched | swapped | balanced | 예측 변경 비율 |','|---|---:|---:|---:|---:|']
for name,scores in metrics['scores'].items():
    values=[scores[c]['accuracy'] for c in ('matched','swapped','balanced')]
    lines.append(f"| {name} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} | {metrics['flip_rates'][name]:.3f} |")
lines+=['','Colab 원격 실행이나 여러 seed·전체 MNIST test 성능 검증은 수행하지 않았습니다. 2주차 balanced test는192개 물체의 두 배경이므로 독립 물체384개로 해석하지 않습니다.']
(ROOT/'cnn_concept_web/현재_실습검증_보고서.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Current week 1 and week 2 notebooks: PASS')
