"""Execute the exact distributed notebooks in fresh kernels and empty directories."""
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime
import json
import platform
import sys
import time
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'_검증/concept_notebooks'
REPORT.mkdir(parents=True,exist_ok=True)

CHECKS = {
    'week01_cnn_basics': '''
assert len(history['train_loss']) == EPOCHS
assert history['train_loss'][-1] < history['train_loss'][0]
assert all(np.isfinite(history[k]).all() for k in history)
assert test_result['cm'].sum().item() == len(y_test)
assert test_result['accuracy'] > .8, '기본 예제에서 학습이 이루어졌는지 점검'
for col in range(3):
    assert (toy[:3,col:col+3]*kernel).sum() == response[0,col]
assert torch.equal(F.relu(torch.tensor([-1.,0.,1.])),torch.tensor([0.,0.,1.]))
assert torch.allclose(shift_right(cup,2)[...,2:],cup[...,:-2])
assert Path('outputs/week01_concept/observations.json').exists()
''',
    'week02_cnn_advanced': '''
assert all(np.isfinite(h['train_loss']).all() for h in [biased_history,balanced_history])
assert all(torch.equal(initial_a[k],initial_b[k]) for k in initial_a)
assert torch.equal(matched[foreground],swapped[foreground])
assert not torch.equal(matched[~foreground],swapped[~foreground])
assert results['biased']['matched']['accuracy'] > .9
assert results['balanced']['swapped']['accuracy'] > .8
assert results['biased']['matched']['accuracy'] > results['biased']['swapped']['accuracy'] + .3
for group in results.values():
    for condition,r in group.items():
        assert int(r['cm'].sum()) == len(test_conditions[condition][1])
assert Path('outputs/week02_concept/report.json').exists()
'''}

results=[]
for folder, checks in CHECKS.items():
    path=ROOT/folder/'concept_practice.ipynb'
    nb=nbformat.read(path,as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs=[]; cell.execution_count=None
    original_count=len(nb.cells)
    nb.cells.append(nbformat.v4.new_code_cell(checks + '''
import platform
validation = {'metrics':metrics, 'python':platform.python_version(), 'torch':torch.__version__,
    'numpy':np.__version__, 'device':str(DEVICE)}
Path('validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
'''))
    print('EXECUTE',folder,flush=True)
    start=time.perf_counter()
    with TemporaryDirectory(prefix='cnn-lab-') as tmp:
        temp=Path(tmp)
        spec=temp/'kernels/cnn-lab'; spec.mkdir(parents=True)
        (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
            'display_name':'CNN validation','language':'python'}),encoding='utf-8')
        km=KernelManager(kernel_name='cnn-lab',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(temp/'kernels')]))
        try:
            NotebookClient(nb,km=km,timeout=180,resources={'metadata':{'path':str(temp)}}).execute()
            result=json.loads((temp/'validation.json').read_text(encoding='utf-8'))
            result.update(file=f'{folder}/concept_practice.ipynb',seconds=round(time.perf_counter()-start,2),status='PASS')
        finally:
            if km.has_kernel:
                km.shutdown_kernel(now=True)
    nb.cells=nb.cells[:original_count]
    nbformat.validate(nb)
    nbformat.write(nb,path)
    result['cells']=len(nb.cells)
    result['code_cells']=sum(c.cell_type=='code' for c in nb.cells)
    result['figures']=sum('image/png' in o.get('data',{}) for c in nb.cells if c.cell_type=='code' for o in c.outputs)
    results.append(result)
    (REPORT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)

lines=['# CNN 개념 실습 노트북 검증', '', f'검증: {datetime.now():%Y-%m-%d %H:%M} (로컬).', '',
    '배포하는 파일을 서로 다른 새 커널과 빈 작업 폴더에서 위부터 끝까지 실행했습니다. 외부 데이터 다운로드·주차 간 실행 상태 없이 동작하며 실행 출력과 그림을 노트북에 저장했습니다.', '',
    '| 파일 | 셀 / 코드 셀 | 그림 | 실행 시간(초) | 상태 |','|---|---:|---:|---:|---|']
for r in results:
    lines.append(f"| {r['file']} | {r['cells']} / {r['code_cells']} | {r['figures']} | {r['seconds']} | PASS |")
a,b=results
lines += ['', '## 실행 환경과 범위','',f"Python {a['python']} / PyTorch {a['torch']} / NumPy {a['numpy']} / CPU.",
    '시간은 이 PC의 관측값이며 Colab 시간 보장이 아닙니다. 실제 Colab 원격 실행은 수행하지 않았습니다. Python 3 커널 메타데이터와 독립 실행 구성을 사용합니다.', '',
    '## 의미 있는 확인','',
    '- 1주차: 손계산과 conv2d 일치, 반전 부호, ReLU, 이동, 필터 가중치 변경, 손실 감소, test 집계, 결과 JSON 저장.',
    '- 2주차: 두 모델의 동일 초기값·예산·전경·라벨, 배경 균형, paired test의 전경 보존과 배경 변경, 혼동 행렬 집계, 결과 JSON 저장.',
    '- 기본 수업 예제가 실제로 학습되고 배경 지름길을 관찰할 수 있는지도 검사했습니다. 성능은 한 seed의 합성 데이터 결과입니다.', '',
    '## 실제 관측값','',
    f"1주차: test accuracy {a['metrics']['test_accuracy']:.3f}, 가림 후 {a['metrics']['occluded_accuracy']:.3f} (폭 {a['metrics']['occlusion_width']}).", '',
    '| 2주차 모델 | matched | swapped | balanced | 배경 교체 시 예측 변경 비율 |','|---|---:|---:|---:|---:|']
for name,group in b['metrics']['scores'].items():
    values=[group[c]['accuracy'] for c in ('matched','swapped','balanced')]
    lines.append(f"| {name} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} | {b['metrics']['flip_rates'][name]:.3f} |")
lines += ['', 'paired test는 같은 192개 물체의 배경 조건을 비교합니다. balanced 조건 384장을 독립 물체 384개로 해석하지 않습니다. 이 합성 실험을 인간 시각 또는 실제 사물 인식의 재현으로 주장하지 않습니다.', '',
    '재생성: `python cnn_concept_web/src/build_notebooks.py` → `python cnn_concept_web/src/validate_notebooks.py`.',
    '검증 실행에는 nbformat, nbclient, ipykernel이 추가로 필요합니다. 원시 결과는 `_검증/concept_notebooks/results.json`에 있습니다.']
(ROOT/'cnn_concept_web/실습_검증_보고서.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
