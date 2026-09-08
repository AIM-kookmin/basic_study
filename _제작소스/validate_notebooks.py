import os,sys,time,json,traceback,ast
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'_검증'; REPORT.mkdir(exist_ok=True)
os.environ.setdefault('AIM_DATA_ROOT',str(Path('D:/AIM/.study-data')))
os.environ['MPLBACKEND']='module://matplotlib_inline.backend_inline'
mode='smoke' if os.environ.get('AIM_SMOKE')=='1' else 'default'
paths=sorted(ROOT.glob('week*/*.ipynb'))
if len(sys.argv)>1: paths=[p for p in paths if sys.argv[1] in str(p)]
report_path=REPORT/f'notebook_{mode}.json'
previous=json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else []
results=[]
for path in paths:
    print('START',path.parent.name,path.name,flush=True); start=time.perf_counter()
    nb=nbformat.read(path,as_version=4); nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type=='code': ast.parse(cell.source)
    km=KernelManager(kernel_name='python3')
    km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    client=NotebookClient(nb,timeout=900,km=km,resources={'metadata':{'path':str(path.parent)}})
    result=dict(file=str(path.relative_to(ROOT)),mode=mode)
    try:
        client.execute()
        outputs=sum(len(c.get('outputs',[])) for c in nb.cells)
        result.update(status='PASS',seconds=round(time.perf_counter()-start,2),outputs=outputs,
                      code_cells=sum(c.cell_type=='code' for c in nb.cells))
        if mode=='default': nbformat.write(nb,path)
        else: nbformat.write(nb,REPORT/(path.parent.name+'_'+path.name))
    except Exception as e:
        result.update(status='FAIL',seconds=round(time.perf_counter()-start,2),error=str(e)[-6000:])
        traceback.print_exc()
    finally:
        if km.has_kernel: km.shutdown_kernel(now=True)
        km.cleanup_resources()
    results.append(result)
    merged={r['file']:r for r in previous}
    merged.update({r['file']:r for r in results})
    report_path.write_text(json.dumps(sorted(merged.values(),key=lambda r:r['file']),ensure_ascii=False,indent=2),encoding='utf-8')
    print('RESULT',json.dumps(result,ensure_ascii=False),flush=True)
print('TOTAL',len(results),'PASS',sum(r['status']=='PASS' for r in results),flush=True)
sys.exit(0 if all(r['status']=='PASS' for r in results) else 1)
