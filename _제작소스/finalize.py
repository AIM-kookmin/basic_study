from pathlib import Path
import json,base64,platform,zipfile,re
import nbformat
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'_검증'
results=json.loads((REPORT/'notebook_default.json').read_text(encoding='utf-8'))
vae_metrics=json.loads((ROOT/'week05_vae_basics/outputs/vae_metrics.json').read_text())['test']
slides=json.loads((REPORT/'slides.json').read_text(encoding='utf-8'))
assert len(results)==18 and all(r['status']=='PASS' for r in results)
assert len(slides)==6 and all(s['slides']==40 and not s['overflow'] for s in slides)
stats=[]
for p in sorted(ROOT.glob('week*/*.ipynb')):
    nb=nbformat.read(p,4)
    # Adjacent progress stream messages are compacted; code and plotted results stay intact.
    for c in nb.cells:
        if c.cell_type!='code': continue
        grouped=[]
        for o in c.outputs:
            if o.output_type=='stream' and grouped and grouped[-1].output_type=='stream' and grouped[-1].name==o.name:
                grouped[-1].text+=o.text
            else: grouped.append(o)
        for o in grouped:
            if o.output_type=='stream' and '\r' in o.text:
                parts=o.text.split('\n'); o.text='\n'.join(s.split('\r')[-1] for s in parts)
        c.outputs=grouped
    nbformat.validate(nb); nbformat.write(nb,p)
    stats.append(dict(file=str(p.relative_to(ROOT)),cells=len(nb.cells),code_lines=sum(len(c.source.splitlines()) for c in nb.cells if c.cell_type=='code')))
    if p.name=='practice.ipynb':
        pics=[o.data['image/png'] for c in nb.cells for o in c.get('outputs',[]) if 'image/png' in o.get('data',{})]
        for i,img in enumerate(pics[-3:]):
            (REPORT/f'{p.parent.name}_result_{i}.png').write_bytes(base64.b64decode(img))
(REPORT/'artifact_stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')

lines=['# 검증 보고서','', '검증 완료: 2026-09-07 (Asia/Seoul).', '',
       '## 실행 환경', '', '- Windows / Python 3.13.9', '- PyTorch 2.14.0+cpu / torchvision 0.29.0+cpu',
       '- NumPy 2.3.5 / Matplotlib 3.10.6', '- CPU, torch 스레드 최대 4개',
       '- 실제 FashionMNIST·CIFAR10·MNIST와 내장 교육용 문자 코퍼스 사용',
       '- 모든 노트북 QUICK=True, SMOKE=False 기본 설정을 실행', '',
       '## 결과', '', '**18개 노트북 전체 셀 실행 성공. 6개 PPT 총 240장 PowerPoint 열기·PDF 변환 성공.**',
       'PPT 전체 텍스트 상자에서 실제 렌더링 높이가 상자 높이를 넘는 항목 0개를 확인했습니다. 대표 슬라이드와 결과 그림도 시각 확인했습니다.', '',
       '| 파일 | 상태 | 실행 시간(초) | 코드 셀 수 |', '|---|---|---:|---:|']
for r in results: lines.append(f"| {r['file'].replace(chr(92),'/')} | {r['status']} | {r['seconds']} | {r['code_cells']} |")
lines += ['', '실행 시간은 이 PC에서의 관측값이며 초기 커널 구동·렌더링을 포함합니다. 2주차 자율 과제의 최초 실행에는 CIFAR10 다운로드 시간이 포함되어 더 깁니다. Colab이나 다른 CPU/GPU의 시간 보장은 아닙니다.', '',
          '## 실제로 확인한 계약', '',
          '- CNN: 출력 shape·파라미터 수, forward/backward, validation checkpoint, confusion matrix·오분류, test 평가.',
          '- Attention: 손계산, 행 합 1, 미래 가중치 0, SDPA와 수치 일치, 정상·누출 모델 구분.',
          '- Transformer: head·block shape, gradient, causal 불변성, top-k/top-p, checkpoint 복원 후 logits 일치.',
          '- VAE: Gaussian KL 닫힌식, 재매개화 평균·분산·gradient, loss reduction과 배치 복제 불변성.',
          '- CVAE: 조건부 shape·값 범위·라벨 변화의 logits 차이, 공통 β=1 평가, 차원별 KL.', '',
          '미완성 학생 과제의 TODO는 안내 후 건너뛰도록 설계했습니다. 학생 과제의 실행 성공이 TODO의 구현 완료를 의미하지는 않습니다. 동일 문제의 **리더용 해설 구현과 검사까지 별도로 모두 실행**했습니다.', '',
          '## 관측된 예시와 해석', '',
          '- 1주차: 이 짧은 설정에서는 MLP가 validation loss로 선택됐고 test 부분집합 정확도는 약 77.15%였습니다. CNN의 보편적 열세를 의미하지 않습니다.',
          '- 2주차: base 조건이 선택됐고 test 부분집합 정확도는 약 38.67%였습니다. 증강이 항상 즉시 이득을 주지 않는 실제 비교 사례입니다.',
          '- 3주차: 선택한 attention 모델의 test CE 약 1.157, PPL 약 3.180을 관측했습니다.',
          '- 4주차: TinyTransformer의 test CE 약 0.274, PPL 약 1.315를 관측했습니다. 단순 조합 문장 코퍼스의 결과이며 실제 자연어 능력 지표로 외삽하지 않습니다.',
          f"- 5주차: MNIST test 부분집합의 1-sample MC negative ELBO 약 {vae_metrics['nelbo']:.3f}을 관측했습니다. recon 약 {vae_metrics['recon']:.3f}, KL 약 {vae_metrics['kl']:.3f}입니다.",
          '- 6주차: β=0.25/1/4 조건의 val recon·KL을 각각 측정하고 같은 R+K 기준으로 비교했습니다. β 비교는 Z=8·8 epoch, 조건부 생성은 Z=2·20 epoch를 사용합니다. CVAE의 실제 격자에서 라벨별 숫자 변화가 확인됐으며, 두 실험은 서로 다른 과제이므로 직접 성능 순위를 비교하지 않습니다.', '',
          '모든 성능 수치는 실행된 노트북 출력에서 읽은 값입니다. 이미지 test는 1,024개 고정 부분집합이고, 여러 seed 통계 검증은 하지 않았습니다.', '',
          '## 검증 범위의 한계', '',
          '- 실제 Google Colab 또는 CUDA GPU에서 원격 실행하지는 않았습니다. 노트북은 Colab 독립 실행 형식이며 CUDA 분기는 코드에 포함됩니다.',
          '- QUICK=False의 확대 데이터·장기 학습, 다른 버전 조합은 별도로 실행하지 않았습니다.',
          '- PowerPoint COM 텍스트 검사와 대표 렌더링 확인은 폰트가 다른 PC의 모든 시각 차이를 보장하지 않습니다. 제공 PDF는 검증 환경 렌더링입니다.',
          '- 외부 데이터 다운로드 서버 응답 시간은 환경에 따라 달라집니다. 수업 전 다운로드를 권장합니다.', '',
          '상세 기계 판독 결과는 `_검증/notebook_default.json`, `_검증/slides.json`, `_검증/artifact_stats.json`에 있습니다.']
(ROOT/'검증_보고서.md').write_text('\n'.join(lines),encoding='utf-8')

files=sorted(REPORT.glob('week*/slide_*.png'))
canvas=Image.new('RGB',(1600,((len(files)+3)//4)*245),'#dfe5ee'); d=ImageDraw.Draw(canvas)
for i,p in enumerate(files):
    x=(i%4)*400;y=(i//4)*245;canvas.paste(Image.open(p).resize((390,219)),(x,y))
    d.text((x+5,y+222),p.parent.name+' / '+p.stem,fill='black')
canvas.save(REPORT/'contact_sheet.png')
# 수강생 배포 ZIP은 해설·실행 산출물·제작소스를 제외한다. 원본 폴더에는 모두 보관.
with zipfile.ZipFile(ROOT/'AIM_2026_2_수강생배포.zip','w',zipfile.ZIP_DEFLATED) as z:
    z.writestr('README.md', '# AIM 2026 가을학기 수강생 자료\n\n6주 × 90분 과정입니다. 주차별 lecture.pptx 또는 lecture.pdf로 복습하고 practice.ipynb를 Colab에 업로드해 실행하세요. homework_optional.ipynb는 자율 과제입니다. 환경설정.md에서 실행 방법을 확인하세요.\n\n주차별 README에 소개된 leader_solution.ipynb와 leader_guide.md는 리더가 별도로 보관합니다. 이 수강생 배포본에는 포함하지 않았습니다.\n\n1–2주 CNN / 3–4주 Attention·언어 모델 / 5–6주 VAE. 각 강의는 40장(핵심 28 + 부록 12)입니다.\n')
    z.write(ROOT/'환경설정.md','환경설정.md')
    for folder in sorted(ROOT.glob('week*')):
        for name in ['lecture.pptx','lecture.pdf','practice.ipynb','homework_optional.ipynb','README.md']:
            p=folder/name; z.write(p,str(p.relative_to(ROOT)))
print('Finalized',len(stats),'notebooks;',sum(s['slides'] for s in slides),'slides')
