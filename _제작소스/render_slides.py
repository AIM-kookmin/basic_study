import json,time
from pathlib import Path
import win32com.client
ROOT=Path(__file__).resolve().parents[1]
report=ROOT/'_검증'; report.mkdir(exist_ok=True)
app=win32com.client.DispatchEx('PowerPoint.Application')
results=[]
try:
    for path in sorted(ROOT.glob('week*/lecture.pptx')):
        print('RENDER',path.parent.name,flush=True)
        pres=app.Presentations.Open(str(path),ReadOnly=True,Untitled=False,WithWindow=False)
        output=report/path.parent.name; output.mkdir(exist_ok=True)
        overflow=[]
        for s in pres.Slides:
            for sh in s.Shapes:
                if sh.HasTextFrame and sh.TextFrame.HasText:
                    try:
                        # BoundHeight is actual rendered glyph height, compared with box height.
                        if sh.TextFrame2.TextRange.BoundHeight > sh.Height+2:
                            overflow.append(dict(slide=s.SlideIndex,text=sh.TextFrame.TextRange.Text[:90],height=sh.Height,bound=sh.TextFrame2.TextRange.BoundHeight))
                    except Exception: pass
            if s.SlideIndex in [1,3,7,10,15,20,24,29,35,36,37,38,39]:
                s.Export(str(output/f'slide_{s.SlideIndex:02d}.png'),'PNG',1600,900)
        # The revised CNN PDFs are produced by cnn_concept_web/src/render_and_check.py.
        revised_cnn = path.parent.name in ('week01_cnn_basics', 'week02_cnn_advanced') and (ROOT/'cnn_concept_web/index.html').exists()
        pdf_path = path.with_name('lecture_legacy.pdf') if revised_cnn else path.with_suffix('.pdf')
        pres.SaveAs(str(pdf_path),32)
        results.append(dict(week=path.parent.name,slides=pres.Slides.Count,overflow=overflow,pdf=True))
        pres.Close()
        (report/'slides.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
finally:
    app.Quit()
print(json.dumps(results,ensure_ascii=False),flush=True)
