from pathlib import Path
import json, textwrap
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR
from curriculum import WEEKS
from visuals import draw,TITLES

ROOT = Path(__file__).resolve().parents[1]
BG='F6F8FC'; INK='13243A'; MUTED='53657D'; WHITE='FFFFFF'
COLORS=['147D92','147D92','6454BF','6454BF','CA643B','CA643B']

def box(slide,x,y,w,h,text='',size=20,color=INK,fill=None,bold=False,font='맑은 고딕'):
    sh=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)) if fill else slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if fill:
        sh.fill.solid(); sh.fill.fore_color.rgb=RGBColor.from_string(fill); sh.line.fill.background()
    tf=sh.text_frame; tf.word_wrap=True
    tf.margin_left=tf.margin_right=Inches(.14); tf.margin_top=Inches(.08); tf.margin_bottom=Inches(.04)
    for i,line in enumerate(text.split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph(); p.text=line
        p.font.name=font; p.font.size=Pt(size); p.font.bold=bold; p.font.color.rgb=RGBColor.from_string(color)
        p.space_after=Pt(10)
    return sh

def parse(raw):
    # title | three key explanations | focal equation / diagram labels | presenter explanation
    a=[v.strip() for v in raw.split('|')]
    assert len(a)==4, a
    return dict(title=a[0], bullets=a[1].split('~'), focal=a[2], note=a[3])

def make_deck(wi,week):
    out=ROOT/week['folder']; out.mkdir(parents=True,exist_ok=True)
    prs=Presentation(); prs.slide_width=Inches(13.333); prs.slide_height=Inches(7.5)
    accent=COLORS[wi-1]; records=[]
    def slide(title,bullets,focal,note,section='핵심',kind='normal',time=''):
        s=prs.slides.add_slide(prs.slide_layouts[6]); s.background.fill.solid(); s.background.fill.fore_color.rgb=RGBColor.from_string(BG)
        box(s,.38,.25,10,.35,f'AIM  /  DEEP LEARNING STUDIO  /  WEEK {wi:02d}  /  {section}',11,accent,bold=True)
        box(s,.5,.86,12.25,.95,title,29,bold=True)
        if kind=='cover':
            box(s,.65,2.15,7.3,3.85,'\n'.join(bullets),22)
            box(s,8.5,2.05,3.85,2.65,focal,32,WHITE,accent,True)
        elif kind=='visual':
            _,note=draw(s,wi,int(focal),accent,box)
        elif focal.startswith('FLOW:'):
            labels=focal[5:].split('>'); gap=.22; width=(11.95-gap*(len(labels)-1))/len(labels)
            for j,label in enumerate(labels):
                box(s,.65+j*(width+gap),2.02,width,1.12,label.replace(';','\n'),17,WHITE,accent,True)
            for j,b in enumerate(bullets): box(s,.72,3.55+j*.8,11.9,.72,f'{j+1:02d}   {b}',20)
        elif focal.startswith('CODE:'):
            for j,b in enumerate(bullets): box(s,.62,2.05+j*1.12,5.6,1.03,b,21)
            box(s,6.5,2.05,6.12,3.98,focal[5:].replace(';;','\n'),17,WHITE,INK,font='Consolas')
        else:
            for j,b in enumerate(bullets): box(s,.7,1.98+j*.87,11.92,.83,f'{j+1:02d}   {b}',22)
            box(s,.76,4.95,11.82,1.03,focal,22,WHITE,accent,True)
        box(s,.68,6.55,11.8,.46,time or '생각하기 → 손으로 계산하기 → 코드로 검증하기',12,MUTED)
        box(s,.5,7.04,11,.23,week['title']+'  •  2026 가을학기  •  PyTorch / Colab',9,MUTED)
        box(s,12.14,6.98,.7,.35,f'{len(prs.slides):02d}',12,accent,bold=True)
        s.notes_slide.notes_text_frame.text=f'[{section}] {time}\n\n{note}\n\n진행 팁: 설명 후 5~10초 기다리고, 한 명에게 근거를 포함한 답을 요청한다. 코드 결과의 수치는 실행 환경과 시드에 따라 달라질 수 있다.'
        records.append(dict(n=len(prs.slides),title=title,section=section,time=time,bullets=bullets,focal=focal,note=note))
    slide(week['title'],week['goals'],week['tag'],week['intro'],'오프닝','cover','00–02분 · 이번 주에 만들 결과물 확인')
    slide('90분 수업 지도', ['00–05 도입 · 05–20 개념 · 20–35 수식과 손계산','35–50 구현 설계 · 50–75 실습 · 75–85 분석','85–90 회고 · 부록과 자율 과제는 수업 이후'], '90 MIN  /  개념 45 + 실습 25 + 분석·회고 20', '시작 전에 실습의 환경 셀과 데이터 다운로드를 실행해 둔다. 핵심 구간의 종료 시간을 지키고 심화 질문은 부록으로 연결한다. 실습은 실행만 하는 시간이 아니라 예측, 관찰, 설명까지 포함한다.','운영',time='02–05분 · 실습 노트북 열기')
    sections=[('개념','05–20분'),('수식','20–35분'),('설계','35–50분'),('실습','50–75분'),('분석','75–85분'),('회고','85–90분'),('부록','수업 외 심화')]
    for key,time in sections:
        for raw in week[key]:
            d=parse(raw); slide(**d,section=key,time=time + (' · practice.ipynb와 함께 진행' if key=='실습' else ''))
    refs=week['refs']
    family='cnn' if wi<3 else 'att' if wi<5 else 'vae'
    for vi in range(4):
        slide(TITLES[family][vi],[],str(vi),'시각 자료는 수식과 텐서의 관계를 복습하는 보충 자료다. 설명용 숫자와 실제 학습 결과를 구분한다. 각 도식의 축, 조건, 합산 방향을 참가자가 직접 설명하게 한다.',section='시각 보충',kind='visual',time='수업 외 복습 · 해당 개념 설명 시 함께 활용 가능')
    slide('읽을거리와 출처', [r[0] for r in refs[:3]],'원문 링크는 발표자 노트와 주차별 README에 수록', '\n'.join(f'{name}\n{url}' for name,url in refs)+'\n\n슬라이드 문장·도식과 실습 코드는 이 수업을 위해 새로 작성했다. 원문 그림을 복제하지 않았다.','참고')
    prs.save(out/'lecture.pptx')
    guide=[f'# {wi}주차 — {week["title"]}', '', '## 목표', *['- '+x for x in week['goals']], '', '## 준비와 운영',week['intro'], '', '수업은 PPT의 핵심 구간으로 90분입니다. 부록은 사전 읽기, 질문 대응, 추가 세션에 사용합니다. 실습 노트북은 위에서 아래로 실행하며, 과제는 자율입니다.', '', '## 파일', '- `lecture.pptx`: 발표자 노트 포함 강의안', '- `practice.ipynb`: 설명·실행 코드·관찰 질문이 있는 독립 실행 실습', '- `homework_optional.ipynb`: 독립 실행 준비 코드 + 단계별 자율 과제', '- `leader_solution.ipynb`: 리더용 과제 해설 및 실행 가능한 참고 구현', '- `leader_guide.md`: 슬라이드별 설명과 질문 답안', '', '## 참고 문헌', *[f'- [{n}]({u})' for n,u in refs]]
    (out/'README.md').write_text('\n'.join(guide),encoding='utf-8')
    detail=[f'# {wi}주차 리더 진행 가이드', '', 'PPT의 발표자 노트와 동일한 설명을 검색 가능한 형태로 제공합니다. 정확도·손실의 고정 정답은 없으며 실습 결과를 직접 비교합니다.', '']
    for r in records: detail.extend([f'## {r["n"]:02d}. {r["title"]}',f'**{r["section"]} · {r["time"]}**','',r['note'],''])
    (out/'leader_guide.md').write_text('\n'.join(detail),encoding='utf-8')
    (ROOT/'_제작소스'/f'slides_week{wi:02d}.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    return len(prs.slides)

if __name__=='__main__':
    print({w['folder']:make_deck(i,w) for i,w in enumerate(WEEKS,1)})
