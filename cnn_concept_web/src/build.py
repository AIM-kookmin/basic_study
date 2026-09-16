from pathlib import Path
from html import escape
import json
from content import W1,W2,SOURCES
from diagrams import diagram
ROOT=Path(__file__).resolve().parents[1]
COURSE=ROOT.parent

def schedule(week):
    rows=([(15,'분류·인식·검출·분할 구분'),(15,'사람의 시각 단서와 맥락'),(30,'비유로 익히는 CNN 기초 용어'),(25,'실제 MNIST의 입력·학습·예측'),(5,'개념 연결과 출구 질문')] if week==1 else [(5,'지난주 구조 복원'),(15,'단서·지름길·데이터'),(15,'증강과 이동의 차이'),(15,'깊이·잔차·규제'),(15,'평가와 근거 확인'),(15,'데이터·시험지 설계 활동'),(10,'해설과 종합')])
    out='<div class="schedule">';start=0
    for duration,label in rows:
        out+=f'<div class="schedule-row"><span class="duration">{start:02d}–{start+duration:02d}</span><span class="label">{label}</span><span class="minutes">{duration}분</span></div>';start+=duration
    return out+'</div><p class="schedule-caption">코딩 실습은 별도 선택 확장입니다. 이 시간표는 개념 수업만으로 90분을 구성합니다.</p>'

def source_links(keys):
    return ' · '.join(f'<a href="{SOURCES[k][1]}" target="_blank" rel="noopener">{escape(SOURCES[k][0].split(" · ")[0])}</a>' for k in keys)

def render_page(p,week,num):
    mast=f'<div class="masthead"><span class="smallmark">AIM <span style="font-weight:350">/ VISUAL THINKING</span></span><span class="chapter">CNN · WEEK 0{week}</span><span>CONCEPT EDITION · 2026</span></div>'
    head=f'<header class="page-head"><div class="kicker">{escape(p["kicker"])}</div><h2>{escape(p["title"]).replace(chr(10),"<br>")}</h2><p class="lead">{p["lead"]}</p></header>'
    paragraphs='<div class="copy">'+''.join(f'<p>{x}</p>' for x in p['paragraphs'])+'</div>'
    foot=f'<footer class="footer"><span class="sources">'+(source_links(p['sources']) if p['sources'] else 'AIM / CNN 개념 스터디 · 설명용 예제와 도식')+f'</span><span class="folio">{num:02d} / 28</span></footer>'
    attr=f'class="sheet {p["layout"]}" id="w{week}-p{num:02d}" data-page="{num}" data-week="{week}"'
    if p['layout']=='cover':
        return f'<section {attr}>{mast}{head}{paragraphs}<div class="cover-art">{diagram(p["visual"])}</div><div class="cover-bottom">{p["takeaway"]}</div>{foot}</section>'
    if p['layout']=='agenda': visual=schedule(week)
    elif p['layout']=='references':
        visual='<ol class="reference-list">'+''.join(f'<li><a href="{SOURCES[k][1]}" target="_blank" rel="noopener">{i+1:02d} / {SOURCES[k][0]}</a><span class="url">{SOURCES[k][1]}</span></li>' for i,k in enumerate(p['sources']))+'</ol>'
    else:
        visual,caption=diagram(p['visual']);visual+=f'<figcaption>{caption}</figcaption>'
    takeaway=f'<div class="takeaway"><span>Remember</span><p>{p["takeaway"]}</p></div>'
    ask=f'<aside class="ask"><strong>함께 생각하기</strong>{p["ask"]}</aside>' if p['ask'] else ''
    return f'<section {attr}>{mast}{head}<div class="sheet-content">{paragraphs}<figure class="visual">{visual}</figure></div>{takeaway}{ask}{foot}</section>'

def main():
    articles=[];toc=[];manifest=[]
    W2[24]['layout']='statement'
    for week,pages in enumerate([W1,W2],1):
        articles.append(f'<div class="book" data-week="{week}" aria-label="CNN {week}주차">'+''.join(render_page(p,week,n) for n,p in enumerate(pages,1))+'</div>')
        toc.extend(f'<li><a href="#w{week}-p{n:02d}" data-week="{week}"><span>{week}주 · {n:02d}</span>{escape(p["title"].replace(chr(10)," "))}</a></li>' for n,p in enumerate(pages,1))
        manifest.extend(dict(week=week,page=n,title=p['title'],layout=p['layout'],sources=list(p['sources'])) for n,p in enumerate(pages,1))
        guide=[f'# CNN {week}주차 개념 수업 · 리더 가이드','', '새 PDF와 웹페이지의 페이지 번호를 따릅니다. 기존 코드 중심 PPT의 페이지 번호와 다릅니다. 90분은 관찰·개념 설명·종이 활동·토론으로 구성합니다. 노트북 실행은 별도 선택 활동입니다.','',
               '## 운영 원칙','- 먼저 그림을 보고 예상한 뒤 설명을 읽습니다.','- 인간 시각과 CNN의 대응은 한계를 포함해 말합니다.','- 설명용 숫자·가상 사례를 실제 모델 측정값으로 소개하지 않습니다.','- 활동 해설 페이지는 개인·짝 계산 후 공개합니다.','']
        for n,p in enumerate(pages,1):
            guide.extend([f'## {n:02d}. {p["title"].replace(chr(10)," ")}',p['notes'],'',('질문: '+p['ask']) if p['ask'] else '', '핵심: '+p['takeaway'],''])
        (ROOT/f'CNN_{week}주차_리더가이드.md').write_text('\n'.join(guide),encoding='utf-8')
    html='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="인간의 시각에서 출발해 CNN의 구조, 학습, 일반화를 이해하는 2주 개념 스터디. 읽기·관찰·토론으로 구성한 웹북과 PDF."><title>AIM · CNN — 보는 것에서 배우는 것으로</title><link rel="stylesheet" href="styles.css"></head><body data-week="1"><a class="skip-link" href="#reader">본문 바로가기</a><header class="toolbar"><a class="brand" href="index.html">AIM <span>VISUAL THINKING</span></a><nav aria-label="자료 탐색"><button type="button" data-week-select="1" aria-pressed="true">01 숫자 분류</button><button type="button" data-week-select="2" aria-pressed="false">02 배우는 법</button><button type="button" data-week-select="all" aria-pressed="false">전체</button><button type="button" id="open-toc">목차</button><a class="pdf-link" href="CNN_1주차_개념강의.pdf">1주차 PDF</a></nav></header><div class="welcome"><div><div class="edition">A CONCEPT-FIRST GUIDE TO CONVOLUTIONAL NEURAL NETWORKS</div><h1>보는 것에서, 배우는 것으로.</h1><p>각 90분 · 2주 · 개념과 시각적 사고를 위한 56페이지<br>필터를 움직이고, 단서를 바꾸고, 분류의 근거를 질문해 보세요.</p></div><div class="small-links"><a href="CNN_개념강의_통합.pdf">통합 PDF ↗</a><a href="README.md">자료 안내 ↗</a></div></div><main id="reader">'''+''.join(articles)+'''</main><p class="screen-help">← → 키로 페이지 이동 · 모바일에서는 위아래로 읽기 · PDF는 A4 가로형<br>설명용 위젯은 모델을 학습하거나 실제 분류 성능을 측정하지 않습니다.</p><div class="page-dock" aria-label="페이지 이동"><button id="previous" aria-label="이전 페이지">←</button><output id="position" aria-live="off">1 / 28</output><button id="next" aria-label="다음 페이지">→</button></div><dialog id="toc" aria-labelledby="toc-heading"><header><h2 id="toc-heading">두 번의 시각 수업</h2><button id="close-toc" aria-label="목차 닫기">×</button></header><ol class="toc-list">'''+''.join(toc)+'''</ol></dialog><script src="app.js"></script></body></html>'''
    (ROOT/'index.html').write_text(html,encoding='utf-8')
    (ROOT/'src/page_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Built 56 concept pages and 2 leader guides')

if __name__=='__main__':main()
