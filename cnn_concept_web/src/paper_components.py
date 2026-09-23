"""Original teaching diagrams and layouts; paper screenshots stay unaltered."""
from html import escape as e

PAPER_URL = 'https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf'

def cards(items, columns=3):
    return f'<div class="reading-cards cols-{columns}">' + ''.join(
        f'<article class="reading-card"><span class="card-index">{i:02d}</span><h3>{title}</h3><p>{body}</p></article>'
        for i, (title, body) in enumerate(items, 1)) + '</div>'

def prose(*items):
    return '<div class="reading-prose">' + ''.join(f'<p>{x}</p>' for x in items) + '</div>'

def split(left, right, ratio='normal'):
    return f'<div class="reading-split {ratio}"><div>{left}</div><div>{right}</div></div>'

def capture(name, page, label, height=250):
    return (f'<figure class="paper-capture"><div class="capture-top"><span>ORIGINAL PAPER</span>'
            f'<a href="{PAPER_URL}#page={page}" target="_blank" rel="noopener">p. {page} · {label} ↗</a></div>'
            f'<img src="assets/alexnet/{name}.png" alt="AlexNet 원문 {label}, {page}페이지 캡처" '
            f'style="max-height:{height}px"><figcaption>원문 PDF 직접 캡처 · 주변 본문/캡션 생략 · 아래 설명은 수업용 해설</figcaption></figure>')

def label(text, tone=''):
    return f'<div class="reading-label {tone}">{text}</div>'

def callout(title, body):
    return f'<div class="reading-callout"><strong>{title}</strong><p>{body}</p></div>'

def formula(main, description):
    return f'<div class="formula-panel"><div class="formula">{main}</div><p>{description}</p></div>'

def table(headers, rows, caption=''):
    return '<div class="reading-table-wrap"><table class="reading-table"><thead><tr>'+''.join(f'<th>{h}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{c}</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table>'+ (f'<p class="table-note">{caption}</p>' if caption else '')+'</div>'

def flow(items):
    return '<div class="reading-flow">' + ''.join(f'<div><span>{i:02d}</span><b>{a}</b><small>{b}</small></div>' for i,(a,b) in enumerate(items,1))+'</div>'

def svg_start(label, width=620, height=310):
    return f'<svg role="img" aria-label="{e(label)}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg"><defs><marker id="arr-{label}" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#087e76"/></marker></defs>'

def text(x,y,value,size=16,color='#1d302d',anchor='start',weight=550):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{e(str(value))}</text>'

def box(x,y,w,h,fill='#e2e9dc',stroke='none',rx=8):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'

def svg(kind):
    s=svg_start(kind)
    if kind=='convolution':
        s+=text(20,30,'작은 영역을 보고, 같은 질문을 옮긴다',21)
        for r in range(5):
            for c in range(5):
                s+=box(25+c*34,60+r*34,31,31,'#087e76' if c>=2 else '#dce4d8',rx=2)
                s+=text(40+c*34,81+r*34,1 if c>=2 else 0,14,'#fff' if c>=2 else '#61716b','middle')
        s+=box(22,57,104,104,'none','#e9643d',3)
        s+=text(225,142,'×',32)
        for r in range(3):
            for c in range(3):
                s+=box(270+c*35,81+r*35,32,32,'#f0dacb',rx=2)+text(286+c*35,103+r*35,[-1,0,1][c],16,anchor='middle')
        s+=text(403,142,'=',32)+box(459,86,115,83,'#087e76')+text(516,143,'3',38,'#fff','middle')
        s+=text(25,254,'입력의 3 × 3 영역',16)+text(273,216,'필터의 가중치',16)+text(450,215,'한 위치의 반응',16)
        s+=text(25,289,'수업용 손계산 · 실제 학습에서는 필터 값 자체가 바뀐다.',14,'#61716b')
    elif kind=='channels':
        for k,(name,color) in enumerate([('R','#bc6753'),('G','#488879'),('B','#618294')]):
            x=35+k*66;y=90-k*12
            s+=box(x,y,110,110,color)+text(x+55,y+60,name,24,'white','middle')
        s+=text(330,134,'→',40,'#087e76')
        for k in range(4):
            s+=box(420+k*19,87-k*10,98,110,['#aec8b5','#80ad9a','#488879','#087e76'][k])
        s+=text(30,35,'RGB는 입력 채널, 필터 개수는 출력 채널',21)
        s+=text(140,251,'입력: 3개 채널',18,anchor='middle')+text(480,251,'출력: 여러 특징 맵',18,anchor='middle')
        s+=text(30,290,'하나의 11 × 11 × 3 필터 → 한 출력 채널',16,'#61716b')
    elif kind=='relu':
        s+=text(18,27,'값을 통과시키는 규칙이 학습을 바꾼다',21)
        s+='<path d="M60 200 H285 M140 240 V55 M340 200 H570 M425 240 V55" stroke="#a6b2a6" fill="none"/>'
        s+='<path d="M65 200 H140 L270 70" stroke="#087e76" stroke-width="5" fill="none"/>'
        s+='<path d="M345 254 C400 254 400 210 425 200 C450 190 450 140 565 140" stroke="#e9643d" stroke-width="5" fill="none"/>'
        s+=text(170,282,'ReLU',20,'#087e76','middle')+text(465,282,'포화되는 활성화의 예',18,'#e9643d','middle')
        s+=text(160,91,'양수 구간: 기울기 1',13,'#087e76')+text(465,117,'변화가 작아짐',13,'#e9643d')
    elif kind=='pool':
        for r in range(4):
            for c in range(5):
                s+=box(30+c*43,63+r*43,39,39,'#e2e9dc',rx=3)+text(50+c*43,89+r*43,[1,2,4,1,0,0,7,3,2,1,1,5,2,6,3,0,1,2,1,0][r*5+c],17,anchor='middle')
        s+=box(26,59,129,129,'none','#e9643d',3)+box(112,59,129,129,'none','#087e76',3)
        s+=text(294,144,'→',40,'#087e76')+box(383,81,83,83,'#e9643d')+box(473,81,83,83,'#087e76')
        s+=text(424,136,'7',34,'white','middle')+text(514,136,'6',34,'white','middle')
        s+=text(28,29,'3 × 3 창을 2칸씩 이동: 영역이 겹친다',21)
        s+=text(30,275,'수업용 예제 · 같은 채널 안에서 주변 반응을 요약',16,'#61716b')
    elif kind=='dropout':
        s+=text(20,28,'일부 동료가 없어도 쓸 수 있는 단서',21)
        for col in range(3):
            x=95+col*205
            for r in range(4):
                active=col==2 or (r+col)%3!=0
                for rr in range(4):
                    if col<2:
                        s+=f'<line x1="{x+13}" y1="{80+r*43}" x2="{x+192}" y2="{80+rr*43}" stroke="#cbd4c8" stroke-width="1"/>'
                s+=f'<circle cx="{x}" cy="{80+r*43}" r="14" fill="{"#087e76" if active else "#d5dacf"}"/>'
                if not active:s+=text(x,86,'×',21,'#8a958d','middle')
            s+=text(x,284,['학습 시도 A','학습 시도 B','평가: 모두 사용'][col],16,anchor='middle')
    elif kind=='parameters':
        s+=text(24,30,'공간 위치마다 새 가중치가 필요한가?',22)
        s+=box(25,60,570,87,'#e2e9dc')+text(48,91,'같은 필터를 재사용하는 합성곱',18)+text(48,129,'(3 × 3 × 1 + 1) × 8 = 80',27,'#087e76')
        s+=box(25,167,570,87,'#f0dacb')+text(48,198,'같은 출력 크기에 완전연결을 사용한다면',18)+text(48,237,'(28 × 28 + 1) × (28 × 28 × 8)',24,'#ae502e')
        s+=text(25,289,'수업용 비교 · 두 모델의 성능이 같다는 뜻은 아니다.',14,'#61716b')
    elif kind=='hierarchy':
        names=[('입력','픽셀'),('초기 층','국소 반응'),('중간 층','반응의 조합'),('후반 층','분류에 유용한 표현')]
        for i,(a,b) in enumerate(names):
            x=12+i*153
            s+=box(x,72,135,139,'#087e76' if i==3 else '#e2e9dc')
            s+=text(x+67,122,a,24,'white' if i==3 else '#1d302d','middle')
            s+=text(x+67,168,b,12,'white' if i==3 else '#61716b','middle')
        s+=text(20,28,'뒤쪽 층은 앞쪽 층의 반응을 입력으로 받는다',21)
        s+=text(20,265,'층을 쌓으면 한 반응에 영향을 주는 입력 범위도 커진다.',16)
        s+=text(20,293,'특정 채널에 사람이 아는 부품 이름이 반드시 붙는 것은 아니다.',14,'#61716b')
    return '<div class="original-diagram">'+s+'</svg><span>수업용 재구성 · 논문 원문 그림과 구별</span></div>'

def render(p, num, total):
    source=f'<a href="{PAPER_URL}#page={p["paper_page"]}" target="_blank" rel="noopener">Krizhevsky et al. · NeurIPS 2012 · {p["locator"]} ↗</a>'
    phases=['읽기 준비','초록·서론','문제 정의','핵심 개념','수식 확인','실험 분석','결론','나의 정리']
    progress=''.join(f'<span class="{"current" if i==p["phase"] else ""}">{v}</span>' for i,v in enumerate(phases))
    ask=f'<aside class="ask"><strong>{p.get("ask_label","함께 읽기")}</strong>{p["ask"]}</aside>' if p['ask'] else ''
    return f'''<section class="sheet paper-sheet {p.get('theme','')}" id="w2-p{num:02d}" data-page="{num}" data-week="2">
<div class="masthead"><span class="smallmark">AIM <span style="font-weight:350">/ PAPER READING</span></span><span class="chapter">CNN · WEEK 02</span><span>ALEXNET / 2012 → 2026</span></div>
<div class="reading-progress">{progress}</div>
<header class="page-head"><div class="kicker">{e(p['kicker'])} <span class="page-minutes">{p['minutes']} MIN</span></div><h2>{p['title']}</h2><p class="lead">{p['lead']}</p></header>
<div class="paper-body">{p['body']}</div>
<div class="takeaway"><span>Read &amp; think</span><p>{p['takeaway']}</p></div>{ask}
<footer class="footer"><span class="sources">{source}</span><span class="folio">{num:02d} / {total}</span></footer></section>'''
