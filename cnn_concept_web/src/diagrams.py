from html import escape
import math
from itertools import count

C={'ink':'#1d302d','muted':'#61716b','line':'#bfc9bd','paper':'#f4f2e9','teal':'#087e76','orange':'#e9643d','pale':'#e2e9dc','blue':'#4158a0'}
IDS=count()
def text(x,y,s,size=18,fill='ink',anchor='start',weight=500):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{C.get(fill,fill)}" text-anchor="{anchor}" font-weight="{weight}">{escape(str(s))}</text>'
def rect(x,y,w,h,fill='pale',rx=8,stroke=None,extra=''):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{C.get(fill,fill)}"'+(f' stroke="{C.get(stroke,stroke)}" stroke-width="2"' if stroke else '')+f' {extra}/>'
def line(x,y,xx,yy,color='line',width=2,dash=''):
    return f'<path d="M{x},{y} L{xx},{yy}" fill="none" stroke="{C.get(color,color)}" stroke-width="{width}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>'
def arrow(x,y,xx,yy,color='teal'):
    norm=math.hypot(xx-x,yy-y); ux=(xx-x)/norm; uy=(yy-y)/norm
    ax,ay=xx-ux*8,yy-uy*8
    return line(x,y,xx,yy,color,2)+f'<path d="M{ax-uy*5},{ay+ux*5} L{xx},{yy} L{ax+uy*5},{ay-ux*5}" fill="none" stroke="{C[color]}" stroke-width="2"/>'
def circle(x,y,r,fill='teal',stroke=None):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{C.get(fill,fill)}"'+(f' stroke="{C.get(stroke,stroke)}" stroke-width="2"' if stroke else '')+'/>'
def cup(x,y,s=1,color='teal',rotate=0):
    return f'<g transform="translate({x} {y}) scale({s}) rotate({rotate} 60 60)" fill="{C.get(color,color)}"><path d="M14 30 H94 V74 Q92 105 56 105 Q18 105 14 74Z"/><path d="M91 40H106 Q134 42 126 71 Q119 91 92 83 V71 Q108 80 115 66 Q122 49 93 52Z"/><ellipse cx="54" cy="29" rx="40" ry="11" fill="{C['paper']}"/><ellipse cx="54" cy="31" rx="34" ry="7" fill="{C['ink']}"/><path d="M32 18 Q18 9 32 -5 M58 15 Q43 3 60 -12" fill="none" stroke="{C.get(color,color)}" stroke-width="3" opacity=".35"/></g>'
def bottle(x,y,s=1,color='orange'):
    return f'<g transform="translate({x} {y}) scale({s})" fill="{C.get(color,color)}"><path d="M40 8H70 V36 Q93 50 93 65V112Q93 121 83 121H26Q16 121 16 112V65Q16 50 40 36Z"/>{rect(39,0,32,12,"ink",3)}{rect(20,73,69,24,"paper",0)}</g>'
def cat(x,y,s=1,color='orange',pattern=False):
    pid=f'cat-{next(IDS)}'
    path='M14 66Q6 47 20 30L18 2L43 19Q57 15 72 20L98 3L95 34Q110 57 97 81Q80 103 54 103Q25 101 14 66Z'
    fill=C.get(color,color)
    defs=''
    if pattern:
        defs=f'<defs><pattern id="{pid}" width="18" height="18" patternUnits="userSpaceOnUse" patternTransform="rotate(-24)"><rect width="18" height="18" fill="{C["orange"]}"/><rect width="7" height="18" fill="{C["ink"]}"/></pattern></defs>'
        fill=f'url(#{pid})'
    return f'<g transform="translate({x} {y}) scale({s})">{defs}<path d="{path}" fill="{fill}"/>'+circle(37,52,4,'paper')+circle(76,52,4,'paper')+f'<path d="M50 66H64L57 74Z" fill="{C["paper"]}"/>'+line(57,74,44,81,'paper',2)+line(57,74,70,81,'paper',2)+'</g>'
def grid(vals,x,y,cell=35,mode='number',highlight=None):
    out=''
    for r,row in enumerate(vals):
        for c,v in enumerate(row):
            fill='paper'
            if mode=='binary': fill='ink' if v else 'pale'
            elif mode=='heat': fill='teal' if v>0 else 'orange' if v<0 else 'pale'
            out+=rect(x+c*cell,y+r*cell,cell-2,cell-2,fill,3,extra=f'data-r="{r}" data-c="{c}"')
            if mode!='pixel':out+=text(x+c*cell+(cell-2)/2,y+r*cell+cell*.65,v,min(18,cell*.44),'paper' if fill in ['ink','teal','orange'] else 'ink','middle',600)
    return out
def labelbox(x,y,w,title,sub='',fill='pale'):
    out=rect(x,y,w,68 if sub else 52,fill)+text(x+w/2,y+28,title,17,'paper' if fill in ['teal','orange','ink'] else 'ink','middle',650)
    if sub:out+=text(x+w/2,y+51,sub,12,'paper' if fill in ['teal','orange','ink'] else 'muted','middle')
    return out
def svg(body,title='',height=380):
    return f'<svg viewBox="0 0 560 {height}" role="img" aria-label="{escape(title)}" xmlns="http://www.w3.org/2000/svg"><title>{escape(title)}</title>{body}</svg>'

def diagram(kind):
    from mnist_diagrams import KINDS,render
    if kind in KINDS:
        return render(kind)
    if kind.startswith('mnist-photo-'):
        name=kind.removeprefix('mnist-photo-')
        captions={'samples':'실제 MNIST test 표본 · 열은 정답 0–9, 각 열의 세 표본',
          'pixels':'실제 MNIST test 표본 · 6×6 픽셀 밝기 확대',
          'filters':'실제 입력에 수작업 필터를 적용한 계산 · 학습된 필터 아님',
          'learning':'실측 학습 곡선 · train 6,000 / validation 1,000 · seed 17 · CPU',
          'prediction':'실제 test 표본의 모델 출력 · 6epoch 최종 모델',
          'errors':'실제 고정 test 오분류 앞 6장 · 전체를 대표하는 표본 아님',
          'features':'실제 학습된 첫 합성곱 층의 ReLU 반응'}
        return f'<img class="mnist-figure" src="assets/mnist_{name}.png" alt="{escape(captions[name])}">',captions[name]
    b=''; caption='수업용 개념도 · 실제 모델의 측정 결과가 아닙니다.'
    if kind.startswith('cover'):
        if kind=='cover-eye':
            b=f'<path d="M10 195 Q280 -95 550 195 Q280 475 10 195Z" fill="{C["pale"]}"/>'+circle(280,195,129,'teal')+circle(280,195,99,'ink')+circle(245,150,22,'paper')
            for r in range(9):
                for c in range(9):
                    if (r-4)**2+(c-4)**2<15:b+=rect(207+c*16,121+r*16,11,11,'orange' if (r+c)%4==0 else 'teal',1)
        else: b=cat(97,45,3.25,'orange',True)+text(280,421,'SHAPE  /  TEXTURE  /  EVIDENCE',15,'paper','middle',600)
        return svg(b,kind,460)
    if kind in ['cups','variations']:
        if kind=='cups':
            b=cup(132,75,2.4)+line(365,205,490,152,'orange')+text(493,147,'손잡이',18,'orange','end')+line(255,137,82,68,'orange')+text(52,59,'입구',18,'orange')+line(252,290,410,329,'orange')+text(414,350,'몸체',18,'orange')
        else:
            for i,(x,y,s,col,rot) in enumerate([(25,55,1.1,'teal',0),(214,25,1.6,'orange',0),(406,65,.95,'ink',-15),(54,232,.8,'blue',18),(241,204,1.15,'teal',0),(416,240,.85,'orange',0)]):b+=cup(x,y,s,col,rot)
            b+=rect(231,272,128,22,'paper',0)+text(280,196,'같은 범주 / 달라진 픽셀',15,'muted','middle')
    elif kind=='parts':
        b=cat(23,78,1.6)+text(105,294,'전체의 배치',16,'ink','middle')
        for x,y,r in [(337,124,9),(429,175,9)]:b+=circle(x,y,r,'teal')
        b+=f'<path d="M310 229L325 188L349 221 M436 81L451 32L472 80" fill="{C["orange"]}"/>'+text(394,294,'같은 부분, 다른 관계',16,'ink','middle')+line(260,50,260,325,'line',1,'5 7')
    elif kind=='pixels':
        b=cup(17,76,1.45)+arrow(193,171,237,171)
        vals=[[18,18,23,26,25],[19,38,178,191,25],[25,41,184,204,171],[29,39,163,186,151],[18,22,39,45,27]]
        for r,row in enumerate(vals):
            for c,v in enumerate(row):b+=rect(259+c*52,52+r*52,49,49,f'rgb({v},{v},{v})',2)+text(283+c*52,84+r*52,v,14,'#fff' if v<120 else '#172523','middle')
        b+=text(389,345,'한 채널을 확대한 숫자 배열',15,'muted','middle')
    elif kind in ['neighbors','template']:
        vals=[[0,0,1,1,0],[0,1,1,0,0],[0,1,0,0,0],[0,1,1,0,0],[0,0,1,1,0]]
        b=grid(vals,37,65,44,'binary')
        if kind=='neighbors':
            b+=rect(77,105,135,135,'none',4,'orange')+arrow(275,170,320,170)+grid([[1,1,0],[1,0,0],[1,1,0]],345,108,51,'binary')+text(143,322,'위치를 가진 이웃 관계',15,'muted','middle')+text(420,322,'일부 영역 확대',15,'muted','middle')
        else:
            vals2=[[0]+r[:-1] for r in vals];b+=grid(vals2,308,65,44,'binary')+text(143,322,'원본',16,'muted','middle')+text(415,322,'오른쪽 1칸 이동',16,'muted','middle')+arrow(272,175,296,175)
    elif kind=='orientation':
        for j,(angle,val) in enumerate([(0,.18),(45,.55),(90,.94),(135,.39)]):
            x=75+j*137;b+=circle(x,112,45,'pale')+f'<path d="M{x-31} 112H{x+31}" transform="rotate({angle} {x} 112)" stroke="{C["ink"]}" stroke-width="9"/>'+rect(x-21,300-val*115,42,val*115,'teal',2)+text(x,335,f'{angle}°',16,'muted','middle')
        b+=text(15,25,'자극 방향',13,'muted')+text(15,192,'예시 반응',13,'muted')
    elif kind=='receptive':
        for r in range(8):
            for c in range(8):b+=rect(20+c*33,45+r*33,30,30,'pale',2)
        b+=rect(82,106,102,102,'none',3,'orange')+line(187,108,428,145,'orange')+line(187,210,428,180,'orange')+circle(448,161,35,'teal')+text(445,170,'반응',17,'paper','middle')+text(141,337,'입력 중 영향을 주는 영역',15,'muted','middle')
    elif kind=='timeline1':
        b+=line(59,52,59,312,'teal',3)
        for y,year,title,sub in [(70,'1962','시각피질의 지역 반응','생물학적 관찰'),(172,'1980','Neocognitron','계층적 패턴 인식 모형'),(277,'1998','학습 가능한 문서 인식','합성곱 구조와 gradient 학습')]:
            b+=circle(59,y,9,'orange')+text(88,y+7,year,24,'teal',weight=750)+text(197,y+7,title,19,weight=600)+text(197,y+35,sub,14,'muted')
    elif kind=='principles':
        for j,(label,sub) in enumerate([('LOCAL','작게 연결'),('SHARED','같은 기준 재사용'),('HIERARCHICAL','반응들을 다시 조합')]):
            y=27+j*116;b+=text(25,y+24,f'0{j+1}',29,'orange',weight=300)+text(100,y+23,label,21,'teal',weight=800)+text(100,y+56,sub,18)+line(100,y+80,535,y+80)
    elif kind in ['convolution','conv-interactive','worksheet','worksheet-answer']:
        vals=[[0,0,1,1,1] for _ in range(5)];f=[[-1,0,1] for _ in range(3)]
        if kind=='convolution':
            b=grid([[0,0,1] for _ in range(3)],28,82,48,'binary')+text(201,163,'×',32,'orange','middle')+grid(f,236,82,48)+text(410,163,'=',32,'orange','middle')+text(483,174,'3',67,'teal','middle',750)+text(98,264,'현재 패치',16,'muted','middle')+text(309,264,'필터',16,'muted','middle')+text(280,324,'(−0+1) + (−0+1) + (−0+1) = 3',18,'ink','middle')
        else:
            b=grid(vals,15,65,40,'binary')+text(110,35,'입력 5 × 5',16,'muted','middle')+grid(f,269,70,33)+text(316,35,'공유 필터',16,'muted','middle')
            output=[[3,3,0] for _ in range(3)] if kind=='worksheet-answer' else [['?']*3 for _ in range(3)]
            if kind=='conv-interactive':output=[[3,3,0] for _ in range(3)]
            b+=grid(output,417,77,39,'heat' if kind in ['conv-interactive','worksheet-answer'] else 'number')+text(473,35,'출력 3 × 3',16,'muted','middle')
            b+=rect(13,63,120,120,'none',4,'orange',extra='class="focus-patch"')
            b+=arrow(223,129,251,129)+arrow(371,129,400,129)
            b+=text(280,321,'padding = 0  ·  stride = 1  ·  bias = 0',15,'muted','middle')
            if kind=='conv-interactive':
                controls='<div class="control"><label>필터 위치 <input class="conv-range" type="range" min="0" max="8" value="0" aria-label="합성곱 필터 위치"></label><output class="conv-output" aria-live="polite">행 1 · 열 1 → 반응 3</output></div>'
                caption='숫자는 위 입력으로 직접 계산한 값입니다. 필터는 위치가 바뀌어도 같습니다.'
                return '<div class="conv-demo">'+svg(b,kind)+controls+'</div>',caption
    elif kind=='channels':
        for j,(lab,col) in enumerate([('R','#d77764'),('G','#84a791'),('B','#6e8db8')]):b+=rect(35+j*14,67-j*9,135,170,col,5)+text(61+j*41,268,lab,20,col,'middle',700)
        b+=arrow(220,145,282,145)
        for j in range(4):b+=rect(310+j*38,48+j*18,92,135,'teal' if j%2==0 else 'ink',4)+line(320+j*38,130+j*18,390+j*38,80+j*18,'paper',6)
        b+=text(113,314,'입력의 색 채널',16,'muted','middle')+text(414,314,'출력의 특징 채널',16,'muted','middle')
    elif kind=='sharing':
        for x,y in [(42,72),(207,72),(372,72)]:b+=grid([[-1,0,1]]*3,x,y,38)
        b+=text(280,244,'9개의 값, 위치마다 재사용',26,'teal','middle',700)+text(280,291,'출력 위치가 늘어도 새 필터가 생기는 것은 아니다.',16,'muted','middle')
        for x in [90,255,420]:b+=line(x,196,280,217,'orange')
    elif kind=='relu':
        b=line(47,238,526,238,'line')+line(255,36,255,336,'line')+f'<path d="M51 238H255L461 64" fill="none" stroke="{C["teal"]}" stroke-width="6"/>'+text(483,267,'입력',15,'muted')+text(270,36,'출력',15,'muted')+text(115,215,'음수는 0',18,'orange')+text(380,137,'양수는 통과',18,'teal')+text(279,352,'ReLU(x) = max(0, x)',21,'ink','middle')
    elif kind=='pooling':
        b=grid([[1,4],[2,3]],29,89,70)+arrow(192,158,259,158)+labelbox(287,77,215,'MAX → 4','가장 큰 반응만 남김','teal')+labelbox(287,203,215,'AVERAGE → 2.5','전체 크기를 평균으로 요약')+text(104,289,'원래 위치 정보',17,'muted','middle')
    elif kind=='hierarchy':
        for j,(n,lab) in enumerate([(3,'한 층 / 3×3'),(5,'두 층 / 5×5'),(7,'세 층 / 7×7')]):
            cell=20;x=30+j*181; y=153-n*cell/2
            b+=grid([[1]*n for _ in range(n)],x,y,cell,'binary')+text(x+70,289,lab,16,'muted','middle')
        b+=text(280,347,'stride 1 · dilation 1의 3×3 합성곱을 연결한 예',14,'muted','middle')
    elif kind=='pipeline':
        b=cup(32,41,.9)
        for j in range(3): b+=rect(216+j*15,42-j*9,83,90,'pale' if j%2 else 'teal',3)
        b+=arrow(172,90,202,90)+arrow(358,90,388,90)+labelbox(399,51,127,'표현 조합','여러 층','ink')
        b+=line(460,137,460,195,'teal')+line(460,195,95,195,'teal')
        for j,(label,w) in enumerate([('컵',315),('병',127),('그릇',73)]):
            y=218+j*44;b+=text(39,y+19,label,16)+rect(93,y,w,27,'teal' if j==0 else 'pale',3)
        b+=text(440,281,'후보별',18,'muted','middle')+text(440,310,'점수',18,'muted','middle')
    elif kind=='softmax':
        for j,(v,p,lab) in enumerate([(2,.665,'컵'),(1,.245,'병'),(0,.09,'그릇')]):
            y=65+j*96;b+=text(34,y+29,lab,19)+text(143,y+32,v,37,'orange','middle',750)+arrow(182,y+17,245,y+17)+rect(274,y,p*300,37,'teal',4)+text(283+p*300,y+27,f'{p:.3f}',17,'ink')
        b+=text(143,364,'logits',15,'muted','middle')+text(380,364,'합이 1인 상대적 확률',15,'muted','middle')
    elif kind in ['learning','feedback']:
        b=labelbox(35,58,189,'현재 모델','필터와 분류 가중치','teal')+labelbox(331,58,194,'예측과 정답 비교','손실 측정')+arrow(234,92,318,92)+labelbox(331,252,194,'변화 방향 계산','gradient')+labelbox(35,252,189,'작은 가중치 수정','다음 예시에서 다시 계산','orange')+line(430,137,430,240,'teal')+arrow(320,287,235,287)+line(129,243,129,137,'teal')
        if kind=='feedback':b+=text(280,204,'연쇄 법칙으로 영향 전달',18,'ink','middle')
    elif kind=='texture':
        b=cat(24,78,1.85,'teal')+cat(318,78,1.85,'orange',True)+text(129,318,'윤곽 + 단색',16,'muted','middle')+text(415,318,'같은 윤곽 + 다른 무늬',16,'muted','middle')
    elif kind in ['shortcut','crossed-data']:
        for r in range(2):
            for c in range(2):
                x=38+c*271;y=25+r*167;fill='orange' if (kind=='shortcut' and c==0) or (kind=='crossed-data' and r==0) else 'teal'
                b+=rect(x,y,232,143,fill,5)
                b+=cup(x+50,y+24,.8,'paper') if c==0 else bottle(x+70,y+7,.9,'paper')
        b+=text(280,373,'라벨과 배경이 묶임' if kind=='shortcut' else '같은 물체를 서로 다른 배경에',17,'ink','middle')
    elif kind=='splits':
        for j,(x,w,lab,sub,col) in enumerate([(23,220,'TRAIN','가중치 학습','teal'),(250,137,'VAL','설정 선택','orange'),(394,143,'TEST','마지막 평가','ink')]):
            b+=rect(x,101,w,120,col,3)+text(x+w/2,149,lab,22,'paper','middle',800)+text(x+w/2,187,sub,15,'paper','middle')
        b+=text(280,290,'공유 원본·물체·장면이 split 경계를 넘는지 확인',17,'muted','middle')+line(400,253,520,253,'orange',3)
    elif kind=='augment-interactive':
        b=rect(30,28,500,222,'pale',8)+f'<g class="demo-object">{cup(192,67,1.55)}</g>'+rect(240,132,125,54,'paper',0,extra='class="occluder" opacity="0"')
        buttons='<div class="control transform-controls" role="group" aria-label="이미지 변환"><button type="button" data-transform="original" aria-pressed="true">원본</button><button type="button" data-transform="shift" aria-pressed="false">위치 이동</button><button type="button" data-transform="flip" aria-pressed="false">좌우 반전</button><button type="button" data-transform="occlude" aria-pressed="false">일부 가림</button></div><p class="transform-result" aria-live="polite">원본: 물체의 단서를 먼저 관찰하세요.</p>'
        for j in range(4):
            x=20+j*138
            b+=rect(x,275,122,92,'pale',4)
            small=cup(x+(26 if j==1 else 6),282,.64,'teal',0)
            if j==2:small=f'<g transform="translate({2*(x+61)} 0) scale(-1 1)">{small}</g>'
            b+=small
            if j==3:b+=rect(x+28,323,61,15,'paper',0)
            b+=text(x+61,388,['원본','이동','반전','가림'][j],13,'muted','middle')
        return '<div class="augment-demo">'+svg(b,kind,395)+buttons+'</div>','변환 관찰 도구입니다. 모델을 실행하거나 분류 결과를 측정하지 않습니다.'
    elif kind=='equivariance':
        for y in [57,220]:
            b+=rect(22,y,158,95,'pale')+cup(37+(30 if y>100 else 0),y+2,.67)+arrow(192,y+45,252,y+45)
            if y<100:
                b+=rect(272,y,243,95,'pale')+rect(302,y+19,16,55,'teal',1)+rect(352,y+19,16,55,'orange',1)+text(394,y+56,'반응도 이동',15)
            else:b+=labelbox(273,y+17,242,'최종 분류: 컵','','teal')
        b+=text(282,183,'위: 등변성   /   아래: 원하는 분류 불변성',15,'muted','middle')
    elif kind=='aliasing':
        for r,active in enumerate([3,4]):
            y=72+r*118
            for i in range(10): b+=rect(32+i*48,y,44,64,'teal' if i==active else 'pale',2)
            for i in range(0,10,2):b+=line(53+i*48,y-23,53+i*48,y-4,'orange',3)
            b+=text(273,y+95,'가는 선이 샘플 위치 사이에 놓임' if r==0 else '한 칸 이동하면 샘플 위치와 겹침',16,'muted','middle')
    elif kind in ['capacity','depth']:
        if kind=='capacity':
            for r,n in [(0,2),(1,5)]:
                for j in range(n):b+=rect(73+j*22,50+r*151-j*8,124,88,'teal' if j%2 else 'pale',3)
                b+=text(316,106+r*151,'적은 채널' if r==0 else '더 많은 채널',24,'ink')
        else:
            for row,n in enumerate([2,5]):
                for j in range(n):
                    b+=rect(36+j*101,75+row*144,76,62,'teal' if j%2==0 else 'pale',4)
                    if j<n-1:b+=arrow(118+j*101,106+row*144,130+j*101,106+row*144)
            b+=text(282,350,'더 많은 변환 단계 ≠ 자동으로 더 좋은 학습',17,'muted','middle')
    elif kind=='residual':
        b=labelbox(15,166,105,'x','','ink')+labelbox(212,166,136,'F(x)','','teal')+circle(436,192,28,'orange')+text(436,200,'+',27,'paper','middle')+arrow(128,192,201,192)+arrow(357,192,400,192)+arrow(468,192,547,192)+line(159,191,159,64,'orange')+line(159,64,436,64,'orange')+line(436,64,436,154,'orange')+text(305,43,'입력을 보존하는 경로',16,'orange','middle')+text(280,318,'y = x + F(x)',36,'ink','middle',600)
    elif kind=='normalization':
        for side in range(2):
            x=45+side*300
            b+=line(x,269,x+184,269)
            for j,h in enumerate([175,70,130,210] if side==0 else [118,77,101,133]):b+=rect(x+14+j*44,269-h,26,h,'orange' if side==0 else 'teal',2)
            b+=text(x+97,319,'조정 전' if side==0 else '평균·스케일 조정',16,'muted','middle')
        b+=arrow(255,164,318,164)
    elif kind=='dropout':
        for r in range(3):
            for c in range(5):
                x=72+c*104;y=70+r*104; dropped=(r,c) in [(0,1),(1,3),(2,0),(2,4)]
                b+=circle(x,y,22,'pale' if dropped else 'teal')
                if dropped:b+=line(x-13,y-13,x+13,y+13,'orange',3)+line(x-13,y+13,x+13,y-13,'orange',3)
        b+=text(280,351,'학습 중 무작위로 일부 활성화를 비움',17,'muted','middle')
    elif kind=='curves':
        b=line(57,304,530,304)+line(57,28,57,304)+text(43,22,'손실',14,'muted')+text(474,336,'학습 진행',14,'muted')
        b+=f'<path d="M67 65C113 199 201 260 516 285" fill="none" stroke="{C["teal"]}" stroke-width="5"/><path d="M67 93C151 242 246 242 309 203S443 109 516 93" fill="none" stroke="{C["orange"]}" stroke-width="5"/>'+text(425,274,'train',17,'teal')+text(392,88,'validation',17,'orange')+text(280,372,'설명용 합성 곡선 / 실제 학습 기록 아님',13,'muted','middle')
    elif kind=='loss-accuracy':
        for j,(lab,p,col,loss) in enumerate([('예측 A',.6,'teal','0.511'),('예측 B',.9,'orange','0.105')]):
            y=58+j*153;b+=text(34,y+25,lab,20)+rect(146,y,p*298,37,col,3)+text(151+p*298,y+27,f'{p}',18,'ink')+text(146,y+80,f'정답: 컵  ·  CE ≈ {loss}',17,'muted')
    elif kind=='confusion':
        vals=[[8,1,1],[2,6,2],[0,3,7]]
        for r,row in enumerate(vals):
            b+=text(130,117+r*79,['컵','병','그릇'][r],17,'ink','end')
            for c,v in enumerate(row):b+=rect(159+c*84,65+r*79,78,74,'teal' if r==c else 'pale',4)+text(198+c*84,112+r*79,v,29,'paper' if r==c else 'ink','middle',700)
        for c,t in enumerate(['컵','병','그릇']):b+=text(198+c*84,47,t,17,'muted','middle')
        b+=text(283,349,'행 = 실제 정답 / 열 = 예측',17,'muted','middle')+text(491,114,'8/10',20,'orange','middle')+text(491,193,'6/10',20,'orange','middle')+text(491,272,'7/10',20,'orange','middle')
    elif kind=='shift':
        b=rect(25,38,225,271,'pale')+cup(76,103,1.3)+rect(308,38,225,271,'ink')+cup(349,129,1.19,'#9b9580',-12)+text(133,352,'밝은 촬영실',18,'muted','middle')+text(422,352,'어두운 다른 장소',18,'muted','middle')
    elif kind=='heatmap':
        b=cat(121,29,2.8,'ink')
        for x,y,r,opacity in [(224,120,57,.48),(276,148,79,.32),(319,212,42,.4)]:b+=f'<circle cx="{x}" cy="{y}" r="{r}" fill="{C["orange"]}" opacity="{opacity}"/>'
        b+=text(280,358,'가상 heatmap / 특정 모델의 Grad-CAM 결과 아님',14,'muted','middle')
    elif kind=='intervention':
        for j in range(3):
            x=12+j*187;b+=rect(x,77,173,180,'orange' if j<2 else 'teal')+cup(x+23,111,1,'paper')
            if j==1:b+=rect(x+19,122,137,112,'ink',0)
            b+=text(x+86,307,['원본','물체 가림','배경 변경'][j],16,'muted','middle')
    elif kind=='ablation':
        for j,y in enumerate([68,244]):
            b+=labelbox(18,y,95,'x')+labelbox(212,y,132,'같은 F','동일 가중치')+arrow(123,y+26,197,y+26)+arrow(354,y+26,401,y+26)+circle(435,y+26,24,'orange' if j==0 else 'teal')+text(435,y+34,'+' if j==0 else '→',26,'paper','middle')
            if j==0:b+=line(157,y+26,157,y-37,'orange')+line(157,y-37,435,y-37,'orange')+line(435,y-37,435,y-4,'orange')
        b+=text(282,199,'body·초기화·분할·예산을 고정',17,'ink','middle')
    elif kind in ['experiment','experiment-answer']:
        b=rect(29,38,230,211,'pale')+rect(297,38,230,211,'pale')+cup(87,68,1)+bottle(353,61,1)+line(41,195,247,195,'ink',7)+line(308,165,517,165,'ink',7)+text(142,281,'컵 / 책상 / 낮',17,'muted','middle')+text(412,281,'병 / 선반 / 낮',17,'muted','middle')+text(280,350,'예산: 추가 사진 40장',24,'orange','middle',750)
    elif kind=='bridge':
        b=circle(126,166,94,'teal')+rect(344,70,179,189,'ink',8)+text(126,158,'인간의',25,'paper','middle')+text(126,196,'시각',25,'paper','middle')+text(434,178,'CNN',35,'paper','middle',750)+arrow(235,166,329,166)+text(280,312,'영감과 공통점 / 차이와 검증',19,'muted','middle')
    elif kind in ['recap1','report','next','glossary']:
        words={'recap1':['숫자','지역 패턴','공유와 조합','범주 점수','학습'],'report':['질문','비교 방법','관측 결과','한계','다음 실험'],'next':['무엇을 보나','무엇을 버리나','무엇을 가정하나','어떻게 배우나','어떻게 검증하나'],'glossary':['INPUT','FEATURES','DECISION','LEARNING']}[kind]
        for j,w in enumerate(words):b+=text(32,42+j*69,f'{j+1:02}',23,'orange')+text(106,42+j*69,w,28,'teal',weight=650)+line(108,62+j*69,530,62+j*69)
    else: b=text(280,190,'AIM / VISUAL THINKING',28,'teal','middle')
    return svg(b,kind),caption
