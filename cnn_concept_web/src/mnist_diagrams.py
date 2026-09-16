"""New native diagrams for task distinctions and human-to-computation analogies."""
from diagrams import text,rect,line,arrow,svg,cup,bottle,labelbox,C,grid

KINDS={'cover-mnist','task-family','task-outputs','digit-task','human-sevens','human-parts','human-context',
       'term-bridge','stride-padding','relu-pool','mnist-intro','mnist-split','mnist-architecture'}

def digit(x,y,value='7',scale=1,color='teal'):
    return f'<g transform="translate({x} {y}) scale({scale})">'+text(0,115,value,142,color,'start',750)+'</g>'

def render(kind):
    b=''
    if kind=='cover-mnist':
        b=rect(130,40,300,300,'teal',24)+digit(203,79,'7',1.65,'paper')
        for i in range(8):b+=rect(65+i*55,371,38,38,'orange' if i==6 else 'pale',5)
        return svg(b,'숫자 분류를 상징하는 표지 도식',460)
    if kind=='task-family':
        b=rect(20,25,520,310,'pale',20)+text(280,65,'객체 인식 · 넓은 표현',25,'ink','middle',700)
        for y,t,s in [(98,'분류 CLASSIFICATION','무엇인가 · 이미지의 라벨'),(173,'검출 DETECTION','무엇이 어디에 · 물체별 상자'),(248,'분할 SEGMENTATION','어떤 픽셀인가 · 영역')]:
            b+=labelbox(65,y,430,t,s,'teal' if y==98 else 'paper')
    elif kind=='task-outputs':
        for j,title in enumerate(['분류','검출','분할']):
            x=8+j*187;b+=rect(x,50,174,225,'pale',10)+text(x+87,32,title,22,'ink','middle',700)
            b+=cup(x+16,103,.7)+bottle(x+100,105,.6)
            if j==0:
                b+=rect(x+16,102,83,86,'none',3,'orange')+text(x+87,252,'잘라 낸 컵 → 컵',13,'ink','middle')
            elif j==1:
                b+=rect(x+12,100,88,90,'none',2,'orange')+rect(x+101,102,58,80,'none',2,'orange')
                b+=text(x+44,94,'컵',14,'orange','middle')+text(x+128,96,'병',14,'orange','middle')+text(x+87,252,'라벨 + 위치 상자',13,'ink','middle')
            else:
                b+=cup(x+16,103,.7,'orange')+bottle(x+100,105,.6,'blue')+text(x+87,252,'픽셀마다 영역',13,'ink','middle')
        b+=text(280,330,'예시는 과제별 정답 형태를 설명하는 도식',15,'muted','middle')
    elif kind=='digit-task':
        b=rect(25,80,150,150,'ink',15)+digit(62,80,'7',.85,'paper')+arrow(191,154,282,154)+labelbox(300,115,225,'예측 라벨 7','후보: 0 1 2 3 4 5 6 7 8 9','teal')
        b+=text(280,294,'한 장 → 후보 중 하나',26,'ink','middle',700)
    elif kind=='human-sevens':
        for j,(angle,w) in enumerate([(-10,10),(7,15),(0,9)]):
            b+=f'<g transform="translate({40+j*175} 85) rotate({angle} 60 80)"><path d="M10 25H112L45 188" fill="none" stroke="{C["teal"]}" stroke-width="{w}" stroke-linecap="round"/>'
            if j==2:b+=line(16,110,89,110,'orange',7)
            b+='</g>'
        b+=text(280,342,'굵기 · 기울기 · 획의 추가',21,'muted','middle')
    elif kind=='human-parts':
        b=digit(65,64,'3',1.4)+digit(352,64,'8',1.4,'orange')+arrow(242,173,324,173)
        b+=text(135,312,'열린 쪽',20,'teal','middle')+text(423,312,'닫힌 고리',20,'orange','middle')
    elif kind=='human-context':
        b=rect(24,46,512,110,'pale')+rect(24,210,512,110,'pale')
        b+=text(130,119,'0',66,'muted','middle')+text(430,119,'2',66,'muted','middle')
        b+=text(130,286,'H',66,'muted','middle')+text(430,286,'J',66,'muted','middle')
        b+=line(280,70,280,130,'teal',13)+line(280,235,280,295,'teal',13)
        b+=text(280,186,'같은 획, 다른 문맥',19,'orange','middle')
    elif kind=='term-bridge':
        for i,(a,c) in enumerate([('작은 질문','필터 / 가중치'),('답을 적은 지도','특징 맵 / 배열'),('여러 관점','채널 / 표현의 축'),('영향을 받는 범위','수용 영역 / 연결')]):
            y=30+i*85;b+=labelbox(15,y,216,a,fill='pale')+arrow(243,y+26,307,y+26)+labelbox(321,y,225,c,fill='teal')
    elif kind=='stride-padding':
        b=grid([[0]*5 for _ in range(5)],15,76,32)+rect(15,76,96,96,'none',2,'orange')
        b+=arrow(40,265,104,265)+text(94,302,'stride: 이동 간격',16,'muted','middle')
        b+=grid([[0]*7 for _ in range(7)],306,43,30)+rect(336,73,150,150,'none',2,'orange')
        b+=text(390,302,'padding: 둘레 추가',16,'muted','middle')
        b+=text(280,351,'영향 범위 · 이동 간격 · 경계 처리',19,'ink','middle')
    elif kind=='relu-pool':
        b=labelbox(24,40,226,'−2  /  0  /  3','입력 반응')+arrow(262,70,304,70)+labelbox(320,40,210,'0  /  0  /  3','ReLU','teal')
        b+=grid([[1,4],[2,3]],70,181,65)+arrow(230,243,315,243)+labelbox(335,213,166,'MAX = 4','2×2 요약','orange')
    elif kind=='mnist-intro':
        for j,n in enumerate('0123456789'):
            x=15+(j%5)*110;y=20+(j//5)*157
            b+=rect(x,y,94,124,'ink',12)+text(x+47,y+93,n,83,'paper','middle',750)
        b+=text(280,364,'실제 표본은 다음 페이지에서',16,'muted','middle')
    elif kind=='mnist-split':
        b=labelbox(15,25,530,'공식 train 60,000','클래스별 층화 선택 · seed 17','ink')
        for j,(title,sub) in enumerate([('학습 6,000','600 / class'),('검증 1,000','100 / class'),('미사용 53,000','수업 예산 밖')]):
            b+=labelbox(15+j*181,142,168,title,sub,'teal' if j==0 else 'pale')
        b+=labelbox(15,270,230,'공식 test 10,000','학습에 사용하지 않음')+arrow(258,302,300,302)+labelbox(316,270,229,'고정 평가 1,000','100 / class','orange')
    elif kind=='mnist-architecture':
        boxes=[('이미지','1 × 28 × 28'),('Conv + ReLU','8 × 28 × 28'),('Max pool','8 × 14 × 14'),('Conv + ReLU','16 × 14 × 14'),('Max pool','16 × 7 × 7'),('Flatten + Linear','10 logits')]
        for i,(t,s) in enumerate(boxes):
            x=8+(i%3)*185;y=57+(i//3)*166;b+=labelbox(x,y,170,t,s,'teal' if i<5 else 'orange')
            if i%3<2:b+=arrow(x+172,y+33,x+183,y+33)
        b+=line(463,135,463,184,'orange')+line(463,184,93,184,'orange')+arrow(93,184,93,211,'orange')
        b+=text(280,353,'학습할 파라미터 9,098개',21,'ink','middle',700)
    return svg(b,kind),'개념 설명용 도식 · 실제 MNIST 표본·측정 결과는 별도 표기'
