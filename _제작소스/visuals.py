from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR

TITLES={
 'cnn':['시각 보충 · 패치와 필터의 곱','시각 보충 · 해상도와 채널의 흐름','시각 보충 · 혼동 행렬 읽기','시각 보충 · 공정한 실험의 두 경로'],
 'att':['시각 보충 · 미래를 가리는 삼각형','시각 보충 · head 분할과 결합','시각 보충 · next-token shift','시각 보충 · temperature와 확률'],
 'vae':['시각 보충 · 학습과 생성의 두 경로','시각 보충 · 재매개화의 gradient 경로','시각 보충 · reduction의 단위','시각 보충 · 조건부 생성 격자의 축']}

def draw(slide,wi,idx,accent,box):
    family='cnn' if wi<3 else 'att' if wi<5 else 'vae'
    def b(x,y,w,h,t,size=20,fill=None,color='13243A'):
        return box(slide,x,y,w,h,t,size,color,fill)
    def link(x1,y1,x2,y2):
        sh=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(y1),Inches(x2),Inches(y2))
        sh.line.color.rgb=RGBColor.from_string(accent); sh.line.width=Pt(2)
    def grid(values,x,y,cell=.65,highlight=None):
        for r,row in enumerate(values):
            for c,v in enumerate(row):
                active=highlight(r,c) if highlight else False
                b(x+c*cell,y+r*cell,cell-.04,cell-.04,str(v),18,accent if active else 'E3EAF4','FFFFFF' if active else '13243A')
    if family=='cnn' and idx==0:
        b(.8,1.9,3,.5,'입력 패치'); grid([[1,2],[3,4]],1.2,2.65,.9)
        b(3.4,2.95,.5,.6,'×',32); b(4.05,1.9,3,.5,'공유 필터'); grid([[1,0],[0,-1]],4.5,2.65,.9)
        b(6.7,2.95,.5,.6,'=',32); b(7.55,2.45,4.7,1.7,'1×1 + 2×0\n+ 3×0 + 4×(−1) = −3',22,'E3EAF4')
        b(.95,5.05,11.3,1,'bias=1을 더하면 −2  /  같은 필터를 다음 패치로 이동한다.',23,accent,'FFFFFF')
        note='숫자 4개씩을 직접 곱해 합한다. PyTorch의 cross-correlation 방식이며 필터를 뒤집지 않는다. 출력 한 칸의 계산과 전체 특징 맵을 구분한다.'
    elif family=='cnn' and idx==1:
        stages=[('1 × 28 × 28',1.6,2.2),('16 × 14 × 14',1.3,2.5),('32 × 7 × 7',1.,2.8),('64',.7,3.1),('10 logits',.5,3.3)] if wi==1 else [('3 × 32 × 32',1.6,2.2),('32 × 16 × 16',1.3,2.5),('64 × 8 × 8',1.,2.8),('GAP: 64',.7,3.1),('10 logits',.5,3.3)]
        for j,(label,h,y) in enumerate(stages):
            x=.8+j*2.45
            for offset in [0.12,.06,0]: b(x+offset,y-offset,1.6,h,'',fill='E3EAF4' if offset else accent)
            b(x-.2,4.7,2.4,.8,label,19)
            if j<4: link(x+1.8,3.6,x+2.3,3.6)
        b(.8,5.8,11.7,.5,'B는 모든 단계에서 유지된다. 채널 수와 공간 해상도를 서로 바꾸어 읽지 않는다.',18)
        note='도형 크기는 해상도 감소를 나타내는 개념도이며 실제 텐서 크기의 정밀한 비율은 아니다. 각 단계의 B,C,H,W를 말하게 한다.'
    elif family=='cnn' and idx==2:
        b(1,1.95,4,.55,'설명용 3-class confusion matrix',18)
        grid([[8,1,1],[2,6,2],[0,3,7]],1.5,2.8,.85,lambda r,c:r==c)
        b(1.5,2.35,3,.4,'예측 A      B      C',16); b(.55,3.05,.8,2.6,'정답\nA\nB\nC',17)
        b(5.15,2.5,6.8,1.5,'B recall = 6 / (2+6+2) = 0.60\nB precision = 6 / (1+6+3) = 0.60',22,'E3EAF4')
        b(5.15,4.35,6.8,1.45,'대각선: 정답\n행 합: 실제 클래스 표본 수\n열 합: 그 클래스로 예측한 수',21)
        note='학습 결과가 아닌 계산 설명용 숫자다. 이 예시에서 precision과 recall이 우연히 같지만 분모는 다르다. B행에서 B를 A로 2개, C로 2개 혼동했다.'
    elif family=='cnn':
        for y,label,change in [(2.15,'A · 기준','고정 전처리'),(4.25,'B · 변경','train 증강만 추가')]:
            for j,t in enumerate(['같은 분할',change,'같은 모델·예산','같은 validation']):
                b(.8+j*3.05,y,2.8,1,t,20,accent if j==1 else 'E3EAF4','FFFFFF' if j==1 else '13243A')
                if j<3: link(3.63+j*3.05,y+.5,3.82+j*3.05,y+.5)
            b(.8,y-0.5,4,.45,label,18)
        b(.85,5.9,11.5,.5,'기법의 이름보다 무엇을 고정했고 무엇을 바꿨는지 먼저 기록한다.',20)
        note='두 경로의 차이를 한 칸으로 한정한 통제 실험 도식이다. seed를 맞추더라도 실제 난수 호출 흐름과 GPU 비결정성은 별도로 고려해야 한다.'
    elif family=='att' and idx==0:
        grid([['1' if c<=r else '0' for c in range(5)] for r in range(5)],1,2.05,.75,lambda r,c:c<=r)
        b(5.55,2.25,6.7,1.5,'행 = query / 열 = key\n1 = 허용 / 0 = 차단 (논리 mask)\nsoftmax 후 각 행의 확률 합 = 1',21,'E3EAF4')
        b(5.55,4.15,6.7,1.5,'위치 2는 0·1·2만 참조한다.\n입력의 현재 토큰은 허용하고\n그 다음 토큰을 정답으로 예측한다.',21)
        note='0-based index에서 세 번째 행을 가리킨다. 대각선을 차단하지 않는다. 첫 행도 자기 자신을 볼 수 있어 all-masked softmax NaN을 피한다.'
    elif family=='att' and idx==1:
        b(.8,2.3,2.4,1.4,'입력\n[B,T,64]',23,accent,'FFFFFF')
        for j in range(4):
            b(4.2,1.85+j*.92,4.4,.72,f'head {j+1}  ·  [B,T,16]',20,'E3EAF4')
            link(3.25,3.,4.12,2.2+j*.92); link(8.65,2.2+j*.92,9.5,3.)
        b(9.6,2.3,2.7,1.5,'concat + W_O\n[B,T,64]',22,accent,'FFFFFF')
        b(.9,5.85,11.6,.55,'head 축을 복원할 때 transpose → contiguous → view 순서를 확인한다.',20)
        note='4개 head로 나누어도 최종 D는 64다. 실제 Q,K,V는 각각 [B,4,T,16]으로 계산한다. 이 도식은 head별 결과 결합을 보여 준다.'
    elif family=='att' and idx==2:
        labels=['t','h','e',' ','c','a','t']
        b(.7,2.2,1.8,.55,'원본',20); grid([labels],2.4,2.1,.85)
        b(.7,3.5,1.8,.55,'입력 x',20); grid([labels[:-1]],2.4,3.4,.85)
        b(.7,4.8,1.8,.55,'정답 y',20); grid([labels[1:]],2.4,4.7,.85)
        b(8.8,3.15,3.65,1.9,'같은 열에서\nx_t → y_t\n정답은 원본의 다음 위치',22,'E3EAF4')
        note='입력 길이 T에 대해 원본 토큰 T+1개가 필요하다. 다음 토큰 정답은 입력의 바로 다음 열이므로 causal mask가 없으면 정답을 볼 수 있다.'
    elif family=='att':
        import math
        for j,tau in enumerate([.5,1.,2.]):
            vals=[math.exp(x/tau) for x in [2,1,0]]; probs=[v/sum(vals) for v in vals]
            x=.9+j*4.15; b(x,1.95,3.6,.5,f'τ={tau}  / logits=[2,1,0]',18)
            for k,p in enumerate(probs):
                b(x+k*1.05,5-p*2.55,.78,p*2.55,'',fill=accent)
                b(x+k*1.05-.08,5.1,1,.4,f'{p:.2f}',16)
        b(.9,6,11.6,.5,'설명용 확률 계산. 온도는 모델 가중치를 바꾸지 않고 선택 분포를 바꾼다.',19)
        note='실제 학습 모델의 결과가 아닌 설명용 logits의 정확한 softmax 계산이다. 온도를 높이면 분포가 완만해진다. τ=0은 허용하지 않는다.'
    elif family=='vae' and idx==0:
        for y,labels in [(2.3,['입력 x','encoder q(z∣x)','z 샘플 또는 μ','decoder → 재구성']), (4.6,['입력 이미지 없음','prior N(0,I)','z 샘플','decoder → 새 생성'])]:
            for j,t in enumerate(labels):
                b(.8+j*3.06,y,2.8,1.12,t,20,accent if j==1 else 'E3EAF4','FFFFFF' if j==1 else '13243A')
                if j<3:link(3.62+j*3.06,y+.56,3.84+j*3.06,y+.56)
        b(.8,1.78,5,.45,'재구성 경로: 입력 정보 사용',18); b(.8,4.06,5,.45,'prior 생성 경로: 특정 입력 없음',18)
        note='재구성과 prior 샘플을 같은 용어로 부르지 않는다. CVAE라면 두 경로의 encoder/decoder에 라벨 y가 추가된다. μ 재구성은 확률 평균 그림이며 MC 평가와 구분한다.'
    elif family=='vae' and idx==1:
        b(.8,2.8,2.7,1.1,'encoder',25,accent,'FFFFFF')
        b(4.1,2.05,2.7,.9,'μ(x)',25,'E3EAF4'); b(4.1,3.6,2.7,.9,'log σ²(x)',25,'E3EAF4')
        b(7.45,2.8,4.7,1.25,'z = μ + exp(½logvar) ⊙ ε',21,accent,'FFFFFF')
        link(3.5,3.35,4.08,2.5); link(3.5,3.35,4.08,4.05)
        link(6.85,2.5,7.4,3.2); link(6.85,4.05,7.4,3.7)
        b(7.5,4.65,4.6,.75,'ε ~ N(0,I)  ·  외부 잡음',22,'E3EAF4'); link(9.8,4.6,9.8,4.1)
        b(.8,5.75,6.4,.65,'gradient는 μ·logvar 경로를 따라 encoder로 흐른다.',19)
        note='샘플링 잡음 ε를 외부에서 뽑되 μ와 표준편차로 변환하는 과정은 미분 가능하다. logvar를 exp한 값은 분산이므로 표준편차에는 0.5가 필요하다.'
    elif family=='vae' and idx==2:
        b(.8,2.1,5.5,1.25,'reconstruction\n[B,1,28,28] → 픽셀 합 → [B]',22,'E3EAF4')
        b(7,2.1,5.5,1.25,'KL\n[B,Z] → latent 합 → [B]',22,'E3EAF4')
        link(3.5,3.4,6.65,4.15);link(9.75,3.4,6.65,4.15)
        b(3.4,4.2,6.55,1.1,'mean_B(recon + β × KL)',25,accent,'FFFFFF')
        b(.9,5.85,11.5,.6,'recon을 픽셀 평균으로 바꾸면 상대 가중치가 784배 달라진다.',20)
        note='28×28 grayscale의 픽셀 수는 784다. recon만 평균으로 줄이고 KL을 그대로 두면 KL이 상대적으로 784배 강해진다. 배치 복제에 대한 loss 불변량을 확인한다.'
    else:
        b(.8,2,4,.5,'개념도 · 실제 생성 이미지 아님',18)
        for r in range(4):
            b(.8,2.8+r*.67,1.3,.55,f'y={r}',17)
            for c in range(5):b(2.45+c*1.85,2.8+r*.67,1.62,.56,f'dec(z{c}, {r})',16,accent if c==2 else 'E3EAF4','FFFFFF' if c==2 else '13243A')
        for c in range(5):b(2.45+c*1.85,2.15,1.6,.5,f'z{c} 고정',17)
        b(.85,6,11.5,.5,'아래로: 같은 z, 다른 y  /  옆으로: 같은 y, 다른 z',21)
        note='행 라벨을 0~9까지 확장하면 실습의 10×8 격자가 된다. 같은 z의 스타일이 모든 라벨에서 같다는 보장은 없다. 실제 조건 충실도는 생성 이미지를 확인해야 한다.'
    return TITLES[family][idx],note
