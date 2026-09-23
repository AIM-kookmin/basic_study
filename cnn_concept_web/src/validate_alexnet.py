"""Validate original-paper captures, reading order, timings and source coverage."""
from pathlib import Path
import hashlib
import json
import re
import fitz
from content import W2
from week2_alexnet_content import SCHEDULE
from prepare_alexnet import CROPS, URL

ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'assets/alexnet/provenance.json').read_text(encoding='utf-8'))
paper=ROOT/'_preview/alexnet_2012.pdf'
assert hashlib.sha256(paper.read_bytes()).hexdigest()==manifest['source_pdf_sha256']
assert len(W2)==22 and sum(p['minutes'] for p in W2)==90
assert [p['phase'] for p in W2]==sorted(p['phase'] for p in W2)
assert [sum(p['minutes'] for p in W2 if p['phase']==i) for i in range(len(SCHEDULE))]==[x[0] for x in SCHEDULE]
body=''.join(p['body'] for p in W2)
used=set(re.findall(r'assets/alexnet/([^\"]+\.png)',body))
assert len(used)==9
assert sum(p['phase']==1 for p in W2)==6
assert used <= {x['file'] for x in manifest['captures']}
with fitz.open(paper) as doc:
    for item in manifest['captures']:
        path=ROOT/'assets/alexnet'/item['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        rerender=doc[item['page']-1].get_pixmap(matrix=fitz.Matrix(4,4),clip=fitz.Rect(item['crop_pdf_points']),alpha=False)
        stored=fitz.Pixmap(path)
        assert rerender.samples==stored.samples, item['file']
    # Check the numerical evidence against selectable text in the primary source.
    results=doc[6].get_text()
    for term in ['25.7%', '17.0%', '18.2%', '15.3%', '26.2%', '7 CNNs*']:
        assert term in results,term
assert len(manifest['captures'])==12
assert '18.2%' in W2[16]['body'] and '검증' in W2[16]['body']
assert '15.3%' in W2[16]['body'] and '시험' in W2[16]['body']
assert all(p['locator'] and 1<=p['paper_page']<=9 for p in W2)
assert not any('미완성' in p['body'] or 'TODO' in p['body'] for p in W2)
report=['# AlexNet 논문 리딩 자료 검증','',
        '2026-09-23 · 22페이지 · 90분. 초반 CNN 복습 6페이지를 포함한 논문 중심 개정판입니다.','',
        '## 확인 결과','',
        '- 공식 원문 PDF의 SHA-256 일치. 보관한 12개 캡처를 원문 좌표에서 다시 렌더링하여 픽셀 일치 확인.',
        '- 현재 발표에는 제목·Figures 1–4·Tables 1–2·LRN 수식 등 9개 캡처를 사용. 원문 페이지 링크 확인.',
        '- 발표 22페이지, 복습 6페이지, 전체 90분과 구간별 합계 확인.',
        '- 핵심 결과 수치를 원문 Table 1·2 텍스트와 대조.',
        '- 단일 모델 Top-5 검증 18.2%와 앙상블 Top-5 시험 15.3%의 조건 구분 확인.','',
        '| 발표 페이지 | 원문 | 확인한 구분 |','|---|---|---|',
        '| 2–7 | §§1–5 연계 복습 | 수업용 도식·손실 표기는 원문 캡처와 구분 |',
        '| 8–9 | Abstract, §§1, 2, 3.5 | 문제와 제안, 독자가 재구성한 문제 정의 |',
        '| 10–11 | Figures 2, 3 | GPU 분할과 앙상블, 가중치와 특징 맵 구분 |',
        '| 12 | Figure 1 | CIFAR-10 4층 CNN의 훈련 오류 대 epoch |',
        '| 13–15 | §§3–5 | LRN·pooling, 증강·dropout, 학습 조건 |',
        '| 16–17 | Tables 1–2 | 2010/2012, val/test, 단일/앙상블, 추가 사전학습 |',
        '| 18 | §§3.2–3.4, 7 | 구성 요소 비교의 조건과 효과 합산의 한계 |',
        '| 19–20 | Figure 4 | 예측 사례와 은닉 표현의 최근접 이미지 분석 |',
        '| 21–22 | §7 및 독자 기록 | 저자의 관찰·미래 방향·독자의 질문 구분 |','',
        '레이아웃·PDF·웹 조작은 [웹북 검증 보고서](검증_보고서.md)를 참고하세요. 자동 검사는 모든 문장의 의미를 검증하지 않으며, 위 구분은 원문과 직접 대조한 항목입니다.','',
        f'[공식 원문]({URL}) · [캡처 기록](assets/alexnet/provenance.json)']
(ROOT/'AlexNet_자료검증_보고서.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
print('PASS: 22 pages / 6 review pages / 90 min / 9 used captures, 12 stored captures verified.')
