"""Validate original-paper captures, reading order, timings and source coverage."""
from pathlib import Path
import hashlib
import json
import fitz
from content import W2
from week2_alexnet_content import SCHEDULE
from prepare_alexnet import CROPS, URL

ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'assets/alexnet/provenance.json').read_text(encoding='utf-8'))
paper=ROOT/'_preview/alexnet_2012.pdf'
assert hashlib.sha256(paper.read_bytes()).hexdigest()==manifest['source_pdf_sha256']
assert len(W2)==40 and sum(p['minutes'] for p in W2)==90
assert [p['phase'] for p in W2]==sorted(p['phase'] for p in W2)
assert [sum(p['minutes'] for p in W2 if p['phase']==i) for i in range(8)]==[x[0] for x in SCHEDULE]
body=''.join(p['body'] for p in W2)
with fitz.open(paper) as doc:
    for item in manifest['captures']:
        path=ROOT/'assets/alexnet'/item['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        assert f'assets/alexnet/{item["file"]}' in body
        rerender=doc[item['page']-1].get_pixmap(matrix=fitz.Matrix(4,4),clip=fitz.Rect(item['crop_pdf_points']),alpha=False)
        stored=fitz.Pixmap(path)
        assert rerender.samples==stored.samples, item['file']
    # Check the numerical evidence against selectable text in the primary source.
    results=doc[6].get_text()
    for term in ['25.7%', '17.0%', '18.2%', '15.3%', '26.2%', '7 CNNs*']:
        assert term in results,term
assert len(manifest['captures'])==12
assert '18.2%' in W2[30]['body'] and '검증' in W2[30]['body']
assert '15.3%' in W2[30]['body'] and '시험' in W2[30]['body']
assert all(p['locator'] and 1<=p['paper_page']<=9 for p in W2)
assert not any('미완성' in p['body'] or 'TODO' in p['body'] for p in W2)
report=['# AlexNet 논문 리딩 자료 검증','',
        '2026-09-22 · 40페이지 · 90분. 이 보고서는 발표 자료의 근거·구성을 확인하며 모델 재현 실험 보고서가 아닙니다.','',
        '## 자동 확인','',
        '- 공식 PDF SHA-256 일치. 12개 캡처를 원문 좌표에서 다시 렌더링하여 저장 이미지와 픽셀 단위 일치 확인.',
        '- 12개 캡처가 모두 발표 HTML에 포함되어 있으며 원문 위치를 보존.',
        '- 40페이지, 전체90분, 8단계의 순서와 단계별 시간 합계 확인.',
        '- 핵심 비교 수치가 원문 Table 1·2 텍스트에 존재함을 확인.',
        '- 단일 모델 검증18.2%와 앙상블 시험15.3%의 조건이 발표 원고에 구분되어 있음을 확인.','',
        '## 원문과 대조한 읽기 포인트','',
        '| 발표 페이지 | 근거 | 확인한 구분 |','|---|---|---|',
        '| 10–13 | §§1, 2, 3.5 | 별도 Problem statement 절 없이 독자가 재구성 |',
        '| 19–20 | Figure 2 / §3.5 | 가중치 있는8개 층, 두GPU 분할과 앙상블은 다름 |',
        '| 26–28 | §3.5 / §3.3 | 합성곱·CE는 수업용 표기, LRN은 원문 수식 캡처 |',
        '| 29 | Figure 1 / §3.1 | CIFAR-10 4층CNN, epoch와 훈련 오류, ReLU/tanh |',
        '| 30 | Table 1 | ILSVRC-2010 시험, 25.7→17.0은8.7%p |',
        '| 31–32 | Table 2 / §6 | ILSVRC-2012, val/test·추가사전학습·앙상블 구분 |',
        '| 34–35 | Figure 4 / §6.1 | 왼쪽 예측, 오른쪽 은닉표현의 가까운 훈련 이미지 |',
        '| 36 | §7 | 마지막 절의 실제 제목은 Discussion |',
        '| 39 | §3.5 / Figure 2 | 224/55 크기 문제를 구현 확인 질문으로 남김 |','',
        '레이아웃·PDF 페이지 수·링크·모바일·웹 조작 검사는 [웹북 검증 보고서](검증_보고서.md)를 확인하세요. 자동 검사는 모든 문장의 의미를 검증하지 않으며, 위 구분은 원문을 직접 대조한 검토 항목입니다.','',
        f'[공식 원문]({URL}) · [캡처 출처 기록](assets/alexnet/provenance.json)']
(ROOT/'AlexNet_자료검증_보고서.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
print('PASS: 40 pages / 90 min / 12 pixel-identical original-paper captures / key evidence conditions.')
