"""Download the primary paper and make traceable, tightly cropped PDF captures."""
from pathlib import Path
import hashlib
import json
import urllib.request
import fitz

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf'
PDF = ROOT / '_preview/alexnet_2012.pdf'
OUT = ROOT / 'assets/alexnet'
# Page numbers below are one-based, matching the paper's printed page numbers.
CROPS = {
    'title': (1, (122, 105, 490, 150), 'Title'),
    'abstract_excerpt': (1, (142, 275, 364, 290), 'Abstract, opening eight words only'),
    'introduction': (1, (106, 445, 194, 462), 'Section 1 heading'),
    'relu_plot': (3, (331, 111, 506, 267), 'Figure 1, graph only'),
    'lrn_equation': (4, (207, 171, 404, 212), 'Section 3.3, LRN equation'),
    'architecture': (5, (106, 80, 506, 210), 'Figure 2, diagram only'),
    'filters': (6, (347, 426, 505, 489), 'Figure 3, learned kernels only'),
    'table2010': (7, (340, 229, 505, 283), 'Table 1, table only'),
    'table2012': (7, (250, 427, 503, 505), 'Table 2, table only'),
    'predictions': (8, (109, 81, 292, 232), 'Figure 4 left, predictions'),
    'neighbors': (8, (294, 81, 504, 232), 'Figure 4 right, feature neighbors'),
    'discussion': (8, (106, 553, 184, 571), 'Section 7 heading'),
}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PDF.parent.mkdir(exist_ok=True)
    if not PDF.exists():
        urllib.request.urlretrieve(URL, PDF)
    digest = hashlib.sha256(PDF.read_bytes()).hexdigest()
    records = []
    with fitz.open(PDF) as doc:
        assert len(doc) == 9
        for name, (number, bounds, label) in CROPS.items():
            page = doc[number - 1]
            pix = page.get_pixmap(matrix=fitz.Matrix(4, 4), clip=fitz.Rect(bounds), alpha=False)
            path = OUT / f'{name}.png'
            pix.save(path)
            records.append(dict(file=path.name, page=number, crop_pdf_points=list(bounds),
                                source_url=URL, locator=label, width=pix.width, height=pix.height,
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    manifest = dict(title='ImageNet Classification with Deep Convolutional Neural Networks',
                    authors=['Alex Krizhevsky', 'Ilya Sutskever', 'Geoffrey E. Hinton'],
                    year=2012, source_url=URL, source_pdf_sha256=digest,
                    method='PyMuPDF PDF rendering at 4 pixels per PDF point; crop only; no redrawing or retouching.',
                    captures=records)
    (OUT / 'provenance.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'README.md').write_text('''# AlexNet 원문 캡처

Krizhevsky, Sutskever & Hinton (2012), *ImageNet Classification with Deep Convolutional Neural Networks*, NeurIPS 25.

원문: ''' + URL + '''

교육용 논문 분석에서 확인할 도표·수식·짧은 구절을 원문 PDF에서 직접 캡처했습니다. 본문이나 캡션 전체를 복제하지 않았습니다. 원 논문 저자·권리자의 자료이며, 자체 제작 도식과 구별합니다. 원문 전체 PDF는 로컬 `_preview/`에만 보관합니다.

`provenance.json`에 원문 URL, 원문 SHA-256, 페이지, PDF 좌표, 캡처 SHA-256이 있습니다. 좌표 원점은 페이지 좌상단이며 단위는 PDF point입니다. 재생성: `python src/prepare_alexnet.py`.

그림 아래 한국어 설명과 강조 표시는 수업용 해설입니다. 원문 캡처 자체의 숫자·곡선·사진은 수정하지 않았습니다. 원문의 수치와 수업용 계산 예제를 혼동하지 않도록 각각 표시합니다.
''', encoding='utf-8')
    print(f'Prepared {len(records)} original-paper captures; SHA-256 {digest}')

if __name__ == '__main__':
    main()
