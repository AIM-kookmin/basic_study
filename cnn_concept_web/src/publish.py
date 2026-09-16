"""Publish validated concept PDFs and self-contained distribution archives."""
from pathlib import Path
from datetime import date
import json
import shutil
import zipfile

WEB = Path(__file__).resolve().parents[1]
ROOT = WEB.parent
report = json.loads((WEB / '_preview/validation.json').read_text(encoding='utf-8'))
expected = {'CNN_1주차_개념강의.pdf': 28, 'CNN_2주차_개념강의.pdf': 28, 'CNN_개념강의_통합.pdf': 56}
assert {p['file']: p['pages'] for p in report['pdfs']} == expected
assert not any(report[k] for k in ('screen_layout_issues', 'print_layout_issues', 'javascript_errors'))
assert report['mobile_width']['scroll'] <= report['mobile_width']['client']
assert len(report['interactions']) == 5
for item in report['pdfs']:
    assert (WEB / item['file']).stat().st_size == item['bytes'], 'Re-run rendering after PDF edits.'
for week, folder in ((1, 'week01_cnn_basics'), (2, 'week02_cnn_advanced')):
    shutil.copy2(WEB / f'CNN_{week}주차_개념강의.pdf', ROOT / folder / 'lecture.pdf')

lines = ['# CNN 개념 웹북 검증 보고서', '', f'검증·배포: {date.today().isoformat()}.', '',
         'HTML을 Chromium으로 렌더링하여 A4 가로 PDF를 생성했습니다. 웹과 PDF의 한글은 포함된 Noto Sans KR 폰트를 사용합니다.', '',
         '| 파일 | 페이지 | PDF 내 링크 |', '|---|---:|---:|']
for item in report['pdfs']:
    lines.append(f"| {item['file']} | {item['pages']} | {item['links']} |")
lines += ['', '## 확인 결과', '',
          '- 화면·인쇄 레이아웃 자동 검사에서 넘침 0건.',
          '- 390px 모바일 화면에서 가로 스크롤 없음.',
          '- 브라우저 JavaScript 오류 0건.',
          '- 합성곱 창의 위치와 계산값(3 → 0), 증강의 반전·가림·초기화, 주차 탭, 목차 이동 확인.',
          '- PDF 텍스트 추출·페이지 수·링크 확인. 전체 페이지 축소 미리보기와 주요 페이지 확대 시각 검토.',
          '- 주차 폴더의 lecture.pdf에 개정판 복사. 배포 ZIP 무결성 검사.', '',
          '## 범위', '',
          '웹 관찰 도구는 설명용 수치를 조작합니다. 1주차 MNIST 이미지·필터·학습·예측 그림은 별도로 실행한 실제 데이터와 모델 결과입니다. 개정 MNIST 노트북 검증은 MNIST_실습검증_보고서.md를 확인하세요.',
          '자동 검사는 제공된 Chromium 환경 기준이며 모든 브라우저·인쇄 드라이버를 검사한 것은 아닙니다.',
          '원시 검사 결과는 Git에서 제외하는 `_preview/validation.json`에 있습니다. 재검증은 `src/render_and_check.py`로 수행합니다.']
(WEB / '검증_보고서.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

web_files = [p for p in WEB.iterdir() if p.is_file() and p.suffix in {'.html', '.css', '.js', '.md', '.pdf'}]
web_files += [p for p in (WEB / 'assets').rglob('*') if p.is_file()]
labs = [ROOT/folder/'concept_practice.ipynb' for folder in ('week01_cnn_basics','week02_cnn_advanced')]
for path in labs:
    notebook = json.loads(path.read_text(encoding='utf-8'))
    assert all(c.get('execution_count') is not None for c in notebook['cells'] if c['cell_type']=='code'), 'Run validate_notebooks.py first.'
    assert not any(o['output_type']=='error' for c in notebook['cells'] for o in c.get('outputs',[]))
standalone = WEB / 'CNN_개념강의_웹북.zip'
with zipfile.ZipFile(standalone, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(web_files):
        archive.write(path, path.relative_to(WEB).as_posix())
    for path in sorted((WEB / 'src').glob('*')):
        if path.is_file() and path.suffix in {'.py', '.json'}:
            archive.write(path, path.relative_to(WEB).as_posix())
    for path in labs:
        archive.write(path, f'notebooks/{path.parent.name}_concept_practice.ipynb')

student = ROOT / 'AIM_2026_2_수강생배포.zip'
with zipfile.ZipFile(student, 'w', zipfile.ZIP_DEFLATED) as archive:
    archive.writestr('README.md', '# AIM 2026 가을학기 수강생 자료\n\n6주 × 90분 과정입니다. CNN 1·2주차는 cnn_concept_web/index.html 또는 주차별 lecture.pdf로 공부합니다. 각각 28페이지 개념 강의이며 기존 CNN PPT는 이전 판본입니다. 3–6주차는 기존 PPT·PDF를 사용합니다.\n\nCNN 코딩 실습은 수업 외 선택 활동입니다. practice.ipynb를 Colab에 업로드하고 homework_optional.ipynb는 자율 과제로 사용하세요. 환경설정.md에 실행 방법이 있습니다.\n\n리더 해설은 별도 보관하며 이 수강생 ZIP에는 포함하지 않았습니다.\n')
    archive.write(ROOT / '환경설정.md', '환경설정.md')
    archive.writestr('CNN_실습_안내.md', '# CNN 개념 연계 실습\n\n1·2주차 폴더의 concept_practice.ipynb를 Colab에서 열고 위부터 실행하세요. CPU로 실행합니다. 1주차는 실제 MNIST 약11MB 최초 다운로드가 필요하며, 2주차는 내부 합성 데이터를 사용합니다.\n\n1주차: 과제 구분·사람의 단서·CNN 용어·실제 MNIST 분류. 2주차: 배경과 라벨의 상관·배경 교체·혼동 행렬·통제 실험.\n\n1주차 전체 실습은 별도45–50분이며 준비·학습 후6–7절을 짧게 시연할 수 있습니다. 2주차는 준비0–2절을 미리 실행하고 핵심3–5절15분을 진행합니다. 기존 practice.ipynb는 실제 이미지 확장 실습입니다.\n')
    for folder in sorted(ROOT.glob('week*')):
        if not folder.is_dir():
            continue
        for name in ('lecture.pptx', 'lecture.pdf', 'practice.ipynb', 'homework_optional.ipynb', 'README.md'):
            path = folder / name
            archive.write(path, path.relative_to(ROOT).as_posix())
        if (folder/'concept_practice.ipynb').exists():
            archive.write(folder/'concept_practice.ipynb', f'{folder.name}/concept_practice.ipynb')
    for path in sorted(web_files):
        if '리더가이드' not in path.name:
            archive.write(path, path.relative_to(ROOT).as_posix())

for path in (standalone, student):
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        assert len([n for n in archive.namelist() if n.endswith('concept_practice.ipynb')]) == 2
        print(f'{path.name}: {len(archive.namelist())} files, {path.stat().st_size:,} bytes')
print('Published CNN PDFs to week01/week02 and validated both archives.')
