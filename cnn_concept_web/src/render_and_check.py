import json,shutil
from pathlib import Path
from playwright.sync_api import sync_playwright
import fitz
from content import W1,W2
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'_preview';OUT.mkdir(exist_ok=True)
def check_layout(page):
    return page.evaluate('''() => [...document.querySelectorAll('.sheet')].filter(s=>getComputedStyle(s.closest('.book')).display!=='none').flatMap(s=>{
      const issues=[],r=s.getBoundingClientRect();
      for(const sel of ['.masthead','.page-head','.copy','.takeaway','.ask','.footer']){
        const el=s.querySelector(sel); if(!el)continue;const b=el.getBoundingClientRect();
        if(b.bottom>r.bottom+1||b.right>r.right+1||b.left<r.left-1)issues.push({page:s.id,element:sel,reason:'outside page',bottom:b.bottom-r.bottom});
      }
      if(s.classList.contains('paper-sheet')){
        const body=s.querySelector('.paper-body'),b=body.getBoundingClientRect(),head=s.querySelector('.page-head').getBoundingClientRect(),take=s.querySelector('.takeaway').getBoundingClientRect();
        if(body.scrollHeight>body.clientHeight+2)issues.push({page:s.id,element:'.paper-body',reason:'content too tall',extra:body.scrollHeight-body.clientHeight});
        for(const child of body.children){const q=child.getBoundingClientRect();if(q.top<head.bottom-1||q.bottom>take.top+1||q.right>r.right-10||q.left<r.left+10)issues.push({page:s.id,element:child.className,reason:'body overlaps another region',top:q.top-head.bottom,bottom:q.bottom-take.top});}
      }else if(!s.classList.contains('cover')){
        const c=s.querySelector('.sheet-content').getBoundingClientRect(),copy=s.querySelector('.copy').getBoundingClientRect(),head=s.querySelector('.page-head').getBoundingClientRect(),visual=s.querySelector('.visual').getBoundingClientRect(),take=s.querySelector('.takeaway').getBoundingClientRect();
        if(copy.top<head.bottom-1||copy.bottom>take.top+1)issues.push({page:s.id,element:'.copy',reason:'overlaps header/takeaway',top:copy.top-head.bottom,bottom:copy.bottom-take.top});
        if(visual.bottom>take.top+1)issues.push({page:s.id,element:'.visual',reason:'overlaps takeaway',bottom:visual.bottom-take.top});
      }
      return issues;
    })''')

with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1)
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto((ROOT/'index.html').as_uri(),wait_until='networkidle');page.evaluate('document.fonts.ready')
    screen_issues=check_layout(page)
    page.locator('#w1-p01').screenshot(path=str(OUT/'web_cover.png'))
    # Meaningful interactions: convolution, paper reading answer, week tabs and TOC.
    slider=page.locator('.book[data-week="1"] .conv-range')
    slider.fill('2');slider.dispatch_event('input')
    assert '반응 0' in page.locator('.book[data-week="1"] .conv-output').inner_text()
    slider.fill('0');slider.dispatch_event('input')
    assert '반응 3' in page.locator('.book[data-week="1"] .conv-output').inner_text()
    page.locator('[data-week-select="2"]').click()
    assert page.locator('.book[data-week="2"] .sheet').count()==40
    page.locator('.paper-answer summary').click()
    assert page.locator('.paper-answer').evaluate('(x)=>x.open')
    assert page.locator('.paper-answer p').is_visible()
    assert page.locator('.paper-capture img').evaluate_all('(xs)=>xs.every(x=>x.complete&&x.naturalWidth>0)')
    screen_issues+=check_layout(page)
    page.locator('.paper-answer summary').click()
    page.locator('#open-toc').click();assert page.locator('#toc').evaluate('(x)=>x.open')
    page.locator('#toc a[href="#w1-p13"]').click();assert page.locator('body').get_attribute('data-week')=='1'
    assert not page.locator('#toc').evaluate('(x)=>x.open')
    screen_issues+=check_layout(page)
    # PDF uses a fresh load so interactive states are deterministic.
    pdfs=[];print_issues=[]
    for week,name,expected in [('1','CNN_1주차_개념강의.pdf',len(W1)),('2','CNN_2주차_개념강의.pdf',len(W2)),('all','CNN_개념강의_통합.pdf',len(W1)+len(W2))]:
        page.goto((ROOT/'index.html').as_uri()+f'?week={week}',wait_until='networkidle');page.evaluate('document.fonts.ready');page.emulate_media(media='print')
        page.evaluate("document.querySelectorAll('.paper-answer').forEach(x=>x.open=true)")
        print_issues+=check_layout(page)
        path=ROOT/name
        page.pdf(path=str(path),format='A4',landscape=True,print_background=True,prefer_css_page_size=True,display_header_footer=False)
        doc=fitz.open(path)
        assert len(doc)==expected,(name,len(doc))
        assert all(len(pg.get_text().strip())>70 for pg in doc), 'blank or text missing'
        assert all(abs(pg.rect.width-841.89)<2 and abs(pg.rect.height-595.28)<2 for pg in doc)
        pdfs.append({'file':name,'pages':len(doc),'bytes':path.stat().st_size,'links':sum(len(pg.get_links()) for pg in doc)})
        if week!='all':
            for n,pg in enumerate(doc,1):
                pg.get_pixmap(matrix=fitz.Matrix(1.2,1.2),alpha=False).save(OUT/f'week{week}_{n:02d}.png')
        doc.close();page.emulate_media(media='screen')
    # Phone layout: keep text readable instead of shrinking desktop pages.
    page.set_viewport_size({'width':390,'height':844})
    page.goto((ROOT/'index.html').as_uri(),wait_until='networkidle');page.evaluate('document.fonts.ready')
    mobile_width=page.evaluate('({scroll:document.documentElement.scrollWidth,client:document.documentElement.clientWidth})')
    assert mobile_width['scroll']<=mobile_width['client']+1,mobile_width
    page.screenshot(path=str(OUT/'mobile.png'))
    page.goto((ROOT/'index.html').as_uri()+'?week=2',wait_until='networkidle');page.evaluate('document.fonts.ready')
    mobile_week2=page.evaluate('({scroll:document.documentElement.scrollWidth,client:document.documentElement.clientWidth})')
    assert mobile_week2['scroll']<=mobile_week2['client']+1,mobile_week2
    page.screenshot(path=str(OUT/'mobile_week2.png'))
    assert not errors,errors
    browser.close()

for week in [1,2]:
    files=sorted(OUT.glob(f'week{week}_*.png'))
    canvas=Image.new('RGB',(1200,((len(files)+3)//4)*232),'#d9dfd5');draw=ImageDraw.Draw(canvas)
    for i,f in enumerate(files):
        x=i%4*300;y=i//4*232
        canvas.paste(Image.open(f).resize((290,205)),(x,y));draw.text((x+5,y+208),f'W{week} / {i+1:02d}',fill='#1d302d')
    canvas.save(OUT/f'contact_week{week}.png')
report={'pdfs':pdfs,'screen_layout_issues':screen_issues,'print_layout_issues':print_issues,'mobile_width':mobile_width,'mobile_week2':mobile_week2,'javascript_errors':errors,'interactions':['convolution response 3 -> 0','filter position','paper answer reveal/hide and original captures loaded','week tabs','TOC navigation'],'font':'Noto Sans KR (SIL OFL bundled)'}
(OUT/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
