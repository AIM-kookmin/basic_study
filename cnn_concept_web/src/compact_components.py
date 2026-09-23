"""Restrained paper-reading layouts; original figures remain unmodified."""
from html import escape
from paper_components import PAPER_URL, svg, table

def paragraphs(*items):
    return '<div class="body-copy">'+''.join(f'<p>{x}</p>' for x in items)+'</div>'

def points(*items):
    return '<ul class="paper-points">'+''.join(f'<li>{x}</li>' for x in items)+'</ul>'

def columns(left, right, kind=''):
    return f'<div class="paper-columns {kind}"><div>{left}</div><div>{right}</div></div>'

def figure(name, page, caption, height=360):
    return f'<figure class="paper-capture"><img src="assets/alexnet/{name}.png" alt="AlexNet 원문 {caption}" style="max-height:{height}px"><figcaption><a href="{PAPER_URL}#page={page}" target="_blank" rel="noopener">{caption} · 원문 {page}쪽 ↗</a></figcaption></figure>'

def equation(text, caption):
    return f'<div class="compact-equation"><div>{text}</div><p>{caption}</p></div>'

def note(text):
    return f'<p class="paper-note">{text}</p>'

def strip(*items):
    return '<div class="process-strip">'+''.join(f'<div><b>{a}</b><span>{b}</span></div>' for a,b in items)+'</div>'

def render(p,num,total):
    source=f'<a href="{PAPER_URL}#page={p["paper_page"]}" target="_blank" rel="noopener">Krizhevsky, Sutskever &amp; Hinton (2012) · {p["locator"]} ↗</a>'
    return f'''<section class="sheet paper-sheet {p.get('theme','')}" id="w2-p{num:02d}" data-page="{num}" data-week="2">
<div class="masthead"><span>AIM · CNN 2주차</span><span>AlexNet / 2012</span></div>
<header class="page-head"><div class="kicker">{escape(p['kicker'])}</div><h2>{p['title']}</h2><p class="lead">{p['lead']}</p></header>
<div class="paper-body">{p['body']}</div>
<footer class="footer"><span class="sources">{source}</span><span class="folio">{num:02d} / {total:02d}</span></footer></section>'''
