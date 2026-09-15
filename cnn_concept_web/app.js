(() => {
  const body=document.body, books=[...document.querySelectorAll('.book')];
  const sheets=()=>[...document.querySelectorAll('.sheet')].filter(s=>getComputedStyle(s.closest('.book')).display!=='none');
  function setWeek(week,scroll=true){
    body.dataset.week=week;
    document.querySelectorAll('[data-week-select]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.weekSelect===week)));
    const pdf=document.querySelector('.pdf-link');
    pdf.href=week==='all'?'CNN_개념강의_통합.pdf':`CNN_${week}주차_개념강의.pdf`;
    pdf.textContent=week==='all'?'통합 PDF':`${week}주차 PDF`;
    if(scroll) sheets()[0]?.scrollIntoView();
    updatePosition();
  }
  document.querySelectorAll('[data-week-select]').forEach(b=>b.addEventListener('click',()=>setWeek(b.dataset.weekSelect)));
  const nearest=()=>sheets().reduce((best,s)=>Math.abs(s.getBoundingClientRect().top-95)<Math.abs(best.getBoundingClientRect().top-95)?s:best,sheets()[0]);
  function updatePosition(){const list=sheets(),current=nearest();document.querySelector('#position').textContent=`${list.indexOf(current)+1} / ${list.length}`;}
  function move(delta){const list=sheets(),idx=list.indexOf(nearest());list[Math.max(0,Math.min(list.length-1,idx+delta))]?.scrollIntoView();}
  document.querySelector('#previous').addEventListener('click',()=>move(-1));document.querySelector('#next').addEventListener('click',()=>move(1));
  let raf=false;window.addEventListener('scroll',()=>{if(!raf){raf=true;requestAnimationFrame(()=>{updatePosition();raf=false;});}},{passive:true});
  window.addEventListener('keydown',e=>{if(['INPUT','BUTTON','TEXTAREA','SELECT'].includes(document.activeElement.tagName)||document.querySelector('dialog').open)return;if(e.key==='ArrowRight'){e.preventDefault();move(1);}if(e.key==='ArrowLeft'){e.preventDefault();move(-1);}});
  const dialog=document.querySelector('#toc');
  document.querySelector('#open-toc').addEventListener('click',()=>dialog.showModal());
  document.querySelector('#close-toc').addEventListener('click',()=>dialog.close());
  dialog.querySelectorAll('a').forEach(a=>a.addEventListener('click',()=>{setWeek(a.dataset.week,false);dialog.close();}));
  document.querySelectorAll('.conv-demo').forEach(d=>{
    const input=Array.from({length:5},()=>[0,0,1,1,1]);
    d.querySelector('input').addEventListener('input',e=>{
      const n=Number(e.target.value),row=Math.floor(n/3),col=n%3;
      let response=0;for(let i=0;i<3;i++)response+=input[row+i][col+2]-input[row+i][col];
      const focus=d.querySelector('.focus-patch');focus.setAttribute('x',13+col*40);focus.setAttribute('y',63+row*40);
      d.querySelector('output').textContent=`행 ${row+1} · 열 ${col+1} → 반응 ${response}`;
    });
  });
  document.querySelectorAll('.augment-demo').forEach(d=>{
    const messages={original:'원본: 물체의 단서를 먼저 관찰하세요.',shift:'위치 이동: 같은 물체가 오른쪽으로 이동했습니다.',flip:'좌우 반전: 컵의 손잡이가 반대쪽으로 바뀌었습니다.',occlude:'일부 가림: 몸체의 정보가 줄었습니다. 단서는 얼마나 남았나요?'};
    d.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{
      const name=b.dataset.transform;d.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));
      d.querySelector('.demo-object').setAttribute('transform',name==='shift'?'translate(100 0)':name==='flip'?'translate(560 0) scale(-1 1)':'translate(0 0)');
      d.querySelector('.occluder').setAttribute('opacity',name==='occlude'?'1':'0');d.querySelector('.transform-result').textContent=messages[name];
    }));
  });
  const requested=new URLSearchParams(location.search).get('week');
  if(['1','2','all'].includes(requested))setWeek(requested,false);
  updatePosition();
})();
