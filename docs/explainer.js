/* OmniMem Figure 2 style, 50-second method walkthrough. */
(() => {
'use strict';
const canvas=document.getElementById('film'),ctx=canvas.getContext('2d');
const W=1200,H=660,DURATION=50;
const C={ink:'#080808',muted:'#555555',line:'#c9cfd6',paper:'#ffffff',blue:'#45648e',query:'#dbe8f7'};
const oranges=['#ff753e','#ff9466','#ffac89','#ffc2a9','#ffd2be','#ffdfd0','#ffe9df','#fff1e9'];
const chapters=[
{at:0,label:'Full history',title:'Full-resolution KV history',desc:'Generate a video chunk by chunk, while retaining access to detailed KV from earlier chunks.'},
{at:5,label:'Three attention paths',title:'Sliding Window · Compression · Selection',desc:'Recent KV, pooled historical KV and retrieved full-resolution KV feed parallel attention branches. Compression scores guide selection; learned gates fuse the outputs.'},
{at:15,label:'Window exclusion',title:'Adaptive Window Exclusion',desc:'Once sufficient distant history exists, remove recent-window blocks from the Top-K candidates. Sliding-window attention still uses recent KV.'},
{at:24,label:'Query sharing',title:'Query-Shared KV Selection',desc:'Average compression-attention scores within neighboring query groups, then share Top-K selection within each group and head.'},
{at:33,label:'Per-head access',title:'Per-Head Scattered KV Access',desc:'Heads follow their own selected block pointers without a padded cross-head union buffer. Missing chunks are loaded from the CPU cache into GPU memory.'},
{at:44,label:'Full framework',title:'The complete OmniMem framework',desc:'Three complementary attention branches combine with adaptive retrieval, query sharing and scattered access. Original Figure 2 from the paper.'}
];
const figure=new Image();figure.src='assets/framework.jpg';
const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,v)),lerp=(a,b,p)=>a+(b-a)*p;
function ease(v){v=clamp(v);return v*v*(3-2*v)}
const prog=(s,t,d=.4)=>ease((s-t)/d);
function text(t,x,y,size=23,color=C.ink,weight=400,align='left',italic=false){ctx.font=`${italic?'italic ':''}${weight} ${size}px "Times New Roman", Times, serif`;ctx.textAlign=align;ctx.textBaseline='middle';ctx.fillStyle=color;ctx.fillText(t,x,y)}
function subText(a,b,x,y,size=23,color=C.ink,weight=400,italic=false){ctx.font=`${italic?'italic ':''}${weight} ${size}px "Times New Roman", Times, serif`;const aw=ctx.measureText(a).width;ctx.font=`${italic?'italic ':''}${weight} ${size*.65}px "Times New Roman", Times, serif`;const bw=ctx.measureText(b).width,xx=x-(aw+bw)/2;text(a,xx,y,size,color,weight,'left',italic);text(b,xx+aw,y+size*.25,size*.65,color,weight,'left',italic)}
function box(x,y,w,h,color=C.blue,r=12,fill='white',lw=2){ctx.beginPath();ctx.roundRect(x,y,w,h,r);ctx.fillStyle=fill;ctx.fill();ctx.lineWidth=lw;ctx.strokeStyle=color;ctx.stroke()}
function line(x1,y1,x2,y2,c=C.ink,w=1.5,d=[]){ctx.beginPath();ctx.setLineDash(d);ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.lineWidth=w;ctx.strokeStyle=c;ctx.stroke();ctx.setLineDash([])}
function path(p,c=C.ink,w=1.5,d=[]){ctx.beginPath();ctx.setLineDash(d);p.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.lineWidth=w;ctx.strokeStyle=c;ctx.stroke();ctx.setLineDash([])}
function arrow(x1,y1,x2,y2,c=C.ink,w=1.5,d=[]){line(x1,y1,x2,y2,c,w,d);const a=Math.atan2(y2-y1,x2-x1);ctx.beginPath();ctx.moveTo(x2,y2);ctx.lineTo(x2-8*Math.cos(a-.45),y2-8*Math.sin(a-.45));ctx.lineTo(x2-8*Math.cos(a+.45),y2-8*Math.sin(a+.45));ctx.closePath();ctx.fillStyle=c;ctx.fill()}
function alpha(a,fn){ctx.save();ctx.globalAlpha*=clamp(a);fn();ctx.restore()}
function tile(x,y,w=68,h=w,fill=oranges[0],grid=true){ctx.fillStyle=fill;ctx.fillRect(x,y,w,h);ctx.strokeStyle=C.ink;ctx.lineWidth=1.25;ctx.strokeRect(x,y,w,h);if(grid){line(x+w/2,y,x+w/2,y+h,C.ink,1);line(x,y+h/2,x+w,y+h/2,C.ink,1)}}
function cells(x,y,n,w=31,h=29,ids=null){for(let i=0;i<n;i++)tile(x+i*w,y,w,h,oranges[ids?ids[i]%8:i%8],false)}
function dotflow(points,s,color=C.blue){const lens=points.slice(1).map((p,i)=>Math.hypot(p[0]-points[i][0],p[1]-points[i][1])),sum=lens.reduce((a,b)=>a+b,0);let d=((s*.65)%1)*sum;for(let i=0;i<lens.length;i++){if(d<=lens[i]){let x=lerp(points[i][0],points[i+1][0],d/lens[i]),y=lerp(points[i][1],points[i+1][1],d/lens[i]);ctx.beginPath();ctx.arc(x,y,3.7,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();break}d-=lens[i]}}
function header(n,title){text(String(n).padStart(2,'0'),40,42,23,C.blue,700);text(title,92,42,30,C.ink,700);line(40,78,1160,78,C.line,1)}
function caption(a,b){line(40,583,1160,583,C.line,1);text(a,600,609,24,C.ink,500,'center');if(b)text(b,600,638,19,C.muted,400,'center')}
function history(y=117,h=191,s=9){box(50,y,1100,h,C.ink,13,'white',2.3);text('Full-Resolution KV Cache',600,y+26,26,C.ink,700,'center');text('Older Chunks',80,y+64,23);text('Recent Chunks',999,y+64,23,C.ink,400,'center');for(let i=0;i<8;i++){let p=prog(s,i*.085,.24),x=93+i*132;alpha(p,()=>{tile(x,y+94+(1-p)*12,63,63,oranges[i]);i===4?text('…',x+31,y+173,20,C.ink,400,'center'):subText('Chunk',i<4?String(i+1):['T−2','T−1','T'][i-5],x+31,y+173,20)})}}
function scene0(s){header(1,'Keep the full video history within reach.');history(108,219,s);text('Current Chunk Queries',600,381,25,C.blue,700,'center');tile(567,412,66,66,C.query);text('q',544,447,30,C.ink,400,'center',true);alpha(prog(s,.55),()=>{path([[565,445],[331,445],[331,345]],C.ink,1.5,[4,4]);arrow(331,345,331,314);path([[635,445],[891,445],[891,345]],C.ink,1.5,[4,4]);arrow(891,345,891,314);dotflow([[565,445],[331,445],[331,330]],s);dotflow([[635,445],[891,445],[891,330]],s+.5)});text('Query-relevant details may be far in the past.',600,528,25,C.ink,400,'center');caption('Each new chunk can access detailed memory from earlier chunks.','Autoregressive long-video generation · conceptual KV blocks');}
function scene1(s){header(2,'Three parallel attention paths.');history(98,193,9);
const centers=[215,600,985],titles=['Sliding Window','Compression','Selection'];
centers.forEach((x,i)=>{box(x-165,314,330,134,C.blue,12);text(titles[i],x,338,26,C.blue,700,'center');});
alpha(.35+.65*prog(s,.15),()=>{for(let i=0;i<4;i++)tile(82+i*75,358,59,59,oranges[i+4]);text('Recent KV',226,431,20,C.ink,400,'center')});
alpha(.35+.65*prog(s,.75),()=>{for(let i=3;i>=0;i--)tile(441+i*7,371-i*7,51,51,oranges[3-i]);arrow(521,393,549,393);cells(560,377,6,28,32);text('Pooled KV',638,431,20,C.ink,400,'center')});
alpha(.35+.65*prog(s,1.45),()=>{for(let i=0;i<4;i++)tile(830+i*77,358,58,58,oranges[[0,2,5,7][i]]);text('Selected full-resolution KV',974,432,20,C.ink,400,'center')});
/* Full cache feeds all paths; the recent and retrieved sources are named explicitly. */
path([[600,294],[600,301],[215,301],[215,306]]);arrow(215,302,215,308);arrow(600,294,600,308);path([[600,301],[985,301],[985,306]]);arrow(985,302,985,308);
alpha(prog(s,2.2),()=>{text('Top-K',792,348,15,C.ink,400,'center');text('Scores',792,370,15,C.ink,400,'center');arrow(768,394,817,394,C.ink,1.2,[3,3]);dotflow([[768,394],[817,394]],s);});
const names=['Sliding-Window Attention','Compression Attention','Selection Attention'];centers.forEach((x,i)=>{text(names[i],x,469,21,C.ink,700,'center');line(x,488,x,513);ctx.beginPath();ctx.arc(x,520,10,0,Math.PI*2);ctx.fillStyle='white';ctx.fill();ctx.strokeStyle=C.ink;ctx.lineWidth=1.5;ctx.stroke();text('·',x,519,27,C.ink,700,'center');text(['gˢʷᵃ','gᶜᵐᵖ','gˢˡᶜ'][i],x+20,514,22,C.ink,400,'left',true);path([[x,530],[x,548],[600,548]]);});
alpha(prog(s,3),()=>{box(496,535,208,35,C.ink,8);text('Gated Output',600,554,23,C.blue,700,'center')});caption('Recent context + pooled history + retrieved detail.','The same current-chunk queries feed all three branches; learned gates fuse the outputs.');}
function scene2(s){header(3,'Adaptive Window Exclusion');box(50,105,1100,428,C.blue,13);text('Historical candidates',354,145,26,C.ink,400,'center');text('Near-window candidates',947,145,26,C.ink,400,'center');let p=prog(s,1.4,.55);for(let i=0;i<8;i++){let x=93+i*132;tile(x,184,66,66,oranges[i]);if(i>4){alpha(p,()=>{ctx.fillStyle='white';ctx.fillRect(x-1,183,68,68);ctx.setLineDash([4,3]);ctx.strokeStyle=C.ink;ctx.lineWidth=1.2;ctx.strokeRect(x,184,66,66);ctx.setLineDash([]);line(x+20,202,x+46,232,'#777',1.5);line(x+46,202,x+20,232,'#777',1.5)});}}path([[91,269],[91,280],[687,280],[687,269]],C.ink,1);path([[753,269],[753,280],[1095,280],[1095,269]],C.ink,1);text('Top-K selection',96,420,25);const ids=p<.5?[4,5,6,7]:[0,2,3,4];for(let i=0;i<4;i++){let source=93+ids[i]*132,dx=374+i*117,dy=389;alpha(prog(s,.2+i*.12),()=>{arrow(source+33,293,dx+31,374,C.ink,1.1,[3,4]);tile(dx,dy,61,61,oranges[ids[i]]);dotflow([[source+33,293],[dx+31,374]],s+i*.21,'#e17b4b')})}text('Recent KV remains available to Sliding-Window Attention.',600,498,23,C.blue,400,'center');caption('Reserve the sparse retrieval budget for long-range history.','Activate exclusion only once sufficient distant history is available.');}
function scene3(s){header(4,'Query-Shared KV Selection');box(50,105,1100,435,C.blue,13);text('Neighboring queries',211,145,26);for(let i=0;i<6;i++){alpha(prog(s,i*.07),()=>tile(219+i*128,188,54,54,C.query,false))}let gp=prog(s,.55,.45);[0,1,2].forEach(g=>alpha(gp,()=>{ctx.setLineDash([6,4]);ctx.strokeStyle=C.ink;ctx.lineWidth=1.3;ctx.strokeRect(206+g*256,174,204,83);ctx.setLineDash([]);subText('G',String(g+1),308+g*256,286,32,C.ink,400,true)}));
alpha(prog(s,1.1),()=>{text('Compression-attention scores',600,325,24,C.ink,400,'center');line(204,345,994,345,C.ink,1,[3,4]);});
[0,1,2].forEach(g=>{let x=308+g*256;alpha(prog(s,1.4+g*.12),()=>{arrow(x,351,x,374);text('Average → Top-K',x,399,23,C.ink,400,'center');arrow(x,419,x,443);cells(x-91,459,5,36.5,32,[[0,2,3,5,7],[1,2,4,5,6],[0,3,4,6,7]][g]);dotflow([[x,350],[x,375]],s+g*.1);});});text('Queries',120,214,25,C.ink,400,'center');arrow(170,215,204,215);caption('One shared Top-K selection within each query group.','Average scores before selection. Groups and attention heads remain independent.');}
function scene4(s){header(5,'Per-Head Scattered KV Access');box(50,105,1100,290,C.blue,13);text('Selected block slots',103,149,25);const slotX=[409,583,757,931];for(let i=0;i<4;i++){box(slotX[i],127,138,36,C.ink,0,'white',1.2);text(`slot-${i+1}`,slotX[i]+69,147,24,C.ink,400,'center')}
const rows=[[1,4],[2,3,4],[1,3]],ys=[217,274,331];rows.forEach((ids,r)=>{alpha(prog(s,r*.35,.35),()=>{text(`Head ${r+1}:`,123,ys[r],25);let x=315;text('[',293,ys[r],30);ids.forEach((id,k)=>{box(x+k*148,ys[r]-19,126,36,C.ink,0,'white',1.2);text(`slot-${id}`,x+63+k*148,ys[r],24,C.ink,400,'center');if(k<ids.length-1)text(',',x+136+k*148,ys[r]+5,24)});text(']',x+ids.length*148-9,ys[r],30);});});
alpha(prog(s,1.4),()=>{text('Native KV block reads',965,218,22,C.blue,400,'center');text('No padded cross-head',965,280,22,C.ink,400,'center');text('union buffer.',965,313,22,C.ink,400,'center');});
alpha(prog(s,2.6),()=>{box(50,420,1100,140,C.blue,13);text('On-Demand KV Residency',600,444,27,C.ink,700,'center');box(75,468,318,74,'#ffe5e5',10,'#ffe5e5',1);text('GPU Hot Cache',234,487,25,C.ink,400,'center');text('[slot-2, slot-5, slot-8]',234,526,24,C.ink,400,'center');box(807,468,318,74,'#eaf1fd',10,'#eaf1fd',1);text('CPU Offload Cache',966,487,25,C.ink,400,'center');text('[slot-1, slot-4, …]',966,526,24,C.ink,400,'center');text('miss: reload selected chunk',600,487,20,C.ink,400,'center');arrow(791,506,409,506);text('full: evict cold chunk',600,533,20,C.ink,400,'center');arrow(409,550,791,550);dotflow([[791,506],[409,506]],s);dotflow([[409,550],[791,550]],s+.3)});caption('Each head follows its own non-contiguous selection.','Load missing chunks onto the GPU; retain full-resolution history in the CPU cache.');}
function scene5(s){header(6,'OmniMem: the complete framework');if(figure.complete&&figure.naturalWidth){const ratio=figure.naturalWidth/figure.naturalHeight,w=1100,h=w/ratio;ctx.drawImage(figure,50,102,w,h)}text('Figure 2',53,557,18,C.muted);caption('Explicit full-range memory. Sparse, query-relevant access.','Scalable and Adaptive Memory Retrieval for Long Video Generation');}
const scenes=[scene0,scene1,scene2,scene3,scene4,scene5];
let time=0,playing=!matchMedia('(prefers-reduced-motion: reduce)').matches,last=performance.now(),current=-1,inView=true;
let params=new URLSearchParams(location.search);
if(params.has('embed'))document.body.classList.add('embedded');
if(params.has('t'))time=clamp(Number(params.get('t'))||0,0,DURATION);
if(params.has('paused'))playing=false;
if(matchMedia('(prefers-reduced-motion: reduce)').matches&&!params.has('t'))time=45;
const play=document.getElementById('play'),seek=document.getElementById('seek'),clock=document.getElementById('clock');
function format(s){return `${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`}
function index(t){let i=0;chapters.forEach((c,n)=>{if(t>=c.at)i=n});return i}
const nav=document.querySelector('.chapters');
chapters.forEach((ch,i)=>{let b=document.createElement('button');b.className='chapter';b.innerHTML=`<span>${String(i+1).padStart(2,'0')} / ${format(ch.at)}</span>${ch.label}`;b.setAttribute('aria-label',`${ch.label}, ${format(ch.at)}`);b.addEventListener('click',()=>{time=ch.at;last=performance.now();render()});nav.appendChild(b)});
function render(){let i=index(time);ctx.setTransform(canvas.width/W,0,0,canvas.height/H,0,0);ctx.fillStyle=C.paper;ctx.fillRect(0,0,W,H);scenes[i](time-chapters[i].at);seek.value=String(time);clock.textContent=`${format(time)} / 0:50`;play.textContent=playing?'Ⅱ Pause':'▶ Play';play.setAttribute('aria-label',playing?'Pause animation':'Play animation');if(current!==i){current=i;document.getElementById('chapter-title').textContent=chapters[i].title;document.getElementById('chapter-description').textContent=chapters[i].desc;[...nav.children].forEach((b,n)=>b.setAttribute('aria-current',String(n===i)));canvas.setAttribute('aria-label',chapters[i].title+'. '+chapters[i].desc);reportHeight()}}
function tick(now){let delta=Math.min((now-last)/1000,.15);last=now;if(playing&&inView&&!document.hidden){time+=delta;if(time>=DURATION)time=0;}render();requestAnimationFrame(tick)}
play.onclick=()=>{playing=!playing;last=performance.now();render()};
document.getElementById('replay').onclick=()=>{time=0;playing=true;last=performance.now();render()};
seek.oninput=()=>{time=Number(seek.value);last=performance.now();render()};
document.getElementById('fullscreen').onclick=()=>{if(document.fullscreenElement)document.exitFullscreen();else document.querySelector('.shell').requestFullscreen().catch(()=>window.open('explainer.html','_blank'))};
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;if(e.code==='Space'){e.preventDefault();play.click()}if(e.code==='ArrowRight'){time=clamp(time+5,0,DURATION);render()}if(e.code==='ArrowLeft'){time=clamp(time-5,0,DURATION);render()}});
document.addEventListener('visibilitychange',()=>last=performance.now());
new IntersectionObserver(entries=>{inView=entries[0].isIntersecting;last=performance.now()},{threshold:.15}).observe(canvas);
function reportHeight(){if(window.parent!==window)window.parent.postMessage({kind:'omnimem-height',height:Math.ceil(document.querySelector('.shell').getBoundingClientRect().height)},location.origin)}
new ResizeObserver(reportHeight).observe(document.querySelector('.shell'));
figure.onload=()=>render();
render();requestAnimationFrame(tick);
})();
