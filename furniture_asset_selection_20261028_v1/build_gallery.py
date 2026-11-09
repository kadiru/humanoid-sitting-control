"""Build an offline contact sheet and selection gallery from publisher previews."""
import json,html,datetime,textwrap
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent
allrows=json.loads((ROOT/'polyhaven_selection.json').read_text())+json.loads((ROOT/'poly_pizza_selection.json').read_text())
lookup={r['asset_id']:r for r in allrows}
# Contact opportunities are visual hypotheses, not physics validation.
choices=[
('6MoOyPtetL','Box-arm couch','Couches','Seat · arms · back','Simple straight couch; a useful first candidate for rotating through seat, side and rear approaches.',True),
('1kwsjhpY84','L-shaped couch','Couches','Seat · chaise · back','Adds an inside corner and a long chaise; useful for testing which surface is reachable.',True),
('mWgQ94zhDZ','Rounded couch','Couches','Seat · arms · back','Sloped arms and a curved back make the contact location less obvious.',False),
('fFfoi1LNKY','Open-arm couch','Couches','Seat · arms · back','Open arm gaps distinguish a real support surface from empty space.',False),
('l84E3NzGSF','Rolled-arm sofa','Couches','Seat · arms · back','Rounded arm ends provide a more difficult perch than a flat ledge.',False),
('sofa_02','Tufted leather sofa','Couches','Seat · arms · back','Low continuous back with rounded sides; check rear and top geometry after import.',False),
('sofa_03','Carved sofa','Couches','Seat · arms · back','Higher ornate back and curved arms create irregular upper edges.',False),
('chinese_sofa','Wooden frame sofa','Couches','Seat · arms · back','Rigid-looking straight frame; useful alongside a padded couch to compare shapes.',True),
('painted_wooden_sofa','High-back wooden sofa','Couches','Seat · arms · back','Tall shaped back offers a different leaning height; its top is a narrow edge.',False),
('myd1WSucAz','Box armchair','Chairs','Seat · arms · back','Compact armchair with solid-looking sides and back.',False),
('modern_arm_chair_01','Wood-and-leather chair','Chairs','Seat · arm rails · back','Exposed narrow arm rails; useful for distinguishing broad support from a thin rail.',False),
('mid_century_lounge_chair','Reclined lounge chair','Chairs','Seat · arms · back','Reclined seat and back plus a pedestal base; base motion must be specified in simulation.',False),
('plastic_monobloc_chair_01','Plastic garden chair','Chairs','Seat · arms · back','Curved shell and thin arms; open back slots matter for collision geometry.',False),
('dining_chair_02','Upholstered dining chair','Chairs','Seat · back','Armless narrow seat with a tall back.',False),
('painted_wooden_chair_01','Straight wooden chair','Chairs','Seat · back rail','Simple seat, slatted back and no arms.',False),
('gallinera_chair','Wooden armchair','Chairs','Seat · arms · back','Solid seat with rails and a tall slatted back.',False),
('painted_wooden_bench','Wooden bench','Seats & stools','Seat · sides · back','Wide rigid-looking seat with a curved upper back.',False),
('wooden_stool_01','Round wooden stool','Seats & stools','Round seat','Backless target; focuses on controlled lowering and balance.',False),
('folding_wooden_stool','Folding stool','Seats & stools','Flat seat','Rectangular seat with gaps; the model does not establish real hinge behavior.',False),
('metal_stool_02','Round metal stool','Seats & stools','Round seat','Small circular target with thin legs.',False),
('wooden_stool_02','Very low wooden stool','Seats & stools','Low seat','Published overall height is about 18 cm; likely a difficult reach at native scale.',False),
('bar_chair_round_01','Bar stool','Seats & stools','High seat · low back','Higher round seat and a small back support.',False),
('modern_coffee_table_01','Rectangular coffee table','Tables','Low top · straight edges','Broad rectangular top and low height; useful first tabletop candidate.',True),
('modern_coffee_table_02','Square coffee table','Tables','Low top · corners','Low square top; inspect transparent areas and underlying structure before contact tests.',False),
('chinese_tea_table','Square tea table','Tables','Top · corners','Compact square top with a raised border.',False),
('coffee_table_round_01','Round coffee table','Tables','Round top · curved edge','Wide circular top tests approach direction without rectangular corners.',False),
('industrial_coffee_table','Industrial coffee table','Tables','Top · straight edges','Square top with an open metal support frame.',False),
('wooden_table_02','Rectangular wooden table','Tables','High top · edges','Higher rectangular tabletop for comparing leaning with perching.',False),
('round_wooden_table_02','Round pedestal table','Tables','High top · curved edge','Circular higher tabletop supported by a pedestal.',False),
('chinese_console_table','Narrow console table','Tables','Narrow top · long edge','Long but shallow top: a useful edge-leaning challenge.',False),
]
rows=[]
for n,(aid,title,category,surfaces,note,recommended) in enumerate(choices,1):
    r=dict(lookup[aid]);r.update(number=n,title=title,category=category,surfaces=surfaces,note=note,
        recommended=recommended,style='Simplified' if r['source']=='Poly Pizza' else 'Realistic',
        simulation_validated=False,scale_verified=False)
    rows.append(r)
manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'Visual furniture selection before simulator import',
 'license_scope':'CC0 refers to 3D models. Publisher preview images remain credited to their sources.',
 'surfaces_scope':'Potential contact surfaces inferred from previews, not validated robot affordances.',
 'model_download_scope':'Six complete GLB models downloaded; 24 Poly Haven mesh descriptors checked, dependencies not yet downloaded.',
 'scientific_work':'No simulator import, training modification, GPU job or robot trial performed.',
 'assets':rows}
(ROOT/'catalog.json').write_text(json.dumps(manifest,indent=2))

def font(size,bold=False):return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial'+(' Bold' if bold else '')+'.ttf',size)
def fit_text(d,text,maxwidth,size=24):
    f=font(size,True)
    while d.textlength(text,font=f)>maxwidth:
        size-=1;f=font(size,True)
    return f

def sheet(data,destination,cols=5):
    width=1800;margin=45;gap=16;cw=(width-2*margin-(cols-1)*gap)//cols;ch=310
    header=182;footer=62;nrows=(len(data)+cols-1)//cols
    im=Image.new('RGB',(width,header+nrows*(ch+gap)+footer),'#f2efe8');d=ImageDraw.Draw(im)
    d.text((margin,29),'Choose the robot’s next furniture',font=font(47,True),fill='#172b2a')
    d.text((margin,94),'30 real 3D assets  /  Couches, chairs, seats and tables  /  Reply with the numbers you like',font=font(24),fill='#4c615d')
    d.text((margin,133),'Outlined cards = suggested first tests. Images are publisher previews; pictures are not to a common scale.',font=font(21),fill='#65756e')
    for i,r in enumerate(data):
        x=margin+(i%cols)*(cw+gap);y=header+(i//cols)*(ch+gap)
        d.rounded_rectangle((x,y,x+cw,y+ch),radius=16,fill='white',outline='#19836d' if r['recommended'] else '#dadfd9',width=3 if r['recommended'] else 1)
        t=Image.open(ROOT/r['preview_local']).convert('RGBA')
        t.thumbnail((cw-26,194),Image.Resampling.LANCZOS)
        im.paste(t,(x+(cw-t.width)//2,y+15+(194-t.height)//2),t)
        d.rounded_rectangle((x+13,y+13,x+60,y+50),radius=10,fill='#173f38')
        d.text((x+22,y+19),f"{r['number']:02d}",font=font(22,True),fill='white')
        d.text((x+14,y+218),r['title'],font=fit_text(d,r['title'],cw-28),fill='#17332e')
        d.text((x+14,y+250),r['surfaces'],font=font(19),fill='#45675d')
        credit='Poly Haven' if r['source']=='Poly Haven' else r['authors'][0]+' / Poly Pizza'
        d.text((x+14,y+280),credit,font=font(16),fill='#7a857e')
    d.text((margin,im.height-42),'CC0 models • Source links and downloads in gallery.html • Geometry, scale and physical contact behavior still need validation.',font=font(20),fill='#52645d')
    im.save(ROOT/destination)
sheet(rows,'furniture_collage.jpg')

cards=[]
for r in rows:
    dims=r.get('dimensions_mm')
    dimtext=('Published overall extents: '+' × '.join(f'{v/1000:.2f}' for v in dims)+' m (source axes). Not seat height; verify after import.') if dims else 'Physical dimensions are not supplied here; set and verify metric scale before testing.'
    badges=''.join('<span class="tag">'+html.escape(x)+'</span>' for x in r['surfaces'].split(' · '))
    cards.append(f'''<article class="card {'recommended' if r['recommended'] else ''}" data-category="{html.escape(r['category'])}" data-number="{r['number']}">
<button class="pick" data-pick="{r['number']}" aria-pressed="false" aria-label="Select {html.escape(r['title'])}"><span class="number">{r['number']:02d}</span><span class="picktext">Select</span></button>
<button class="picture" data-view="{r['number']}" aria-label="Enlarge {html.escape(r['title'])}"><img src="{r['preview_local']}" alt="Publisher preview of {html.escape(r['name'])}" loading="eager"></button>
<div class="body"><div class="eyebrow">{r['style']} · {r['category']}{' · START HERE' if r['recommended'] else ''}</div><h2>{r['title']}</h2><div class="tags">{badges}</div><p class="note">{r['note']}</p>
<div class="credit">{html.escape(', '.join(r['authors']))} · {r['source']}</div>
<details><summary>Size & download details</summary><p>{dimtext}</p><p>Original asset name: {r['name']}<br>Formats: {', '.join(r['formats'])}<br>Model license: CC0<br>Download check: {r['download_check']}.</p><p>Physics validation: pending. A furniture mesh does not specify friction, softness, mass or stability.</p></details>
<a class="source" href="{r['source_url']}" target="_blank" rel="noopener">Source & download ↗</a></div></article>''')

page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Furniture for sitting & leaning — 30 choices</title>
<style>
:root{--ink:#183b32;--muted:#64766d;--green:#167c62;--paper:#f5f2eb;--line:#dce2da}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}button,a,input{font:inherit}button,a{touch-action:manipulation}button{cursor:pointer}a{color:var(--green)}header,main{max-width:1460px;margin:auto;padding:35px 36px}header{padding-bottom:18px}.kicker{font-size:12px;font-weight:750;letter-spacing:.14em;color:var(--green)}h1{font-size:clamp(32px,4vw,54px);line-height:1.08;letter-spacing:-.035em;max-width:880px;margin:14px 0}header>p{max-width:880px;color:var(--muted);font-size:18px;margin:15px 0}.meta{display:flex;flex-wrap:wrap;gap:12px;color:#68796d;font-size:13px}.meta span{border:1px solid var(--line);padding:5px 11px;border-radius:99px}.toolbar{position:sticky;top:0;z-index:3;background:#f5f2ebf5;border-block:1px solid var(--line);backdrop-filter:blur(8px)}.toolbar-inner{max-width:1460px;margin:auto;padding:12px 36px;display:flex;flex-wrap:wrap;gap:8px;align-items:center}.filter{background:transparent;border:1px solid #c9d4ca;color:var(--ink);border-radius:99px;padding:7px 16px}.filter[aria-pressed=true]{background:var(--ink);color:white;border-color:var(--ink)}.count{margin-left:auto;color:var(--muted);font-size:13px}.intro{margin:0 0 24px;color:var(--muted);max-width:1050px}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:20px}.card{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:15px;background:white;box-shadow:0 3px 12px #183b3205}.card.recommended{border-color:#67ad96}.card.chosen{outline:3px solid var(--green);outline-offset:2px}.card[hidden]{display:none}.pick{position:absolute;top:12px;left:12px;z-index:1;display:flex;align-items:center;border:0;padding:0;border-radius:10px;background:#173f38;color:white;box-shadow:0 1px 6px #0002}.number{font-weight:750;font-size:16px;padding:5px 9px}.picktext{font-size:12px;padding:5px 10px;border-left:1px solid #ffffff38}.picture{display:block;border:0;width:100%;height:215px;padding:32px 10px 8px;background:#fbfbf8}.picture img{width:100%;height:100%;object-fit:contain}.body{padding:16px}.eyebrow{font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:700}h2{font-size:20px;line-height:1.2;margin:7px 0 11px}.tags{display:flex;gap:5px;flex-wrap:wrap}.tag{font-size:12px;background:#edf5ee;color:#406454;padding:3px 7px;border-radius:5px}.note{color:#526a5e;font-size:13px;min-height:58px;margin:12px 0}.credit{font-size:11px;color:#748176}.source{display:inline-block;margin-top:12px;font-size:13px;font-weight:650;text-decoration:none}details{border-top:1px solid #eef0eb;margin-top:12px;padding-top:9px;font-size:12px;color:var(--muted)}summary{cursor:pointer}details p{margin-bottom:0}.next{margin:42px 0 104px;border:1px solid #bdcfc3;border-radius:16px;padding:25px;background:#eaf1e9}.next h2{font-size:26px}.next p{max-width:1000px;color:#4c6757}.steps{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}.step{padding:16px;background:#fff8;border-radius:10px}.step b{display:block}.step span{font-size:13px;color:#617469}footer{font-size:12px;color:#748176;margin-top:25px}.tray{position:fixed;bottom:0;left:0;right:0;background:#173f38;color:#f6faf6;z-index:4;box-shadow:0 -4px 25px #143c3822}.tray-inner{max-width:1460px;margin:auto;padding:15px 36px;display:flex;gap:20px;align-items:center}.tray label{font-size:12px;color:#b3d3c3}.tray strong{display:block;font-size:17px;max-width:700px}.tray button{border:1px solid #6f9785;padding:8px 15px;background:transparent;color:white;border-radius:8px;white-space:nowrap}.tray .copy{background:#f0f7ed;color:#173f38;border-color:#f0f7ed;margin-left:auto}dialog{border:0;border-radius:18px;padding:0;width:min(850px,94vw);max-height:92vh;background:var(--paper)}dialog::backdrop{background:#08251ec7}.modalhead{display:flex;justify-content:space-between;align-items:center;padding:16px 24px;gap:12px}.modalhead button{border:1px solid #bdcfc3;background:white;border-radius:8px;padding:5px 12px}dialog img{display:block;width:100%;height:min(56vh,530px);object-fit:contain;background:#fdfdfb}dialog p{margin:14px 24px 22px;color:var(--muted)}:focus-visible{outline:3px solid #e7a845;outline-offset:4px}@media(max-width:1100px){.grid{grid-template-columns:repeat(3,1fr)}}@media(max-width:800px){.grid{grid-template-columns:repeat(2,1fr)}header,main{padding:24px 18px}.toolbar-inner,.tray-inner{padding:12px 18px}.steps{grid-template-columns:repeat(2,1fr)}.note{min-height:0}.tray strong{font-size:14px}.count{display:none}}@media(max-width:460px){.grid{grid-template-columns:1fr}.tray-inner{gap:8px}.tray button{padding:6px 8px;font-size:12px}}
</style></head><body>
<header><div class="kicker">ROBOT SITTING & LEANING / ASSET SELECTION</div><h1>One piece of furniture.<br>Many places to find support.</h1><p>Choose a couch, chair or table for the next experiment. The idea is to rotate it behind the robot and test the seat, armrests, side and rear upper edge.</p><div class="meta"><span>30 downloadable models</span><span>CC0 model licenses</span><span>Actual publisher previews</span><span>Pictures are not to scale</span></div></header>
<nav class="toolbar" aria-label="Filter furniture"><div class="toolbar-inner"><button class="filter" data-filter="All" aria-pressed="true">All 30</button><button class="filter" data-filter="Couches" aria-pressed="false">Couches</button><button class="filter" data-filter="Chairs" aria-pressed="false">Chairs</button><button class="filter" data-filter="Seats & stools" aria-pressed="false">Seats & stools</button><button class="filter" data-filter="Tables" aria-pressed="false">Tables</button><span class="count" id="count">30 choices</span></div></nav>
<main><p class="intro"><strong>Suggested first choices: 01, 02, 08 or 23.</strong> They give us distinct seats, arms, backs or table edges to work with. Green outlines mark these choices. Click a picture to enlarge it; select any numbers you like.</p><section class="grid" aria-label="Furniture choices">__CARDS__</section>
<section class="next"><div class="kicker">AFTER YOU CHOOSE</div><h2>Same object, different approaches</h2><div class="steps"><div class="step"><b>Front / 0°</b><span>Seat or tabletop facing the robot.</span></div><div class="step"><b>Side / 90°</b><span>Armrest or side edge facing the robot.</span></div><div class="step"><b>Rear / 180°</b><span>Furniture back facing the robot’s back.</span></div><div class="step"><b>Diagonal / 45°</b><span>A corner or mixed surfaces within reach.</span></div></div><p>First we verify metric size, collision geometry and reachable support surfaces. Then we can define sitting and leaning trials at several distances. These surface tags are proposals from the pictures; no model has passed a robot contact test yet.</p><p>Teaching the robot to choose a surface will also require observations of the furniture’s shape and pose, plus training for that choice. Adding a mesh alone will not provide that ability.</p><footer>Models and previews: <a href="https://polyhaven.com/">Poly Haven</a>, <a href="https://poly.pizza/u/Quaternius">Quaternius via Poly Pizza</a> and <a href="https://poly.pizza/m/myd1WSucAz">CreativeTrio via Poly Pizza</a>. All links checked 28 October 2026. CC0 covers models; publisher preview images remain credited to their sources. This local gallery does not alter existing training jobs. <a href="catalog.json">Full source manifest</a> · <a href="furniture_collage.jpg">Full collage</a></footer></section></main>
<div class="tray"><div class="tray-inner"><div><label>YOUR SHORTLIST · reply with these numbers in chat</label><strong id="selected">No choices selected yet</strong></div><button class="copy" id="copy">Copy choices</button><button id="clear">Clear</button></div></div>
<dialog id="viewer"><div class="modalhead"><strong id="viewtitle"></strong><button id="close">Close</button></div><img id="viewimage" alt=""><p id="viewnote"></p></dialog>
<script>const assets=__DATA__;const chosen=new Set();const storeKey='robotFurnitureSelection20261028v1';try{JSON.parse(localStorage.getItem(storeKey)||'[]').forEach(n=>chosen.add(n))}catch{}const cards=[...document.querySelectorAll('.card')];function update(){cards.forEach(c=>{const on=chosen.has(+c.dataset.number);c.classList.toggle('chosen',on);c.querySelector('.pick').setAttribute('aria-pressed',on);c.querySelector('.picktext').textContent=on?'Selected':'Select'});document.getElementById('selected').textContent=chosen.size?[...chosen].sort((a,b)=>a-b).map(n=>String(n).padStart(2,'0')).join(', '):'No choices selected yet';try{localStorage.setItem(storeKey,JSON.stringify([...chosen]))}catch{}}document.querySelectorAll('[data-pick]').forEach(b=>b.onclick=()=>{const n=+b.dataset.pick;chosen.has(n)?chosen.delete(n):chosen.add(n);update()});document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{document.querySelectorAll('[data-filter]').forEach(x=>x.setAttribute('aria-pressed',x===b));let n=0;cards.forEach(c=>{c.hidden=b.dataset.filter!=='All'&&c.dataset.category!==b.dataset.filter;if(!c.hidden)n++});document.getElementById('count').textContent=n+' choices'});document.getElementById('clear').onclick=()=>{chosen.clear();update()};document.getElementById('copy').onclick=async()=>{const s=[...chosen].sort((a,b)=>a-b).map(n=>String(n).padStart(2,'0')).join(', ');if(!s)return;try{await navigator.clipboard.writeText('My furniture choices: '+s);document.getElementById('copy').textContent='Copied';setTimeout(()=>document.getElementById('copy').textContent='Copy choices',1500)}catch{prompt('Copy these choices:','My furniture choices: '+s)}};const viewer=document.getElementById('viewer');document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{const r=assets.find(x=>x.number===+b.dataset.view);document.getElementById('viewtitle').textContent=String(r.number).padStart(2,'0')+' / '+r.title;document.getElementById('viewimage').src=r.preview_local;document.getElementById('viewimage').alt=r.name;document.getElementById('viewnote').textContent=r.note+' Source: '+r.authors.join(', ')+' / '+r.source+'.';viewer.showModal()});document.getElementById('close').onclick=()=>viewer.close();update();</script></body></html>'''
page=page.replace('__CARDS__','\n'.join(cards)).replace('__DATA__',json.dumps(rows).replace('</','<\\/'))
(ROOT/'gallery.html').write_text(page)
assert len(rows)==30 and len({r['number'] for r in rows})==30
assert all((ROOT/r['preview_local']).exists() for r in rows)
print(json.dumps({'assets':len(rows),'gallery':str(ROOT/'gallery.html'),'collage':str(ROOT/'furniture_collage.jpg')}))
