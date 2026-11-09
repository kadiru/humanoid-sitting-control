"""Actual mesh + exact standing-reset preview. No invented learned behavior."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,hashlib,html
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('/home/kadir/humanoid-sitting');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'sitting_task/src'),str(ROOT/'experiments/furniture_saved_review_20261104_v1')]
from humanoid_sitting.shared_frame import BodyModel,apply,transform
from video import paint
read=lambda p:json.loads(Path(p).read_text())
BG=(245,247,250);INK='#183444';MUTED='#567080'
def font(n,b=False):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if b else '')+'.ttf',n)
def text(d,xy,s,n=23,col=INK,b=False):d.text(xy,s,font=font(n,b),fill=col)
right=np.array([.85,np.sqrt(1-.85**2),0.]);eye=np.array([right[1]*.96,-right[0]*.96,.28]);eye/=np.linalg.norm(eye)
VIEW=np.stack([right,-np.cross(eye,right),eye],axis=1)

def main():
    p=read(HERE/'protocol.json');cases={c['id']:c for c in read(HERE/'cases.json')};model=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json')
    visuals={}
    with np.load(ROOT/'outputs/capability_review_20261105_v1/visuals.npz') as z:
        for k in z.files:
            if k.endswith('_v'):visuals[k[:-2]]=(z[k],z[k[:-2]+'_f'])
    (HERE/'previews').mkdir(exist_ok=True);cards=[]
    for index,v in enumerate(p['variants']):
        c=cases[v['case']]
        with np.load(c['mesh_npz']) as z:fv=z['vertices'];ff=z['faces']
        with np.load(HERE/'banks'/(v['id']+'_preview.npz')) as z:q=z['q'];rp=z['root_pose']
        root=transform(Rotation.from_quat(rp[3:7]).as_matrix(),rp[:3]);fk=model.fk(q)
        fc=np.tile((np.array(c['color'])*255).astype(np.uint8),(len(ff),1));region=read(HERE/'scenes'/v['case']/'target_region.json')['triangle_indices'];fc[region]=[220,164,65]
        meshes=[(fv,ff,fc)]
        for name,(vert,face) in visuals.items():
            if name not in fk:continue
            col=[94,126,148] if name in ['head_link','pelvis'] else [207,219,229]
            if 'ankle' in name or 'palm' in name:col=[57,78,91]
            meshes.append((apply(root@fk[name],vert),face,np.tile(col,(len(face),1))))
        pts=np.concatenate([x[0] for x in meshes])@VIEW;lo,hi=pts.min(0),pts.max(0)
        W,H=580,440;scale=min((W-60)/(hi[0]-lo[0]),(H-55)/(hi[1]-lo[1]));offset=np.array([W/2,H/2,0])-(lo+hi)*.5*scale
        rgb=np.full((H,W,3),BG,np.uint8);depth=np.full((H,W),-np.inf,np.float32)
        for vert,face,col in meshes:
            paint(rgb,depth,W,H,np.ascontiguousarray(vert@VIEW*scale+offset,np.float32),np.ascontiguousarray(face,np.int32),np.ascontiguousarray(col,np.uint8),len(face))
        im=Image.new('RGB',(600,565),'white');im.paste(Image.fromarray(rgb),(10,55));d=ImageDraw.Draw(im)
        text(d,(20,15),f"{index+1:02d}  {v['title']}",25,b=True)
        text(d,(20,496),v['task'].capitalize()+' / '+v['id'].split('_')[-1]+(' / saved pass' if v.get('reuse') else ' / untested'),21)
        text(d,(20,532),f"x {v['x']:+.3f}m  y {v['y']:+.3f}m  angle {v['yaw_deg']:+g}°",18,MUTED)
        im.save(HERE/'previews'/(v['id']+'.jpg'),quality=91);cards.append((v,im))
    for task,cols in [('sitting',5),('leaning',4)]:
        selected=[im for v,im in cards if v['task']==task];rows=(len(selected)+cols-1)//cols
        sheet=Image.new('RGB',(cols*620+20,rows*585+170),BG);d=ImageDraw.Draw(sheet)
        text(d,(30,22),f"{len(selected)} {task} scenarios | five furniture objects overall",40,b=True)
        text(d,(30,80),'STARTING LAYOUTS — amber marks the intended surface; no new physics trials shown.',24,MUTED)
        for j,im in enumerate(selected):sheet.paste(im,(20+j%cols*620,140+j//cols*585))
        sheet.save(HERE/(task+'_scenarios.jpg'),quality=94)
    body=''.join(f'<figure><img src="previews/{v["id"]}.jpg"><figcaption>{html.escape(v["id"])} — '+('saved verified example' if v.get('reuse') else 'unverified')+'</figcaption></figure>' for v,_ in cards)
    (HERE/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Furniture examples</title><style>body{font:18px system-ui;background:#f5f7fa;color:#183444;margin:36px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}figure{margin:0;background:white;padding:8px;border-radius:10px}img{width:100%}figcaption{font-size:13px;color:#567080}p{max-width:950px}</style><h1>10 sitting + 12 leaning scenarios</h1><p>Five distinct objects: box armchair, box-arm couch, L-shaped couch, stool and coffee table. These show the actual mesh and exact standing reset. Amber marks the intended contact surface. They are prepared test scenarios, not successful demonstrations or a learned-policy video.</p><main>'+body+'</main>')
    (HERE/'preview_manifest.json').write_text(json.dumps(dict(classification='CPU_STATIC_LAYOUTS_NOT_PHYSICAL_DEMONSTRATIONS',cards=len(cards),protocol_sha256=hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest()),indent=2)+'\n')
    print('Rendered22 actual starting layouts, no simulation',flush=True)
if __name__=='__main__':main()
