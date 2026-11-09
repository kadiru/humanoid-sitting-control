"""CPU replay of saved physical poses, showing all20 trials and measured holds."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,hashlib,subprocess,functools
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('/home/kadir/humanoid-sitting');OUT=ROOT/'outputs/furniture_trials_review_20261105_v1';SRC=ROOT/'experiments/furniture_examples_20261105_v2'
sys.path[:0]=[str(ROOT/'experiments/furniture_saved_review_20261104_v1'),str(SRC)]
from video import paint
from audit_rollout import original
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
W,H,FPS=1920,1080,25;BG=(245,247,250);INK='#183444';MUTED='#546d7c';GREEN='#14765b';RED='#b04237'
@functools.lru_cache(None)
def font(n,b=False):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if b else '')+'.ttf',n)
def text(d,xy,s,n=25,col=INK,b=False):d.text(xy,s,font=font(n,b),fill=col)
right=np.array([.85,np.sqrt(1-.85**2),0.]);eye=np.array([right[1]*.96,-right[0]*.96,.28]);eye/=np.linalg.norm(eye)
VIEW=np.stack([right,-np.cross(eye,right),eye],axis=1)
visual={}
with np.load(ROOT/'outputs/capability_review_20261105_v1/visuals.npz') as z:
    for k in z.files:
        if k.endswith('_v'):
            name=k[:-2];col=[207,219,229]
            if name in ['pelvis','head_link']:col=[83,115,138]
            if 'ankle' in name or 'palm' in name:col=[57,78,91]
            faces=z[name+'_f'];visual[name]=(z[k],faces,np.tile(col,(len(faces),1)).astype(np.uint8))
cases={c['id']:c for c in read(SRC/'cases.json')}
variants={v['id']:v for v in read(SRC/'protocol.json')['variants']}

class Scene:
    def __init__(self,row):
        self.row=row;self.w=900;self.h=330;self.folder=Path(row['source']);self.static=not row['full20s']
        if self.static:
            self.card=Image.open(SRC/'previews'/(row['id']+'.jpg')).convert('RGB').crop((10,55,590,495));return
        sit=row['task']=='sitting';name='furniture_trajectory.npz' if sit else 'trajectory.npz'
        audit=read(self.folder/'independent_audit.json');assert sha(self.folder/name)==audit['files'][name]
        capture=read(self.folder/('furniture_capture.json' if sit else 'results.json'))
        with np.load(self.folder/name) as z:d=dict(z)
        self.pose=d['body'][:,0].copy();self.pose[:,:,:3]-=np.array(capture['origins'][0])
        r=original.physical(d);held=0;self.hold=[]
        for good in r['good'][:,0]:held=held+1 if good and not audit['warm_failure'] else 0;self.hold.append(held*.02)
        self.failure=np.maximum.accumulate(r['failure'][:,0]);self.slip=np.maximum.accumulate(~r['no_slip'][:,0]);self.legal=np.minimum.accumulate(r['legal'][:,0])
        names=capture['body_names'];self.parts=[(names.index(n),m) for n,m in visual.items() if n in names]
        case=cases[variants[row['id']]['case']]
        with np.load(case['mesh_npz']) as z:self.v=z['vertices'];self.f=z['faces']
        self.c=np.tile((np.array(case['color'])*255).astype(np.uint8),(len(self.f),1))
        bounds=[self.v@VIEW]
        for j,(v,f,c) in self.parts:
            lo,hi=v.min(0),v.max(0);corners=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
            R=Rotation.from_quat(self.pose[:,j,3:7]).as_matrix();pts=np.einsum('tij,kj->tki',R,corners)+self.pose[:,j,None,:3];bounds.append(pts.reshape(-1,3)@VIEW)
        pts=np.concatenate(bounds);lo,hi=pts.min(0),pts.max(0)
        self.scale=min((self.w-65)/(hi[0]-lo[0]),(self.h-35)/(hi[1]-lo[1]));self.offset=np.array([self.w/2,self.h/2,0])-(lo+hi)*self.scale/2
        self.base=np.full((self.h,self.w,3),BG,np.uint8);self.depth=np.full((self.h,self.w),-np.inf,np.float32)
        # The simulated floor top is z=0; show it to make foot height legible.
        ground=np.array([[-2.5,-2.5,0],[2.5,-2.5,0],[2.5,2.5,0],[-2.5,2.5,0]])
        gf=np.array([[0,1,2],[0,2,3]],np.int32)
        paint(self.base,self.depth,self.w,self.h,self.project(ground),gf,np.tile([235,240,244],(2,1)).astype(np.uint8),2)
        paint(self.base,self.depth,self.w,self.h,self.project(self.v),np.ascontiguousarray(self.f,np.int32),self.c,len(self.f))
    def project(self,points):return np.ascontiguousarray(points@VIEW*self.scale+self.offset,np.float32)
    def frame(self,tick):
        if self.static:
            im=Image.new('RGB',(self.w,self.h),BG);card=self.card.resize((435,330));im.paste(card,(232,0));return im
        rgb=self.base.copy();depth=self.depth.copy()
        for j,(v,f,c) in self.parts:
            pose=self.pose[tick,j];world=v@Rotation.from_quat(pose[3:7]).as_matrix().T+pose[:3]
            paint(rgb,depth,self.w,self.h,self.project(world),np.ascontiguousarray(f,np.int32),c,len(f))
        return Image.fromarray(rgb)

def page(title,subtitle):
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im);text(d,(45,28),title,40,b=True);text(d,(45,83),subtitle,23,MUTED)
    d.line((45,1014,1875,1014),fill='#d5dee5',width=2);text(d,(45,1033),'Saved simulation poses | demonstration controllers | unchanged rules | 4090 only',21,MUTED);return im

def main():
    result=read(OUT/'results.json');rows=result['rows'];assert len(rows)==20
    for p,h in result['bindings'].items():assert sha(p)==h
    target=OUT/'furniture_trials_20.mp4'
    cmd=['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
    process=subprocess.Popen(cmd,stdin=subprocess.PIPE);count=0
    def emit(im):
        nonlocal count
        process.stdin.write(im.tobytes());count+=1
    intro=page('20 new furniture trials','10 sitting +10 leaning | five distinct furniture objects')
    d=ImageDraw.Draw(intro);text(d,(90,215),'A pass requires:',36,b=True)
    for j,s in enumerate(['20 seconds from a standing start','At least5 seconds of continuous support at the end','No loaded slipping, illegal contact or physical-limit failure']):text(d,(90,290+j*74),s,32)
    text(d,(90,610),f"{result['full20s_rollouts']} complete rollouts at 2.5×; {result['planning_rejections_without_motion']} starts blocked by planning.",31)
    text(d,(90,680),'Hold timers report qualifying support, not merely touching furniture.',27,MUTED)
    text(d,(90,790),'Two older armchair passes are excluded from these20 results.',26,MUTED)
    intro.save(OUT/'intro.jpg',quality=93)
    for _ in range(4*FPS):emit(intro)
    for g in range(5):
        group=rows[g*4:(g+1)*4];scenes=[Scene(r) for r in group]
        for frame in range(8*FPS):
            tick=min(999,int(frame/(8*FPS-1)*999))
            subtitle=f'Simulation time {(tick+1)*.02:04.1f}/20.0s | playback 2.5×' if any(r['full20s'] for r in group) else 'Starting layouts only — planner rejected these starts before motion'
            im=page(f'New trials {g*4+1}–{g*4+4} of 20',subtitle)
            for j,(scene,row) in enumerate(zip(scenes,group)):
                x=40+(j%2)*940;y=134+(j//2)*438
                panel=Image.new('RGB',(900,438),BG);panel.paste(scene.frame(tick),(0,37));d=ImageDraw.Draw(panel)
                color=GREEN if row['passed'] else RED
                text(d,(8,0),f"{g*4+j+1:02d}  {row['furniture']} • {row['task']} / {row['start']}",24,b=True)
                hold='not run' if scene.static else f'{scene.hold[tick]:.2f}s'
                final='—' if row['final_hold_s'] is None else f"{row['final_hold_s']:.2f}s"
                verdict='PASS' if row['passed'] else ('FAIL' if row['full20s'] else 'NOT RUN')
                text(d,(8,372),f"{verdict}   Hold now {hold} | final {final}",24,color,True)
                reason=row['reason'].replace('final hold shorter than5s','hold<5s')
                if len(reason)>78:reason=reason[:75]+'…'
                text(d,(8,407),reason,17,MUTED)
                im.paste(panel,(x,y))
            emit(im)
            if frame==199:im.save(OUT/f'group_{g+1}.jpg',quality=93)
        print('Rendered group',g+1,flush=True)
    end=page(f"Results: {result['new_passes']} passed / {result['failed_complete_rollouts']} failed / {result['planning_rejections_without_motion']} not run",'Final qualifying hold times • all 20 planned starts; reused examples excluded')
    d=ImageDraw.Draw(end)
    for i,r in enumerate(rows):
        x=55+(i//10)*950;y=150+(i%10)*76;color=GREEN if r['passed'] else RED
        hold='not run' if r['final_hold_s'] is None else f"{r['final_hold_s']:.2f}s"
        verdict='PASS' if r['passed'] else ('FAIL' if r['full20s'] else 'NOT RUN')
        text(d,(x,y),f"{i+1:02d} {verdict}  {r['furniture']}",24,color,True)
        text(d,(x+45,y+31),f"{r['task']} / {r['start']} — final hold {hold}",21,MUTED)
    end.save(OUT/'results.jpg',quality=94)
    for _ in range(8*FPS):emit(end)
    process.stdin.close();assert process.wait()==0;assert count==52*FPS
    (OUT/'video_manifest.json').write_text(json.dumps(dict(path=str(target),sha256=sha(target),duration_s=52,frames=count,width=W,height=H,fps=FPS,playback_speed=2.5,source='saved physical poses; rejected starts show static layouts; no resimulation',new_planned_starts=20,complete_motion_rollouts=result['full20s_rollouts'],source_results_sha256=sha(OUT/'results.json'),renderer_sha256=sha(Path(__file__)),visuals_sha256=sha(ROOT/'outputs/capability_review_20261105_v1/visuals.npz'),source_frozen_inputs_sha256=sha(SRC/'freeze.json'),gpu_used=False),indent=2)+'\n')
    print('Video complete',count,'frames',flush=True)
if __name__=='__main__':main()
