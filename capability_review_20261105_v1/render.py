"""A clean, provenance-labeled capability reel from existing trajectories only."""
import sys, json, hashlib, subprocess, functools
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.spatial.transform import Rotation

ROOT=Path('/home/kadir/humanoid-sitting')
OUT=ROOT/'outputs/capability_review_20261105_v1'
sys.path.insert(0,str(ROOT/'experiments/furniture_saved_review_20261104_v1'))
from video import paint
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
W,H,FPS=1920,1080,25
BG=(246,248,250); INK='#152b3b'; MUTED='#596c79'; GREEN='#08785e'; RED='#b3443d'
@functools.lru_cache(None)
def font(size,bold=False):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if bold else '')+'.ttf',size)
def text(d,xy,s,size=28,color=INK,bold=False):
    d.text(xy,s,font=font(size,bold),fill=color)
def badge(d,xy,s,color=GREEN):
    b=d.textbbox((0,0),s,font=font(21,True)); x,y=xy
    d.rounded_rectangle((x,y,x+b[2]+32,y+40),radius=8,fill=color)
    text(d,(x+16,y+6),s,21,'white',True)
def base(chapter,title,subtitle):
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
    text(d,(56,30),chapter,23,GREEN,True)
    text(d,(54,72),title,47,INK,True)
    text(d,(56,139),subtitle,25,MUTED)
    d.line((56,1010,1864,1010),fill='#dce3e8',width=2)
    text(d,(56,1030),'Recorded simulation  /  original scoring rules  /  no new training or physics',21,MUTED)
    return im

right=np.array([.85,np.sqrt(1-.85**2),0.])
eye=np.array([right[1]*.96,-right[0]*.96,.28]);eye/=np.linalg.norm(eye)
up=np.cross(eye,right);VIEW=np.stack([right,-up,eye],axis=1)
visual={}
with np.load(OUT/'visuals.npz') as z:
    for k in z.files:
        if k.endswith('_v'):
            name=k[:-2];v=z[k];f=z[name+'_f']
            col=[211,224,233]
            if any(n in name for n in ['ankle','palm','elbow','five','six','three','four','one','two','zero']):col=[64,83,97]
            if name in ['head_link','pelvis']:col=[81,111,133]
            visual[name]=(v,f,np.tile(col,(len(f),1)).astype(np.uint8))
cases={c['id']:c for c in read(ROOT/'experiments/furniture_test_20261028_v1/cases.json')}
bindings={}

class Scene:
    def __init__(self,folder,case_id,width,height,current=False):
        self.folder=folder;self.w=width;self.h=height;self.current=current
        self.audit=read(folder/'independent_audit.json')
        expected=self.audit['source_hashes']['trajectory.npz'] if current else self.audit['trajectory_sha256']
        assert sha(folder/'trajectory.npz')==expected
        assert self.audit['mapping_validated'] if current else self.audit['contact_mapping_validated']
        assert self.audit['qualifying_lean'] if current else self.audit['clean'] is False
        bindings[str(folder/'trajectory.npz')]=expected
        bindings[str(folder/'independent_audit.json')]=sha(folder/'independent_audit.json')
        self.result=read(folder/'results.json');self.manifest=read(folder/'manifest.json')
        bindings[str(folder/'manifest.json')]=sha(folder/'manifest.json')
        self.names=self.result['body_names']
        with np.load(folder/'trajectory.npz') as z:
            self.pose=z['body'][1::2,0].copy()
            self.hold=z['held_ticks'][1::2,0]*.02 if current else np.zeros(500)
        assert len(self.pose)==500
        self.ids=[(self.names.index(k),m) for k,m in visual.items() if k in self.names]
        case=cases[case_id]
        with np.load(case['mesh_npz']) as z:
            self.fv=z['vertices'];self.ff=np.ascontiguousarray(z['faces'],np.int32)
        self.fc=np.tile([76,144,148] if 'myd1' in case_id else [177,133,101],(len(self.ff),1)).astype(np.uint8)
        bounds=[self.fv@VIEW]
        # Bound actual visual extents, not body origins: no clipped head or feet.
        for j,(v,f,c) in self.ids:
            lo,hi=v.min(0),v.max(0)
            corners=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
            R=Rotation.from_quat(self.pose[:,j,3:7]).as_matrix()
            pts=np.einsum('tij,kj->tki',R,corners)+self.pose[:,j,None,:3]
            bounds.append(pts.reshape(-1,3)@VIEW)
        bounds=np.concatenate(bounds);lo,hi=bounds.min(0),bounds.max(0)
        self.scale=min((width-120)/(hi[0]-lo[0]),(height-100)/(hi[1]-lo[1]))
        self.offset=np.array([width/2,height/2,0])-(lo+hi)*.5*self.scale
        self.base=np.full((height,width,3),BG,np.uint8)
        self.depth=np.full((height,width),-np.inf,np.float32)
        # Subtle floor grid with the physical z=0 plane.
        floor=Image.fromarray(self.base);d=ImageDraw.Draw(floor)
        for v in np.arange(-2.5,2.6,.5):
            for pts in [np.array([[v,-2.5,0],[v,2.5,0]]),np.array([[-2.5,v,0],[2.5,v,0]])]:
                uv=self.project(pts)[:,:2];d.line([tuple(a) for a in uv],fill='#e7ecef',width=1)
        self.base=np.array(floor)
        paint(self.base,self.depth,width,height,self.project(self.fv),self.ff,self.fc,len(self.ff))
    def project(self,v):
        return np.ascontiguousarray((v@VIEW)*self.scale+self.offset,np.float32)
    def frame(self,i):
        rgb,depth=self.base.copy(),self.depth.copy()
        for j,(v,f,c) in self.ids:
            p=self.pose[i,j];R=Rotation.from_quat(p[3:7]).as_matrix()
            world=v@R.T+p[:3]
            paint(rgb,depth,self.w,self.h,self.project(world),f,c,len(f))
        return Image.fromarray(rgb)

def main():
    center=Scene(ROOT/'outputs/box_armchair_lean_20261104_v4/trial_0','myd1WSucAz_rear_back',1260,780,True)
    repeats=[Scene(ROOT/f'outputs/box_armchair_lean_20261104_v4/trial_{j}','myd1WSucAz_rear_back',880,600,True) for j in [1,2]]
    failures=[Scene(ROOT/f'outputs/furniture_perception_repair_20261029_v1/robot_{k}',k,880,600) for k in ['myd1WSucAz_seat_oblique','6MoOyPtetL_arm']]
    def intro():
        im=base('CAPABILITY REVIEW','One repeatable lean. A broader skill still to build.','What the saved evidence supports today')
        im.paste(center.frame(499),(20,200));d=ImageDraw.Draw(im)
        text(d,(1260,340),'PROVEN',27,GREEN,True)
        text(d,(1260,391),'Armchair back lean',32,INK,True)
        text(d,(1260,449),'3 / 3 nearby starts',30)
        text(d,(1260,525),'STILL OPEN',27,RED,True)
        text(d,(1260,577),'General sitting, leaning',27)
        text(d,(1260,619),'and recovery',27)
        return im
    def success(i):
        im=base('01 / DEMONSTRATED NOW','Find the chair. Move back. Hold the lean.','Current camera-guided controller  •  centered standing start  •  complete 20-second attempt')
        im.paste(center.frame(i),(10,196));d=ImageDraw.Draw(im)
        t=(i+1)/FPS;hold=float(center.hold[i])
        badge(d,(1300,253),'5-SECOND HOLD PASSED' if hold>=5 else 'APPROACH & SETTLE')
        text(d,(1300,333),'Continuous hold',28,MUTED)
        text(d,(1295,378),f'{hold:.2f} s',74,GREEN if hold>=5 else INK,True)
        d.rounded_rectangle((1300,492,1800,504),6,fill='#dce5e8')
        if hold>0:d.rounded_rectangle((1300,492,1300+500*min(hold/5,1),504),6,fill=GREEN)
        text(d,(1300,543),'Uses the rear camera',27)
        text(d,(1300,586),'Balances contact forces',27)
        text(d,(1300,662),'Built controller;',25,MUTED)
        text(d,(1300,698),'no new PPO learning.',25,MUTED)
        text(d,(1300,809),f'Trial time  {t:05.2f} / 20 s',25,MUTED)
        return im
    def repeat(i):
        im=base('02 / REPEATABILITY','The same controller passed two more starts.','Final 7 seconds of each saved trial  •  real-time playback  •  original action limits')
        for j,(scene,title) in enumerate(zip(repeats,['Start shifted +15 mm','Start shifted −15 mm'])):
            x=60+930*j;im.paste(scene.frame(i),(x,240));d=ImageDraw.Draw(im)
            text(d,(x+22,210),title,29,INK,True)
            text(d,(x+22,851),f'Hold now  {scene.hold[i]:.2f} s',35,GREEN,True)
            text(d,(x+22,908),f'Final result  {scene.hold[-1]:.2f} s  /  PASS',26)
        return im
    def failure(i):
        im=base('03 / EARLIER PPO FAILURES','Reliable furniture sitting is still unresolved.','Archived PPO-based system  •  29 October  •  these are not tests of the new lean controller')
        rows=[('Armchair seat, angled approach','Unwanted body contact at 8.08 s'),('Couch arm','Fall, unwanted contact and joint-limit violation')]
        for j,(scene,(title,reason)) in enumerate(zip(failures,rows)):
            x=60+930*j;im.paste(scene.frame(i),(x,240));d=ImageDraw.Draw(im)
            text(d,(x+22,210),title,29,INK,True)
            t=(i+1)/FPS
            triggered=t>=scene.audit['first_recorded_failure_s']
            badge(d,(x+22,842),'FAILURE RECORDED' if triggered else 'ATTEMPT IN PROGRESS',RED if triggered else MUTED)
            text(d,(x+22,904),reason,22,RED if triggered else MUTED)
            text(d,(x+22,945),'Final qualifying hold: 0.00 s  /  FAIL',24,INK,True)
        text(ImageDraw.Draw(im),(1620,32),f'{(i+1)/FPS:05.2f} / 20 s',22,MUTED)
        return im
    def end():
        im=base('04 / WHAT REMAINS','Turn a specific success into a reliable general skill.','The armchair result is a starting point, with clear limits.')
        d=ImageDraw.Draw(im)
        rows=[('PROVEN','One chair back; centered and ±15 mm starts.','Three clean holds: 6.84 s, 6.82 s, 6.84 s.',GREEN),
              ('UNRESOLVED','Original sitting benchmark: 64 / 72 clean.','Eight failures still lack an accepted repair.',RED),
              ('UNTESTED','New controller on other furniture or wider offsets.','Push recovery, longer holds and real hardware.',MUTED),
              ('NEXT STEP','Teach and test a broader skill from this success.','Preserve successful behavior while expanding the task.',INK)]
        for j,(tag,a,b,col) in enumerate(rows):
            y=256+j*167;text(d,(70,y),tag,26,col,True)
            text(d,(400,y-2),a,34,INK,True);text(d,(400,y+52),b,28,MUTED)
        return im
    segments=[('intro',2*FPS,lambda i:intro()),('current_lean',20*FPS,success),('repeat_starts',7*FPS,lambda i:repeat(325+i)),('earlier_ppo_failures',20*FPS,failure),('limits',8*FPS,lambda i:end())]
    if '--preview' in sys.argv:
        for name,im in [('intro',intro()),('lean_start',success(0)),('lean_hold',success(499)),('repeats',repeat(499)),('failures_mid',failure(250)),('failures_final',failure(499)),('limits',end())]:
            im.save(OUT/(name+'.png'))
        print('Preview frames ready',flush=True);return
    path=OUT/'capabilities_and_limits.mp4';assert not path.exists()
    p=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r',str(FPS),'-i','-','-an','-c:v','libx264','-threads','3','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    for name,n,fn in segments:
        cached=fn(0) if name in ['intro','limits'] else None
        for i in range(n):
            im=cached if cached is not None else fn(i)
            p.stdin.write(im.tobytes())
        print(name,'complete',flush=True)
    p.stdin.close();assert p.wait()==0
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=width,height,nb_read_frames,duration','-of','json',str(path)]))
    assert int(probe['streams'][0]['nb_read_frames'])==1425 and float(probe['streams'][0]['duration'])==57
    manifest=dict(video_sha256=sha(path),renderer_sha256=sha(__file__),visual_mesh_sha256=sha(OUT/'visuals.npz'),streams=probe['streams'],source_bindings=bindings,segments=[{'name':name,'seconds':n/FPS} for name,n,_ in segments],physics_steps=0,ppo_updates=0,rendering='CPU-only replay of exact saved poses using simplified authored visual meshes, neutral colors and a display floor grid. No collision overlays, no force arrows, no altered motion.',scope='Current model-based lean controller and earlier PPO failure clips are explicitly distinct. Wider current-controller capability is untested.')
    (OUT/'video_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('VIDEO_COMPLETE',path,flush=True)

if __name__=='__main__':main()
