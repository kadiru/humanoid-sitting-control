"""CPU replay of saved physical poses, showing all20 trials and measured holds."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,hashlib,subprocess,functools
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('/home/kadir/humanoid-sitting');OUT=ROOT/'outputs/furniture_collision_review_20261106_v1';SRC=ROOT/'experiments/furniture_examples_20261105_v2'
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
        self.row=row;self.w=900;self.h=560;self.folder=Path(row['source']);self.static=not row['full20s']
        if self.static:
            self.card=Image.open(SRC/'previews'/(row['id']+'.jpg')).convert('RGB').crop((10,55,590,495));return
        sit=False;name='furniture_trajectory.npz' if sit else 'trajectory.npz'
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
    OUT.mkdir(parents=True,exist_ok=True)
    root=ROOT/'outputs/furniture_camera_coverage_20261106_v1/calibration_demos'
    rows=[];scenes=[];signals=[];data=[]
    for name in ['sit_1kwsjhpY84_center','sit_1kwsjhpY84_angled']:
        folder=root/(name+'_local_patch_normal_landing');a=read(folder/'independent_audit.json')
        row=dict(id=name,task='sitting',source=str(folder),full20s=True,passed=a['admitted'],final_hold_s=a['final_hold_s'],first_failure_s=a['first_failure_s'])
        rows.append(row);scenes.append(Scene(row));signals.append(read(folder/'controller_signals.json'))
        with np.load(folder/'trajectory.npz') as z:data.append(dict(z))
    target=OUT/'l_couch_two_full_trials.mp4'
    cmd=['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
    process=subprocess.Popen(cmd,stdin=subprocess.PIPE);count=0
    def emit(im):
        nonlocal count
        process.stdin.write(im.tobytes());count+=1
    intro=page('L-couch: two complete physical simulation attempts','Model collision checks passed; neither attempt passed the physical sitting test.')
    d=ImageDraw.Draw(intro)
    for j,line in enumerate(['Both20-second starts are shown at normal speed.','Pass needs5 seconds of final support with no loaded slipping or illegal contact.','The rear-camera view was tilted from45 to54 degrees for these two trials.','Action limits and success thresholds were unchanged.']):text(d,(75,260+j*95),line,30)
    for _ in range(2*FPS):emit(intro)
    for frame in range(20*FPS):
        tick=min(999,round(frame/(20*FPS-1)*999));im=page('L-couch — center start / angled start',f'Saved physical poses | simulation time {(tick+1)*.02:04.2f}/20.00s | playback1×')
        for j,(scene,row,sg,z) in enumerate(zip(scenes,rows,signals,data)):
            x=40+j*940;im.paste(scene.frame(tick),(x,155));d=ImageDraw.Draw(im)
            text(d,(x+10,125),'CENTER' if j==0 else 'ANGLED3°',27,b=True)
            s=sg[tick];force=z['physical_force_w'][tick,0];normal=z['physical_normal_w'][tick,0]
            load=float(((force*normal).sum(-1).clip(min=0)*z['physical_inside_patch'][tick,0])[:3].sum()/z['physical_bodyweight_N'][tick,0])
            text(d,(x+10,740),f"Hold: {scene.hold[tick]:.2f}s /5.00s   |   validated load: {load:.3f} bodyweight",25)
            text(d,(x+10,786),f"Camera: {'fresh' if s['measured_geometry_valid'] else 'LOST PATCH'}   |   controller: {s['phase']}",25,GREEN if s['measured_geometry_valid'] else RED)
            label='loaded slip detected' if scene.slip[tick] else 'no loaded slip yet'
            text(d,(x+10,832),label,25,RED if scene.slip[tick] else MUTED)
            text(d,(x+10,890),f"Final result: FAIL — {row['final_hold_s']:.2f}s hold",28,RED,True)
            text(d,(x+10,940),'Late contact mapping is ambiguous; zero validated load ≠ no contact.',20,MUTED)
        emit(im)
        if frame in [175,499]:im.save(OUT/f'frame_{frame}.jpg',quality=94)
    end=page('Result: 0 /2 successful demonstrations','These trials are evidence of failure, not training examples labelled successful.')
    d=ImageDraw.Draw(end)
    for j,line in enumerate(['The contact patch left the camera image at about6 seconds.','Both ankle-roll joints kept moving; the stationary landing guard never opened.','Loaded slipping began at12.42s (center) and15.20s (angled).','The stool still has no admitted path. PPO and the72-case evaluation did not run.']):text(d,(70,230+j*115),line,30,RED if j==2 else INK)
    end.save(OUT/'results.jpg',quality=94)
    for _ in range(4*FPS):emit(end)
    process.stdin.close();assert process.wait()==0
    manifest=dict(path=str(target),sha256=sha(target),duration_s=26,frames=count,width=W,height=H,fps=FPS,playback_speed=1,source='saved full20s physical poses, no resimulation',physical_trials=2,successful_demos=0,rows=rows,source_files={str(Path(r['source'])/'trajectory.npz'):sha(Path(r['source'])/'trajectory.npz') for r in rows},renderer_sha256=sha(Path(__file__)),GPU=False)
    (OUT/'video_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(video=str(target),frames=count,duration_s=26)),flush=True)
if __name__=='__main__':main()
