"""A clean, provenance-labeled capability reel from existing trajectories only."""
import sys, json, hashlib, subprocess, functools
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.spatial.transform import Rotation

ROOT=Path('/home/kadir/humanoid-sitting')
OUT=ROOT/'outputs/armchair_rest_policy_video_20261105_v1'
RUN=ROOT/'outputs/armchair_rest_ppo_20261105_v1'
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
    text(d,(56,1030),'Saved simulation replay  /  original success rules  /  4090-only training',21,MUTED)
    return im

right=np.array([.85,np.sqrt(1-.85**2),0.])
eye=np.array([right[1]*.96,-right[0]*.96,.28]);eye/=np.linalg.norm(eye)
up=np.cross(eye,right);VIEW=np.stack([right,-up,eye],axis=1)
visual={}
with np.load(ROOT/'outputs/capability_review_20261105_v1/visuals.npz') as z:
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
    def __init__(self,folder,index,width,height,mesh=True):
        self.folder=folder;self.w=width;self.h=height;self.index=index
        self.audit=read(folder/'independent_audit.json');assert self.audit.get('passed',self.audit.get('passed_integrity'))
        expected=(self.audit.get('files') or self.audit['source_hashes'])['trajectory.npz'];assert sha(folder/'trajectory.npz')==expected
        bindings[str(folder/'trajectory.npz')]=expected;bindings[str(folder/'independent_audit.json')]=sha(folder/'independent_audit.json')
        self.result=read(folder/'results.json');self.row=self.result['episodes'][index];self.names=self.result['body_names']
        origin=np.array(self.result['origins'][index])
        with np.load(folder/'trajectory.npz') as z:
            self.pose=z['body'][1::2,index].copy();self.pose[:,:,:3]-=origin
            self.hold=z['held_ticks'][1::2,index]*.02
            self.failure=z['failure'][1::2,index].copy();self.slip=~z['no_slip'][1::2,index];self.good=z['good'][1::2,index]
        assert len(self.pose)==500
        self.ids=[(self.names.index(k),m) for k,m in visual.items() if k in self.names]
        if mesh:
            case=cases['myd1WSucAz_rear_back']
            with np.load(case['mesh_npz']) as z:self.fv=z['vertices'];self.ff=np.ascontiguousarray(z['faces'],np.int32)
        else:
            g=read(ROOT/'outputs/full_body_contact_imitation_v4/dataset/sitting/geometry.json')[index]
            v=np.array([[x,y,z] for z in [-.1,0.] for y in [-g['width']/2,g['width']/2] for x in [-g['depth']/2,g['depth']/2]])
            self.fv=v@Rotation.from_quat(g['rotation_xyzw']).as_matrix().T+g['top_center']
            self.ff=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]],np.int32)
        self.fc=np.tile([76,144,148] if mesh else [153,137,113],(len(self.ff),1)).astype(np.uint8)
        bounds=[self.fv@VIEW]
        for j,(v,f,c) in self.ids:
            lo,hi=v.min(0),v.max(0);corners=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
            R=Rotation.from_quat(self.pose[:,j,3:7]).as_matrix();pts=np.einsum('tij,kj->tki',R,corners)+self.pose[:,j,None,:3];bounds.append(pts.reshape(-1,3)@VIEW)
        bounds=np.concatenate(bounds);lo,hi=bounds.min(0),bounds.max(0)
        self.scale=min((width-90)/(hi[0]-lo[0]),(height-65)/(hi[1]-lo[1]));self.offset=np.array([width/2,height/2,0])-(lo+hi)*.5*self.scale
        self.base=np.full((height,width,3),BG,np.uint8);self.depth=np.full((height,width),-np.inf,np.float32)
        floor=Image.fromarray(self.base);d=ImageDraw.Draw(floor)
        for v in np.arange(-2.5,2.6,.5):
            for pts in [np.array([[v,-2.5,0],[v,2.5,0]]),np.array([[-2.5,v,0],[2.5,v,0]])]:d.line([tuple(a) for a in self.project(pts)[:,:2]],fill='#e7ecef',width=1)
        self.base=np.array(floor);paint(self.base,self.depth,width,height,self.project(self.fv),self.ff,self.fc,len(self.ff))
    def project(self,v):return np.ascontiguousarray((v@VIEW)*self.scale+self.offset,np.float32)
    def frame(self,i):
        rgb,depth=self.base.copy(),self.depth.copy()
        for j,(v,f,c) in self.ids:
            p=self.pose[i,j];world=v@Rotation.from_quat(p[3:7]).as_matrix().T+p[:3];paint(rgb,depth,self.w,self.h,self.project(world),f,c,len(f))
        return Image.fromarray(rgb)

def main():
    OUT.mkdir(exist_ok=True)
    learned=RUN/'learned_armchair11';sitting=RUN/'learned_sitting72'
    result=read(learned/'results.json');audit=read(learned/'independent_audit.json');prior=read(RUN/'prior_armchair11/independent_audit.json')
    passes=audit.get('validated_clean',audit['clean']);priorpasses=prior.get('validated_clean',prior['clean'])
    sitrows=read(sitting/'results.json')['episodes'];protect=read(ROOT/'experiments/motion_prior_ppo_20261029_v1/protected_sitting_cases.json')
    repairs=[r['id'] for r in sitrows if r['id'] in protect['failed'] and r['clean']];regress=[r['id'] for r in sitrows if r['id'] in protect['protected'] and not r['clean']]
    sitpass=next((i for i,r in enumerate(sitrows) if r['clean']),0);sitfail=next((i for i,r in enumerate(sitrows) if r['id'] in protect['failed'] and not r['clean']),4)
    # Fixed examples show central, unlearned-angle and reserved starts, whether pass or fail.
    indices=[0,5,7,10];titles=['Center / training start','+3 degrees / training start','Near-left / reserved start','Far-right / reserved start']
    scenes=[Scene(learned,i,880,610) for i in indices]
    seats=[Scene(sitting,i,880,610,False) for i in [sitpass,sitfail]]
    pelvis=Scene(RUN/'demo_rear_pelvis_center',0,1240,760)
    def intro():
        im=base('POLICY REVIEW','Expanded armchair training: measured capabilities','Frozen motion prior + learned 23-joint PPO residual  / 128 updates / seed 65011');d=ImageDraw.Draw(im)
        im.paste(scenes[0].frame(499),(180,220));d=ImageDraw.Draw(im)
        text(d,(1300,310),'ARMCHAIR STARTS',25,GREEN,True)
        text(d,(1300,365),f'{sum(passes)} / 11 clean',45,INK,True)
        text(d,(1300,441),f'Before PPO: {sum(priorpasses)} / 11',28,MUTED)
        text(d,(1300,559),'ORIGINAL SITTING',25,GREEN,True)
        text(d,(1300,611),f'{sum(r["clean"] for r in sitrows)} / 72 clean',40,INK,True)
        text(d,(60,840),'A clean trial needs a final five-second hold, with no slipping or illegal contact.',29)
        text(d,(60,894),'These are recorded simulations. The furniture mesh and robot motion are preserved.',26,MUTED)
        return im
    def rest(i,reserved=False):
        im=base('02 / RESERVED STARTS' if reserved else '01 / LEARNED ARMCHAIR POLICY',
            'New positions and angles, held out of PPO.' if reserved else 'Complete standing-start attempts after PPO.',
            '20-second attempts at 2x playback / original physical scoring' if reserved else '20-second attempts at real speed / original physical scoring')
        offset=2 if reserved else 0
        for j in range(2):
            sc=scenes[offset+j];title=titles[offset+j];x=60+930*j;im.paste(sc.frame(i),(x,246));d=ImageDraw.Draw(im)
            text(d,(x+18,209),title,27,INK,True)
            ok=passes[indices[offset+j]];result='PASS' if ok else 'FAIL';color=GREEN if ok else RED
            text(d,(x+18,867),f'Hold now {sc.hold[i]:.2f}s / final {sc.hold[-1]:.2f}s',30,color,True)
            text(d,(x+18,921),f'Final result: {result}',26,color,True)
        text(ImageDraw.Draw(im),(1630,34),f'{(i+1)/25:05.2f} / 20s',23,MUTED)
        return im
    def sit(i):
        im=base('03 / PRESERVED SITTING','The original sitting skill is kept separate.','Frozen demonstrated sitting branch  /  new PPO residual is disabled for sitting  /  complete 20-second attempts at 2x playback')
        for j,(sc,title) in enumerate(zip(seats,['Previously successful case','Previously failing case'])):
            x=60+j*930;im.paste(sc.frame(i),(x,244));d=ImageDraw.Draw(im)
            text(d,(x+20,209),title+'  '+sc.row['id'],26,INK,True)
            color=GREEN if sc.row['clean'] else RED
            text(d,(x+20,865),f'Hold now {sc.hold[i]:.2f}s / final {sc.hold[-1]:.2f}s',30,color,True)
            text(d,(x+20,920),'PASS: no slipping or illegal contact' if sc.row['clean'] else 'FAIL: slipping occurred earlier in the trial',24,color)
        return im
    def prototype(i):
        im=base('04 / UNSOLVED POSTURE','Straighter legs need a different approach path.','Separate demonstration prototype; excluded from PPO training  /  first 6 seconds at real speed')
        im.paste(pelvis.frame(i),(0,210));d=ImageDraw.Draw(im)
        text(d,(1260,320),'WHAT WENT WRONG',25,RED,True)
        text(d,(1260,384),'Torso hits the upper edge',28,INK,True)
        text(d,(1260,429),'before pelvis support.',28,INK,True)
        text(d,(1260,515),'First invalid contact: 3.28 s',25,RED)
        text(d,(1260,586),'The safety logic then',26,MUTED)
        text(d,(1260,626),'freezes commands.',26,MUTED)
        text(d,(1260,752),'No straight-leg rest',28,RED,True)
        text(d,(1260,794),'has been validated.',28,RED,True)
        return im
    def end():
        im=base('05 / RESULTS AND LIMITS','What changed — and what remains open','Original action limits and success criteria were preserved.');d=ImageDraw.Draw(im)
        lines=[('ARMCHAIR',f'Prior {sum(priorpasses)}/11 → learned {sum(passes)}/11 clean.',f'Training starts {sum(passes[:7])}/7; reserved starts {sum(passes[7:])}/4.',RED if sum(passes)<sum(priorpasses) else GREEN),('SITTING',f'{sum(r["clean"] for r in sitrows)}/72 clean; {len(regress)} regressions.',f'{len(repairs)} of the eight prior failures repaired.',GREEN if not regress else RED),('STILL OPEN','Straight-leg pelvis support and recovery.','Other furniture and wider placements remain unproven.',RED),('METHOD','Motion prior + PPO on nearby armchair starts.','Simulation tests; motor effort is an energy proxy.',MUTED)]
        for j,(tag,a,b,col) in enumerate(lines):
            y=255+170*j;text(d,(65,y),tag,26,col,True);text(d,(380,y-2),a,32,INK,True);text(d,(380,y+50),b,27,MUTED)
        return im
    segments=[('intro',3*FPS,lambda i:intro()),('armchair_training_starts',20*FPS,rest),('armchair_reserved_starts',10*FPS,lambda i:rest(2*i+1,True)),('sitting',10*FPS,lambda i:sit(2*i+1)),('pelvis_prototype',6*FPS,prototype),('results',7*FPS,lambda i:end())]
    if '--preview' in sys.argv:
        for name,im in [('intro',intro()),('armchair_start',rest(0)),('armchair_final',rest(499)),('reserved_final',rest(499,True)),('sitting_final',sit(499)),('pelvis',prototype(120)),('results',end())]:im.save(OUT/(name+'.png'))
        return
    path=OUT/'learned_policy_and_limits.mp4';assert not path.exists()
    p=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','25','-i','-','-an','-c:v','libx264','-threads','3','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    for name,n,fn in segments:
        cached=fn(0) if name in ['intro','results'] else None
        for i in range(n):p.stdin.write((cached if cached is not None else fn(i)).tobytes())
        print(name,'complete',flush=True)
    p.stdin.close();assert p.wait()==0
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,duration','-of','json',str(path)]))
    assert probe['streams'][0]['nb_read_frames']=='1400' and float(probe['streams'][0]['duration'])==56
    write=dict(video_sha256=sha(path),renderer_sha256=sha(__file__),visual_mesh_sha256=sha(ROOT/'outputs/capability_review_20261105_v1/visuals.npz'),probe=probe,bindings=bindings,selected_armchair_indices=indices,selected_sitting_indices=[sitpass,sitfail],segments=[dict(name=n,seconds=t/FPS) for n,t,_ in segments],rendering='CPU projection of saved body poses and real robot visuals, neutral materials, no motion editing or physics rerun',ppo_checkpoint_sha256=sha(RUN/'training_65011/final.pt'),physics_steps=0,simulation_only=True)
    (OUT/'video_manifest.json').write_text(json.dumps(write,indent=2)+'\n');print(path,flush=True)
if __name__=='__main__':main()
