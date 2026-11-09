"""CPU render of recorded collision probes and body-shaped load fixtures."""
import argparse, functools, hashlib, json, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

W,H,FPS=2560,1440,10
CW,CH=628,310
INK='#233b4d';BG='#edf2f5';RED='#bf403e';GREEN='#167657'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
@functools.lru_cache(None)
def font(n,bold=False):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if bold else '')+'.ttf',n)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assets=a.root/'experiments/furniture_test_20261028_v1';old=a.root/'outputs/furniture_validation_20261028_v2/probes';new=a.root/'outputs/motion_prior_ppo_20261029_v2/body_support'
    cases=read(assets/'cases.json');small=read(old/'measurements.json');sr=read(old/'result.json');body=read(new/'measurements.json');br=read(new/'result.json');manifest=read(new/'manifest.json')
    assert br['complete'] and read(new/'independent_audit.json')['passed']
    a.output.mkdir(exist_ok=True,parents=True);path=a.output/'furniture_support_and_forces.mp4';assert not path.exists()
    view=np.array([[.8,.6,0.],[-.18,.24,.954]]);depth=np.cross(view[0],view[1]);cache={}
    for c in cases:
        z=np.load(assets/'scenes'/c['id']/'mesh.npz');v=z['vertices'];tri=v[z['faces']]
        uv=np.vstack([v,[[.5,-.35,0],[.5,.35,1.35]]])@view.T
        sc=min((CW-42)/np.ptp(uv[:,0]),185/np.ptp(uv[:,1]));of=np.array([CW/2-sc*(uv[:,0].max()+uv[:,0].min())/2,255+sc*uv[:,1].min()])
        def proj(v,sc=sc,of=of):
            q=np.asarray(v)@view.T;return np.stack([of[0]+sc*q[...,0],of[1]-sc*q[...,1]],-1)
        im=Image.new('RGB',(CW,CH),'white');d=ImageDraw.Draw(im)
        for t in tri[np.argsort(tri.mean(1)@depth)]:d.polygon([tuple(x) for x in proj(t)],fill='#c6b99d',outline='#a59d91')
        cache[c['id']]=(im,proj)
    def arrow(d,proj,point,force,scale):
        f=np.asarray(force);p=np.asarray(point)
        if not np.isfinite(p).all() or np.linalg.norm(f)<.1:return
        start,end=proj(np.array([p,p+f*scale]));d.line([tuple(start),tuple(end)],fill=RED,width=4)
        delta=end-start;length=np.linalg.norm(delta)
        if length>4:
            u=delta/length;v=np.array([-u[1],u[0]]);d.polygon([tuple(end),tuple(end-9*u+4*v),tuple(end-9*u-4*v)],fill=RED)
    def grid(kind,tick):
        im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im);ri,k=divmod(tick,60)
        title='Small probes: three surface presses and one empty-space control' if kind=='small' else 'Robot body shapes: controlled load tests, not autonomous sitting'
        d.text((22,14),title,font=font(30,True),fill=INK)
        d.text((22,55),'Recorded physics at 1x speed. Furniture lines show collision mesh; red arrows show measured contact forces.',font=font(22),fill=INK)
        d.text((22,88),'Small-probe arrow scale: 2 cm/N. Body-fixture arrow scale: 2 mm/N. Original robot success criteria unchanged.',font=font(20),fill=INK)
        for ci,c in enumerate(cases):
            panel,proj=cache[c['id']];panel=panel.copy();q=ImageDraw.Draw(panel)
            q.text((12,8),(c['title']+' / '+c['approach'].replace('_',' '))[:48],font=font(18,True),fill=INK)
            row=(small if kind=='small' else body)[ri][k*5][ci]
            result=next(x for x in (sr if kind=='small' else br)['checks'] if x['case']==c['id'] and x['round']==ri)
            if kind=='small':
                xy=proj(row['position']);q.ellipse((xy[0]-5,xy[1]-5,xy[0]+5,xy[1]+5),fill='#3089a2',outline=INK,width=2)
                mechanical=all(v for key,v in result['checks'].items() if key!='target_mapping')
                label=('PRESS '+str(ri+1) if ri<3 else 'EMPTY CONTROL')+' | '+('SOLID' if mechanical else 'CHECK FAILED')
                patch=result['checks'].get('target_mapping');scale=.02;fallback=row['position']
            else:
                pose=np.asarray(row['pose']);rot=Rotation.from_quat(pose[3:]).as_matrix();origin=np.array([0.,ci*8.,0.]);drawtri=[]
                for s in manifest['bodies'][row['body']]:
                    v=np.asarray(s['vertices'])@rot.T+pose[:3]-origin
                    for tr in v[np.asarray(s['faces'])]:drawtri.append((float(tr.mean(0)@depth),tr))
                for _,tr in sorted(drawtri,key=lambda x:x[0]):q.polygon([tuple(x) for x in proj(tr)],fill='#92b8cc',outline='#46748e')
                mechanical=result['physical_support'];patch=result['intended_contact_patch'];scale=.002;fallback=pose[:3]-origin
                label=row['body'].replace('_link','')+' | '+('SUPPORTS LOAD' if mechanical else 'SUPPORT FAIL')
            cp=np.mean(row['points'],axis=0) if row['points'] else fallback
            arrow(q,proj,cp,row['force'],scale)
            q.text((12,34),label,font=font(17,True),fill=GREEN if mechanical else RED)
            if patch is not None:q.text((12,58),'Intended patch: '+('PASS' if patch else 'EDGE / OTHER FACE'),font=font(16,True),fill=GREEN if patch else RED)
            q.text((12,278),f"t={row['t']:4.2f}s   measured force {np.linalg.norm(row['force']):6.1f} N",font=font(17),fill=INK)
            q.rectangle((1,1,CW-2,CH-2),outline='#becbd3',width=2);im.paste(panel,(14+(CW+8)*(ci%4),135+(CH+8)*(ci//4)))
        x=14+2*(CW+8);y=135+3*(CH+8)
        lines=['WHAT THIS TEST MEANS','Bodies are force-controlled fixtures.',f"Target load: {manifest['load_N']:.1f} N (half robot weight).",'Gravity is disabled; orientation held.','Furniture is rigid and fixed in place.','Support PASS is not robot success.','Red patch label means scoring/reach','needs attention at that location.']
        for i,line in enumerate(lines):d.text((x+12,y+12+i*30),line,font=font(20,i==0),fill=INK)
        return im
    def slate(lines):
        im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
        for i,line in enumerate(lines):d.text((110,250+i*110),line,font=font(40 if i==0 else 30,i==0),fill=INK)
        return im
    proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    def hold(lines,count):
        frame=slate(lines)
        for _ in range(count):proc.stdin.write(frame.tobytes())
    hold(['Can this furniture physically support the robot?','14 placements: seats, arms, backs, stools and tables.','First: small calibrated probes. Then: actual G1 body collision shapes.','Saved simulation replay; no invented motion or forces.'],30)
    for kind,count in [('small',240),('body',180)]:
        if kind=='body':hold(['Body-shape capacity is separate from whole-robot balance.',f"42 controlled fixtures press with {manifest['load_N']:.1f} N each.",'A physically solid surface may still be unusable by the current policy.'],20)
        for tick in range(count):
            frame=grid(kind,tick);proc.stdin.write(frame.tobytes())
            if tick in [35,count-1]:frame.save(a.output/f'{kind}_{tick:03d}.png')
    support=sum(x['physical_support'] for x in br['checks']);patch=sum(x['intended_contact_patch'] for x in br['checks'])
    hold(['Measured results',f"Body fixtures supporting target load: {support}/42. Intended patch: {patch}/42.",'Whole-robot furniture balance and softness are not established by this test.','All 24 seconds of small probes and 18 seconds of body tests shown at 1x.'],30)
    proc.stdin.close();assert proc.wait()==0
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=width,height,nb_frames','-of','json',str(path)]))
    assert float(info['format']['duration'])==50 and int(info['streams'][0]['nb_frames'])==500
    report=dict(path=str(path),sha256=sha(path),duration_s=50,frames=500,small_probe_seconds=24,body_fixture_seconds=18,playback_speed=1,extra_simulation_steps=0,robot_success_claimed=False,body_supports=support,intended_patches=patch,
        inputs={str(p):sha(p) for p in [old/'measurements.json',old/'result.json',new/'measurements.json',new/'result.json',new/'independent_audit.json']})
    (a.output/'video_manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(report)

if __name__=='__main__':main()
