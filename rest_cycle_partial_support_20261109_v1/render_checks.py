"""CPU-only evidence video. No invented motion or successful-cycle claim."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys,json,hashlib,subprocess,time,signal,resource
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw,ImageFont

R=Path('/home/kadir/humanoid-sitting');E=R/'experiments';H=Path(__file__).resolve().parent
S=R/'outputs'/H.name/'camera_and_wrench_capture';O=R/'outputs'/H.name/'video'
sys.path.insert(0,str(E/'furniture_saved_review_20261104_v1'))
from video import paint,group_shapes
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
W,HEIGHT,FPS=1280,800,10
BG=(245,247,250);INK='#183444';RED='#a53b36';GREEN='#13745a';GRAY='#526675'
@lru_cache(None)
def font(n,bold=False):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if bold else '')+'.ttf',n)
def text(d,xy,s,n=23,color=INK,bold=False):d.text(xy,s,font=font(n,bold),fill=color)

def main():
    start=time.monotonic();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
    def deadline(*a):raise TimeoutError('45_SECOND_RENDER_CAP')
    signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,45)
    for p,h in read(H/'render_freeze.json')['files'].items():assert sha(p)==h,p
    O.mkdir(exist_ok=False)
    audit=read(S/'capture_audit.json');edge=read(S/'edge_sensor.json')
    assert audit['edge_passed'] and audit['edge_envelope_passed'] and not audit['complete_cycle']
    with np.load(R/'outputs/rest_cycle_edge_visible_20261108_v1/sit_edge_visible/frame_0.npz') as z:
        before=Image.fromarray(z['rgb'][...,:3].astype(np.uint8))
    with np.load(S/'frame_0.npz') as z:
        rgb=z['rgb'][...,:3].astype(np.uint8).copy()
        shade=np.where(np.isfinite(z['depth']),1-np.clip((z['depth']-.05)/1.5,0,1),0)
    with np.load(S/'edge_arrays.npz') as z:
        mask=z['plane_mask'];segments=z['edge_segments']
    marked=rgb.copy();marked[mask]=.65*marked[mask]+.35*np.array([35,215,145])
    annotated=Image.fromarray(marked);ad=ImageDraw.Draw(annotated)
    for row in segments:
        r,c=map(int,row[6:8]);ad.rectangle((c-1,r-1,c+1,r+1),fill=(255,210,25))
    depth=Image.fromarray(np.repeat((255*shade).astype(np.uint8)[...,None],3,-1))
    with np.load(S/'standing_calibration.npz') as z:
        poses=z['body'][:,0];load=z['support_bw'][:,0];force=z['force'][:,0]
    names=read(S/'results.json')['body_names']
    shapes=group_shapes(read(E/'furniture_perception_repair_20261029_v1/render_colliders.json')['shapes'])
    with np.load(E/'rest_cycle_high_support_20261108_v1/assets/scenes/high_support_table_050/mesh.npz') as z:
        fv,ff=z['vertices'],np.ascontiguousarray(z['faces'],np.int32)
    right=np.array([.96,.28,0.]);eye=np.array([.28,-.96,.15]);eye/=np.linalg.norm(eye)
    view=np.stack([right,-np.cross(eye,right),eye],axis=1)
    pw,ph=780,510
    pts=[fv@view]
    for name,(v,f,c) in shapes.items():
        p=poses[0,names.index(name)];pts.append((v@Rotation.from_quat(p[3:]).as_matrix().T+p[:3])@view)
    pts=np.concatenate(pts);lo,hi=pts.min(0),pts.max(0)
    scale=min((pw-70)/(hi[0]-lo[0]),(ph-40)/(hi[1]-lo[1]))
    offset=np.array([pw/2,ph/2,0])-(lo+hi)*scale/2
    def project(v):return np.ascontiguousarray(v@view*scale+offset,np.float32)
    def mesh(img,depth,v,f,col):paint(img,depth,pw,ph,project(v),np.ascontiguousarray(f,np.int32),np.tile(col,(len(f),1)).astype(np.uint8),len(f))
    base=np.full((ph,pw,3),BG,np.uint8);zb=np.full((ph,pw),-np.inf,np.float32)
    mesh(base,zb,fv,ff,[183,153,112])
    @lru_cache(None)
    def scene(i):
        img,db=base.copy(),zb.copy()
        for name,(v,f,c) in shapes.items():
            p=poses[i,names.index(name)];col=[166,188,205]
            if name in ('pelvis','left_hip_yaw_link','right_hip_yaw_link'):col=[43,151,134]
            if name in ('left_hip_roll_link','right_hip_roll_link'):col=[214,90,67]
            mesh(img,db,v@Rotation.from_quat(p[3:]).as_matrix().T+p[:3],f,col)
        im=Image.fromarray(img);d=ImageDraw.Draw(im)
        for side,j in [('left',5),('right',6)]:
            pos=poses[i,names.index(side+'_ankle_roll_link'),:3]
            a,b=project(np.stack([pos,pos+force[i,j]*.001]))[:,:2]
            d.line([tuple(a),tuple(b)],fill='#277ccc',width=4)
        return im
    target=O/'camera_and_release_checks.mp4'
    proc=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{HEIGHT}',
        '-r',str(FPS),'-i','-','-an','-c:v','libx264','-threads','1','-preset','fast','-crf','20',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(target)],stdin=subprocess.PIPE)
    try:
        for frame in range(180):
            t=frame/FPS;im=Image.new('RGB',(W,HEIGHT),BG);d=ImageDraw.Draw(im)
            text(d,(30,22),'Rear-camera and contact checks',34,bold=True)
            text(d,(30,74),'Actual saved simulation evidence | no new sitting motion',22,RED)
            if t<6:
                im.paste(before.resize((576,432)),(32,150));im.paste(annotated.resize((576,432)),(670,150));d=ImageDraw.Draw(im)
                text(d,(32,114),'BEFORE: robot hides the front edge',22,RED,True)
                text(d,(670,114),'AFTER: true edge detected in depth',22,GREEN,True)
                text(d,(32,615),'Virtual camera moved 34.7 mm rearward and re-aimed.',26,bold=True)
                text(d,(32,659),'Green: support plane. Yellow: observed depth drop-off.',24)
                text(d,(32,702),f"Side estimates differ by {edge['side_center_difference_m']*1000:.3f} mm; limit 3 mm.",23,GREEN)
            elif t<12:
                # One actual second at1x, followed by its explicitly frozen last frame.
                i=min(49,int((t-6)*50))
                im.paste(scene(i),(8,118));im.paste(depth.resize((384,288)),(870,160));d=ImageDraw.Draw(im)
                text(d,(30,115),'Measured standing poses + collision shapes',22,bold=True)
                text(d,(870,123),'Measured rear depth',21,bold=True)
                text(d,(870,478),'Blue: measured foot forces',19)
                text(d,(870,516),'Table support: 0%',23)
                text(d,(30,650),'Correctly recognized: robot is clear of the finite table.',25,GREEN,True)
                text(d,(30,695),'1-second standing replay at1x' if t<7 else 'Standing replay ended - final measured frame held',22,GRAY)
            else:
                im.paste(scene(49),(8,118));d=ImageDraw.Draw(im)
                text(d,(825,164),'PASSED',25,GREEN,True)
                text(d,(825,207),'Standing camera edge check',21)
                text(d,(825,247),'Standing separation check',21)
                text(d,(825,287),'Standing force mapping',21)
                text(d,(825,370),'NOT YET DEMONSTRATED',23,RED,True)
                text(d,(825,413),'Clearance-safe sitting path',21)
                text(d,(825,453),'Loaded release / stand-up',21)
                text(d,(825,493),'New five-second rest cycle',21)
                text(d,(30,650),'New perch construction failed its model checks; it was not executed.',24,RED,True)
                text(d,(30,695),'Partial legal hip / thigh support is enough. The full cycle remains unfinished.',23)
            text(d,(30,760),'4090 capture only | 4070 untouched | unchanged action limits and physical success criteria',19,GRAY)
            proc.stdin.write(im.tobytes())
            if frame in (30,65,150):im.save(O/f'frame_{frame}.jpg',quality=95)
        proc.stdin.close();assert proc.wait()==0
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
    result=dict(video_sha256=sha(target),duration_s=18,frames=180,fps=FPS,
        new_simulation_steps=0,new_solver_calls=0,GPU_used=False,complete_cycle=False,
        content='Actual old/current camera snapshots, current measured1second standing replay at1x then labelled frozen frame; no invented sitting or stand-up.',
        elapsed_seconds=time.monotonic()-start,source_freeze_sha256=sha(H/'render_freeze.json'))
    (O/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    signal.setitimer(signal.ITIMER_REAL,0)
if __name__=='__main__':main()
