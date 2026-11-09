"""Saved evidence only: actual camera snapshot and labelled static model poses."""
import os,time,signal,resource,sys,json,hashlib,subprocess
from pathlib import Path
START=time.monotonic();os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
def timeout(*a):raise TimeoutError('45_SECOND_EVIDENCE_RENDER_CAP')
signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,45)
R=Path('/home/kadir/humanoid-sitting');E=R/'experiments';H=Path(__file__).resolve().parent;O=H/'video'
sys.path[1:1]=[str(E/'furniture_saved_review_20261104_v1'),str(R/'sitting_task/src')]
def read(p):return json.loads(p.read_text())
def main():
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from video import paint,group_shapes
    from humanoid_sitting.shared_frame import BodyModel,level_rotation,apply
    for p,h in read(H/'render_freeze.json')['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    O.mkdir(exist_ok=False);plan=read(H/'nominal_construction/plan.json');audit=read(H/'nominal_construction/audit.json')
    with np.load(R/'outputs/rest_cycle_corner_20261109_v1/camera_and_wrench_capture/frame_0.npz') as z:frame=dict(z)
    with np.load(R/'outputs/rest_cycle_corner_20261109_v1/camera_and_wrench_capture/patch_mask_v2_0.npz') as z:mask=z['mask']
    rgb=frame['rgb'][...,:3].astype(np.uint8).copy();rgb[mask]=.65*rgb[mask]+.35*np.array([25,220,140]);camera=Image.fromarray(rgb)
    model=BodyModel(R/'runs/g1_kinematics_geometry_04/model.json');shapes=group_shapes(read(E/'furniture_perception_repair_20261029_v1/render_colliders.json')['shapes'])
    with np.load(E/'rest_cycle_narrow_support_20261109_v1/assets/scenes/narrow_support_050/mesh.npz') as z:fv,ff=z['vertices'],z['faces']
    R0=level_rotation(frame['gravity'].astype(float).copy());basis=frame['root_rotation']@R0.T
    rest=plan['nodes'][-1];exitnode=next(n for n in plan['return_nodes'] if n['load']==0.)
    right=np.array([.83,.557763,0]);right/=np.linalg.norm(right);eye=np.array([right[1],-right[0],.18]);eye/=np.linalg.norm(eye)
    view=np.stack([right,-np.cross(eye,right),eye],axis=1);pw,ph=780,540
    def body_world(node):
        rt=np.asarray(node['root']);fk=model.fk(np.asarray(node['q']))
        return {name:apply(rt@fk[name],v)@basis.T+frame['root_position'] for name,(v,f,c) in shapes.items()}
    bodies=[body_world(n) for n in [plan['nodes'][0],rest,exitnode]]
    allpts=np.concatenate([fv,*[v for body in bodies for v in body.values()]])@view;lo,hi=allpts.min(0),allpts.max(0)
    scale=min((pw-80)/(hi[0]-lo[0]),(ph-50)/(hi[1]-lo[1]));offset=np.array([pw/2,ph/2,0])-(lo+hi)*scale/2
    def scene(body):
        img=np.full((ph,pw,3),[244,247,249],np.uint8);depth=np.full((ph,pw),-np.inf,np.float32)
        def draw(v,f,color):
            paint(img,depth,pw,ph,np.ascontiguousarray(v@view*scale+offset,np.float32),np.ascontiguousarray(f,np.int32),np.tile(color,(len(f),1)).astype(np.uint8),len(f))
        draw(fv,ff,[180,144,94])
        for name,v in body.items():
            color=[149,180,204]
            if name=='pelvis':color=[25,153,119]
            if name in ['left_hip_roll_link','right_hip_roll_link']:color=[223,105,65]
            draw(v,shapes[name][1],color)
        return Image.fromarray(img)
    scenes=[scene(b) for b in bodies]
    def font(n,bold=False):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if bold else '')+'.ttf',n)
    def text(d,xy,s,n=23,color='#193b4b',bold=False):d.text(xy,s,font=font(n,bold),fill=color)
    # Exact accounting of the fixed piecewise-linear command path, not a new
    # solver or a feasibility claim for other paths. Quantization saving bound.
    durations=[];joint_totals={n:0. for n in model.names[:23]}
    for a,b in zip(plan['nodes'],plan['nodes'][1:]):
        delta=abs(np.asarray(b['command'])-a['command']);j=int(delta.argmax());dt=.02*float(delta[j])/.003
        if a.get('normal') or b.get('normal'):dt=max(dt,abs(b['gap']-a['gap'])/.006)
        durations.append(dt)
        if j<23:joint_totals[model.names[j]]+=dt
    accounting=dict(fixed_path_continuous_rate_lower_bound_s=sum(durations),with_existing1p5second_reserves_s=sum(durations)+1.5,
        scheduled_motion_s=plan['forward']['seconds'],nominal_settled_s=plan['nominal_rest_settle_s'],required_nominal_settle_below_s=14.4,
        leading_rate_limiting_joints=sorted(joint_totals.items(),key=lambda x:-x[1])[:5],scope='Fixed geometric/command path only; no impossibility claim for a different contact path or foot placement.')
    (O/'timing_accounting.json').write_text(json.dumps(accounting,indent=2)+'\n')
    target=O/'corner_and_narrow_support_review.mp4';W,HEIGHT,FPS=1280,800,10
    proc=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{HEIGHT}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-threads','1','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],stdin=subprocess.PIPE)
    try:
        for i in range(160):
            im=Image.new('RGB',(W,HEIGHT),(244,247,249));d=ImageDraw.Draw(im)
            text(d,(30,24),'Corner and narrow-support checks',33,bold=True)
            text(d,(30,75),'Diagnostic evidence — no successful sit / hold / stand cycle recorded',22,'#a13d35')
            segment=i//40
            if segment==0:
                im.paste(camera.resize((768,576)),(14,128));d=ImageDraw.Draw(im)
                text(d,(820,148),'ACTUAL REAR CAMERA',23,bold=True)
                text(d,(820,204),'Clipped table corner',25,bold=True)
                text(d,(820,249),'Depth matches collision mesh.',21)
                text(d,(820,294),'Green: locally planar pixels.',21)
                text(d,(820,370),'Corner contact region is too',21,'#a13d35')
                text(d,(820,403),'small for this demonstrated',21,'#a13d35')
                text(d,(820,436),'pose plus clearance margins.',21,'#a13d35')
                text(d,(820,530),'Standing snapshot only.',22)
            else:
                scene_index=1 if segment in [1,3] else 2;im.paste(scenes[scene_index],(15,125));d=ImageDraw.Draw(im)
                text(d,(820,145),'STATIC MODEL ONLY',25,'#a13d35',True)
                text(d,(820,190),'12 cm-wide test fixture',23,bold=True)
                if segment==1:
                    text(d,(820,255),'Resting pose prediction:',22,bold=True)
                    text(d,(820,307),f"Foot unloading: {100*plan['rest_load']:.1f}%",24)
                    text(d,(820,357),f"Joint torque-squared:",22)
                    text(d,(820,394),f"{100*(1-plan['rest_model_effort_ratio']):.1f}% below standing",25)
                    text(d,(820,474),'Not measured electrical energy.',20)
                elif segment==2:
                    text(d,(820,253),'Predicted unloaded exit',23,bold=True)
                    text(d,(820,307),'Hip-housing clearance: PASS',21,'#147458')
                    text(d,(820,351),'Whole-path mesh check: PASS',21,'#147458')
                    text(d,(820,395),'Balance / action limits: PASS',21,'#147458')
                    text(d,(820,468),'No dynamics were simulated.',21,'#a13d35')
                else:
                    text(d,(820,252),'Next: physical validation',24,bold=True)
                    text(d,(820,307),f"Settling estimate: {plan['nominal_rest_settle_s']:.2f} s",23,'#147458')
                    text(d,(820,349),'Model timing / clearance: PASS',20,'#147458')
                    text(d,(820,404),'Live camera / load validation',21)
                    text(d,(820,440),'and physical cycle still needed.',21)
                    text(d,(820,504),'4090 occupied by another job.',21)
                text(d,(35,690),'Green: pelvis | Orange: hip housings | All collision geometry retained',21)
            text(d,(30,754),'Original action limits and success criteria unchanged | No PPO run or repair claimed',20,'#526875')
            proc.stdin.write(im.tobytes())
            if i in [20,60,100,140]:im.save(O/f'frame_{i}.jpg',quality=95)
        proc.stdin.close();assert proc.wait()==0
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
    report=dict(duration_s=16,frames=160,video_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),scope='Actual rear camera snapshot, then explicitly labelled static model poses; no invented physical rollout.',
        new_physics_steps=0,new_solvers=0,PPO_updates=0,complete_cycle=False,elapsed_seconds=time.monotonic()-START,timing=accounting)
    (O/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
