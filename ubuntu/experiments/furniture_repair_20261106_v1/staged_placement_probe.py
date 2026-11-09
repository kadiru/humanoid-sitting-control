"""Two NEW demonstration placements, 5cm closer to the observed furniture.

This is not a replay or a claimed repair of the original20 starts. The human
authorized varying positions and orientations. Camera-relative furniture points
are translated analytically; fresh camera/reset checks are mandatory in physics.
"""
import os,sys,json,time,hashlib,signal
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path('/home/kadir/humanoid-sitting');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'model_v4'),str(ROOT/'scripts'),str(ROOT/'sitting_task/src')]
import numpy as np
from humanoid_sitting.shared_frame import BodyModel,level_rotation
from seat_plan import SeatPlan
from audit_seat_plan import audit
read=lambda p:json.loads(Path(p).read_text())
def encode(v):
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
def write(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2,default=encode,allow_nan=False)
def deadline(*args):raise TimeoutError('45_SECOND_CASE_CAP')
signal.signal(signal.SIGALRM,deadline)
freeze=read(HERE/'staged_placement_freeze.json')
for n,h in freeze['files'].items():assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==h,n
out=HERE/'staged_placement_results';out.mkdir(exist_ok=False)
model=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json')
inertia=read('/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json')
shapes=read(ROOT/'experiments/furniture_perception_repair_20261029_v1/render_colliders.json')['shapes']
rows=[]
for name in freeze['source_cases']:
    start=time.monotonic();signal.alarm(45)
    try:
        p=ROOT/'outputs/furniture_examples_20261105_v2'/name
        with np.load(p/'frame_0.npz') as z,np.load(p/'startup.npz') as init:
            level=level_rotation(z['gravity'].astype(float));translation=level.T@np.array([.05,0.,0.])
            points=z['selected_points'].copy()+translation
            geometry=z['geometry'].astype(float).copy();geometry[5]-=geometry[2:5]@translation;geometry[6:9]+=translation
            plan=SeatPlan(model,inertia,shapes,z['q'],z['gravity'],geometry,points,init['command'][0])
        construction=plan.solve();write(out/(name+'_closer.plan.json'),construction)
        check=audit(plan,construction)
        row=dict(case=name+'_closer',source_case=name,robot_stance_translation_in_gravity_frame_m=[-.05,0.,0.],
                 original_start_repaired=False,model_audit=check)
        write(out/(name+'_closer.audit.json'),row)
    except Exception as e:row=dict(case=name+'_closer',error=repr(e),model_audit_pass=False)
    finally:signal.alarm(0)
    row['seconds']=time.monotonic()-start;rows.append(row);print(json.dumps(row,default=encode),flush=True)
write(out/'receipt.json',dict(cases=rows,simulation_steps=0,ppo_updates=0,physical_success=False))
