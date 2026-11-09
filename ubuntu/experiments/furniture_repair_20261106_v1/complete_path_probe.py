"""Complete two saved new-placement paths; no repeated approach solve."""
import os,sys,json,time,signal,hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path('/home/kadir/humanoid-sitting');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'model_v4'),str(ROOT/'scripts'),str(ROOT/'sitting_task/src')]
import numpy as np
from humanoid_sitting.shared_frame import BodyModel,level_rotation
from seat_plan import SeatPlan
from complete_seat_path import complete
from audit_complete_path import audit as audit_commands
from audit_mesh_clearance import audit as audit_mesh
read=lambda p:json.loads(Path(p).read_text())
def encode(v):
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
def write(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2,default=encode,allow_nan=False)
def deadline(*args):raise TimeoutError('45_SECOND_COMPLETION_CAP')
signal.signal(signal.SIGALRM,deadline)
freeze=read(HERE/'complete_path_freeze.json')
for n,h in freeze['files'].items():assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==h,n
out=HERE/'complete_path_results';out.mkdir(exist_ok=False)
model=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json')
inertia=read('/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json')
shapes=read(ROOT/'experiments/furniture_perception_repair_20261029_v1/render_colliders.json')['shapes']
case=next(c for c in read(ROOT/'experiments/furniture_examples_20261105_v1/cases.json') if c['id']=='modern_coffee_table_01_top')
meshpath=Path(case['mesh_npz']);furniture=dict(np.load(meshpath));allowed=read(meshpath.with_name('target_region.json'))['triangle_indices']
rows=[]
for name in freeze['source_cases']:
    start=time.monotonic();signal.alarm(45)
    try:
        p=ROOT/'outputs/furniture_examples_20261105_v2'/name
        with np.load(p/'frame_0.npz') as z,np.load(p/'startup.npz') as init:
            level=level_rotation(z['gravity'].astype(float));translation=level.T@np.array([.05,0.,0.])
            geometry=z['geometry'].astype(float).copy();geometry[5]-=geometry[2:5]@translation;geometry[6:9]+=translation
            plan=SeatPlan(model,inertia,shapes,z['q'],z['gravity'],geometry,z['selected_points']+translation,init['command'][0])
            basis=z['root_rotation']@level.T;initial=z['root_position']-basis@np.array([.05,0.,0.])
        construction=read(HERE/'staged_placement_results'/(name+'_closer.plan.json'))
        path=complete(plan,construction);write(out/(name+'.complete.json'),path)
        commands=audit_commands(plan,construction,path)
        nodes=construction['nodes']+[dict(n,phase='landing') for n in path['landing'][1:]]
        clearance=audit_mesh(model,shapes,furniture,allowed,nodes,basis,initial)
        row=dict(case=name+'_closer',command_audit=commands,clearance_audit=clearance,
                 model_admitted=commands['passed'] and clearance['passed'],physical_success=False)
        write(out/(name+'.audit.json'),row)
    except Exception as e:row=dict(case=name+'_closer',error=repr(e),model_admitted=False)
    finally:signal.alarm(0)
    row['seconds']=time.monotonic()-start;rows.append(row);print(json.dumps(row,default=encode),flush=True)
write(out/'receipt.json',dict(cases=rows,approach_solver_calls=0,maximum_new_landing_fits=32,simulation_steps=0,ppo_updates=0))
