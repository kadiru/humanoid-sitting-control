"""One cold construction each for the remaining eight NEW closer seat starts."""
import os,sys,json,time,signal,hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path('/home/kadir/humanoid-sitting');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(ROOT/'experiments/furniture_repair_20261106_v1/model_v4'),str(ROOT/'experiments/furniture_repair_20261106_v1'),str(ROOT/'scripts'),str(ROOT/'sitting_task/src')]
import numpy as np
from humanoid_sitting.shared_frame import BodyModel,level_rotation
from seat_plan import SeatPlan
from standing_placement import closer_stance
from audit_seat_plan import audit as audit_geometry
from complete_seat_path import complete
from audit_complete_path import audit as audit_commands
from audit_mesh_clearance_fast import audit as audit_mesh
read=lambda p:json.loads(Path(p).read_text())
def encode(v):
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
def write(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2,default=encode,allow_nan=False)
def deadline(*args):raise TimeoutError('45_SECOND_TOTAL_CONSTRUCTION_AND_AUDIT_CAP')
signal.signal(signal.SIGALRM,deadline)
freeze=read(HERE/'stool_fast_audit_freeze.json')
for n,h in freeze['files'].items():assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==h,n
out=HERE/'seat_models_stool_fast';out.mkdir(exist_ok=False)
model=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json')
inertia=read('/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json')
shapes=read(ROOT/'experiments/furniture_perception_repair_20261029_v1/render_colliders.json')['shapes']
assets=ROOT/'experiments/furniture_examples_20261105_v1';cases=read(assets/'cases.json');rows=[]
for variant in freeze['variants']:
    name=variant['id'];start=time.monotonic();signal.alarm(45)
    try:
        case=next(c for c in cases if c['id']==variant['case']);meshpath=Path(case['mesh_npz'])
        furniture=dict(np.load(meshpath));allowed=read(meshpath.with_name('target_region.json'))['triangle_indices']
        p=ROOT/'outputs/furniture_examples_20261105_v2'/name
        with np.load(p/'frame_0.npz') as z,np.load(p/'startup.npz') as init:
            level=level_rotation(z['gravity'].astype(float));basis=z['root_rotation']@level.T
            shift,placement=closer_stance(model,shapes,furniture,z['q'],z['root_rotation'],z['root_position'],basis)
            write(out/(name+'.placement.json'),placement)
            translation=level.T@np.array([shift,0.,0.])
            geometry=z['geometry'].astype(float).copy();geometry[5]-=geometry[2:5]@translation;geometry[6:9]+=translation
            plan=SeatPlan(model,inertia,shapes,z['q'],z['gravity'],geometry,z['selected_points']+translation,init['command'][0])
            basis=z['root_rotation']@level.T;initial=z['root_position']-basis@np.array([shift,0.,0.])
        construction=plan.solve();write(out/(name+'.plan.json'),construction)
        geometry_check=audit_geometry(plan,construction)
        path=complete(plan,construction);write(out/(name+'.complete.json'),path)
        commands=audit_commands(plan,construction,path)
        nodes=construction['nodes']+[dict(n,phase='landing') for n in path['landing'][1:]]
        clearance=audit_mesh(model,shapes,furniture,allowed,nodes,basis,initial)
        geometry_pass=all(v for k,v in geometry_check['tests'].items() if k!='time_for_original_hold')
        row=dict(case=name+'_closer',closer_m=shift,placement=placement,geometry_audit=geometry_check,command_audit=commands,clearance_audit=clearance,
                 model_admitted=bool(geometry_pass and commands['passed'] and clearance['passed']),physical_success=False,
                 time_admission='Exact discrete original-action-contract schedule replaces the preliminary conservative estimate; the14.4s cutoff is unchanged.')
        write(out/(name+'.audit.json'),row)
    except Exception as e:row=dict(case=name+'_closer',error=repr(e),model_admitted=False)
    finally:signal.alarm(0)
    row['seconds']=time.monotonic()-start;rows.append(row)
    print(json.dumps(dict(case=row['case'],model_admitted=row['model_admitted'],seconds=row['seconds'],error=row.get('error'),
                          failed_checks={group:[k for k,v in row[group]['tests'].items() if not v] for group in ['geometry_audit','command_audit'] if group in row},
                          clearance_intersections=row.get('clearance_audit',{}).get('forbidden_or_early_clearance_intersection_count')),default=encode),flush=True)
write(out/'receipt.json',dict(cases=rows,simulation_steps=0,ppo_updates=0,original_starts_repaired=0))
