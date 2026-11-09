"""Cold plans associated with actual saved near-placement camera observations."""
import os,sys,json,time,signal,hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path('/home/kadir/humanoid-sitting');HERE=Path(__file__).resolve().parent;EXP=ROOT/'experiments'
sys.path[:0]=[str(EXP/n) for n in ['furniture_collision_collection_20261106_v2','furniture_collision_repair_20261106_v5','furniture_collision_repair_20261106_v4','furniture_load_repair_20261106_v3','furniture_repair_20261106_v1/model_v4','furniture_repair_20261106_v1','furniture_perception_repair_20261029_v1']]+[str(ROOT/'scripts'),str(ROOT/'sitting_task/src')]
import numpy as np
from json_types import encode
from humanoid_sitting.shared_frame import BodyModel,level_rotation
from seat_plan import SeatPlan
from spatial_seat_plan import SeatPlan as SpatialPlan
from audit_seat_plan import audit as geometry_audit
from complete_seat_path import complete
from audit_complete_path import audit as commands_audit
from audit_mesh_clearance_fast import audit as mesh_audit
from perception import observe
read=lambda p:json.loads(Path(p).read_text())
def write(p,d):
 with p.open('x') as f:json.dump(d,f,default=encode,allow_nan=False,indent=2)
def deadline(*args):raise TimeoutError('45_SECOND_TOTAL_CASE_CAP')
signal.signal(signal.SIGALRM,deadline)
freeze=read(HERE/'freeze.json')
for n,h in freeze['files'].items():assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==h,n
model=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json')
inertia=read('/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json')
shapes=read(EXP/'furniture_perception_repair_20261029_v1/render_colliders.json')['shapes']
assets=EXP/'furniture_examples_20261105_v1';variants=read(assets/'protocol.json')['variants'];cases=read(assets/'cases.json')
out=HERE/'models';out.mkdir(exist_ok=False);rows=[]
for job in freeze['jobs']:
 tick=time.monotonic();signal.alarm(45);name=job['source_case'];row=dict(case=name,model_admitted=False,physics_steps=0)
 try:
  variant=next(v for v in variants if v['id']==name);case=next(c for c in cases if c['id']==variant['case'])
  z=dict(np.load(Path(job['fresh_frame'])));init=dict(np.load(assets/'banks'/name/'reset_candidates.npz'))
  T=np.eye(4);T[:3,:3]=np.array([[0.,.707106781,-.707106781],[1.,0.,0.],[0.,-.707106781,-.707106781]]);T[:3,3]=[-.07,0.,.38]
  np.testing.assert_allclose(z['root_rotation']@T[:3,:3],z['camera_rotation'],atol=2e-7,rtol=0)
  value,info=observe(model,z['K'],T,z['depth'],z['q'],z['gravity'].astype(float),variant['intent']);assert value is not None
  args=[model,inertia,shapes,z['q'],z['gravity'],value,info['selected_points'],init['command'][0,0]]
  local=bool(job.get('support_mode')=='pelvis')
  plan=SpatialPlan(*args,variant['intent']) if local else SeatPlan(*args)
  construction=plan.solve();write(out/(name+'.plan.json'),construction)
  path=complete(plan,construction);write(out/(name+'.complete.json'),path)
  geometry=geometry_audit(plan,construction);commands=commands_audit(plan,construction,path)
  mesh=dict(np.load(case['mesh_npz']));allowed=read(Path(case['mesh_npz']).with_name('target_region.json'))['triangle_indices']
  nodes=construction['nodes']+[dict(n,phase='landing') for n in path['landing'][1:]]
  clearance=mesh_audit(model,shapes,mesh,allowed,nodes,z['root_rotation']@level_rotation(z['gravity'].astype(float)).T,z['root_position'])
  admitted=all(v for k,v in geometry['tests'].items() if k!='time_for_original_hold') and commands['passed'] and clearance['passed']
  row.update(model_admitted=bool(admitted),geometry_audit=geometry,command_audit=commands,clearance_audit=clearance)
  write(out/(name+'.audit.json'),row)
 except Exception as e:row['error']=repr(e)
 finally:signal.alarm(0)
 row['seconds']=time.monotonic()-tick;rows.append(row)
 print(json.dumps(dict(case=name,model_admitted=row['model_admitted'],seconds=row['seconds'],error=row.get('error'),failed={g:[k for k,v in row[g]['tests'].items() if not v] for g in ['geometry_audit','command_audit'] if g in row},intersections=row.get('clearance_audit',{}).get('forbidden_or_early_clearance_intersection_count')),default=encode),flush=True)
write(out/'receipt.json',dict(cases=rows,physics_steps=0,PPO_updates=0,source='actual_saved_near_start_depth_not_translated_old_cloud'))
