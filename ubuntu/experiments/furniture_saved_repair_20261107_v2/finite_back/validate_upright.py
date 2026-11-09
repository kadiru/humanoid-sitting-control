import os,sys,json,time,signal,hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path('/home/kadir/humanoid-sitting');HERE=Path(__file__).resolve().parent;EXP=ROOT/'experiments'
sys.path[:0]=[str(HERE),str(EXP/'armchair_rest_ppo_20261105_v1'),str(ROOT/'sitting_task/src')]
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply
from upright_lean_plan import LeanPlan
from audit_lean_mesh import audit
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
def timeout(*_):raise TimeoutError('45_SECOND_TOTAL_CONSTRUCTION_AUDIT_CAP')
signal.signal(signal.SIGALRM,timeout)
freeze=read(HERE/'freeze.json')
for p,h in freeze['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
out=HERE/'upright_models';out.mkdir(exist_ok=False)
m=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json');rows=[]
shapes=read(EXP/'furniture_perception_repair_20261029_v1/render_colliders.json')['shapes']
case=next(c for c in read(EXP/'furniture_examples_20261105_v1/cases.json') if c['id']=='myd1WSucAz_rear_back')
mesh=dict(np.load(case['mesh_npz']));allowed=read(Path(case['mesh_npz']).with_name('target_region.json'))['triangle_indices']
for name in ['demo_rear_back_center']:
 start=time.monotonic();signal.alarm(45)
 try:
  p=ROOT/'outputs/armchair_rest_ppo_20261105_v1'/name;z=dict(np.load(p/'frame_0.npz'))
  with np.load(p/'trajectory.npz') as d:previous=d['previous'][0,0].astype(float)
  plan=LeanPlan(m,z['q'].astype(float),z['gravity'].astype(float),z['geometry'].astype(float),z['selected_points'],previous)
  nodes,times=plan.solve();write(out/(name+'.plan.json'),dict(nodes=nodes,times=times.tolist(),posture_selection=plan.posture_selection))
  # Independently re-evaluate every full-body pose and discrete .1-filter action.
  allnodes=[];feet=[];gaps=[];cops=[];commands=[]
  for n in nodes:
   q=plan.q.copy();q[plan.ids]+=np.array(n['q'])-plan.start;fk=m.fk(q);rt=plan.pose(np.array(n['q']))[0]
   allnodes.append(dict(q=q.tolist(),root=rt.tolist(),phase='landing' if n['gap']<=.006 else 'approach'))
   for s in ['left','right']:
    initial=plan.pose(plan.start)[0]@m.fk(plan.q)[s+'_ankle_roll_link'];now=rt@fk[s+'_ankle_roll_link']
    feet.append(max(float(np.linalg.norm(now[:3,3]-initial[:3,3])),float(np.linalg.norm(Rotation.from_matrix(initial[:3,:3].T@now[:3,:3]).as_rotvec()))))
   gap=plan.geometry(np.array(n['q']))[0];gaps.append(gap);cops.append(min(n['cop']-plan.foot_bounds[0],plan.foot_bounds[1]-n['cop']))
   command=previous.copy();command[plan.ids]+=np.array(n['cmd'])-plan.prev;commands.append(command)
  commands=np.array(commands);prior=previous.copy();peak=0.;margin=1e6
  limits={j['name']:np.deg2rad(j['limits']) for j in m.joints if j['name'] in m.names}
  for t in np.arange(.02,times[-1]+.019999,.02):
   target=np.array([np.interp(t,times,commands[:,i]) for i in range(37)]);desired=prior+(target-prior)/.1
   peak=max(peak,float(abs(target-prior).max()))
   margin=min(margin,min(min(desired[i]-limits[m.names[i]][0],limits[m.names[i]][1]-desired[i]) for i in range(23)))
   prior=target
  clearance=audit(m,shapes,mesh,allowed,allnodes,z['root_rotation']@plan.level.T,z['root_position'])
  tests=dict(both_feet_fixed=max(feet)<.003,physical_contact_geometry=min(gaps)>-.002,CoP_inset=min(cops)>=.005,
   action_increment=peak<=.003,desired_margin=margin>=.03,fingers_fixed=bool(np.all(commands[:,23:]==previous[23:])),hold_time_reserved=times[-1]<14.4,full_mesh=clearance['passed'])
  row=dict(case=name,finite_surface_diagnosis=plan.diagnosis,posture_selection=plan.posture_selection,tests=tests,model_pass=all(tests.values()),max_increment=peak,minimum_desired_margin=margin,maximum_foot_error=max(feet),clearance=clearance,
   physical_success=False,physics_launch_ready=False,remaining='Current-sensor force/ankle/controller admission and full camera-path witness remain mandatory')
 except Exception as e:row=dict(case=name,model_pass=False,error=repr(e),physical_success=False,physics_launch_ready=False)
 finally:signal.alarm(0)
 row['total_seconds']=time.monotonic()-start;write(out/(name+'.audit.json'),row);rows.append(row);print(json.dumps(row,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()),flush=True)
write(out/'receipt.json',dict(rows=rows,physics_steps=0,PPO_updates=0,new_demos=0))
