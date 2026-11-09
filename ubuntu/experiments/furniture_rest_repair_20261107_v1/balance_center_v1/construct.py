import os,sys,json,time,hashlib,signal
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path('/home/kadir/humanoid-sitting');HERE=Path(__file__).resolve().parent;EXP=ROOT/'experiments'
sys.path[:0]=[str(HERE),str(EXP/'furniture_collision_repair_20261106_v5'),str(EXP/'furniture_collision_repair_20261106_v4'),str(EXP/'furniture_load_repair_20261106_v3'),str(EXP/'furniture_repair_20261106_v1/model_v4'),str(EXP/'furniture_repair_20261106_v1'),str(ROOT/'scripts'),str(ROOT/'sitting_task/src')]
import numpy as np
from humanoid_sitting.shared_frame import BodyModel,level_rotation
from spatial_seat_plan import SeatPlan
from complete_seat_path import complete
from audit_seat_plan import audit as geometry_audit
from audit_complete_path import audit as command_audit
from audit_mesh_clearance_fast import audit as collision_audit
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
def timeout(*_):raise TimeoutError('45_SECOND_TOTAL_CONSTRUCTION_AUDIT_CAP')
signal.signal(signal.SIGALRM,timeout)
freeze=read(HERE/'freeze.json')
for p,h in freeze['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
out=HERE/'model';out.mkdir(exist_ok=False);start=time.monotonic();signal.alarm(45)
try:
 p=ROOT/'outputs/furniture_camera_coverage_20261106_v1/calibration_demos'/freeze['case']
 z=dict(np.load(p/'construction_inputs.npz'));frame=dict(np.load(p/'frame_0.npz'))
 m=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json');inertia=read('/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json')
 shapes=read(EXP/'furniture_perception_repair_20261029_v1/render_colliders.json')['shapes'];case=read(p/'manifest.json')['case']
 plan=SeatPlan(m,inertia,shapes,z['q'],z['gravity'],z['geometry'],z['selected_points'],z['previous'],'seat')
 construction=plan.solve();path=complete(plan,construction);write(out/'plan.json',construction);write(out/'complete.json',path)
 geometry=geometry_audit(plan,construction);commands=command_audit(plan,construction,path)
 nodes=construction['nodes']+[dict(n,phase='landing') for n in path['landing'][1:]]
 collision=collision_audit(m,shapes,dict(np.load(case['mesh_npz'])),read(Path(case['mesh_npz']).with_name('target_region.json'))['triangle_indices'],nodes,frame['root_rotation']@level_rotation(z['gravity'].astype(float)).T,frame['root_position'])
 row=dict(geometry=geometry,commands=commands,collision=collision,common_foot_interval=plan.common_foot_interval,
  target_balance=construction['target_balance'],model_admitted=bool(all(v for k,v in geometry['tests'].items() if k!='time_for_original_hold') and commands['passed'] and collision['passed']),physical_success=False,physics_launch_ready=False)
except Exception as e:row=dict(model_admitted=False,error=repr(e),physical_success=False,physics_launch_ready=False)
finally:signal.alarm(0)
row['seconds']=time.monotonic()-start;row.update(physics_steps=0,PPO_updates=0)
write(out/'receipt.json',row);print(json.dumps(row,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()))
