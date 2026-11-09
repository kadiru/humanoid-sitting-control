import os,sys,time,json,hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';START=time.monotonic()
R=Path('/home/kadir/humanoid-sitting');E=R/'experiments';H=Path(__file__).resolve().parent
sys.path[:0]=[str(H),str(E/'furniture_perception_repair_20261029_v1'),str(R/'sitting_task/src')]
for p,h in json.loads((H/'observed_clearance_v2_freeze.json').read_text())['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
import numpy as np
from observed_clearance_v2 import observed_mesh,check
from humanoid_sitting.shared_frame import BodyModel,transform,level_rotation
read=lambda p:json.loads(Path(p).read_text())
f=dict(np.load(R/'outputs/armchair_rest_ppo_20261105_v1/demo_rear_back_center/frame_0.npz'))
model=BodyModel(R/'runs/g1_kinematics_geometry_04/model.json');shapes=read(E/'furniture_perception_repair_20261029_v1/render_colliders.json')['shapes']
T=np.eye(4);T[:3,:3]=[[0,.707106781,-.707106781],[1,0,0],[0,-.707106781,-.707106781]];T[:3,3]=[-.07,0,.38]
mesh,eligible,metadata=observed_mesh(model,f['depth'],f['K'],T,f['q'],f['gravity'],f['geometry'])
saved=read(H/'encoded_results/lean_center/plan.json')
nodes=[dict(z,phase='landing' if z['gap']<=.006 else 'approach') for z in saved['nodes']]
bad=check(model,shapes,mesh,eligible,nodes,'leaning')
initial=dict(q=f['q'].tolist(),root=transform(level_rotation(f['gravity'].astype(float))).tolist(),phase='approach')
standing=check(model,shapes,mesh,eligible,[initial],'leaning')
missing_rejected=False
try:observed_mesh(model,np.full_like(f['depth'],np.nan),f['K'],T,f['q'],f['gravity'],f['geometry'])
except ValueError:missing_rejected=True
result=dict(passed=standing['passed'] and not bad['passed'] and missing_rejected,metadata=metadata,standing=standing,rejected_path=bad,missing_depth_rejected=missing_rejected,scope='Saved-camera veto validation only. Rejected path used solely as a negative test, never as an initializer or actor motion. No optimizer, simulation or PPO.',physical_repair=False,deployed=False,seconds=time.monotonic()-START)
with (H/'observed_clearance_v2_validation.json').open('x') as out:json.dump(result,out,indent=2)
print(json.dumps({k:result[k] for k in ['passed','metadata','missing_depth_rejected','seconds']}),flush=True)
