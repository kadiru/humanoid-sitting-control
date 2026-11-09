import os,time,json,sys,hashlib
from pathlib import Path
START=time.monotonic();os.environ['CUDA_VISIBLE_DEVICES']=''
H=Path(__file__).resolve().parent;R=Path('/home/kadir/humanoid-sitting');E=R/'experiments'
sys.path[:0]=[str(H),str(E/'furniture_perception_repair_20261029_v1'),str(R/'sitting_task/src')]
for p,h in json.loads((H/'freeze_v2.json').read_text())['files'].items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,level_rotation,transform
from camera_clearance import CameraClearance
from compact_camera import CompactCamera
model=BodyModel(R/'runs/g1_kinematics_geometry_04/model.json')
shapes=json.loads((E/'furniture_perception_repair_20261029_v1/render_colliders.json').read_text())['shapes']
p=R/'outputs/armchair_rest_ppo_20261105_v1/demo_rear_back_center'
f=dict(np.load(p/'frame_0.npz'));data=np.load(p/'trajectory.npz')
T=np.eye(4);T[:3,:3]=[[0,.707106781,-.707106781],[1,0,0],[0,-.707106781,-.707106781]];T[:3,3]=[-.07,0,.38]
R0=level_rotation(f['gravity'].astype(float).copy());n=R0@f['geometry'][2:5];off=f['geometry'][5]
camera=CompactCamera(model,shapes,f['depth'],f['K'],T,f['q'],R0,n,off,'pelvis')
sizes=[len(v) for v in camera.samples.values()];starts=np.r_[0,np.cumsum(sizes)]
assert json.loads((p/'results.json').read_text())['body_names'][0]=='pelvis'
rows=[];base_seconds=compact_seconds=0.
for label,q,root in [('standing',f['q'],transform(R0))]+[(str(i),data['q'][i,0],data['body'][i,0,0]) for i in (499,999)]:
    if np.asarray(root).shape!=(4,4):
        # Saved root world pose is used solely to exercise the auditor at known
        # poses, not as an input to the proposed sensor-only construction.
        basis=f['root_rotation']@R0.T
        root=transform(basis.T@Rotation.from_quat(root[3:7]).as_matrix(),basis.T@(root[:3]-f['root_position']))
    fk=model.fk(q);t=time.monotonic();full=CameraClearance.rows(camera,root,fk,.00601);base_seconds+=time.monotonic()-t
    expected=np.array([full[starts[i]:starts[i+1]].min() for i in range(len(sizes))])
    t=time.monotonic();actual=camera.rows(root,fk,.00601);compact_seconds+=time.monotonic()-t
    error=float(abs(expected-actual).max())
    rows.append(dict(pose=label,maximum_difference_m=error,active_sample_points=camera.last_active_samples,passed=error<=1e-12))
result=dict(passed=all(r['passed'] for r in rows),rows=rows,original_camera_constraint_count=camera.count,compact_constraint_count=len(sizes),original_evaluation_seconds=base_seconds,compact_evaluation_seconds=compact_seconds,seconds=time.monotonic()-START,physics_steps=0,GPU_used=False,scope='Exact constraint-equivalence and resource audit only; no physical repair or optimality claim')
with (H/'compression_audit.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(result),flush=True)
