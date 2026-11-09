"""Finish the independent audit from saved paths; never call a solver."""
import os, sys, json, hashlib, time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path('/home/kadir/humanoid-sitting');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'sitting_task/src')]
import numpy as np
from humanoid_sitting.shared_frame import BodyModel
from seat_plan import SeatPlan
from audit_seat_plan import audit
read=lambda p:json.loads(p.read_text())
def encode(v):
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
freeze=read(HERE/'cpu_freeze.json')
for name,sha in freeze['files'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
model=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json')
inertia=read(Path('/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json'))
shapes=read(ROOT/'experiments/furniture_perception_repair_20261029_v1/render_colliders.json')['shapes']
rows=[]
for name in freeze['cases']:
    start=time.monotonic();p=ROOT/'outputs/furniture_examples_20261105_v2'/name
    with np.load(p/'frame_0.npz') as z,np.load(p/'startup.npz') as init:
        plan=SeatPlan(model,inertia,shapes,z['q'],z['gravity'],z['geometry'],z['selected_points'],init['command'][0])
    saved=HERE/'cpu_results'/(name+'.plan.json');result=audit(plan,read(saved))
    result.update(source_sha256=hashlib.sha256(saved.read_bytes()).hexdigest(),solver_calls=0,
                  reason_for_report_completion='Original audit result used NumPy booleans unsupported by JSON writer; saved path remains unchanged.',
                  seconds=time.monotonic()-start)
    target=HERE/'cpu_results'/(name+'.audit.json')
    with target.open('x') as f:json.dump(result,f,indent=2,default=encode,allow_nan=False)
    rows.append(dict(case=name,**result));print(json.dumps(rows[-1],default=encode),flush=True)
with (HERE/'cpu_results/audit_completion_receipt.json').open('x') as f:
    json.dump(dict(cases=rows,solver_calls=0,simulation_steps=0,ppo_updates=0),f,indent=2,default=encode)
