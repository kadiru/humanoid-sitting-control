"""Exactly two named CPU constructions; no simulation or optimizer training."""
import os, sys, json, time, hashlib, signal
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path('/home/kadir/humanoid-sitting'); HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'sitting_task/src')]
import numpy as np
from humanoid_sitting.shared_frame import BodyModel
from seat_plan import SeatPlan
from audit_seat_plan import audit
read=lambda p:json.loads(Path(p).read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def deadline(*args):raise TimeoutError('45_SECOND_TOTAL_CASE_CAP')
signal.signal(signal.SIGALRM,deadline)

if __name__=='__main__':
    freeze=read(HERE/'cpu_freeze.json')
    for name,sha in freeze['files'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
    out=HERE/'cpu_results';out.mkdir(exist_ok=False)
    model=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json')
    inertia=read('/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json')
    shapes=read(ROOT/'experiments/furniture_perception_repair_20261029_v1/render_colliders.json')['shapes']
    result=[]
    for name in freeze['cases']:
        start=time.monotonic();signal.alarm(45)
        try:
            p=ROOT/'outputs/furniture_examples_20261105_v2'/name
            with np.load(p/'frame_0.npz') as z,np.load(p/'startup.npz') as init:
                plan=SeatPlan(model,inertia,shapes,z['q'],z['gravity'],z['geometry'],z['selected_points'],init['command'][0])
            construction=plan.solve();write(out/(name+'.plan.json'),construction)
            check=audit(plan,construction);write(out/(name+'.audit.json'),check)
            row=dict(case=name,model_audit_pass=check['passed'],audit=check)
        except Exception as e:row=dict(case=name,model_audit_pass=False,error=repr(e))
        finally:signal.alarm(0)
        row['seconds']=time.monotonic()-start;result.append(row);print(json.dumps(row),flush=True)
        write(out/'receipt.json',dict(cases=result,simulation_steps=0,ppo_updates=0,physical_success=False))
