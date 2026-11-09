"""One deterministic camera-patch arithmetic pass; no pose optimizer or physics."""
import json,sys,time,signal,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
sys.path.insert(1,str(ROOT/'sitting_task/src'))
from humanoid_sitting.shared_frame import BodyModel,level_rotation
from local_patch import refine
read=lambda p:json.loads(Path(p).read_text())
def main():
    freeze=read(HERE/'patch_freeze.json')
    for p,h in freeze['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    def timeout(*args):raise TimeoutError('PATCH_ARITHMETIC30S_CAP')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(30);start=time.monotonic()
    model=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json');rows=[]
    for item in freeze['starts']:
        name=item['source_case']
        with np.load(ROOT/'outputs/furniture_examples_20261105_v2'/name/'frame_0.npz') as z:
            # Offline reset displacement; transformed sensor points, never truth surfaces.
            shift=-z['root_rotation'].T@np.array([0.,item['world_y_shift_m'],0.])
            geometry=z['geometry'].astype(float).copy();geometry[5]-=geometry[2:5]@shift;geometry[6:9]+=shift
            try:
                value,points,report=refine(model,z['q'],z['gravity'],geometry,z['selected_points']+shift)
                row=dict(case=name,passed=True,geometry=value.tolist(),report=report)
            except Exception as e:row=dict(case=name,passed=False,error=repr(e))
            rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='report'}),flush=True)
    signal.alarm(0)
    result=dict(cases=rows,seconds=time.monotonic()-start,optimization_solves=0,simulation_steps=0,PPO_updates=0)
    with (HERE/'patch_arithmetic.json').open('x') as f:json.dump(result,f,indent=2)
if __name__=='__main__':main()
