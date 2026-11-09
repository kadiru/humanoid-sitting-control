import os,time,json,sys,hashlib
from pathlib import Path
START=time.monotonic();os.environ['CUDA_VISIBLE_DEVICES']=''
H=Path(__file__).resolve().parent;R=Path('/home/kadir/humanoid-sitting')
sys.path[:0]=[str(H),str(R/'sitting_task/src')]
freeze=json.loads((H/'freeze.json').read_text())
for p,h in freeze['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
prior=json.loads((H.parent/'contact_mode_20261108_v1/contact_mode_audit_recovered.json').read_text())
assert prior['passed'] and prior['aggregate_verdict_observed'] and not prior['scientific_replay']
import numpy as np
from scipy.spatial.transform import Rotation
from contact_mode import FullRest
from humanoid_sitting.shared_frame import transform,apply
obj=FullRest.__new__(FullRest);obj.body='pelvis';obj.n=np.array([1.,0.,0.]);obj.preference_marker=np.array([-.05,0.,0.])
obj.mesh={'pelvis':np.array([[x,y,z] for x in (-.05,.05) for y in (-.03,.03) for z in (-.1,.1)])}
fk={'pelvis':np.eye(4)};target=np.array([-.05,.02,0.]);values=[]
for angle in (-1e-7,0.,1e-7):
    root=transform(Rotation.from_euler('z',angle).as_matrix());s=dict(root=root,fk=fk)
    exposed=apply(root,obj.exposed_local(root,fk));stable=obj.preference_point(s)
    def penalty(p):
        t=p-target;t-=obj.n*(t@obj.n)
        return .05*float(t@t)/.15**2
    values.append(dict(angle=angle,old=penalty(exposed),corrected=penalty(stable),point=stable.tolist()))
old_jump=max(abs(v['old']-values[1]['old']) for v in values)
new_jump=max(abs(v['corrected']-values[1]['corrected']) for v in values)
tests={'fixture_exhibits_support_feature_jump':bool(old_jump>1e-3),
       'fixed_anchor_objective_continuous':bool(new_jump<1e-8),
       'same_preference_at_tie':bool(abs(values[1]['old']-values[1]['corrected'])<1e-15)}
v=dict(passed=all(tests.values()),tests=tests,values=values,old_jump=old_jump,corrected_change=new_jump,
       inherited_contact_work_audit_replayed=False,seconds=time.monotonic()-START,physics_steps=0,GPU_used=False)
with (H/'preference_audit.json').open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
print(json.dumps(v),flush=True)
