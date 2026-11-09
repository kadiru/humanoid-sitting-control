"""JSON list boundary fixture; do not replay the v1 geometry proof."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from seed import chains
names=[s+'_'+n+'_joint' for s in ('left','right') for n in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
joints={n:dict(p1=[0.,0.,0.],p0=[0.,0.,-.1],q0=[0.,0.,0.,1.]) for n in names}
plan=SimpleNamespace(joints=json.loads(json.dumps(joints)),q0=np.zeros(12),model=SimpleNamespace(names=names),feet=[np.eye(4),np.eye(4)])
data=chains(plan)
positive=len(data)==2 and all(np.max(abs(v[2]-[0,0,-.3]))<1e-15 for v in data.values())
plan.joints[names[0]]['p1']=[.001,0.,0.]
negative=False
try:chains(plan)
except AssertionError:negative=True
result=dict(passed=bool(positive and negative),zero_JSON_lists_accepted=bool(positive),nonzero_JSON_lists_rejected=negative,experiment_inputs_read=False,physics_steps=0)
with Path(__file__).with_name('synthetic_audit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result));assert result['passed']
