"""Synthetic fixed-plus-revolute tree, checked against independent FK."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from analytic_model import AnalyticMixin
def transform(R=None,p=None):
    T=np.eye(4)
    if R is not None:T[:3,:3]=R
    if p is not None:T[:3,3]=p
    return T
fixed=transform(Rotation.from_euler('x',.13).as_matrix(),[.1,.03,-.2])
hinge=transform(p=[-.05,0.,-.3]);end=transform(p=[.02,0.,-.1])
model=SimpleNamespace(names=['hinge'],joints=[dict(name='fixed',axis=None,body0=['pelvis'],body1=['fixed_link']),dict(name='hinge',axis='Y',body0=['fixed_link'],body1=['tip'])],frames={'fixed':(fixed,np.eye(4)),'hinge':(hinge,end)})
obj=AnalyticMixin.__new__(AnalyticMixin);obj.model=model
def pose(x):
    root=transform(Rotation.from_rotvec(x[26:29]).as_matrix(),x[23:26])
    return np.array([x[0]]),root,{'pelvis':np.eye(4),'fixed_link':fixed,'tip':fixed@hinge@transform(Rotation.from_euler('y',x[0]).as_matrix())@end}
obj.pose=pose;x=np.zeros(31);x[0]=.4;x[23:26]=[.1,-.04,.03];x[26:29]=[.2,-.1,.12]
world,jets,_=obj.forward_jets(x);error=0.;h=1e-6
for k in [0,23,24,25,26,27,28]:
    a=x.copy();b=x.copy();a[k]-=h;b[k]+=h
    _,ra,fa=pose(a);_,rb,fb=pose(b)
    for body in world:
        numeric=(rb@fb[body]-ra@fa[body])/(2*h)
        error=max(error,float(np.max(abs(numeric-jets[body][:,:,k]))))
result=dict(passed=error<1e-8,maximum_world_transform_derivative_error=error,
    fixed_axis_None_handled=True,experiment_inputs_read=False,physics_steps=0)
with Path(__file__).with_name('synthetic_audit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result));assert result['passed']
