import sys,json
from pathlib import Path
R=Path('/home/kadir/humanoid-sitting')
sys.path[:0]=[str(Path(__file__).resolve().parent),str(R/'experiments/furniture_perception_repair_20261029_v1'),str(R/'sitting_task/src')]
from itertools import product
import numpy as np
from compact_camera import CompactCamera
from camera_clearance import CameraClearance

c=CompactCamera.__new__(CompactCamera);c.K=np.array([[4.,0,3.],[0,4.,3.],[0,0,1.]])
c.T=np.eye(4);c.R0=np.eye(3);c.h=c.w=7;c.body='left_hip_yaw_link'
c.support_bodies=('left_hip_yaw_link','right_hip_yaw_link')
c.valid=np.ones((6,6,2),bool);c.target=c.valid.copy();c.ground=np.zeros_like(c.valid)
c.N=np.broadcast_to([0.,0.,-1.],(6,6,2,3));c.D=np.full((6,6,2),2.)
body='left_hip_pitch_link';pose=np.eye(4);checks=[]
for original_gap,expanded_gap in [(.01,.004),(.005,.004),(.01,-.001),(.006,0.)]:
    V=np.array([[0.,0.,2-original_gap],[.05,0.,2-expanded_gap]])
    c.samples={body:V};c.sample_clearance={body:np.array([.006,0.])}
    c.boxes={body:np.array(list(product(*zip(V.min(0),V.max(0)))))}
    raw=CameraClearance.rows(c,pose,{body:pose},0.)
    compact=CompactCamera.rows(c,pose,{body:pose},0.)
    expected=np.array([original_gap-.006,expanded_gap])
    assert np.max(abs(raw-expected))<1e-12 and abs(compact[0]-min(expected))<1e-12
    checks.append(dict(original_gap=original_gap,expanded_gap=expanded_gap,minimum=float(compact[0])))
assert checks[1]['minimum']<0 and checks[2]['minimum']<0
result=dict(passed=True,cases=checks,scope='Production raw/compact finite-camera rows preserve original margins and enforce already-expanded housing geometry without an extra6mm',robot_states=0,physics_steps=0)
with Path(__file__).with_name('finite_camera_audit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
