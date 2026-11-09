"""Independent known-distance fixture for the new optimizer constraint rows."""
import os,time,json,sys,hashlib
from pathlib import Path
START=time.monotonic();os.environ['CUDA_VISIBLE_DEVICES']=''
H=Path(__file__).resolve().parent;R=Path('/home/kadir/humanoid-sitting');E=R/'experiments'
sys.path[:0]=[str(H),str(E/'furniture_perception_repair_20261029_v1'),str(R/'sitting_task/src')]
for p,h in json.loads((H/'freeze.json').read_text())['files'].items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
import numpy as np
from scipy.spatial.transform import Rotation
from camera_clearance import CameraClearance

class FixtureModel:
    vertices={s+'_ankle_roll_link':np.array([[0.,0.,-5.]]) for s in ('left','right')}
    def fk(self,q):return {b:np.eye(4) for b in ('pelvis','left_ankle_roll_link','right_ankle_roll_link')}
    def self_mask(self,p,fk,margin):return np.ones(len(p),bool)

model=FixtureModel();K=np.array([[100.,0,3.5],[0,100.,3.5],[0,0,1.]])
depth=np.full((7,7),2.);tests=[]
for angles in ([0,0,0],[.3,-.2,.1]):
    T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',angles).as_matrix();T[:3,3]=[.03,-.02,.04]
    R0=Rotation.from_euler('xyz',[-.1,.2,.05]).as_matrix()
    C=np.array([[-.01,-.01,1.99],[.01,-.01,1.99],[0,.01,1.99],[0,0,1.985]])
    vertices=C@T[:3,:3].T+T[:3,3]
    n=R0@T[:3,:3]@np.array([0.,0.,-1.]);offset=2.-n@(R0@T[:3,3])
    args=(model,[dict(body='pelvis',vertices=vertices)],depth,K,T,np.zeros(37),R0,n,offset)
    camera=CameraClearance(*args,'torso_link');root=np.eye(4);root[:3,:3]=R0
    samples=camera.samples['pelvis'];camera_z=((samples-T[:3,3])@T[:3,:3])[:,2]
    expected=2.-camera_z-.006
    actual=camera.rows(root,model.fk(None),0.)
    tests.append(dict(name='independent_axial_plane_distance',error=float(abs(actual-expected).max()),passed=bool(np.allclose(actual,expected,atol=1e-12,rtol=0))))
    root[:3,3]=R0@T[:3,:3]@np.array([0.,0.,.02])
    actual=camera.rows(root,model.fk(None),0.)
    tests.append(dict(name='penetration_rejected',passed=bool(actual.min()<-.015)))
    contact=CameraClearance(*args,'pelvis');root[:3,3]=0.
    relaxed=contact.rows(root,model.fk(None),0.);reserved=contact.rows(root,model.fk(None),.006)
    tests.append(dict(name='only_target_body_phase_clearance',passed=bool(np.allclose(relaxed-reserved,.006,atol=1e-12,rtol=0))))
try:
    CameraClearance(model,[dict(body='pelvis',vertices=vertices)],depth*np.nan,K,T,np.zeros(37),R0,n,offset,'pelvis')
    tests.append(dict(name='missing_depth_rejected',passed=False))
except ValueError as e:tests.append(dict(name='missing_depth_rejected',passed=str(e)=='INSUFFICIENT_OBSERVED_TRIANGLES'))
result=dict(passed=all(t['passed'] for t in tests),tests=tests,seconds=time.monotonic()-START,physics_steps=0,GPU_used=False,scope='Synthetic geometric constraint audit, not a physical stability or motion result')
with (H/'camera_row_audit.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(result),flush=True)
