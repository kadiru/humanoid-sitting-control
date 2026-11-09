"""Synthetic finite-surface checks; no robot trajectories or optimization."""
import sys,json
from pathlib import Path
R=Path('/home/kadir/humanoid-sitting')
sys.path[:0]=[str(Path(__file__).resolve().parent),str(R/'experiments/furniture_perception_repair_20261029_v1'),str(R/'sitting_task/src')]
import numpy as np
from types import SimpleNamespace
from patch_contact import ObservedPatch,connected

K=np.array([[4.,0.,3.],[0.,4.,3.],[0.,0.,1.]])
rows,cols=np.mgrid[:7,:7];rays=np.stack([(cols+.5-3.)/4.,(rows+.5-3.)/4.,np.ones((7,7))],-1)
N=np.array([.1,.0,-1.]);N/=np.linalg.norm(N);d=2.
depth=-d/(rays@N)
mask=np.ones((6,6,2),bool);mask[:,2,:]=False
c=SimpleNamespace(K=K,T=np.eye(4),R0=np.eye(3),h=7,w=7,
 N=np.broadcast_to(N,(6,6,2,3)),D=np.full((6,6,2),d),target=mask,
 valid=np.ones_like(mask),ground=np.zeros_like(mask))
p=ObservedPatch(c,depth,[2,1,0])
assert len(p.component)==24
assert all(col<2 for row,col,side in p.component)
checks=0;maximum=0.
for key in sorted(p.component):
 normal,offset,edges,V=p.facet(key)
 for a,b in [(.1,.2),(.2,.6),(.6,.1),(.333,.333)]:
  hit=(1-a-b)*V[0]+a*V[1]+b*V[2]
  for scale in [.95,1.,1.05]:
   point=hit*scale;n,off,e,info=p.query(point)
   # Independent normal and ray/plane intersection from observed vertices.
   A=np.array(info['vertices']);cross=np.cross(A[1]-A[0],A[2]-A[0]);cross/=np.linalg.norm(cross)
   independent=point*((cross@A[0])/(cross@point))
   coefficients=np.linalg.lstsq(np.stack([A[1]-A[0],A[2]-A[0]],1),independent-A[0],rcond=None)[0]
   error=float(np.max(abs(independent-np.array(info['ray_contact']))));maximum=max(maximum,error)
   assert error<1e-12 and min(coefficients)>=-1e-12 and coefficients.sum()<=1+1e-12
   assert abs(n@independent+off)<1e-12
   checks+=1
rejections=0
for pixel in [(2.2,2.2),(4.2,2.2),(-.2,2.2)]:
 u,v=pixel;point=np.array([(u+.5-3.)/4.,(v+.5-3.)/4.,1.])*2.
 try:p.query(point)
 except ValueError:rejections+=1
 else:raise AssertionError('Unknown/disconnected/outside accepted')
result=dict(passed=True,finite_ray_queries=checks,rejected_missing_disconnected_outside=rejections,
            maximum_independent_intersection_error=maximum,connected_triangles=len(p.component),
            new_robot_states=0,physics_steps=0)
out=Path(__file__).with_name('synthetic_audit.json')
with out.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
