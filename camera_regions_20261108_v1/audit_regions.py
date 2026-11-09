"""Independent original-plane evaluator for synthetic finite-region fixtures."""
import json,time
from pathlib import Path
import numpy as np
from region_certificate import certificate_planes
started=time.monotonic();count=0;minimum=float('inf');behind_rejected=0
fixtures=[([-0.2,.3,-.25,.15],[[0.,0.,-1.],[.25,-.4,-.88]],[.6,.52]),
          ([.4,.7,-.6,-.2],[[-.7,.1,-.7],[.2,.9,-.4]],[.5,.8])]
for bounds,raw,alpha in fixtures:
 N=np.asarray(raw,float);N/=np.linalg.norm(N,axis=1)[:,None];alpha=np.asarray(alpha)
 A,b,detail=certificate_planes(bounds,N,alpha);L,R,T,B=bounds
 for u,v in [(L,(T+B)/2),(R,(T+B)/2),((L+R)/2,T),((L+R)/2,B)]:
  assert np.max(A@(3.*np.array([u,v,1.]))+b)<0,'Tile boundary cannot count as strictly outside'
 for u in np.linspace(L-.1,R+.1,37):
  for v in np.linspace(T-.1,B+.1,37):
   ray=np.array([u,v,1.])
   for z in [detail['front_depth_m']*.8,detail['front_depth_m'],detail['front_depth_m']*1.2,2.]:
    p=z*ray;count+=1
    safe=np.max(A@p+b)>=-1e-13
    if L+1e-10<u<R-1e-10 and T+1e-10<v<B-1e-10 and safe:
     value=float(np.min(N@p+alpha-.006));minimum=min(minimum,value);assert value>=-1e-11
 for j in range(len(N)):
  ray=np.array([(L+R)/2,(T+B)/2,1.]);slope=N[j]@ray
  if slope<0:
   p=ray*((alpha[j]+.01)/-slope)
   assert np.max(A@p+b)<0;behind_rejected+=1
# Explicitly exercise a facet whose plane faces away from every tile ray.
certificate_planes([-1.,-.5,0.,.1],np.array([[-1.,0.,0.],[0.,0.,-1.]]),[.5,.6])
try:certificate_planes([0.,1.,0.,1.],[[0.,0.,-1.]],[.005])
except ValueError:pass
else:raise AssertionError('Missing camera/offset rejection')
result=dict(passed=True,queries=count,original_plane_minimum_margin_residual_m=minimum,
            unsafe_behind_plane_points_rejected=behind_rejected,wall_seconds=time.monotonic()-started,
            scientific_saved_inputs_used=False,physics_steps=0)
with Path(__file__).with_name('synthetic_audit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
