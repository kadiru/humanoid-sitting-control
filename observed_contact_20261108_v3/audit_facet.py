"""Synthetic finite-triangle edge audit against independent barycentric signs."""
import json,sys,time
from pathlib import Path
import numpy as np
R=Path('/home/kadir/humanoid-sitting')
sys.path[:0]=[str(R/'experiments/furniture_perception_repair_20261029_v1'),str(R/'sitting_task/src')]
from contact_facet import edges_for_triangle
start=time.monotonic();count=0
for V in [np.array([[1.,0.,0.],[1.,.2,0.],[1.,0.,.3]]),np.array([[.3,.1,.2],[.7,.6,.4],[-.1,.5,.6]])]:
 N=np.cross(V[1]-V[0],V[2]-V[0]);N/=np.linalg.norm(N)
 E=edges_for_triangle(V,N)
 for a in np.linspace(-.2,1.2,31):
  for b in np.linspace(-.2,1.2,31):
   p=V[0]+a*(V[1]-V[0])+b*(V[2]-V[0]);count+=1
   if min(a,b,1-a-b)>1e-8:assert np.min(E[:,:3]@p+E[:,3])>0
   if min(a,b,1-a-b)<-1e-8:assert np.min(E[:,:3]@p+E[:,3])<0
   # Normal approach leaves the finite-triangle coordinates unchanged.
   assert np.max(abs(E[:,:3]@(p+.012*N)+E[:,3]-(E[:,:3]@p+E[:,3])))<1e-12
result=dict(passed=True,independent_barycentric_queries=count,wall_seconds=time.monotonic()-start,physics_steps=0,saved_inputs_used=False)
with Path(__file__).with_name('synthetic_audit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
