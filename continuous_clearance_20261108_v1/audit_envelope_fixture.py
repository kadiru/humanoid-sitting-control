"""Independent ray/plane sufficient-condition test on synthetic creased mesh."""
import json
from pathlib import Path
import numpy as np
from envelope_camera import front_envelope
camera=np.array([1.,0.,.6]);axis=np.array([1.,0.,0.]);margin=.006
V=np.array([[[0.,-.3,.2],[0.,.3,.2],[0.,-.3,.8]],[[0.,.3,.2],[-.08,.3,.8],[0.,-.3,.8]]])
N=np.cross(V[:,1]-V[:,0],V[:,2]-V[:,0]);N/=np.linalg.norm(N,axis=1)[:,None]
N=np.where((np.sum(N*(camera-V[:,0]),axis=1)<0)[:,None],-N,N);d=-np.sum(N*V[:,0],axis=1)
bound,info=front_envelope(V,N,d,np.ones(2,bool),camera,axis)
minimum=float('inf');samples=0
for i in range(2):
    for u in np.linspace(.01,.98,17):
        for v in np.linspace(.01,1-u,13):
            if u+v>1:continue
            surface=(1-u-v)*V[i,0]+u*V[i,1]+v*V[i,2]
            for inward in (0.,.0001,.01):
                t=(camera@axis-bound-inward)/(camera@axis-surface@axis)
                point=camera+t*(surface-camera)
                minimum=min(minimum,float(point@N[i]+d[i]-margin));samples+=1
wrong=np.max(V@axis)<bound
tests=dict(original_plane_margin_preserved=minimum>=-1e-12,
           actual_surface_points_rejected=bool(wrong),front_envelope_between_surface_and_camera=0.<bound<1.)
result=dict(passed=all(tests.values()),tests=tests,samples=samples,minimum_original_margin_residual_m=minimum,
            envelope=info,experiment_inputs_read=False,physics_steps=0)
with Path(__file__).with_name('synthetic_audit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result));assert result['passed']
