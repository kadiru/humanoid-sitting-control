"""Inspect current-plane conditioning at retained noisy failures (offline only)."""
import json
from pathlib import Path
import numpy as np
from humanoid_sitting.shared_frame import BodyModel, voxel, unproject, cloud_normals, apply, regularize_plane_normals
root=Path(__file__).resolve().parents[1]
with np.load(root/'runs/shared_frame_junction_fresh_capture_01/sensors.npz') as f:
 d={k:f[k] for k in ('depth','K','q','T_P_C')}
rows=json.loads((root/'runs/shared_frame_junction_fresh_noisy_01/estimates.json').read_text())
model=BodyModel(root/'runs/g1_kinematics_geometry_04/model.json');rng=np.random.default_rng(41019)
for k in range(72):
 for i in range(6):
  depth=d['depth'][k,i].copy();depth+=rng.normal(0,.005,depth.shape);depth[rng.random(depth.shape)<.10]=np.nan
  if i!=3 or k not in (68,69,70,71):continue
  T=np.array(rows[k*6+i]['T_O_P']);fk=model.fk(d['q'][k,i])
  p=voxel(unproject(depth,d['K'][i],d['T_P_C']),.012);p=p[model.self_mask(p,fk)];p,n=cloud_normals(p)
  world=apply(T,p);n,planes=regularize_plane_normals(world,n@T[:3,:3].T)
  for pl in planes:
   if not .28<pl['height']<.39:continue
   pts=pl['points'];v=apply(T,model.vertices['pelvis']);coarse=voxel(pts,.06)
   center=coarse[:,:2].mean(0);X=np.c_[coarse[:,:2]-center,np.ones(len(coarse))];Q=np.c_[v[:,:2]-center,np.ones(len(v))]
   bound=3*max(.005,pl['rms'])*np.sqrt(np.einsum('ij,jk,ik->i',Q,np.linalg.pinv(X.T@X),Q)).max()
   print(json.dumps(dict(frame=k,points=len(pts),coarse=len(coarse),span=np.ptp(pts[:,:2],axis=0).tolist(),normal=pl['normal'].tolist(),height=pl['height'],rms=pl['rms'],bound=bound,raw_clearance=float(np.min(v@pl['normal']+pl['offset'])))))
