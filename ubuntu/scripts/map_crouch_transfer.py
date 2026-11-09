"""Offline clearance and gravity-moment map; not a physics validation."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform
from plan_seat_load_targets import clipped_vertices
from build_near_seat_candidates import solve

r=Path('/data/humanoid-sitting-perception/crouch_rest_bridge_01')
p=r/'balanced_standing10_01';t=np.load(p/'trace.npz');results=json.loads((p/'results.json').read_text());gs=json.loads((p/'geometry.json').read_text());inertia=json.loads((p/'static_inertial_model.json').read_text())
model=BodyModel('runs/g1_kinematics_geometry_04/model.json');rows=[]
for case in [8,12]:
 q=t['q'][99,case];pose=t['body'][99,case,0];R=Rotation.from_quat(pose[3:]).as_matrix();origin=pose[:3]-results['origins'][case];fk=model.fk(q)
 feet={s:(R@fk[s+'_ankle_roll_link'][:3,3]+origin,R@fk[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']}
 g=gs[results['episodes'][case]['geometry_index']];n=np.array(g['normal']);d=-n@np.array(g['top_center'])
 for pitch in np.arange(.6,-.01,-.05):
  for x in np.arange(-.14,-.231,-.005):
   qi,root,error=solve(model,q,feet,n,d,x,pitch,gap=.001,y=origin[1]);T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]);f=model.fk(qi);clear=[]
   for b in ['left_hip_pitch_link','right_hip_pitch_link','left_hip_roll_link','right_hip_roll_link']:
    pts=clipped_vertices(apply(T@f[b],model.vertices[b]))
    if len(pts):clear.append([float((pts@n+d).min()),b])
   com=np.average([apply(T@f[b],c) for b,c in zip(inertia['body_names'],inertia['local_com'])],axis=0,weights=inertia['mass'])
   pts=clipped_vertices(apply(T,model.vertices['pelvis']));low=pts[pts@n+d<.004];sx=float(low[:,0].mean()) if len(low) else float('nan')
   fx=float(np.mean([v[0][0] for v in feet.values()]));load=(fx-com[0])/(fx-sx)
   rows.append(dict(case=case,pitch=float(pitch),x=float(x),clearance=sorted(clear),com=com.tolist(),seat_x=sx,estimated_seat_bw=load,error=error))
out=r/'offline_clearance_map_01.json';assert not out.exists();out.write_text(json.dumps(rows,indent=2))
for case in [8,12]:
 for pitch in np.arange(.6,-.01,-.05):
  subset=[v for v in rows if v['case']==case and abs(v['pitch']-pitch)<1.e-6 and min(vv[0] for vv in v['clearance'])>.005 and v['error']<.003]
  print(case,round(pitch,2),[(round(v['x'],3),round(v['estimated_seat_bw'],2)) for v in subset])
