"""Offline geometry diagnosis; truth poses are evaluation labels only."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform
from plan_seat_load_targets import clipped_vertices

root=Path('/data/humanoid-sitting-perception/crouch_rest_bridge_01')
model=BodyModel('runs/g1_kinematics_geometry_04/model.json')
rows=[]
for name in sys.argv[1:]:
 p=root/name
 trace=np.load(p/'trace.npz');g=json.loads((p/'geometry.json').read_text())
 startup=np.load(p/'startup.npz');print(name,'startup',startup.files)
 results=json.loads((p/'results.json').read_text());names=results.get('body_names')
 if names is None:
  print('RESULT KEYS',list(results));continue
 for episode in results['episodes'][::2]:
  i=episode['case'];geom=g[episode['geometry_index']]
  for t in [9.9,10.,10.3,10.5,11.,12.,14.,18.]:
   j=int(np.argmin(abs(trace['t']-t)));body=trace['body'][j,i];q=trace['q'][j,i]
   pelvis=body[names.index('pelvis')];origin=pelvis[:3].copy()
   # Align each trial with the explicit canonical reference-frame reset.
   r0=startup['root'][i] if 'root' in startup else None
   if r0 is None:print('NO ROOT');break
   # World translations are available as evaluation metadata.
   if 'origins' in results:origin-=np.array(results['origins'][i])
   else:print('NO ORIGINS');break
   T=transform(Rotation.from_quat(pelvis[3:]).as_matrix(),origin);fk=model.fk(q)
   n=np.array(geom['normal']);d=-n@np.array(geom['top_center']);clear=[]
   for b in ['pelvis','left_hip_roll_link','right_hip_roll_link','left_hip_pitch_link','right_hip_pitch_link']:
    pts=clipped_vertices(apply(T@fk[b],model.vertices[b]))
    if len(pts):clear.append([b,float((pts@n+d).min())])
   rows.append(dict(probe=name,case=i,t=float(trace['t'][j]),root=origin.tolist(),pitch=float(Rotation.from_quat(pelvis[3:]).as_euler('xyz')[1]),clearance=clear))
print(json.dumps(rows,indent=2))
