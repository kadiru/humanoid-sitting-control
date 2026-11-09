"""Measured collider front-edge clearance at 20 ms; force histories remain authoritative."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();root=Path(__file__).resolve().parents[1];run=Path(a.run);r=json.loads((run/'results.json').read_text());m=json.loads((root/'runs/g1_kinematics_geometry_04/model.json').read_text());d=np.load(run/'rollouts.npz');poses=d['body_pose'];rows=[]
for i,o in enumerate(r['outcomes']):
 g=r['geometry_bank'][o['geometry_index']];center=np.array([g['x'],g['y'],g['height']-.05]);seat_rot=Rotation.from_euler('z',g['yaw']).as_matrix()
 for body in ['left_knee_link','right_knee_link','left_ankle_roll_link','right_ankle_roll_link']:
  points=np.concatenate([np.array(x['points']) for x in m['collisions'] if x['body']==body]);pose=poses[:,i,r['articulation_body_names'].index(body)];rot=Rotation.from_quat(pose[:,3:]).as_matrix()
  # Transform actual exported collision vertices, not an assumed knee sphere.
  local=(np.einsum('tij,nj->tni',rot,points)+pose[:,:3,None].transpose(0,2,1)-center)@seat_rot
  low=local.min(1);high=local.max(1);active=(low[:,2]<=.05)&(high[:,2]>=-.05)&(low[:,1]<=g['width']/2)&(high[:,1]>=-g['width']/2)
  gap=low[:,0]-g['depth']/2
  k=int(np.argmin(np.where(active,gap,np.inf)))
  rows.append(dict(trial=i,geometry_index=o['geometry_index'],body=body,minimum_front_clearance_at_seat_height_m=float(gap[k]) if active.any() else None,time_s=k*float(d['dt']) if active.any() else None))
(run/'front_clearance_audit.json').write_text(json.dumps(dict(rows=rows,pose_sample_dt_s=float(d['dt']),force_sample_dt_s=r['contact_sampling_dt_s'],note='Seat-local projected actual collider bounds. Positive proves front-edge clearance at recorded poses; 5 ms contact-force history defines safety.'),indent=2)+'\n')
valid=[x for x in rows if x['minimum_front_clearance_at_seat_height_m'] is not None];print('Worst',min(valid,key=lambda x:x['minimum_front_clearance_at_seat_height_m']) if valid else None)
