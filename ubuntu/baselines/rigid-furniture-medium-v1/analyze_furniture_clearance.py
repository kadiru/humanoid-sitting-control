"""Conservative actual arm-collider AABBs in each recorded couch's frame."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--model',default='../runs/g1_kinematics_geometry_04/model.json');a=p.parse_args()
r=Path(a.run);metrics=json.loads((r/'results.json').read_text());model=json.loads(Path(a.model).read_text());z=np.load(r/'rollouts.npz')
summary=[]
for i,o in enumerate(metrics['outcomes']):
 g=metrics['geometry_bank'][o['geometry_index']]
 if g['family']!='rigid_couch':continue
 center=np.array([g['x'],g['y'],g['height']-.05]);rotation=Rotation.from_euler('z',g['yaw']).as_matrix()
 boxes=[('back',np.array([-g['depth']/2,-g['width']/2,.05]),np.array([-g['depth']/2+.06,g['width']/2,.4]))]
 for sign in (-1,1):
  lo=np.array([-g['depth']/2,sign*(g['width']/2+.03)-.03,.05]);hi=np.array([g['depth']/2,sign*(g['width']/2+.03)+.03,.25]);boxes.append((f'arm_{sign}',lo,hi))
 for body in ['left_palm_link','right_palm_link','left_one_link','right_one_link','left_two_link','right_two_link']:
  points=np.concatenate([np.array(c['points']) for c in model['collisions'] if c['body']==body]);bounds=np.stack([points.min(0),points.max(0)]);corners=np.array([[bounds[b[j],j] for j in range(3)] for b in itertools.product((0,1),repeat=3)])
  bi=metrics['articulation_body_names'].index(body);poses=z['body_pose'][:,i,bi];rots=Rotation.from_quat(poses[:,3:]).as_matrix();local=(np.einsum('tij,nj->tni',rots,corners)+poses[:,:3,None].transpose(0,2,1)-center)@rotation
  lo=local.min(1);hi=local.max(1)
  for label,blo,bhi in boxes:
   separation=np.maximum(lo-bhi,blo-hi).max(axis=1);k=int(separation.argmin())
   summary.append(dict(trial=i,geometry_index=o['geometry_index'],body=body,furniture_part=label,minimum_conservative_axis_separation_m=float(separation[k]),time_s=k*float(z['dt']),note='Positive proves separated AABBs. Negative is conservative overlap, not proof of mesh contact; measured forces determine safety.'))
(r/'arm_clearance_analysis.json').write_text(json.dumps(summary,indent=2)+'\n');print('Worst conservative bound',min(summary,key=lambda v:v['minimum_conservative_axis_separation_m']))
