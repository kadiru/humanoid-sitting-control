import json,sys,math
from pathlib import Path
import numpy as np
ROOT=Path('/home/kadir/humanoid-sitting');EXP=ROOT/'experiments';HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'sitting_task/src'))
from humanoid_sitting.shared_frame import apply,level_rotation
shapes=json.loads((EXP/'furniture_perception_repair_20261029_v1/render_colliders.json').read_text())['shapes'];pelvis=np.concatenate([s['vertices'] for s in shapes if s['body']=='pelvis'])
rows=[]
for name in ['sit_1kwsjhpY84_center','sit_1kwsjhpY84_angled']:
 plan=json.loads((EXP/'furniture_collision_repair_20261106_v5/seat_models'/(name+'.plan.json')).read_text());root=np.array(plan['nodes'][-1]['root']);n=np.array(plan['surface_normal']);d=plan['surface_offset'];points=apply(root,pelvis);heights=points@n+d;patch=points[heights<=heights.min()+.002]
 p=ROOT/'outputs/furniture_collision_collection_20261106_v2/calibration_demos'/(name+'_local_patch_normal_landing')/'frame_0.npz'
 z=dict(np.load(p));R=level_rotation(z['gravity'].astype(float));rays=patch@R-np.array([-.07,0.,.38]);angles=np.arctan2(-rays[:,2],-rays[:,0]);assert (rays[:,0]<0).all()
 max_row=85.;half=math.atan((max_row+.5-z['K'][1,2])/z['K'][1,1]);needed=math.degrees(float(angles.max())-half)
 current_pixels=z['K'][1,1]*np.tan(angles-math.pi/4)+z['K'][1,2]-.5
 rows.append(dict(case=name,original_contact_patch_pixels_at45deg=current_pixels.tolist(),required_downward_axis_deg=needed,max_permitted_pixel_row=max_row,source='Previously collision/action-admitted endpoint used only to derive camera coverage; no rejected vector used as solver initialization or motion'))
angle=math.ceil(max(r['required_downward_axis_deg'] for r in rows));assert 45<angle<70
out=dict(original_camera_down_deg=45,new_camera_down_deg=angle,camera_translation_unchanged=True,intrinsics_unchanged=True,action_limits_unchanged=True,success_criteria_unchanged=True,physical_validation_pending=True,rows=rows)
(HERE/'coverage_derivation.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(angle=angle,required=[r['required_downward_axis_deg'] for r in rows])))
