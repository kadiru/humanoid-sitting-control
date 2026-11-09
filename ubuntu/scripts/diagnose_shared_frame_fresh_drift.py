"""Offline localization of fresh-bank failures; no runtime or thresholds changed."""
import json
from pathlib import Path
import numpy as np
from humanoid_sitting.shared_frame import (BodyModel,unproject,voxel,cloud_normals,
    depth_edges,apply,regularize_plane_normals,support_patch,transform)
from evaluate_shared_frame import pose,polygon
from run_shared_frame import serial

root=Path(__file__).resolve().parents[1];capture=root/'runs/shared_frame_junction_fresh_capture_01'
out=root/'runs/shared_frame_junction_drift_diagnostic_01';out.mkdir(exist_ok=False)
with np.load(capture/'sensors.npz') as ar:d={k:ar[k] for k in ('depth','K','q','T_P_C')}
with np.load(capture/'truth.npz') as ar:truth={k:ar[k] for k in ('root_pose','seat_pose')}
cfg=json.loads((capture/'manifest.json').read_text())
rows=json.loads((root/'runs/shared_frame_junction_fresh_clean_01/estimates.json').read_text())
errors=json.loads((root/'runs/shared_frame_junction_fresh_clean_01/errors.json').read_text())
model=BodyModel(root/'runs/g1_kinematics_geometry_04/model.json');result=[]
for i,k in [(0,80),(0,107),(0,111),(1,80),(1,95),(1,97),(1,99),(3,80),(3,100),(3,116),(3,135)]:
    row=rows[k*6+i];err=errors[k*6+i];T=np.asarray(row['T_O_P']);fk=model.fk(d['q'][k,i])
    points=voxel(unproject(d['depth'][k,i],d['K'][i],d['T_P_C']),.012)
    points=points[model.self_mask(points,fk)];p,n=cloud_normals(points)
    world=apply(T,p);normals,planes=regularize_plane_normals(world,n@T[:3,:3].T)
    z=row['seat_center_O'][2]
    matches=[p for p in planes if abs(p['height']-z)<.035]
    record=dict(case=i,frame=k,stamp=row['stamp'],map_position_error_cm=err['pelvis_position_error_cm'],
        mapped_seat_center_error_cm=err['seat_visible_center_error_cm'],
        map_forward_error_cm=err['forward_offset_error_cm'],map_lateral_error_cm=err['lateral_offset_error_cm'],
        current_patch=None)
    if matches:
        plane=max(matches,key=lambda x:len(x['points']))
        edges=depth_edges(d['depth'][k,i],d['K'][i],d['T_P_C']);edges=edges[model.self_mask(edges,fk)]
        edges=apply(T,edges);edges=edges[abs(edges@plane['normal']+plane['offset'])<.012]
        patch=support_patch(plane,contour=edges,context_points=world,context_normals=normals)
        if patch:
            relative=np.linalg.inv(patch['T'])@T
            # Ground truth below is used only to assess this diagnostic estimate.
            g=cfg['geometry_bank'][i];poly=polygon(g);center=(poly.min(0)+poly.max(0))/2
            true_seat=pose(truth['seat_pose'][k,i])@transform(p=[*center,.05])
            expected=np.linalg.inv(true_seat)@pose(truth['root_pose'][k,i])
            record['current_patch']=dict(boundary=patch['boundary_evidence'],axis_ambiguous=patch['axis_ambiguous'],
                forward_error_cm=100*abs(relative[0,3]-expected[0,3]),
                lateral_error_cm=100*abs(relative[1,3]-expected[1,3]))
    result.append(serial(record))
(out/'diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
for r in result:print(json.dumps(r),flush=True)
