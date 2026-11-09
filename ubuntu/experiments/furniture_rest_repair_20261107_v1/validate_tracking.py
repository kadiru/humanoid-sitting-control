"""Saved-frame validation only; no simulator/optimizer/action replay."""
import sys,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent;EXP=ROOT/'experiments'
sys.path[:0]=[str(HERE),str(EXP/'furniture_collision_repair_20261106_v5'),str(EXP/'furniture_collision_repair_20261106_v4'),str(EXP/'furniture_perception_repair_20261029_v1'),str(ROOT/'sitting_task/src')]
import numpy as np
from surface_observer import Observer
from humanoid_sitting.shared_frame import BodyModel
from scipy.spatial import cKDTree
model=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json');rows=[]
for folder in sorted((ROOT/'outputs/furniture_camera_coverage_20261106_v1/calibration_demos').iterdir()):
 observer=Observer('seat');T=np.load(folder/'construction_inputs.npz')['rear_camera'];prior=None
 for i in [0,350,650,1000]:
  f=folder/f'frame_{i}.npz';z=dict(np.load(f));old=json.loads(f.with_suffix('.json').read_text())
  value,info=observer(model,z['K'],T,z['depth'],z['q'],z['gravity'].astype(float),prior)
  if value is not None:
   prior=value
   assert np.array_equal(value[9:11],observer.contact_span)
   assert np.array_equal(observer.boundary,observer.contact_boundary)
  rows.append(dict(case=folder.name,frame=i,old_valid=bool(z['geometry'][11]),new_valid=value is not None,
    visible_points=len(info.get('selected_points',[])),rejection=info.get('anchored_patch_rejected'),
    plane_residual_m=info.get('surface_anchor',{}).get('maximum_plane_residual_m'),input_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
    new_normal=None if value is None else value[2:5].tolist(),new_offset=None if value is None else float(value[5]),
    post_physical_failure=i>=650 if 'center' in folder.name else i>=1000))
 # Missing images must never be treated as fresh measured geometry.
 value,info=observer(model,z['K'],T,np.full_like(z['depth'],np.nan),z['q'],z['gravity'].astype(float),prior)
 assert value is None
result=dict(rows=rows,missing_depth_rejected=True,physics_steps=0,PPO_updates=0,new_successful_demos=0,
  thresholds_unchanged=True,physical_validation_pending=True,source_sha256=hashlib.sha256((HERE/'surface_observer.py').read_bytes()).hexdigest())
(HERE/'connected_facet_float32_validation.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))
