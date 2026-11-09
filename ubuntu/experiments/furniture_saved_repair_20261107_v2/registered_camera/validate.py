"""Saved depth/encoder replay; simulator poses only in independent evaluation."""
import os,time,sys,json,hashlib
START=time.monotonic();os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
H=Path(__file__).resolve().parent;R=Path('/home/kadir/humanoid-sitting');E=R/'experiments'
sys.path[:0]=[str(H),str(E/'furniture_collision_repair_20261106_v5'),str(E/'furniture_collision_repair_20261106_v4'),str(E/'furniture_perception_repair_20261029_v1'),str(R/'sitting_task/src')]
freeze=json.load(open(H/'freeze.json'))
for p,h in freeze['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
import numpy as np
from humanoid_sitting.shared_frame import BodyModel,level_rotation
from surface_observer import Observer
m=BodyModel(R/'runs/g1_kinematics_geometry_04/model.json');rows=[];missing=[];negative=[]
for folder in sorted((R/'outputs/furniture_camera_coverage_20261106_v1/calibration_demos').iterdir()):
    observer=Observer('seat');T=np.load(folder/'construction_inputs.npz')['rear_camera'];prior=None
    z0=dict(np.load(folder/'frame_0.npz'));basis=z0['root_rotation']@level_rotation(z0['gravity'].astype(float)).T
    for i in [0,350,650,1000]:
        z=dict(np.load(folder/f'frame_{i}.npz'))
        value,info=observer(m,z['K'],T,z['depth'],z['q'],z['gravity'].astype(float),prior)
        evaluation=None
        if value is not None:
            prior=value
            assert np.array_equal(value[9:11],observer.contact_span)
            assert np.array_equal(observer.boundary,observer.contact_boundary)
            # These saved simulator poses never enter the observer call above.
            true_origin=basis.T@(z['root_position']-z0['root_position'])
            true_n=z['root_rotation'].T@basis@observer.initial_n
            true_d=observer.initial_d+observer.initial_n@true_origin
            true_target=z['root_rotation'].T@(basis@observer.target+z0['root_position']-z['root_position'])
            evaluation=dict(normal_error=float(np.linalg.norm(value[2:5]-true_n)),plane_offset_error_m=float(abs(value[5]-true_d)),target_error_m=float(np.linalg.norm(value[6:9]-true_target)))
        before_failure=i<650 if 'center' in folder.name else i<1000
        rows.append(dict(case=folder.name,frame=i,before_physical_failure=before_failure,valid=value is not None,registration=info.get('surface_anchor'),rejection=info.get('anchored_patch_rejected'),independent_saved_pose_check=evaluation,input_sha256=hashlib.sha256((folder/f'frame_{i}.npz').read_bytes()).hexdigest()))
    for mode,depth in [('missing',np.full_like(z0['depth'],np.nan)),('wrong_distance',z0['depth']+.1)]:
        value,info=observer(m,z0['K'],T,depth,z0['q'],z0['gravity'].astype(float),prior)
        (missing if mode=='missing' else negative).append(dict(case=folder.name,rejected=value is None,reason=info.get('anchored_patch_rejected')))
pre=[r for r in rows if r['before_physical_failure']]
passed=all(r['valid'] and r['independent_saved_pose_check']['plane_offset_error_m']<.003 and r['independent_saved_pose_check']['target_error_m']<.003 and r['independent_saved_pose_check']['normal_error']<.03 for r in pre)
result=dict(rows=rows,missing_depth=missing,wrong_distance=negative,all_saved_pre_failure_checks_passed=passed and all(r['rejected'] for r in missing+negative),full_descent_validated=False,deployed=False,physical_success=False,new_physics_steps=0,new_PPO_updates=0,seconds=time.monotonic()-START)
with (H/'validation.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(result),flush=True)
