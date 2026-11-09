"""Analytical camera-visible standing placement; no motion or pose optimization."""
import os,json,time,signal,resource,hashlib,sys
from pathlib import Path
START=time.monotonic()
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
def timeout(*args):raise TimeoutError('45_SECOND_PREPARATION_CAP')
signal.signal(signal.SIGALRM,timeout);signal.alarm(45)
H=Path(__file__).resolve().parent;R=H.parent.parent
def main():
    import numpy as np
    frozen=json.loads((H/'preparation_freeze.json').read_text())
    for name,digest in frozen['files'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    sys.path[1:1]=[str(R/'sitting_task/src'),str(R/'experiments/furniture_repair_20261106_v1')]
    from humanoid_sitting.shared_frame import BodyModel,level_rotation
    from audit_mesh_clearance import audit
    assets=H.parent/'rest_cycle_high_support_20261108_v1/assets';case=json.loads((assets/'cases.json').read_text())[0]
    with np.load(case['mesh_npz']) as z:furniture=dict(z)
    region=json.loads((assets/'scenes'/case['id']/'target_region.json').read_text())['triangle_indices']
    vertices=furniture['vertices'][furniture['faces'][region]].reshape(-1,3)
    edge=np.array([vertices[:,0].max(),0.,case['target_center'][2]])
    source=R/'outputs/rest_cycle_high_support_20261108_v1/sit_high_support/frame_0.npz'
    with np.load(source) as z:
        Rcam=z['camera_rotation'];K=z['K'];camera=z['camera_position']
        basis=z['root_rotation']@level_rotation(z['gravity'].astype(float).copy()).T
        forward=basis[:,0];initial=(edge-camera)@Rcam;direction=forward@Rcam
        target_pixel=z['depth'].shape[0]-10.5;r=(target_pixel-K[1,2])/K[1,1]
        translation=float((initial[1]-r*initial[2])/(direction[1]-r*direction[2]))+.001
        assert 0<translation<=.1,'REQUIRED_STANDING_PLACEMENT_OUTSIDE_DECLARED_RANGE'
        predicted=initial-translation*direction
        pixel=float(K[1,2]+K[1,1]*predicted[1]/predicted[2])
        oldpixel=float(K[1,2]+K[1,1]*initial[1]/initial[2])
        points=z['selected_points']@z['root_rotation'].T+z['root_position']
        old_edge=float(points[:,0].max())
        root=z['root_position']+translation*forward
        model=BodyModel(R/'runs/g1_kinematics_geometry_04/model.json')
        shapes=json.loads((R/'experiments/furniture_perception_repair_20261029_v1/render_colliders.json').read_text())['shapes']
        node=dict(q=z['q'].tolist(),root=np.eye(4).tolist(),phase='approach')
        clearance=audit(model,shapes,furniture,region,[node,node],z['root_rotation'],root)
    tests=dict(analytical_visible_edge=pixel<=target_pixel,standing_clearance=bool(clearance['passed']))
    result=dict(passed=all(tests.values()),tests=tests,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        original_front_pixel_y=oldpixel,predicted_front_pixel_y=pixel,target_pixel_y=float(target_pixel),
        actual_front_world_x=float(edge[0]),previous_observed_front_world_x=old_edge,
        original_front_extent_error_m=old_edge-float(edge[0]),reset_translation_forward_m=translation,
        closer_m=.05-translation,seed=66302,clearance=clearance,
        geometry_used_only_for='initial scene placement and independent admission; actor uses fresh camera',
        current_camera_validated=False,model_motion_admitted=False,physical_success=False,physics_steps=0,
        elapsed_seconds=time.monotonic()-START)
    (H/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='clearance'}))
    assert result['passed']
if __name__=='__main__':main()
