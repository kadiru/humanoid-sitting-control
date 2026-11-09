"""Calibrated field-of-view mechanism, tested on saved occluded boundary.

The full scientific contract is frozen before this bounded CPU check. Saved
world transforms are used only to evaluate visibility, never as controller input.
"""
import os, sys, time, json, hashlib, ast, resource
from pathlib import Path
START=time.monotonic()
os.environ['CUDA_VISIBLE_DEVICES']=''
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
import numpy as np
H=Path(__file__).resolve().parent;R=H.parent.parent
sys.path.insert(0,str(R/'sitting_task/src'))
sys.path.insert(0,str(R/'experiments/furniture_perception_repair_20261029_v1'))
from perception import cloud


def main():
    contract=json.loads((H/'contract.json').read_text())
    for name,digest in contract['files'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    for name in ['trial.py','construct.py','owner.py']:ast.parse((H/name).read_text())
    K=np.array([[384.*15./39.,0,192.],[0,384.*15./39.,144.],[0,0,1.]])
    T=np.eye(4);T[:3,:3]=[[0.,.707106781,-.707106781],[1.,0.,0.],[0.,-.707106781,-.707106781]];T[:3,3]=[-.07,0.,.38]
    oldpath=R/'outputs/furniture_examples_20261105_v2/sit_modern_coffee_table_01_center/frame_0.npz'
    newpath=R/'outputs/rest_cycle_startup_fix_20261108_v1/sit_coffee_center/frame_0.npz'
    with np.load(oldpath) as old,np.load(newpath) as new:
        p=(old['selected_points']@old['root_rotation'].T+old['root_position']-new['root_position'])@new['root_rotation']
        camera=(p-T[:3,3])@T[:3,:3]
        pixels=camera[:,:2]/camera[:,2,None]*K[0,0]+K[:2,2]
        front=old['selected_points'][:,0]>=old['selected_points'][:,0].max()-.003
        margin=np.minimum.reduce([pixels[:,0],383.5-pixels[:,0],pixels[:,1],287.5-pixels[:,1]])
        visible=(camera[:,2]>0)&(margin>0)
        assert np.all(visible[front]),'PREVIOUS_OBSERVED_FRONT_STILL_OUTSIDE_VIEW'
        oldfx=float(old['K'][0,0]);oldwidth=old['depth'].shape[1]
    # Pixel-center roundtrip at the new resolution; no synthetic data is called
    # a physical or scene validation. The real K must pass again before motion.
    depth=np.full((288,384),.7);points,valid=cloud(depth,K,T)
    camera=(points.reshape(-1,3)-T[:3,3])@T[:3,:3]
    pixels=camera[:,:2]/camera[:,2,None]*K[0,0]+K[:2,2]
    y,x=np.mgrid[:288,:384];expected=np.c_[x.ravel()+.5,y.ravel()+.5]
    error=float(np.max(np.abs(pixels-expected)))
    assert valid.all() and error<1e-5
    result=dict(passed=True,saved_front_boundary_samples=int(front.sum()),
        saved_front_boundary_inside_wider_view=int(visible[front].sum()),
        minimum_saved_front_boundary_image_margin_px=float(margin[front].min()),
        horizontal_FOV_deg=float(2*np.rad2deg(np.arctan(39./(2*15.)))),
        vertical_FOV_deg=float(2*np.rad2deg(np.arctan((39.*288/384)/(2*15.)))),
        original_horizontal_FOV_deg=float(2*np.rad2deg(np.arctan(oldwidth/(2*oldfx)))),
        angular_pixel_density_ratio=float(K[0,0]/oldfx),pixel_center_roundtrip_error_px=error,
        camera=dict(width=384,height=288,horizontal_aperture_mm=39.,focal_length_mm=15.,K=K.tolist()),
        actual_capture_validated=False,occlusion_free_claim=False,mechanical_path_admitted=False,
        complete_cycle=False,physics_steps=0,PPO_updates=0,wall_seconds=time.monotonic()-START)
    with (H/'preflight.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))

if __name__=='__main__':main()
