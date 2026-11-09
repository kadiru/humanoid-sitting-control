"""Actual rear-camera RGB/depth/pointcloud and existing recognizer comparison."""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from common import ROOT,read,write
ASSETS=ROOT/'experiments/furniture_test_20261028_v1'
VALIDATE=ROOT/'experiments/furniture_validation_20261028_v2'
sys.path.extend([str(ROOT/'scripts'),str(ASSETS),str(VALIDATE)])
import run_resting_posture as base
import run_supported_postures as support
from environment import FullBodyEnv
from scene_adapter import make_cfg

def analyze(depth,K,position,rotation,case):
    import trimesh
    z=np.load(case['mesh_npz']);mesh=trimesh.Trimesh(z['vertices'],z['faces'],process=False)
    region=read(ASSETS/'scenes'/case['id']/'target_region.json')
    # Projection uses calibrated optical-z depth, not radial distance.
    H,W=depth.shape;vv,uu=np.mgrid[0:H:3,0:W:3];u=uu.reshape(-1);v=vv.reshape(-1)
    dirs=np.stack(((u-K[0,2])/K[0,0],(v-K[1,2])/K[1,1],np.ones(len(u))),axis=-1)@rotation.T
    origins=np.broadcast_to(position,dirs.shape)
    hits,rays,faces=mesh.ray.intersects_location(origins,dirs,multiple_hits=False)
    expected=np.full(len(u),np.inf);hitface=np.full(len(u),-1)
    expected[rays]=(hits-position)@rotation[:,2];hitface[rays]=faces
    actual=depth[v,u];inview=np.isfinite(expected)&(expected>0)
    finite=np.isfinite(actual)&(actual>0)
    occluded=inview&finite&(actual<expected-.015)
    matching=inview&finite&(abs(actual-expected)<=.015)
    missing=inview&~occluded&~matching
    # The exact region key is validated in static preflight.
    target=np.isin(hitface,np.asarray(region['triangle_indices']))
    points=(dirs*actual[:,None])+position
    return dict(rays=len(u),furniture_projected_pixels=int(inview.sum()),visible_furniture_pixels=int(matching.sum()),
        robot_or_scene_occluded_pixels=int(occluded.sum()),depth_mismatch_pixels=int(missing.sum()),
        visible_target_pixels=int((target&matching).sum()),target_projected_pixels=int((target&inview).sum()),
        max_matching_depth_error_m=float(abs(actual[matching]-expected[matching]).max()) if matching.any() else None,
        camera_geometry_consistent=bool(matching.sum()>=30 and missing.sum()<=max(1,int(.01*(inview&~occluded).sum()))),
        target_observable=bool((target&matching).sum()>=20)),dict(points=points[finite],furniture_points=points[matching],target_points=points[target&matching],sample_u=u,sample_v=v,matching=matching,target=target,missing=missing)

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--output',required=True);base.add_launcher_args(p);a=p.parse_args()
    c=next(c for c in read(ASSETS/'cases.json') if c['id']==a.case)
    a.device='cuda:0';a.num_envs=1;a.seed=64701;a.mode='evaluate';a.depth_check=True;a.enable_cameras=True
    out=Path(a.output);out.mkdir(exist_ok=False);bankroot=ROOT/'outputs/full_body_contact_imitation_v4/dataset'/c['task']
    meta=read(bankroot/'metadata.json');meta['cases']=1;meta['ids']=[c['id']]
    with np.load(bankroot/'bank.npz') as z:
        bank={k:z[k][:1].copy() for k in ['reset_q','reset_qd','reset_command','reset_root_pose','reset_root_velocity','phase_times','admitted']}
        bank['q_post']=z['q_post'][:,:1].copy();bank['reference']=np.zeros(1,int)
    cfg=make_cfg(a,c);cfg.scene.rear_camera.data_types=['rgb','distance_to_image_plane']
    frames=[];start=time.monotonic()
    with base.launch_simulation(cfg,a):
        raw=base.gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped
        env=FullBodyEnv(raw,a,out,bank,meta)
        try:
            from robot_trial import collision_inventory
            collision_inventory(raw,out)
            for frame in range(3):
                assert time.monotonic()-start<900
                if frame:
                    for _ in range(10):
                        raw.step(env.previous);env.simulator_control_steps+=1
                        env.warmup_transitions+=1;env.t+=1
                env._observe(force=True)
                camera=raw.scene['rear_camera'];data=camera.data
                depth=data.output['distance_to_image_plane'].torch[0].cpu().numpy().squeeze()
                rgb=data.output['rgb'].torch[0].cpu().numpy()
                K=data.intrinsic_matrices.torch[0].cpu().numpy();origin=raw.scene.env_origins[0].cpu().numpy()
                position=data.pos_w.torch[0].cpu().numpy()-origin;quat=data.quat_w_ros.torch[0].cpu().numpy()
                R=Rotation.from_quat(quat).as_matrix()
                summary,cloud=analyze(depth,K,position,R,c)
                robot=raw.scene['robot'].data;rootR=Rotation.from_quat(robot.root_quat_w.torch[0].cpu().numpy()).as_matrix()
                root=robot.root_pos_w.torch[0].cpu().numpy()-origin;normal=np.asarray(c['target_normal'])
                g=env.geometry[0];normal_error=float(np.rad2deg(np.arccos(np.clip(g[2:5]@(rootR.T@normal),-1,1))))
                offset_error=float(g[5]-(root-np.asarray(c['target_center']))@normal)
                summary.update(frame=frame,recognizer_valid=bool(g[11]),recognizer_target_normal_error_deg=normal_error,
                    recognizer_target_offset_error_m=offset_error,recognizer_matches_target=bool(g[11] and normal_error<=5 and abs(offset_error)<=.015))
                np.savez_compressed(out/f'frame_{frame}.npz',depth=depth,rgb=rgb,K=K,camera_position=position,camera_quaternion_ros=quat,
                    geometry_observed=g,**cloud)
                frames.append(summary);write(out/'depth_summary.json',dict(case=c,frames=frames,complete=len(frames)==3,
                    acquisition='Actual unchanged rear depth optics, robot and furniture; RGB added only for visualization.',
                    ground_truth_use='Offline mesh-ray and target-association validation only; never sent to actor or recognizer.',
                    diagnostic_thresholds=dict(depth_error_m=.015,target_normal_deg=5,minimum_target_sampled_pixels=20),
                    physical_success_claim=False))
        finally:env.close()

if __name__=='__main__':main()
