"""Render a rear-facing body-mounted RGB-D sensor at stored demonstration starts."""
import argparse,json,sys,shutil
from pathlib import Path
import numpy as np,torch,gymnasium as gym
from scipy.spatial.transform import Rotation
from isaaclab.app import add_launcher_args,launch_simulation
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim
from humanoid_sitting.tasks.geometry_contact_v12 import GeometryEnvCfg
import humanoid_sitting.tasks
from seat_depth_estimator import estimate_body_frame
p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--output',required=True);add_launcher_args(p);a=p.parse_args();a.enable_cameras=True
out=Path(a.output);out.mkdir(parents=True,exist_ok=False);assert shutil.disk_usage(out).free>30*2**30
fixture=Path(a.fixture);r=json.loads((fixture/'results.json').read_text());initial=np.load(fixture/'initial_conditions.npz');N=r['num_trials']
f=np.array([-1.,0.,-1.7]);f/=np.linalg.norm(f);right=np.cross(f,[0,0,1]);right/=np.linalg.norm(right);down=np.cross(f,right);quat=Rotation.from_matrix(np.stack((right,down,f),axis=1)).as_quat();mount=np.array([.05,.18,.55])
cfg=GeometryEnvCfg(geometry_stage=r['geometry_stage'],geometry_seed=r['geometry_seed'],geometry_count=len(r['geometry_bank']),geometry_bank=r['geometry_bank'],include_families=True,arm_clearance=True);cfg.scene.num_envs=N;cfg.sim.device=a.device
cfg.scene.seat_camera=CameraCfg(prim_path='{ENV_REGEX_NS}/Robot/pelvis/SeatCamera',width=320,height=240,data_types=['rgb','distance_to_image_plane'],update_period=0,offset=CameraCfg.OffsetCfg(pos=tuple(mount),rot=tuple(quat),convention='ros'),spawn=sim.PinholeCameraCfg(focal_length=18.,horizontal_aperture=24.,clipping_range=(.05,4.)))
with launch_simulation(cfg,a):
    env=gym.make('Humanoid-Sit-Rigid-Furniture-G1',cfg=cfg);raw=env.unwrapped;raw.reset();robot=raw.scene['robot']
    pose=torch.tensor(initial['root_pose'],device=a.device);position=torch.tensor([[o['x'],o['y'],o['z']] for o in r['outcomes']],device=a.device);pose[:,:3]=raw.scene.env_origins+position
    robot.write_root_pose_to_sim_index(root_pose=pose);robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(N,6,device=a.device));robot.write_joint_position_to_sim_index(position=torch.tensor(initial['joint_pos'],device=a.device));robot.write_joint_velocity_to_sim_index(velocity=torch.zeros(N,37,device=a.device));robot.reset()
    robot.set_joint_position_target(torch.tensor(initial['joint_pos'],device=a.device))
    for _ in range(12):raw.scene.write_data_to_sim();raw.sim.step();raw.scene.update(.005)
    camera=raw.scene['seat_camera'];camera.update(.005,force_recompute=True);d=camera.data
    depth=d.output['distance_to_image_plane'].torch.cpu().numpy();rgb=d.output['rgb'].torch.cpu().numpy();K=d.intrinsic_matrices.torch.cpu().numpy();gravity=robot.data.projected_gravity_b.torch.cpu().numpy()
    estimates=[]
    for i in range(N):
        try:est=estimate_body_frame(depth[i],K[i],mount,quat,gravity[i]);estimates.append(dict(valid=True,**est))
        except Exception as e:estimates.append(dict(valid=False,error=str(e)))
    np.savez_compressed(out/'rgbd.npz',depth=depth,rgb=rgb,K=K,imu_gravity=gravity,mount_position=mount,mount_quaternion=quat)
    (out/'estimates.json').write_text(json.dumps(estimates,indent=2)+'\n');(out/'manifest.json').write_text(json.dumps(dict(arguments=vars(a),frames=N,fixture=str(fixture),sensor='simulated rear-facing body-mounted RGB-D; proposed mount, not verified physical G1 hardware',actor_geometry_uses='depth, camera intrinsics, fixed body extrinsics, IMU gravity; floor inferred from depth',ground_truth_robot_or_seat_pose_used_in_estimator=False,physics_settle_s=.06,valid=sum(x['valid'] for x in estimates)),indent=2)+'\n')
    print('BODY_RGBD_VALID',sum(x['valid'] for x in estimates),'/',N,flush=True);env.close()
