"""Small renderer-backed RGB-D fixture set for a calibrated external observer."""
import argparse,json,shutil
from pathlib import Path
import numpy as np,torch
from isaaclab.app import add_launcher_args,launch_simulation
import isaaclab.sim as sim
from isaaclab_physx.physics import PhysxCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import instantiate
from humanoid_sitting.tasks.furniture import bank,FurnitureSpawnCfg
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=23001);p.add_argument('--count',type=int,default=16);p.add_argument('--wide',action='store_true');add_launcher_args(p);a=p.parse_args();a.enable_cameras=True
out=Path(a.output);out.mkdir(parents=True,exist_ok=False);assert shutil.disk_usage(out).free>30*2**30
cfg=sim.SimulationCfg(device=a.device,dt=.005,physics=PhysxCfg())
with launch_simulation(cfg,a):
    context=sim.SimulationContext(cfg)
    sim.spawn_ground_plane('/World/Ground',sim.GroundPlaneCfg())
    light=sim.DomeLightCfg(intensity=2000);light.func('/World/Light',light)
    geometries=bank(2,a.seed,a.count,True);rng=np.random.default_rng(a.seed+1)
    if a.wide:
        for g in geometries:
            g.update(height=float(rng.uniform(.22,.48)),depth=float(rng.uniform(.35,.65)),width=float(rng.uniform(.8,1.3)),yaw=float(rng.uniform(-np.pi,np.pi)))
    origins=[]
    for i,g in enumerate(geometries):
        origin=np.array([4*(i%4),4*(i//4),0.]);origins.append(origin)
        spawn=FurnitureSpawnCfg(size=(g['depth'],g['width'],.1),family=g['family'],top_height=g['height'],rigid_props=sim.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),collision_props=sim.UsdPhysicsCollisionCfg(),mass_props=sim.MassCfg(mass=50.),visual_material=sim.PreviewSurfaceCfg(diffuse_color=tuple(rng.uniform(.15,.8,3))),activate_contact_sensors=True)
        spawn.func(f'/World/Furniture_{i}',spawn,translation=tuple(origin+[g['x'],g['y'],g['height']-.05]),orientation=(0.,0.,np.sin(g['yaw']/2),np.cos(g['yaw']/2)))
    camera=instantiate(CameraCfg(prim_path='/World/Observer',width=320,height=240,update_period=0,data_types=['rgb','distance_to_image_plane'],update_latest_camera_pose=True,spawn=sim.PinholeCameraCfg(focal_length=20.,horizontal_aperture=24.,clipping_range=(.05,20))))
    context.reset();records=[]
    for i,(g,origin) in enumerate(zip(geometries,origins)):
        for view in range(2):
            pos=origin+np.array([.65,.7 if view==0 else -.7,1.35]);target=origin+[-.4,0,.3]
            camera.set_world_poses_from_view(torch.tensor(pos[None],device=a.device,dtype=torch.float32),torch.tensor(np.array(target)[None],device=a.device,dtype=torch.float32))
            for _ in range(8):context.step();camera.update(.005,force_recompute=True)
            d=camera.data;rgb=d.output['rgb'].torch[0].cpu().numpy();depth=d.output['distance_to_image_plane'].torch[0].cpu().numpy()
            file=f'frame_{i:02d}_{view}.npz';np.savez_compressed(out/file,rgb=rgb,depth=depth,K=d.intrinsic_matrices.torch[0].cpu().numpy(),camera_position=d.pos_w.torch[0].cpu().numpy()-origin,camera_quaternion=d.quat_w_ros.torch[0].cpu().numpy(),capture_time_s=context.get_physics_step_count()*context.get_physics_dt())
            records.append(dict(file=file,geometry_index=i,view=view,ground_truth=g))
            print('CAPTURE',file,'valid_depth',int(np.isfinite(depth).sum()),flush=True)
    (out/'manifest.json').write_text(json.dumps(dict(arguments=vars(a),records=records,depth_kind='distance_to_image_plane metres',camera='320x240 calibrated external observer; two fixed views per geometry; no robot or occluding clutter',inference_uses='depth and calibration only; RGB saved for inspection; GT metadata evaluation only'),indent=2)+'\n')
