"""Contact-binding audit and deliberately invalid knee-contact positive control."""
import argparse,json,re
from pathlib import Path
import numpy as np,torch,gymnasium as gym
from isaaclab.app import add_launcher_args,launch_simulation
from humanoid_sitting.tasks.geometry_contact_v12 import GeometryEnvCfg
from humanoid_sitting.tasks import mdp,full_transition
import humanoid_sitting.tasks
from isaaclab.sim import get_current_stage
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--warnings-log',required=True);add_launcher_args(p);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
cfg=GeometryEnvCfg(geometry_stage=2,geometry_seed=10514,geometry_count=1,include_families=True,arm_clearance=True);cfg.scene.num_envs=1;cfg.sim.device=a.device;cfg.episode_length_s=50.
for name in ('fallen','pelvis_floor','foot_seat','seat_impact','time_out'):setattr(cfg.terminations,name,None)
with launch_simulation(cfg,a):
 env=gym.make('Humanoid-Sit-Rigid-Furniture-G1',cfg=cfg);raw=env.unwrapped;raw.reset();robot=raw.scene['robot']
 from pxr import UsdPhysics
 stage=get_current_stage();expected=[f'/World/envs/env_0/Robot/{name}' for name in robot.body_names]+['/World/envs/env_0/Seat'];missing=[path for path in expected if not stage.GetPrimAtPath(path).HasAPI(UsdPhysics.RigidBodyAPI)];assert not missing,missing
 sensor=raw.scene['all_contacts'];assert set(sensor.body_names)==set(robot.body_names)
 bindings={name:dict(body_names=raw.scene[name].body_names,actual_physx_body_paths=list(raw.scene[name].body_physx_view.prim_paths)) for name in (*mdp.SUPPORT_SENSORS,'all_contacts','seat_contacts','left_foot_seat','right_foot_seat','left_knee_seat','right_knee_seat')}
 for name,b in bindings.items():assert all(path in expected for path in b['actual_physx_body_paths']),(name,b)
 samples=[]
 for step in range(1000):
  raw.step(torch.zeros(1,37,device=a.device))
  if step>850:
   support_vec=sum(raw.scene[name].data.normal_force_matrix_w.torch.sum((1,2)) for name in mdp.SUPPORT_SENSORS)
   seat=raw.scene['seat_contacts'].data.net_normal_forces_w.torch.sum(1)
   samples.append([float((support_vec+seat).norm()),float(mdp.support_fraction(raw)),float(raw.scene['all_contacts'].data.net_normal_forces_w.torch[...,2].sum())])
 samples=np.array(samples);reaction_error=float(samples[:,0].max());assert reaction_error<2.,reaction_error;assert samples[:,1].mean()>.25
 # Positive control is intentionally unsafe and is never counted as a success trial.
 raw.reset();seat_pose=raw.scene['seat'].data.root_pose_w.torch.clone();seat_pose[:,0]=raw.scene.env_origins[:,0];raw.scene['seat'].write_root_pose_to_sim_index(root_pose=seat_pose)
 knee_peak=0.;invalid=False
 for _ in range(25):
  raw.step(torch.zeros(1,37,device=a.device));force=sum(raw.scene[f'{s}_knee_seat'].data.normal_force_matrix_w.torch.sum((1,2)).norm(dim=-1) for s in ('left','right'));knee_peak=max(knee_peak,float(force.max()));invalid|=bool(full_transition.unintended_contact(raw).any())
 assert knee_peak>10 and invalid,(knee_peak,invalid)
 text=Path(a.warnings_log).read_text();paths=set(re.findall("Failed to find rigid body at '([^']+)'",text));actual_missing=[]
 for path in paths:
  prim=stage.GetPrimAtPath(path)
  if prim and prim.HasAPI(UsdPhysics.RigidBodyAPI):actual_missing.append(path)
 assert not actual_missing,actual_missing
 result=dict(expected_rigid_bodies=len(expected),all_expected_usd_rigid_APIs_present=True,actual_sensor_bindings=bindings,warning_unique_paths=len(paths),warnings_on_actual_rigid_body_prims=actual_missing,source='IsaacLab physx_manager.py create_rigid_body_view("/**") includes non-rigid prims',seated_support_mean_bodyweight=float(samples[:,1].mean()),seat_vs_filtered_robot_reaction_max_error_N=reaction_error,mean_total_robot_normal_vertical_force_N=float(samples[:,2].mean()),bodyweight_N=float(robot.data.body_mass.torch.sum()*9.81),deliberate_invalid_knee_contact_peak_N=knee_peak,safety_sensor_correctly_rejected_positive_control=invalid)
 (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print('CONTACT_AUDIT',result,flush=True);env.close()
