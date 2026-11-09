"""Whole-trial evaluation: no automatic resets, 5 ms force peaks, multiple seeded starts."""
import argparse
import itertools
import json
from pathlib import Path
import sys
import gymnasium as gym
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab_visualizers.kit import KitVisualizerCfg
from isaaclab_rl.rsl_rl import (RslRlVecEnvWrapper,create_rsl_rl_runner,
    check_rsl_rl_version,handle_deprecated_rsl_rl_cfg)
from humanoid_sitting.tasks import full_transition as task, mdp
from humanoid_sitting.tasks.curriculum import SeatedStabilityEnvCfg
import humanoid_sitting.tasks
from humanoid_sitting.tasks import furniture, geometry_reference

parser=argparse.ArgumentParser()
parser.add_argument('--checkpoint')
parser.add_argument('--legacy-reference',action='store_true')
parser.add_argument('--geometry-stage',type=int,default=0)
parser.add_argument('--geometry-seed',type=int,default=9310)
parser.add_argument('--geometry-count',type=int,default=12)
parser.add_argument('--families',action='store_true')
parser.add_argument('--arm-clearance',action='store_true')
parser.add_argument('--clearance-pitch',type=float,default=.8)
parser.add_argument('--video-geometry-index',type=int,default=0)
parser.add_argument('--reference-mode',choices=['normal','frozen_standing','phase_lag','gentle_recovery','slower_transition','canonical_geometry'],default='normal')
parser.add_argument('--blind-geometry',action='store_true')
parser.add_argument('--seated-baseline',action='store_true')
parser.add_argument('--seated-initialization',choices=['authored','settled'],default='settled')
parser.add_argument('--seeds',nargs='+',type=int,default=[7,42,123])
parser.add_argument('--seconds',type=float,default=20.)
parser.add_argument('--video',action='store_true')
parser.add_argument('--output',required=True)
add_launcher_args(parser)
args=parser.parse_args();sys.argv=[sys.argv[0]]
out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
(out/'command.json').write_text(json.dumps(vars(args),indent=2)+'\n')
import hashlib, shutil
source=Path(geometry_reference.__file__)
shutil.copy2(source,out/'generator_source.py')
(out/'source_sha256.txt').write_text(hashlib.sha256(source.read_bytes()).hexdigest()+'\n')
configs=list(itertools.product([-.015,0.,.015],[-.035,-.02,-.005],[.715,.725]))
trials=[]
# Every geometry is crossed with every initialization seed and three standing poses.
for gi in range(1 if args.video else args.geometry_count):
    for seed in args.seeds[:1] if args.video else args.seeds:
        rng=np.random.default_rng(seed+gi*10000)
        for ci,(x,pitch,z) in enumerate([(0.,-.02,.72)] if args.video else [(-.015,-.035,.715),(0.,-.02,.72),(.015,-.005,.725)]):
            trials.append(dict(seed=seed,config=ci,geometry_index=gi,x=x+float(rng.uniform(-.003,.003)),
                y=float(rng.uniform(-.008,.008)),z=z,pitch=pitch+float(rng.uniform(-.005,.005)),
                roll=float(rng.uniform(-.005,.005)),yaw=float(rng.uniform(-.005,.005))))
            if args.video:trials[-1].update(x=0.,y=0.,z=.72,pitch=-.02,roll=0.,yaw=0.)
cfg=(furniture.FurnitureEnvCfg if args.legacy_reference else geometry_reference.GeometryEnvCfg)(geometry_stage=args.geometry_stage,geometry_seed=args.geometry_seed,geometry_count=1 if args.video else args.geometry_count,include_families=args.families,reference_mode=args.reference_mode,arm_clearance=args.arm_clearance,clearance_shoulder_pitch=args.clearance_pitch)
if args.video:
    cfg.geometry_bank=[furniture.bank(args.geometry_stage,args.geometry_seed,args.geometry_count,args.families)[args.video_geometry_index]]
    original=cfg.scene.seat.spawn.assets_cfg[0]
    g=cfg.geometry_bank[0];original.size=(g['depth'],g['width'],.1);original.family=g['family'];original.top_height=g['height']
cfg.scene.num_envs=len(trials);cfg.sim.device=args.device
cfg.seed=args.seeds[0];cfg.episode_length_s=args.seconds+10
cfg.scene.seat.spawn.activate_contact_sensors=True
cfg.scene.seat_contacts=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Seat',history_length=4)
cfg.scene.lazy_sensor_update=False
for name in (*mdp.SUPPORT_SENSORS,'left_foot_seat','right_foot_seat','left_knee_seat','right_knee_seat','all_contacts'):
    getattr(cfg.scene,name).history_length=4
# Evaluate one complete attempt per environment. Failures are latched, never reset away.
for name in ('time_out','fallen','pelvis_floor','foot_seat','seat_impact'):
    if hasattr(cfg.terminations,name):setattr(cfg.terminations,name,None)
if args.video:
    args.visualizer=['kit'];args.enable_cameras=True
    cfg.sim.default_visualizer_cfg=KitVisualizerCfg(eye=(1.8,1.8,1.5),lookat=(-.15,0.,.70),origin_type='env',origin_env_index=0)
    cfg.video_recorders=[VideoRecorderCfg(source='visualizer:kit',output_dir=str(out),
        video_length=int(args.seconds/.02),video_interval=100000,
        output_filename_prefix='transition' if not args.seated_baseline else 'seated_baseline',frame_stride=2)]
agent=handle_deprecated_rsl_rl_cfg(task.FullTransitionPPORunnerCfg(),check_rsl_rl_version())
agent.device=args.device
with launch_simulation(cfg,args):
    env=RslRlVecEnvWrapper(gym.make('Humanoid-Sit-Rigid-Furniture-G1',cfg=cfg))
    raw=env.unwrapped;robot=raw.scene['robot'];raw.reset()
    root=robot.data.default_root_pose.torch.clone();q=robot.data.default_joint_pos.torch.clone()
    settled=None
    if args.seated_baseline and args.seated_initialization=='settled':
        settled=np.load('../runs/curriculum_0_policy_eval_final/rollouts.npz')
        settled_root=settled['root_pose'][-1,0].copy()
        settled_q=settled['joint_pos'][-1,0].copy()
    for i,p in enumerate(trials):
        rng=np.random.default_rng(p['seed']+p['config']*1000)
        if args.seated_baseline:
            if settled is not None:
                position=settled_root[:3]+np.array([p['x'],p['y'],0.])
                orientation=(Rotation.from_euler('xyz',[p['roll'],p['pitch']+.02,p['yaw']])*Rotation.from_quat(settled_root[3:])).as_quat()
                root[i,:3]=raw.scene.env_origins[i]+torch.tensor(position,device=raw.device)
                root[i,3:]=torch.tensor(orientation,device=raw.device)
                q[i]=torch.tensor(settled_q,device=raw.device)
            else:
                root[i,:3]=raw.scene.env_origins[i]+torch.tensor([-.3769+p['x'],p['y'],.453],device=raw.device)
                root[i,3:]=torch.tensor(Rotation.from_euler('xyz',[p['roll'],p['pitch']+.02,p['yaw']]).as_quat(),device=raw.device)
        else:
            root[i,:3]=raw.scene.env_origins[i]+torch.tensor([p['x'],p['y'],p['z']],device=raw.device)
            root[i,3:]=torch.tensor(Rotation.from_euler('xyz',[p['roll'],p['pitch'],p['yaw']]).as_quat(),device=raw.device)
        if not args.video:q[i]+=torch.tensor(rng.uniform(-.01,.01,robot.num_joints),device=raw.device,dtype=q.dtype)
    limits=robot.data.joint_pos_limits.torch
    q=torch.maximum(torch.minimum(q,limits[...,1]),limits[...,0])
    robot.write_root_pose_to_sim_index(root_pose=root)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((raw.num_envs,6),device=raw.device))
    robot.write_joint_position_to_sim_index(position=q)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q));robot.reset()
    policy=None
    if args.checkpoint:
        runner=create_rsl_rl_runner(env,agent)
        loaded=torch.load(args.checkpoint,weights_only=False,map_location=raw.device)
        for group in ('actor_state_dict','critic_state_dict'):
            if loaded[group]['mlp.0.weight'].shape[1]==127:
                loaded[group]['mlp.0.weight']=torch.nn.functional.pad(loaded[group]['mlp.0.weight'],(0,2))
        runner.alg.load(loaded,{'actor':True,'critic':True},True)
        policy=runner.get_inference_policy(device=raw.device)
    assert furniture.indices(raw).cpu().tolist()==[p['geometry_index'] for p in trials]
    from pxr import UsdGeom,Usd,Gf
    from isaaclab.sim import get_current_stage
    stage=get_current_stage();audit=[]
    for gi in range(len(cfg.geometry_bank)):
        i=next(i for i,p in enumerate(trials) if p['geometry_index']==gi)
        path=f'/World/envs/env_{i}/Seat/geometry/mesh';prim=stage.GetPrimAtPath(path)
        cube=UsdGeom.Cube(prim);scale=Gf.Transform(UsdGeom.Xformable(prim).GetLocalTransformation()).GetScale()
        actual=[float(cube.GetSizeAttr().Get()*scale[j]) for j in range(3)]
        expected=[cfg.geometry_bank[gi]['depth'],cfg.geometry_bank[gi]['width'],.1]
        assert np.allclose(actual,expected,atol=1e-6),(path,actual,expected)
        seat_pose=raw.scene['seat'].data.root_pose_w.torch[i].cpu().numpy();position=seat_pose[:3]-raw.scene.env_origins[i].cpu().numpy()
        assert np.allclose(position,[cfg.geometry_bank[gi]['x'],cfg.geometry_bank[gi]['y'],cfg.geometry_bank[gi]['height']-.05],atol=2e-5)
        assert abs(Rotation.from_quat(seat_pose[3:]).as_euler('xyz')[2]-cfg.geometry_bank[gi]['yaw'])<2e-5
        audit.append(dict(geometry_index=gi,prim_path=path,actual_seat_dimensions_m=actual,actual_seat_pose=seat_pose.tolist()))
    (out/'geometry_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    np.savez_compressed(out/'initial_conditions.npz',root_pose=root.cpu().numpy(),joint_pos=q.cpu().numpy())
    (out/'geometry_bank.json').write_text(json.dumps(cfg.geometry_bank,indent=2)+'\n')
    obs=env.get_observations()
    zeros=torch.zeros(raw.num_envs,device=raw.device)
    hold=zeros.clone();peak_hold=zeros.clone();first_hold=torch.full_like(hold,float('nan'))
    fell=torch.zeros_like(hold,dtype=torch.bool);wrong=fell.clone();descent_bad=fell.clone();impact_bad=fell.clone();floor_bad=fell.clone()
    max_force=zeros.clone();max_wrong=zeros.clone();max_descent=zeros.clone();standing_ok=fell.clone();torque_bad=fell.clone()
    weight=robot.data.body_mass.torch.sum(dim=1)*9.81
    frames={k:[] for k in ('root_pose','joint_pos','joint_vel','lin_vel','ang_vel','gravity','support','body_forces','foot_seat','knee_seat','actions','seat_force_peak','unintended_peak','hold','body_pose','targets','joint_torque')}
    for step in range(int(args.seconds/raw.step_dt)):
        with torch.inference_mode():
            if args.blind_geometry:
                obs=obs.clone(); obs[:, 120:123]=raw.scene.env_origins+torch.tensor([-.4,0.,.25],device=raw.device)-robot.data.root_pos_w.torch; obs[:, 123:126]=torch.tensor([.5,.9,.1],device=raw.device);obs[:,127:129]=0.
            actions=policy(obs) if policy else torch.zeros((raw.num_envs,env.num_actions),device=raw.device)
            assert torch.isfinite(actions).all()
            obs,reward,done,info=env.step(actions)
            assert not done.any(), 'Automatic resets must remain disabled for whole-trial evaluation.'
            pose=robot.data.root_pose_w.torch.clone();pose[:,:3]-=raw.scene.env_origins
            g=robot.data.projected_gravity_b.torch;lv=robot.data.root_lin_vel_w.torch;av=robot.data.root_ang_vel_w.torch
            support=mdp.support_fraction(raw);delta=robot.data.root_pos_w.torch-raw.scene['seat'].data.root_pos_w.torch
            force=task.seat_peak_force(raw);unexpected=task.unintended_peak_force(raw)
            fell|=(pose[:,2]<.28)|(g[:,2]>-.65)
            wrong|=unexpected>10.;descent_bad|=lv[:,2]<-.8;impact_bad|=force>2.*weight
            pelvis=raw.scene['pelvis_seat'].data
            floor=(pelvis.net_normal_forces_w.torch[:,0]-pelvis.normal_force_matrix_w.torch[:,0].sum(dim=1)).norm(dim=1)>30.
            floor_bad|=floor
            torque_bad|=(robot.data.applied_torque.torch.abs()>robot.data.joint_effort_limits.torch+1.e-3).any(dim=1)
            max_force=torch.maximum(max_force,force);max_wrong=torch.maximum(max_wrong,unexpected);max_descent=torch.maximum(max_descent,-lv[:,2])
            valid=(support>.25)&(delta[:,0].abs()<.22)&(delta[:,1].abs()<.40)&(delta[:,2]>.17)&(delta[:,2]<.30)&(lv.norm(dim=1)<.20)&(av.norm(dim=1)<.35)&(g[:,2]<-.90)
            # Measure physical seated duration independently; whole-trial failure flags still veto success.
            valid&=(unexpected<=10.)&(force<=2.*weight)&(lv[:,2]>=-.8)&~floor
            hold=torch.where(valid,hold+raw.step_dt,zeros);peak_hold=torch.maximum(peak_hold,hold)
            first_hold=torch.where((hold>=3)&torch.isnan(first_hold),torch.full_like(hold,(step+1)*raw.step_dt),first_hold)
            if step==75:
                names=raw.scene['all_contacts'].body_names
                ids=[names.index(f'{s}_ankle_roll_link') for s in ('left','right')]
                feet=raw.scene['all_contacts'].data.net_normal_forces_w.torch[:,ids,2]
                standing_ok=(pose[:,2]>.68)&(g[:,2]<-.9)&(feet.min(dim=1).values>10.)
            values=dict(joint_torque=robot.data.applied_torque.torch,root_pose=pose,joint_pos=robot.data.joint_pos.torch,joint_vel=robot.data.joint_vel.torch,lin_vel=lv,ang_vel=av,gravity=g,support=support,
                body_forces=raw.scene['all_contacts'].data.net_normal_forces_w.torch,
                foot_seat=torch.stack([raw.scene[f'{s}_foot_seat'].data.normal_force_matrix_w.torch.sum(dim=(1,2)) for s in ('left','right')],dim=1),
                knee_seat=torch.stack([raw.scene[f'{s}_knee_seat'].data.normal_force_matrix_w.torch.sum(dim=(1,2)) for s in ('left','right')],dim=1),
                actions=actions,seat_force_peak=force,unintended_peak=unexpected,hold=hold,
                body_pose=robot.data.body_pose_w.torch-torch.cat((raw.scene.env_origins[:,None,:],torch.zeros((raw.num_envs,1,4),device=raw.device)),dim=-1),
                targets=raw.action_manager.get_term('joints').processed_actions)
            for name,value in values.items():
                assert torch.isfinite(value).all(),(step,name)
                frames[name].append(value.cpu().numpy().copy())
    success=(peak_hold>=3)&~(fell|wrong|descent_bad|impact_bad|floor_bad|torque_bad)
    if not args.seated_baseline:success&=standing_ok
    outcomes=[]
    for i,p in enumerate(trials):
        outcomes.append(p|dict(geometry_index=int(furniture.indices(raw)[i]),success=bool(success[i]),fall=bool(fell[i]),unintended_contact=bool(wrong[i]),
            joint_torque_violation=bool(torque_bad[i]),excessive_descent=bool(descent_bad[i]),excessive_impact=bool(impact_bad[i]),pelvis_floor=bool(floor_bad[i]),
            initial_standing_verified=bool(standing_ok[i]),max_seat_impact_N=float(max_force[i]),
            max_unintended_contact_N=float(max_wrong[i]),max_descent_m_s=float(max_descent[i]),
            seated_hold_s=float(peak_hold[i]),time_to_3s_hold_s=float(first_hold[i]) if torch.isfinite(first_hold[i]) else None))
    result=dict(task='seated_baseline' if args.seated_baseline else 'full_transition',checkpoint=args.checkpoint,
        initialization_seeds=args.seeds,num_trials=raw.num_envs,seconds=args.seconds,
        geometry_stage=args.geometry_stage,geometry_seed=args.geometry_seed,geometry_bank=cfg.geometry_bank,
        geometry_assignment=furniture.indices(raw).cpu().tolist(),reference_generator='legacy' if args.legacy_reference else 'geometry_conditioned',reference_mode=args.reference_mode,blind_geometry=args.blind_geometry,arm_clearance=args.arm_clearance,arm_clearance_strategy='flexed_elbows_initial_and_reference' if args.arm_clearance else 'none',elbow_clearance_target_rad=2.2 if args.arm_clearance else None,shoulder_clearance_target_rad=1.6 if args.arm_clearance else None,shoulder_pitch_clearance_target_rad=args.clearance_pitch if args.arm_clearance else None,
        initialization_kind=('validated_settled_seated_state' if settled is not None else 'authored_seated_state') if args.seated_baseline else 'standing_configurations',
        seated_state_source='../runs/curriculum_0_policy_eval_final/rollouts.npz' if settled is not None else None,
        physical_hold_duration_excludes_current_invalid_contact_but_trial_success_vetoes_any_past_failure=True,
        success_rate=float(success.float().mean()),fall_rate=float(fell.float().mean()),joint_torque_violation_rate=float(torque_bad.float().mean()),
        unintended_contact_rate=float(wrong.float().mean()),excessive_impact_rate=float(impact_bad.float().mean()),
        maximum_seat_impact_N=float(max_force.max()),minimum_seated_hold_s=float(peak_hold.min()),
        median_seated_hold_s=float(peak_hold.median()),maximum_seated_hold_s=float(peak_hold.max()),
        seat_impact_limit_N=float((2*weight[0])),maximum_descent_m_s=float(max_descent.max()),
        contact_sampling_dt_s=cfg.sim.dt,control_dt_s=raw.step_dt,automatic_resets=False,
        root_constraints=False,actuators_only=True,gravity=list(cfg.sim.gravity),outcomes=outcomes,
        body_names=raw.scene['all_contacts'].body_names,articulation_body_names=robot.body_names,joint_names=robot.joint_names,
        ankle_effort_limits_Nm={n:float(robot.data.joint_effort_limits.torch[0,j]) for j,n in enumerate(robot.joint_names) if 'ankle_' in n},
        maximum_observed_absolute_ankle_effort_Nm=float(np.abs(np.stack(frames['joint_torque']))[:,:,[j for j,n in enumerate(robot.joint_names) if 'ankle_' in n]].max()),
        maximum_observed_absolute_joint_torque_Nm=float(np.abs(np.stack(frames['joint_torque'])).max()),
        joint_effort_limits_Nm=robot.data.joint_effort_limits.torch[0].cpu().tolist(),
        standing_ankle_gains=dict(stiffness=cfg.scene.robot.actuators['feet'].stiffness,damping=cfg.scene.robot.actuators['feet'].damping),
        success_definition='Initially standing (full transition only), no fall or pelvis-floor contact, unintended seat/arm contact <=10 N at every 5 ms sample, downward speed <=0.8 m/s, total seat force <=2 body weights, and >=3 s continuous original seated contact/posture/stability conditions.')
    (out/'command.json').write_text(json.dumps(vars(args),indent=2)+'\n')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/'rollouts.npz',**{k:np.stack(v) for k,v in frames.items()},dt=raw.step_dt)
    print('FULL_TRANSITION_EVALUATION',json.dumps({k:v for k,v in result.items() if k not in ('outcomes','body_names','joint_names','articulation_body_names')}),flush=True)
    env.close()
