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
from humanoid_sitting.tasks import furniture, geometry_reference, geometry_feedback, balanced_reference, geometry_guarded, geometry_guarded_contact, geometry_guarded_contact_v10, geometry_contact_v11, geometry_contact_v12

parser=argparse.ArgumentParser()
parser.add_argument('--checkpoint')
parser.add_argument('--student-checkpoint')
parser.add_argument('--deployment-checkpoint')
parser.add_argument('--contact-estimator')
parser.add_argument('--disable-contact-feedback',action='store_true')
parser.add_argument('--soft-landing-mode',choices=['phase','actions'],default='phase')
parser.add_argument('--soft-landing-parameters',help='Experimental episode-level stochastic phase parameters; training/evaluation provenance required')
parser.add_argument('--soft-landing-rate',type=float,default=.6)
parser.add_argument('--deployment-perception',help='Folder containing camera estimates.json and fixture manifest')
parser.add_argument('--sensor-delay-steps',type=int,default=1)
parser.add_argument('--seat-position-error',type=float,default=0.)
parser.add_argument('--seat-height-error',type=float,default=0.)
parser.add_argument('--deployment-action-rate',type=float,default=.01)
parser.add_argument('--reset-deployment-memory',action='store_true')
parser.add_argument('--freeze-deployment-feedback',action='store_true',help='Ablation: hold encoder/torque/IMU samples at their initial values')
parser.add_argument('--canonical-deployment-geometry',action='store_true',help='Ablation: replace measured geometry with the training-set mean')
parser.add_argument('--record-deployment-inputs',action='store_true')
parser.add_argument('--deployment-teacher-mix',type=float,default=0.,help='Training collection only: teacher action fraction; never a teacher-independent benchmark')
parser.add_argument('--remove-teacher-reference',action='store_true')
parser.add_argument('--student-ablation',choices=['none','no_clock','blind_geometry','no_balance'],default='none')
parser.add_argument('--record-demonstrations',action='store_true')
parser.add_argument('--record-friction',action='store_true')
parser.add_argument('--dagger-teacher-labels',action='store_true')
parser.add_argument('--compact-recording',action='store_true')
parser.add_argument('--summary-only',action='store_true')
parser.add_argument('--legacy-reference',action='store_true')
parser.add_argument('--feedback-reference',action='store_true')
parser.add_argument('--balanced-reference',action='store_true')
parser.add_argument('--guarded-reference',action='store_true')
parser.add_argument('--contact-reference',action='store_true')
parser.add_argument('--v10-reference',action='store_true')
parser.add_argument('--v11-reference',action='store_true')
parser.add_argument('--v12-reference',action='store_true')
parser.add_argument('--v5-reference',action='store_true')
parser.add_argument('--production-task',action='store_true')
parser.add_argument('--geometry-stage',type=int,default=2)
parser.add_argument('--geometry-seed',type=int,default=9310)
parser.add_argument('--geometry-count',type=int,default=16)
parser.add_argument('--families',action=argparse.BooleanOptionalAction,default=True)
parser.add_argument('--novel-shapes',action='store_true',help='Held-out elliptical/chamfered rigid seat silhouettes; controller unchanged')
parser.add_argument('--arm-clearance',action=argparse.BooleanOptionalAction,default=True)
parser.add_argument('--clearance-pitch',type=float,default=.8)
parser.add_argument('--video-geometry-index',type=int,default=0)
parser.add_argument('--video-benchmark-config',type=int,choices=[0,1,2],help='Record an exact seeded benchmark start instead of the nominal video start')
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
if not 0 <= args.deployment_teacher_mix <= 1:parser.error('Teacher mix must be in [0,1].')
if args.deployment_teacher_mix and not (args.deployment_checkpoint and args.dagger_teacher_labels and args.record_deployment_inputs):parser.error('Teacher mixing requires explicit deployment data collection and teacher labels.')
selection=[name for name,enabled in [
    ('ik-v5',args.v5_reference),('legacy',args.legacy_reference),('feedback-v6',args.feedback_reference),
    ('balanced-v7',args.balanced_reference),('guarded-v8',args.guarded_reference),
    ('contact-v9',args.contact_reference),('contact-v10',args.v10_reference),
    ('contact-v11',args.v11_reference),('contact-v12',args.v12_reference)] if enabled]
if len(selection)>1:parser.error('Select only one reference generator.')
args.generator=selection[0] if selection else 'contact-v12'
if args.generator!='legacy' and args.reference_mode not in ('normal','canonical_geometry'):parser.error('Geometry references support normal and canonical_geometry modes only.')
generators={
    'legacy':(furniture.FurnitureEnvCfg,furniture),
    'ik-v5':(geometry_reference.GeometryEnvCfg,geometry_reference),
    'feedback-v6':(geometry_feedback.GeometryEnvCfg,geometry_feedback),
    'balanced-v7':(balanced_reference.BalancedEnvCfg,balanced_reference),
    'guarded-v8':(geometry_guarded.GeometryEnvCfg,geometry_guarded),
    'contact-v9':(geometry_guarded_contact.GeometryEnvCfg,geometry_guarded_contact),
    'contact-v10':(geometry_guarded_contact_v10.GeometryEnvCfg,geometry_guarded_contact_v10),
    'contact-v11':(geometry_contact_v11.GeometryEnvCfg,geometry_contact_v11),
    'contact-v12':(geometry_contact_v12.GeometryEnvCfg,geometry_contact_v12),
}
cfg_class,reference_module=generators[args.generator]
if args.production_task:
    if args.generator!='contact-v12':parser.error('The registered production task uses contact-v12.')
    from humanoid_sitting.tasks.geometry_curriculum import GeometryV12TrainingEnvCfg
    cfg_class=GeometryV12TrainingEnvCfg
if args.student_checkpoint:
    if args.checkpoint:parser.error('Student and residual checkpoints are mutually exclusive.')
    from humanoid_sitting.tasks.student import StudentEnvCfg
    cfg_class=StudentEnvCfg
if args.deployment_checkpoint:
    if args.student_checkpoint or args.checkpoint:parser.error('Only one controller checkpoint is allowed.')
    if not args.deployment_perception:parser.error('Deployment policy requires measured perception estimates.')
    from humanoid_sitting.tasks.student import DeploymentEnvCfg
    cfg_class=DeploymentEnvCfg
import shutil
if shutil.disk_usage(Path(args.output).resolve().parent).free < 30*2**30:
    raise RuntimeError('Recording forbidden below the 30 GiB free-space floor')
out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
(out/'command.json').write_text(json.dumps(vars(args),indent=2)+'\n')
import hashlib,shutil
source=Path(reference_module.__file__)
shutil.copy2(source,out/'generator_source.py')
(out/'source_sha256.txt').write_text(hashlib.sha256(source.read_bytes()).hexdigest()+'\n')
configs=list(itertools.product([-.015,0.,.015],[-.035,-.02,-.005],[.715,.725]))
trials=[]
# Every geometry is crossed with every initialization seed and three standing poses.
for gi in range(1 if args.video else args.geometry_count):
    for seed in args.seeds[:1] if args.video else args.seeds:
        rng=np.random.default_rng(seed+(args.video_geometry_index if args.video and args.video_benchmark_config is not None else gi)*10000)
        starts=[(-.015,-.035,.715),(0.,-.02,.72),(.015,-.005,.725)]
        if args.video and args.video_benchmark_config is not None:
            # Advance pose jitter for preceding configurations exactly as in the bank.
            for _ in range(args.video_benchmark_config):rng.uniform(-.003,.003);rng.uniform(-.008,.008);rng.uniform(-.005,.005,size=3)
        selected_starts=[starts[args.video_benchmark_config]] if args.video and args.video_benchmark_config is not None else ([(0.,-.02,.72)] if args.video else starts)
        for ci,(x,pitch,z) in enumerate(selected_starts):
            if args.video and args.video_benchmark_config is not None:ci=args.video_benchmark_config
            trials.append(dict(seed=seed,config=ci,geometry_index=gi,x=x+float(rng.uniform(-.003,.003)),
                y=float(rng.uniform(-.008,.008)),z=z,pitch=pitch+float(rng.uniform(-.005,.005)),
                roll=float(rng.uniform(-.005,.005)),yaw=float(rng.uniform(-.005,.005))))
            if args.video and args.video_benchmark_config is None:trials[-1].update(x=0.,y=0.,z=.72,pitch=-.02,roll=0.,yaw=0.)
cfg=cfg_class(geometry_stage=args.geometry_stage,geometry_seed=args.geometry_seed,geometry_count=1 if args.video else args.geometry_count,include_families=args.families,reference_mode=args.reference_mode,arm_clearance=args.arm_clearance,clearance_shoulder_pitch=args.clearance_pitch)
if args.novel_shapes:
    for i,g in enumerate(cfg.geometry_bank):
        g['shape']=('ellipse','chamfered')[i%2];cfg.scene.seat.spawn.assets_cfg[i].shape=g['shape']
if args.video:
    cfg.geometry_bank=[furniture.bank(args.geometry_stage,args.geometry_seed,args.geometry_count,args.families)[args.video_geometry_index]]
    original=cfg.scene.seat.spawn.assets_cfg[0]
    g=cfg.geometry_bank[0];original.size=(g['depth'],g['width'],.1);original.family=g['family'];original.top_height=g['height']
if args.remove_teacher_reference:
    if not args.checkpoint or args.student_checkpoint:parser.error('Reference removal is for the old residual actor only.')
    from humanoid_sitting.tasks.student import FrozenDefaultResidualActionCfg
    cfg.actions.joints=FrozenDefaultResidualActionCfg(asset_name='robot',joint_names=['.*'])
cfg.scene.num_envs=len(trials);cfg.sim.device=args.device
cfg.seed=args.seeds[0];cfg.episode_length_s=args.seconds+10
cfg.scene.seat.spawn.activate_contact_sensors=True
cfg.scene.seat_contacts=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Seat',history_length=4)
cfg.scene.lazy_sensor_update=False
for name in (*mdp.SUPPORT_SENSORS,'left_foot_seat','right_foot_seat','left_knee_seat','right_knee_seat','all_contacts'):
    getattr(cfg.scene,name).history_length=4
if args.dagger_teacher_labels and not ((args.student_checkpoint and args.record_demonstrations) or (args.deployment_checkpoint and args.record_deployment_inputs)):parser.error('DAgger requires student execution and explicit demonstration recording.')
if args.summary_only and args.record_demonstrations:parser.error('Summary-only cannot discard demonstration fields.')
if args.record_friction:
    if not args.record_demonstrations:parser.error('Friction collection requires --record-demonstrations.')
    for name in (*mdp.SUPPORT_SENSORS,'left_foot_seat','right_foot_seat','left_knee_seat','right_knee_seat'):
        getattr(cfg.scene,name).track_friction_forces=True
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
    env=RslRlVecEnvWrapper(gym.make('Humanoid-Sit-Learned-G1' if args.student_checkpoint or args.deployment_checkpoint else ('Humanoid-Sit-Geometry-Conditioned-G1' if args.production_task else 'Humanoid-Sit-Rigid-Furniture-G1'),cfg=cfg))
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
        if not args.video or args.video_benchmark_config is not None:q[i]+=torch.tensor(rng.uniform(-.01,.01,robot.num_joints),device=raw.device,dtype=q.dtype)
    limits=robot.data.joint_pos_limits.torch
    q=torch.maximum(torch.minimum(q,limits[...,1]),limits[...,0])
    robot.write_root_pose_to_sim_index(root_pose=root)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((raw.num_envs,6),device=raw.device))
    robot.write_joint_position_to_sim_index(position=q)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q));robot.reset()
    shadow_teacher=None
    if args.dagger_teacher_labels:
        shadow_teacher=geometry_contact_v12.GeometryAction(geometry_contact_v12.GeometryActionCfg(asset_name='robot',joint_names=['.*']),raw)
    deployment_policy=None;deployment_state=None;deployment_sensor=None
    first_deployment_feedback=None
    if args.deployment_checkpoint:
        from humanoid_sitting.deployment import DeploymentStudent
        from humanoid_sitting.sim_deployment_sensor import SimSensorInterface
        deployment_policy,deployment_metadata=DeploymentStudent.load(args.deployment_checkpoint,raw.device)
        soft_controller=None
        if args.contact_estimator:
            from humanoid_sitting.soft_landing import ContactEstimator,SoftLandingController
            parameters=None
            if args.soft_landing_parameters:
                parameters=torch.tensor(json.loads(Path(args.soft_landing_parameters).read_text())['parameters'],device=raw.device)
                if len(parameters)==1:parameters=parameters.expand(raw.num_envs,-1)
                assert parameters.shape==(raw.num_envs,2) and torch.isfinite(parameters).all()
            soft_controller=SoftLandingController(deployment_policy,ContactEstimator.load(args.contact_estimator,raw.device),args.soft_landing_rate,parameters,args.soft_landing_mode,args.disable_contact_feedback)
            (out/'soft_landing_interface.json').write_text(json.dumps(dict(mode=args.soft_landing_mode,estimator=args.contact_estimator,privileged_contact_inputs=False,teacher_action_inputs=False,actor_sensor_packet_channels=109,foot_feedback_channels=0,controller_has_authored_governor=True,fully_learned_controller=False,estimated_contact_threshold=.65,first_contact_training_label_threshold_mg=.05,physical_safety_limits_unchanged=True),indent=2)+'\n')
            import inspect
            (out/'soft_landing_source.py').write_text(inspect.getsource(__import__('humanoid_sitting.soft_landing',fromlist=['SoftLandingController'])))
        term=raw.action_manager.get_term('joints')
        assert term.__class__.__name__=='StudentJointAction'
        assert not any(hasattr(term,key) for key in ('kinematics','deep','seated','contact_time'))
        (out/'teacher_absence_verification.json').write_text(json.dumps(dict(action_class=term.__class__.__name__,teacher_action_fraction=args.deployment_teacher_mix,no_teacher_apply_actions=True,actor_geometry_source='depth-only estimates',actor_inputs=109,effective_neural_channels=107 if deployment_policy.mask_perception_status else 109,status_channels_used_for_motion=not deployment_policy.mask_perception_status,network_variant=deployment_policy.variant,implicit_recurrent_timing=True),indent=2)+'\n')
        camera_dir=Path(args.deployment_perception);camera_manifest=json.loads((camera_dir/'manifest.json').read_text())
        camera_fixture=Path(camera_manifest['fixture']);camera_results=json.loads((camera_fixture/'results.json').read_text())
        estimates=json.loads((camera_dir/'estimates.json').read_text())
        lookup={(o['geometry_index'],o['seed'],o['config']):e for o,e in zip(camera_results['outcomes'],estimates)}
        selected=[lookup[(args.video_geometry_index if args.video and args.video_benchmark_config is not None else o['geometry_index'],o['seed'],o['config'])] for o in trials]
        deployment_sensor=SimSensorInterface(robot,selected,raw.device,args.sensor_delay_steps,args.seeds[0]+901,args.seat_position_error,args.seat_height_error)
        previous_deployment_target=q.clone()
        (out/'deployment_interface.json').write_text(json.dumps(dict(actor_inputs=109,foot_feedback_channels=0,measured_geometry_source=str(camera_dir),ground_truth_geometry=False,exact_contacts=False,COM_or_root_velocity=False,explicit_clock=False,scripted_balance=False,action_rate_limit_rad_per_control_step=args.deployment_action_rate,delay_ms=args.sensor_delay_steps*20,actor_tensor_shape=[raw.num_envs,109],controller='GRU absolute body joint targets; fixed hand targets'),indent=2)+'\n')
    policy=None
    student_policy=None
    from humanoid_sitting.tasks.student import student_observation
    if args.student_checkpoint:
        term=raw.action_manager.get_term('joints')
        assert term.__class__.__name__=='StudentJointAction'
        assert not any(hasattr(term,key) for key in ('kinematics','deep','seated','contact_time'))
        (out/'teacher_absence_verification.json').write_text(json.dumps(dict(action_class=term.__class__.__name__,no_teacher_trajectory_or_IK=True,full_joint_targets=True),indent=2)+'\n')
        from humanoid_sitting.student_network import Student
        student_policy,student_metadata=Student.load(args.student_checkpoint,raw.device)
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
        scale=Gf.Transform(UsdGeom.Xformable(prim).GetLocalTransformation()).GetScale()
        if prim.IsA(UsdGeom.Mesh):
            points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get());actual=((points.max(0)-points.min(0))*np.asarray(scale)).tolist()
        else:
            cube=UsdGeom.Cube(prim);actual=[float(cube.GetSizeAttr().Get()*scale[j]) for j in range(3)]
        expected=[cfg.geometry_bank[gi]['depth'],cfg.geometry_bank[gi]['width'],.1]
        assert np.allclose(actual,expected,atol=1e-6),(path,actual,expected)
        seat_pose=raw.scene['seat'].data.root_pose_w.torch[i].cpu().numpy();position=seat_pose[:3]-raw.scene.env_origins[i].cpu().numpy()
        assert np.allclose(position,[cfg.geometry_bank[gi]['x'],cfg.geometry_bank[gi]['y'],cfg.geometry_bank[gi]['height']-.05],atol=2e-5)
        assert abs(Rotation.from_quat(seat_pose[3:]).as_euler('xyz')[2]-cfg.geometry_bank[gi]['yaw'])<2e-5
        audit.append(dict(geometry_index=gi,prim_path=path,actual_seat_dimensions_m=actual,actual_seat_pose=seat_pose.tolist()))
    (out/'geometry_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    np.savez_compressed(out/'initial_conditions.npz',root_pose=root.cpu().numpy(),joint_pos=q.cpu().numpy())
    (out/'geometry_bank.json').write_text(json.dumps(cfg.geometry_bank,indent=2)+'\n')
    # Manual benchmark initialization supersedes the root pose sampled by reset.
    if hasattr(raw,'student_anchor'):
        del raw.student_anchor;del raw.student_first_support
    obs=env.get_observations()
    zeros=torch.zeros(raw.num_envs,device=raw.device)
    hold=zeros.clone();peak_hold=zeros.clone();first_hold=torch.full_like(hold,float('nan'))
    fell=torch.zeros_like(hold,dtype=torch.bool);wrong=fell.clone();descent_bad=fell.clone();impact_bad=fell.clone();floor_bad=fell.clone()
    max_force=zeros.clone();max_wrong=zeros.clone();max_descent=zeros.clone();standing_ok=fell.clone();torque_bad=fell.clone()
    weight=robot.data.body_mass.torch.sum(dim=1)*9.81
    frames={k:[] for k in ('root_pose','joint_pos','joint_vel','lin_vel','ang_vel','gravity','support','body_forces','foot_seat','knee_seat','actions','seat_force_peak','unintended_peak','hold','body_pose','targets','joint_torque','sitting_body_velocity')}
    if args.compact_recording:frames.pop('body_pose')
    actor_obs_name='student_actor_observations' if student_policy else 'teacher_actor_observations'
    if args.record_demonstrations:
        for name in ('student_observations',actor_obs_name,'pre_joint_pos','pre_joint_vel','pre_root_pose','all_contact_history','seat_contact_history','support_contact_history','foot_knee_contact_history'):frames[name]=[]
    if args.dagger_teacher_labels:
        frames['expert_targets']=[];frames['safe_training_state']=[]
    if args.record_deployment_inputs:
        assert deployment_policy is not None
        frames['deployment_inputs']=[];frames['deployment_proposed_targets']=[]
    if args.contact_estimator:
        frames['estimated_contact']=[];frames['phase_rate']=[]
    if args.record_friction:
        frames['support_friction_history']=[];frames['foot_knee_friction_history']=[]
    for step in range(int(args.seconds/raw.step_dt)):
        with torch.inference_mode():
            if args.blind_geometry:
                obs=obs.clone(); obs[:, 120:123]=raw.scene.env_origins+torch.tensor([-.4,0.,.25],device=raw.device)-robot.data.root_pos_w.torch; obs[:, 123:126]=torch.tensor([.5,.9,.1],device=raw.device);obs[:,127:129]=0.
            student_obs=student_observation(raw) if args.record_demonstrations or student_policy else None
            if args.record_demonstrations:
                pre_values=dict(student_observations=student_obs,
                    pre_joint_pos=robot.data.joint_pos.torch,pre_joint_vel=robot.data.joint_vel.torch,
                    pre_root_pose=robot.data.root_pose_w.torch)
                pre_values[actor_obs_name]=obs if isinstance(obs,torch.Tensor) else obs['policy']
                for name,value in pre_values.items():frames[name].append(value.cpu().numpy().copy())
            if deployment_policy:
                deployment_obs=deployment_sensor.read(previous_deployment_target)
                if args.freeze_deployment_feedback:
                    if first_deployment_feedback is None:first_deployment_feedback=deployment_obs[:,:75].clone()
                    deployment_obs[:,:75]=first_deployment_feedback
                if args.canonical_deployment_geometry:deployment_obs[:,98:108]=deployment_policy.mean[98:108]
                if args.reset_deployment_memory:deployment_state=None
                body_targets,deployment_state=(soft_controller or deployment_policy)(deployment_obs[:,None,:],deployment_state)
                if soft_controller:
                    frames['estimated_contact'].append(soft_controller.last_contact.cpu().numpy().copy())
                    frames['phase_rate'].append(soft_controller.last_rate.cpu().numpy().copy())
                actions=previous_deployment_target.clone();old=actions[:,deployment_sensor.ids]
                actions[:,deployment_sensor.ids]=body_targets[:,0].clamp(min=old-args.deployment_action_rate,max=old+args.deployment_action_rate)
                previous_deployment_target=actions.clone()
                if args.record_deployment_inputs:
                    frames['deployment_inputs'].append(deployment_obs.cpu().numpy().copy())
                    frames['deployment_proposed_targets'].append(body_targets[:,0].cpu().numpy().copy())
            elif student_policy:
                student_obs=student_obs.clone()
                if args.student_ablation=='no_clock':student_obs[:,92]=0.;student_obs[:,97]=0.
                if args.student_ablation=='blind_geometry':
                    student_obs[:,83:91]=0.;student_obs[:,85]=.25-student_obs[:,91];student_obs[:,86]=.5;student_obs[:,87]=.9;student_obs[:,93]=-.15;student_obs[:,94]=0.
                if args.student_ablation=='no_balance':student_obs[:,95]=0.;student_obs[:,0]=0.
                actions=student_policy(student_obs)
            else:actions=policy(obs) if policy else torch.zeros((raw.num_envs,env.num_actions),device=raw.device)
            if shadow_teacher is not None:
                clock=raw.transition_time.clone()
                shadow_teacher.process_actions(torch.zeros_like(actions))
                raw.transition_time.copy_(clock)
                # No shadow apply_actions call: the actual action term remains student-only.
                frames['expert_targets'].append(shadow_teacher.processed_actions.cpu().numpy().copy())
                frames['safe_training_state'].append((~(fell|wrong|descent_bad|impact_bad|floor_bad|torque_bad)).cpu().numpy().copy())
                if args.deployment_teacher_mix:
                    actions=(1-args.deployment_teacher_mix)*actions+args.deployment_teacher_mix*shadow_teacher.processed_actions
                    previous_deployment_target=actions.clone()
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
            values=dict(sitting_body_velocity=robot.data.body_lin_vel_w.torch[:,[robot.body_names.index(n) for n in ('pelvis','left_hip_yaw_link','right_hip_yaw_link')]],joint_torque=robot.data.applied_torque.torch,root_pose=pose,joint_pos=robot.data.joint_pos.torch,joint_vel=robot.data.joint_vel.torch,lin_vel=lv,ang_vel=av,gravity=g,support=support,
                body_forces=raw.scene['all_contacts'].data.net_normal_forces_w.torch,
                foot_seat=torch.stack([raw.scene[f'{s}_foot_seat'].data.normal_force_matrix_w.torch.sum(dim=(1,2)) for s in ('left','right')],dim=1),
                knee_seat=torch.stack([raw.scene[f'{s}_knee_seat'].data.normal_force_matrix_w.torch.sum(dim=(1,2)) for s in ('left','right')],dim=1),
                actions=actions,seat_force_peak=force,unintended_peak=unexpected,hold=hold,
                body_pose=robot.data.body_pose_w.torch-torch.cat((raw.scene.env_origins[:,None,:],torch.zeros((raw.num_envs,1,4),device=raw.device)),dim=-1),
                targets=raw.action_manager.get_term('joints').processed_actions)
            if args.compact_recording:values.pop('body_pose')
            if args.record_demonstrations:
                values.update(all_contact_history=raw.scene['all_contacts'].data.net_normal_forces_w_history.torch,
                    seat_contact_history=raw.scene['seat_contacts'].data.net_normal_forces_w_history.torch,
                    support_contact_history=torch.stack([raw.scene[name].data.normal_force_matrix_w_history.torch.sum(dim=(2,3)) for name in mdp.SUPPORT_SENSORS],dim=2),
                    foot_knee_contact_history=torch.stack([raw.scene[f'{side}_{part}_seat'].data.normal_force_matrix_w_history.torch.sum(dim=(2,3)) for side in ('left','right') for part in ('foot','knee')],dim=2))
            if args.record_friction:
                values['support_friction_history']=torch.stack([raw.scene[name].data.friction_force_matrix_w_history.torch.sum(dim=(2,3)) for name in mdp.SUPPORT_SENSORS],dim=2)
                values['foot_knee_friction_history']=torch.stack([raw.scene[f'{side}_{part}_seat'].data.friction_force_matrix_w_history.torch.sum(dim=(2,3)) for side in ('left','right') for part in ('foot','knee')],dim=2)
            for name,value in values.items():
                assert torch.isfinite(value).all(),(step,name)
                frames[name].append(value.cpu().numpy().copy())
    success=(peak_hold>=3)&~(fell|wrong|descent_bad|impact_bad|floor_bad|torque_bad)
    if not args.seated_baseline:success&=standing_ok
    late_linear=np.stack(frames['lin_vel'][-int(3/raw.step_dt):]);late_angular=np.stack(frames['ang_vel'][-int(3/raw.step_dt):]);late_gravity=np.stack(frames['gravity'][-int(3/raw.step_dt):])
    linear_rms=np.sqrt(np.square(late_linear).sum(-1).mean(0));angular_rms=np.sqrt(np.square(late_angular).sum(-1).mean(0));upright_min=(-late_gravity[...,2]).min(0)
    outcomes=[]
    for i,p in enumerate(trials):
        outcomes.append(p|dict(geometry_index=int(furniture.indices(raw)[i]),success=bool(success[i]),fall=bool(fell[i]),unintended_contact=bool(wrong[i]),
            joint_torque_violation=bool(torque_bad[i]),excessive_descent=bool(descent_bad[i]),excessive_impact=bool(impact_bad[i]),pelvis_floor=bool(floor_bad[i]),
            initial_standing_verified=bool(standing_ok[i]),max_seat_impact_N=float(max_force[i]),
            max_unintended_contact_N=float(max_wrong[i]),max_descent_m_s=float(max_descent[i]),
            final_3s_linear_rms_m_s=float(linear_rms[i]),final_3s_angular_rms_rad_s=float(angular_rms[i]),final_3s_minimum_upright_cosine=float(upright_min[i]),end_seated_hold_s=float(hold[i]),
            seated_hold_s=float(peak_hold[i]),time_to_3s_hold_s=float(first_hold[i]) if torch.isfinite(first_hold[i]) else None))
    result=dict(task='seated_baseline' if args.seated_baseline else 'full_transition',checkpoint=args.checkpoint,
        deployment_checkpoint=args.deployment_checkpoint,deployment_perception=args.deployment_perception,
        student_checkpoint=args.student_checkpoint,student_ablation=args.student_ablation,teacher_reference_in_action_path=bool(args.deployment_teacher_mix) or not (bool(student_policy) or bool(deployment_policy) or args.remove_teacher_reference),deployment_teacher_action_fraction=args.deployment_teacher_mix,
        initialization_seeds=args.seeds,num_trials=raw.num_envs,seconds=args.seconds,
        geometry_stage=args.geometry_stage,geometry_seed=args.geometry_seed,geometry_bank=cfg.geometry_bank,
        geometry_assignment=furniture.indices(raw).cpu().tolist(),reference_generator=args.generator,production_task=args.production_task,reference_mode=args.reference_mode,blind_geometry=args.blind_geometry,arm_clearance=args.arm_clearance,arm_clearance_strategy='flexed_elbows_initial_and_reference' if args.arm_clearance else 'none',elbow_clearance_target_rad=2.2 if args.arm_clearance else None,shoulder_clearance_target_rad=1.6 if args.arm_clearance else None,shoulder_pitch_clearance_target_rad=args.clearance_pitch if args.arm_clearance else None,
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
    action_term=raw.action_manager.get_term('joints')
    if hasattr(action_term,'plan_audit'):(out/'reference_plans.json').write_text(json.dumps(action_term.plan_audit,indent=2)+'\n')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/'rollouts.npz',**{k:np.stack(v) for k,v in frames.items() if not args.summary_only or k in ('root_pose','support','seat_force_peak','unintended_peak','hold')},dt=raw.step_dt)
    if args.record_deployment_inputs:
        (out/'deployment_demonstration_manifest.json').write_text(json.dumps(dict(schema_version=1,actor_channels=109,body_outputs=23,control_dt_s=raw.step_dt,physics_dt_s=cfg.sim.dt,
            timing='deployment_inputs, proposed targets and expert labels at t; executed targets/state/contact diagnostics on [t,t+20ms]; contacts sampled every 5ms',
            deployment_inputs='Typed sensor packet; no exact root/COM/contact/geometry channels',proposed_targets='Unclipped neural 23-body-joint proposals',targets='Actual full 37-joint targets after command guard, optional teacher mixing and original joint limits',
            teacher_action_fraction=args.deployment_teacher_mix,teacher_assisted_rollout=bool(args.deployment_teacher_mix),expert_targets='Privileged shadow-teacher supervision only; not actor input',safe_training_state='Whole-trial safety prefix at pre-step state',
            exact_state_and_contacts='Diagnostics and evaluation only',source_perception=args.deployment_perception),indent=2)+'\n')
    if args.record_demonstrations:
        manifest=dict(schema_version=1,num_trajectories=raw.num_envs,num_steps=len(frames['targets']),control_dt_s=raw.step_dt,physics_dt_s=cfg.sim.dt,
            timing='pre observations/state at t; targets applied on [t,t+dt]; ordinary states/contacts at t+dt; history index 0 newest then decreasing time at 5 ms intervals',
            synchronized_fields=list(frames),targets_are_full_joint_positions_rad=True,teacher_residual_actions_zero=not bool(policy),
            geometry='geometry_bank.json indexed by results.outcomes.geometry_index; seat top height, depth, width, world XY and yaw',
            outcome='results.json contains whole-trial latched success and all safety outcomes',
            filtered_seat_friction_recorded=args.record_friction,student_inputs=99,actor_inputs=int((obs if isinstance(obs,torch.Tensor) else obs['policy']).shape[1]),all_contact_body_order=raw.scene['all_contacts'].body_names,
            reference='expert_targets contains teacher labels at learner states for DAgger; otherwise targets is the teacher reference when zero residual and no student',dagger_teacher_labels=args.dagger_teacher_labels,actual_action_source='student' if student_policy else 'teacher/residual',
            training_allowed_geometry_seed=args.geometry_seed not in (9612,10514,10515),held_out_baseline_banks=[9612,10514,10515])
        (out/'demonstration_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('FULL_TRANSITION_EVALUATION',json.dumps({k:v for k,v in result.items() if k not in ('outcomes','body_names','joint_names','articulation_body_names')}),flush=True)
    env.close()
