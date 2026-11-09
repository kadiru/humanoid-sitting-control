"""Isolated level-seat reset capture, fixed-target baseline and resting PPO.

Physical labels are confined to metrics/reset/evaluation. Actor and critic use
encoders, torque/IMU equivalents, previous commands and measured depth geometry.
No frozen motion generator is used after reset. Capture mode uses a preserved
teacher solely to obtain dynamically settled full simulator states.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import gymnasium as gym
from scipy.spatial.transform import Rotation
from tensordict import TensorDict
from rsl_rl.env import VecEnv
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationTermCfg
import isaaclab.sim as sim
from isaaclab_rl.rsl_rl import create_rsl_rl_runner, check_rsl_rl_version, handle_deprecated_rsl_rl_cfg
from humanoid_sitting.tasks.geometry_curriculum import GeometryV12TrainingEnvCfg
from humanoid_sitting.tasks.student import StudentJointActionCfg, deployment_encoders
from humanoid_sitting.tasks.full_transition import FullTransitionPPORunnerCfg
from humanoid_sitting.tasks import furniture, full_transition, mdp
from humanoid_sitting.deployment import BODY_JOINT_NAMES
from humanoid_sitting.shared_frame import BodyModel
from humanoid_sitting.support_landing_geometry import extract_scene
from humanoid_sitting.support_camera import acquire_current_camera
from humanoid_sitting.resting_objective import resting_objective, RestingObjectiveCfg
from humanoid_sitting.resting_reward import calibrated_rest_reward
import humanoid_sitting.tasks

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'runs/g1_kinematics_geometry_04/model.json'
DT=.02
CONTACTS=[('pelvis_seat','pelvis',True),('left_thigh_seat','left_hip_yaw_link',True),
          ('right_thigh_seat','right_hip_yaw_link',True),
          ('rest_left_floor','left_ankle_roll_link',False),('rest_right_floor','right_ankle_roll_link',False)]
T_CAMERA=np.eye(4); T_CAMERA[:3,:3]=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])
T_CAMERA[:3,3]=[-.22,0.,.55]


def write_json(path,value):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2)+'\n'); tmp.replace(path)


def metrics(raw):
    """Training/evaluation labels only. Total patch forces include friction."""
    r=raw.scene['robot']; d=r.data; n=raw.num_envs; device=raw.device
    force=[]; position=[]; slip=[]
    for sensor,body,_ in CONTACTS:
        data=raw.scene[sensor].data; j=r.body_names.index(body)
        normal=data.normal_force_matrix_w.torch.sum((1,2))
        assert data.friction_force_matrix_w is not None
        tangent=data.friction_force_matrix_w.torch.sum((1,2))
        f=normal+tangent; force.append(f)
        cp=data.contact_pos_w.torch[:,0,0]
        cp=torch.where(torch.isfinite(cp),cp,d.body_com_pos_w.torch[:,j])
        v=d.body_com_vel_w.torch[:,j]
        contact_velocity=v[:,:3]+torch.cross(v[:,3:],cp-d.body_com_pos_w.torch[:,j],dim=-1)
        slip.append(contact_velocity[:,:2].norm(dim=-1)); position.append(cp)
    force=torch.stack(force,1); position=torch.stack(position,1); slip=torch.stack(slip,1)
    weight=d.body_mass.torch.sum(1)*9.81
    g=torch.tensor([[x[k] for k in ('depth','width','height','x','y')]
                    for x in raw.cfg.geometry_bank],device=device)[furniture.indices(raw)]
    center=raw.scene['seat'].data.root_pos_w.torch
    local=position-center[:,None,:]
    inside=torch.ones(n,5,dtype=torch.bool,device=device)
    inside[:,:3]=(local[:,:3,0].abs()<g[:,None,0]/2-.015)&(local[:,:3,1].abs()<g[:,None,1]/2-.015)
    ids=[r.joint_names.index(name) for name in BODY_JOINT_NAMES]
    limits=d.joint_pos_limits.torch[:,ids]; q=d.joint_pos.torch[:,ids]
    joint_margin=((q>limits[...,0]+.005)&(q<limits[...,1]-.005)).all(1)
    contact=raw.scene['all_contacts']; known={b for _,b,_ in CONTACTS}
    bad_ids=[i for i,b in enumerate(contact.body_names) if b not in known]
    unexpected=contact.data.net_normal_forces_w_history.torch[:,:,bad_ids].norm(dim=-1).flatten(1).max(1).values
    wrong=torch.maximum(unexpected,full_transition.unintended_peak_force(raw))
    # Also reject allowed resting bodies touching anything except their surface.
    for sensor,_,rest in CONTACTS:
        if rest:
            s=raw.scene[sensor].data
            other=(s.net_normal_forces_w.torch.sum(1)-s.normal_force_matrix_w.torch.sum((1,2))).norm(dim=-1)
            wrong=torch.maximum(wrong,other)
    peak=full_transition.seat_peak_force(raw)
    tilt=torch.acos((-d.projected_gravity_b.torch[:,2]).clamp(-1,1))
    rootz=d.root_pos_w.torch[:,2]-raw.scene.env_origins[:,2]
    flags=dict(fall=(rootz<.28)|(tilt>np.deg2rad(55)), forbidden=wrong>10.,
               torque=(d.applied_torque.torch.abs()>d.joint_effort_limits.torch+.001).any(1),
               impact=peak>2*weight, joint_limit=~joint_margin)
    failure=torch.stack(list(flags.values())).any(0)
    normals=torch.zeros_like(force); normals[...,2]=1
    # Fixed original rigid materials. .6 is conservative relative to static .8;
    # this is a label-side traction margin, not actor access to hidden friction.
    m=dict(torque=d.applied_torque.torch,torque_scale=d.joint_effort_limits.torch,
        joint_velocity=d.joint_vel.torch,force_w=force,normal_w=normals,
        friction=torch.full((n,5),.6,device=device),permitted=torch.ones(n,5,dtype=torch.bool,device=device),
        rest_patch=torch.tensor([[True,True,True,False,False]],device=device).expand(n,-1),inside_patch=inside,
        slip_m_s=slip,bodyweight_N=weight,speed_m_s=d.root_lin_vel_w.torch.norm(dim=-1),
        angular_speed_rad_s=d.root_ang_vel_w.torch.norm(dim=-1),trunk_tilt_rad=tilt,
        joint_margin_ok=joint_margin,failure=failure)
    _,good,detail=resting_objective(m,torch.ones(n,dtype=torch.bool,device=device))
    leg=[j for j,name in enumerate(r.joint_names) if any(s in name for s in ('hip_','knee','ankle'))]
    torso=[j for j,name in enumerate(r.joint_names) if j not in leg]
    foot_ids=[contact.body_names.index(f'{s}_ankle_roll_link') for s in ('left','right')]
    foot_net=contact.data.net_normal_forces_w.torch[:,foot_ids].sum(1)
    foot_filtered=sum(raw.scene[s].data.normal_force_matrix_w.torch.sum((1,2)) for s in ('rest_left_floor','rest_right_floor'))
    detail.update(good=good,peak_force_N=peak,wrong_force_N=wrong,
        leg_rms_Nm=d.applied_torque.torch[:,leg].square().mean(1).sqrt(),
        torso_arm_hand_rms_Nm=d.applied_torque.torch[:,torso].square().mean(1).sqrt(),
        foot_load_bw=force[:,3:,2].sum(1)/weight,
        foot_filter_relative_error=(foot_net-foot_filtered).norm(dim=-1)/weight,
        force_balance_relative=(force.sum(1)-torch.stack((torch.zeros_like(weight),torch.zeros_like(weight),weight),1)).norm(dim=-1)/weight,
        max_slip_m_s=torch.where(force.norm(dim=-1)>.01*weight[:,None],slip,0.).max(1).values)
    return m,detail,flags


def make_cfg(args,geometries):
    cfg=GeometryV12TrainingEnvCfg(geometry_bank=geometries,geometry_count=len(geometries),
                                 include_families=False,arm_clearance=True)
    cfg.scene.num_envs=args.num_envs; cfg.sim.device=args.device; cfg.seed=args.seed
    # PhysX GPU cannot filter contact forces against its infinite ground plane.
    # Use an anchored, level cuboid per environment so foot friction is actually
    # measured. Top remains z=0; no overlapping infinite plane is retained.
    cfg.scene.ground=None
    cfg.scene.rest_floor=RigidObjectCfg(prim_path='{ENV_REGEX_NS}/RestFloor',
        spawn=sim.CuboidCfg(size=(2.8,2.8,.1),
            rigid_props=sim.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            collision_props=sim.UsdPhysicsCollisionCfg(),mass_props=sim.MassCfg(mass=1000.),
            physics_material=sim.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.6),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.35,.35,.35)),activate_contact_sensors=True),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.,0.,-.05)))
    cfg.episode_length_s=100000.; cfg.sim.render_interval=20
    for name in vars(cfg.terminations):
        if not name.startswith('_'):setattr(cfg.terminations,name,None)
    for name in vars(cfg.rewards):
        if not name.startswith('_'):setattr(cfg.rewards,name,None)
    for name in vars(cfg.observations.policy):
        if not name.startswith('_') and isinstance(getattr(cfg.observations.policy,name),ObservationTermCfg):
            setattr(cfg.observations.policy,name,None)
    cfg.observations.policy.encoders=ObservationTermCfg(func=deployment_encoders)
    if args.mode!='capture':cfg.actions.joints=StudentJointActionCfg(asset_name='robot',joint_names=['.*'])
    for sensor,body,rest in CONTACTS:
        if not rest:
            setattr(cfg.scene,sensor,ContactSensorCfg(prim_path=f'{{ENV_REGEX_NS}}/Robot/{body}',
                filter_prim_paths_expr=['{ENV_REGEX_NS}/RestFloor']))
        s=getattr(cfg.scene,sensor); s.track_friction_forces=True; s.track_contact_points=True
        s.max_contact_data_count_per_prim=32; s.history_length=4
    quat=Rotation.from_matrix(T_CAMERA[:3,:3]).as_quat()
    cfg.scene.rest_camera=CameraCfg(prim_path='{ENV_REGEX_NS}/Robot/pelvis/RestCamera',width=128,height=96,
        data_types=['distance_to_image_plane'],update_period=1.e6,
        offset=CameraCfg.OffsetCfg(pos=tuple(T_CAMERA[:3,3]),rot=tuple(quat),convention='ros'),
        spawn=sim.PinholeCameraCfg(focal_length=18.,horizontal_aperture=24.,clipping_range=(.05,4.)))
    return cfg


def capture(raw,args,out):
    raw.reset(); r=raw.scene['robot']; history=[]; traces=[]
    with torch.inference_mode():
        for step in range(round(args.seconds/DT)):
            raw.step(torch.zeros(raw.num_envs,r.num_joints,device=raw.device))
            assert not raw.reset_buf.any()
            m,d,flags=metrics(raw)
            if step>=round((args.seconds-4)/DT):history.append(d['good'].cpu().numpy())
            if step%10==0:
                traces.append(dict(t=(step+1)*DT,body=r.data.body_pose_w.torch.cpu().numpy().copy(),
                    torque=d['all_joint_rms_Nm'].cpu().numpy(),support=d['rest_normal_load_bw'].cpu().numpy(),
                    good=d['good'].cpu().numpy()))
            if step%100==99:print('CAPTURE_SECONDS',(step+1)*DT,'good',int(d['good'].sum()),flush=True)
        good_fraction=np.mean(history,0); m,d,flags=metrics(raw)
        # A teacher trajectory is not automatically a valid reset. Preserve every
        # candidate; only candidates passing sustained stable contact may be used.
        valid=(good_fraction>=.95)&d['good'].cpu().numpy()&(~m['failure']).cpu().numpy()&(d['foot_filter_relative_error'].cpu().numpy()<.001)
        pose=r.data.root_pose_w.torch.clone(); pose[:,:3]-=raw.scene.env_origins
        command=raw.action_manager.get_term('joints').processed_actions.clone()
        np.savez_compressed(out/'candidates.npz',root_pose=pose.cpu().numpy(),root_velocity=r.data.root_vel_w.torch.cpu().numpy(),
            q=r.data.joint_pos.torch.cpu().numpy(),qd=r.data.joint_vel.torch.cpu().numpy(),command=command.cpu().numpy(),valid=valid)
        rows=[dict(case=i,height_m=raw.cfg.geometry_bank[int(furniture.indices(raw)[i])]['height'],
                   valid=bool(valid[i]),good_fraction_last4s=float(good_fraction[i]),
                   metrics={k:float(v[i]) for k,v in d.items()},failures={k:bool(v[i]) for k,v in flags.items()}) for i in range(raw.num_envs)]
        write_json(out/'capture.json',dict(rows=rows,valid_count=int(valid.sum()),total=len(valid),joint_names=r.joint_names,
            body_names=r.body_names,origins=raw.scene.env_origins.cpu().numpy().tolist(),geometry=raw.cfg.geometry_bank))
        np.savez_compressed(out/'trace.npz',**{k:np.stack([x[k] for x in traces]) for k in traces[0]})
        print('CAPTURE_RESULT',int(valid.sum()),len(valid),flush=True)


class RestEnv(VecEnv):
    def __init__(self,raw,args,out,bank):
        self.raw=raw; self.cfg=raw.cfg; self.device=raw.device; self.num_envs=raw.num_envs
        self.args=args; self.out=out; self.robot=raw.scene['robot']; self.ids=[self.robot.joint_names.index(n) for n in BODY_JOINT_NAMES]
        self.num_actions=len(self.ids); self.max_episode_length=round(args.seconds/DT)
        self.episode_length_buf=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.model=BodyModel(MODEL); self.bank=bank; self.episodes=[]; self.trace=[]; self.t=0
        self.cfg_reward=RestingObjectiveCfg(effort_weight=1. if args.arm=='effort' else 0.)
        self.geometry=np.zeros((self.num_envs,10),np.float32); self.geometry[:,-1]=1
        self.depth_errors=[]; self.camera_audit=[]; self.reset_records=[]; self.obs_records=[]
        self.active_transitions=0; self.warmup_transitions=0; self.disturbances=[]
        self.reset()

    def reset(self):
        self.raw.reset(); self.t=0
        self.previous=self.robot.data.joint_pos.torch.clone()
        self.hist=None; self.episode_length_buf.zero_(); self.attempt=torch.zeros_like(self.episode_length_buf)
        self.alive=torch.ones(self.num_envs,dtype=torch.bool,device=self.device)
        self.good_ticks=torch.zeros_like(self.episode_length_buf); self.current_hold=self.good_ticks.clone(); self.peak_hold=self.good_ticks.clone()
        self.peak_force=torch.zeros(self.num_envs,device=self.device); self.sums={}; self.flags={}
        self.warming=torch.zeros_like(self.episode_length_buf); self.depth_valid_ticks=self.good_ticks.clone()
        self._reset_rows(torch.arange(self.num_envs,device=self.device),initial=True)
        # Initial global settling is outside the episode/learning budget. Indexed
        # resets during learning settle on ordinary ticks, never hidden world steps.
        for _ in range(25):self.raw.step(self.previous)
        self.warming.zero_(); self._observe(force_depth=True)
        m,d,f=metrics(self.raw)
        root=self.robot.data.root_pose_w.torch.cpu().numpy()
        rotated=np.einsum('nij,vj->nvi',Rotation.from_quat(root[:,3:]).as_matrix(),self.model.vertices['pelvis'])
        heights=np.array([self.cfg.geometry_bank[int(j)]['height'] for j in furniture.indices(self.raw)])
        true_clearance=root[:,2]-self.raw.scene.env_origins[:,2].cpu().numpy()+rotated[:,:,2].min(1)-heights
        # These values are strictly validation labels; _observe does not read them.
        clearance_error=self.geometry[:,0]-true_clearance
        write_json(self.out/'initial_reset_validation.json',dict(
            good=d['good'].cpu().tolist(),failure=m['failure'].cpu().tolist(),
            metrics={k:v.cpu().tolist() for k,v in d.items()},depth_valid=self.geometry[:,8].tolist(),
            warmup_seconds=.5,depth_errors=self.depth_errors,
            true_pelvis_clearance_m=true_clearance.tolist(),estimated_clearance_error_m=clearance_error.tolist()))
        if not bool(d['good'].all()):raise RuntimeError('Initial seated reset failed physical gate; no training allowed')
        if not bool((d['foot_filter_relative_error']<.001).all()):raise RuntimeError('Foot/floor contact filtering mismatch')
        if not np.all(self.geometry[:,8]):raise RuntimeError('Initial observed resting geometry unavailable; no truth fallback')
        if not np.all(np.abs(clearance_error)<.01):raise RuntimeError('Depth/body frame clearance error exceeds 1 cm')
        if not np.all(true_clearance>-.008):raise RuntimeError('Reset pelvis penetrates seat by more than 8 mm')
        np.savez_compressed(self.out/'startup.npz',q=self.robot.data.joint_pos.torch.cpu().numpy(),
            root=self.robot.data.root_pose_w.torch.cpu().numpy()-np.pad(self.raw.scene.env_origins.cpu().numpy(),((0,0),(0,4))),
            qd=self.robot.data.joint_vel.torch.cpu().numpy(),command=self.previous.cpu().numpy(),
            geometry=self.geometry,observation=self.obs['policy'].cpu().numpy())
        return self.obs

    def _reset_rows(self,ids,initial=False):
        # Different environments share one clock but restore complete independent
        # root/joint states and the actual PD command used at capture.
        poses=torch.as_tensor(self.bank['root_pose'],device=self.device)[ids].clone()
        poses[:,:3]+=self.raw.scene.env_origins[ids]
        self.robot.reset(ids)
        self.robot.write_root_pose_to_sim_index(root_pose=poses,env_ids=ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=torch.as_tensor(self.bank['root_velocity'],device=self.device)[ids],env_ids=ids)
        self.robot.write_joint_state_to_sim_index(position=torch.as_tensor(self.bank['q'],device=self.device)[ids],
            velocity=torch.as_tensor(self.bank['qd'],device=self.device)[ids],env_ids=ids)
        self.previous[ids]=torch.as_tensor(self.bank['command'],device=self.device)[ids]
        self.robot.set_joint_position_target(self.previous[ids],env_ids=ids)
        for sensor in self.raw.scene.sensors.values():sensor.reset(ids)
        self.raw.scene.write_data_to_sim()
        self.episode_length_buf[ids]=0; self.alive[ids]=True; self.warming[ids]=25
        for v in (self.good_ticks,self.current_hold,self.peak_hold,self.peak_force,self.depth_valid_ticks):v[ids]=0
        for v in self.sums.values():v[ids]=0
        for v in self.flags.values():v[ids]=False
        self.geometry[ids.cpu().numpy()]=0; self.geometry[ids.cpu().numpy(),-1]=1
        if self.hist is not None:
            frame=self._sensor_frame()
            self.hist[ids]=frame[ids,None].expand(-1,3,-1)
            self.obs['policy']=self.hist.flatten(1)
        if not initial:self.attempt[ids]+=1

    def _sensor_frame(self):
        d=self.robot.data
        q=d.joint_pos.torch[:,self.ids]; qd=d.joint_vel.torch[:,self.ids]
        tau=d.applied_torque.torch[:,self.ids]/d.joint_effort_limits.torch[:,self.ids]
        return torch.cat((q,qd/10.,tau,d.projected_gravity_b.torch,d.root_ang_vel_b.torch/3.,
                         self.previous[:,self.ids],torch.as_tensor(self.geometry,device=self.device)),1)

    def _observe(self,force_depth=False):
        """Only depth, calibration, encoders, IMU, measured torque and command."""
        d=self.robot.data
        if force_depth or self.t%5==0:
            step=self.raw.sim.get_physics_step_count()
            self.raw.sim.render_context.update_scene_state(step)
            cam=self.raw.scene['rest_camera']; acquire_current_camera(cam)
            assert step==self.raw.sim.get_physics_step_count()
            depth=cam.data.output['distance_to_image_plane'].torch.cpu().numpy()
            K=cam.data.intrinsic_matrices.torch.cpu().numpy()
            q=d.joint_pos.torch.cpu().numpy(); gravity=d.projected_gravity_b.torch.cpu().numpy()
            for i in range(self.num_envs):
                self.geometry[i]=0; self.geometry[i,-1]=1
                try:
                    scene=extract_scene(self.model,K[i],T_CAMERA,depth[i],q[i],gravity[i])
                    # Horizontal-only initial task: highest sizeable observed plane
                    # below the pelvis, at least 8 cm above another observed plane.
                    planes=[p for p in scene['planes'] if len(p['points'])>=60 and p['rms']<.008]
                    candidates=[p for p in planes if -.35<p['height']<-.05 and
                                any(p['height']-floor['height']>.08 for floor in planes)]
                    if len(candidates)!=1:raise ValueError(f'Expected one observed support plane; got {len(candidates)}')
                    p=candidates[0]; R=scene['R']; n=R.T@p['normal']; pts=p['points']@R
                    clearance=(self.model.vertices['pelvis']@n+p['offset']).min()
                    self.geometry[i]=[clearance,*n,*pts[:,:2].mean(0),*np.ptp(pts[:,:2],axis=0),1.,0.]
                except (ValueError,np.linalg.LinAlgError) as e:
                    if len(self.depth_errors)<100:self.depth_errors.append(dict(t=self.t*.02,case=i,error=str(e)))
            self.camera_audit.append(dict(step=step,physics_unchanged=True))
        else:self.geometry[:,-1]=np.minimum(1.,self.geometry[:,-1]+DT)
        frame=self._sensor_frame()
        assert frame.shape==(self.num_envs,108) and torch.isfinite(frame).all()
        if self.hist is None:self.hist=frame[:,None].repeat(1,3,1)
        else:self.hist=torch.cat((self.hist[:,1:],frame[:,None]),1)
        self.obs=TensorDict({'policy':self.hist.flatten(1)},batch_size=[self.num_envs])

    def get_observations(self):return self.obs

    def reset_probe(self):
        def snapshot():
            d=self.robot.data
            return dict(q=d.joint_pos.torch[1:].clone(),qd=d.joint_vel.torch[1:].clone(),
                root=d.root_pose_w.torch[1:].clone(),velocity=d.root_vel_w.torch[1:].clone(),
                obs=self.obs['policy'][1:].clone(),command=self.previous[1:].clone(),
                length=self.episode_length_buf[1:].clone())
        before=snapshot(); clock=self.raw.sim.get_physics_step_count()
        self._reset_rows(torch.tensor([0],device=self.device))
        after=snapshot(); errors={k:float((v-after[k]).abs().max()) for k,v in before.items()}
        assert max(errors.values())==0.,errors
        assert clock==self.raw.sim.get_physics_step_count()
        assert not self.geometry[0,8] and self.warming[0]==25
        write_json(self.out/'reset_probe.json',dict(passed=True,errors=errors,physics_clock_unchanged=True,
            selected_geometry_invalidated=True,selected_sensor_history_reinitialized=True))

    def finish(self,ids):
        rows=[]
        for i in ids.tolist():
            ticks=int(self.episode_length_buf[i]); good=int(self.good_ticks[i])
            row=dict(case=i,attempt=int(self.attempt[i]),height_m=self.cfg.geometry_bank[int(furniture.indices(self.raw)[i])]['height'],
                steps=ticks,duration_s=ticks*DT,good_ticks=good,good_fraction=good/max(1,ticks),peak_hold_s=int(self.peak_hold[i])*DT,
                success=bool(self.alive[i] and ticks>=self.max_episode_length and good>=.95*ticks and self.current_hold[i]>=100),
                peak_force_N=float(self.peak_force[i]),depth_availability=float(self.depth_valid_ticks[i])/max(1,ticks),
                failures={k:bool(v[i]) for k,v in self.flags.items()},
                metrics_on_good={k:float(v[i])/good if good else None for k,v in self.sums.items()})
            rows.append(row)
        self.episodes.extend(rows)
        with (self.out/'episodes.jsonl').open('a') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
        return rows

    def step(self,actions):
        with torch.inference_mode():
            assert torch.isfinite(actions).all()
            warm=self.warming>0
            if self.args.impulse_vx:
                due=(self.episode_length_buf==250)&~warm
                ids=due.nonzero().flatten()
                if len(ids):
                    velocity=self.robot.data.root_vel_w.torch[ids].clone()
                    kick=self.args.impulse_vx*torch.where(ids%2==0,1.,-1.)
                    velocity[:,0]+=kick
                    self.robot.write_root_velocity_to_sim_index(root_velocity=velocity,env_ids=ids)
                    self.disturbances.append(dict(t=self.t*DT,cases=ids.tolist(),delta_vx_m_s=kick.tolist()))
            delta=.01*actions.clamp(-1,1)
            delta[warm|~self.alive]=0
            limits=self.robot.data.joint_pos_limits.torch[:,self.ids]
            self.previous[:,self.ids]=(self.previous[:,self.ids]+delta).clamp(min=limits[...,0]+.01,max=limits[...,1]-.01)
            self.raw.step(self.previous); assert not self.raw.reset_buf.any()
            self.t+=1; self.warming=(self.warming-1).clamp_min(0)
            m,d,flags=metrics(self.raw)
            reward,good,terms=calibrated_rest_reward(m,self.alive,self.args.effort_reference,self.cfg_reward.effort_weight)
            reward[warm]=0; good[warm]=False
            active=~warm; self.episode_length_buf+=active
            self.active_transitions+=int(active.sum()); self.warmup_transitions+=int(warm.sum())
            self.alive &= ~(m['failure']&active)
            for k,v in flags.items():
                if k not in self.flags:self.flags[k]=torch.zeros_like(self.alive)
                self.flags[k] |= v&active
            self.good_ticks+=good; self.current_hold=torch.where(good,self.current_hold+1,torch.zeros_like(self.current_hold))
            self.peak_hold=torch.maximum(self.peak_hold,self.current_hold)
            self.peak_force=torch.maximum(self.peak_force,d['peak_force_N']*active)
            for k in ('normalized_torque_squared','all_joint_rms_Nm','leg_rms_Nm','torso_arm_hand_rms_Nm',
                      'absolute_mechanical_power_W','foot_load_bw','rest_normal_load_bw','max_slip_m_s'):
                if k not in self.sums:self.sums[k]=torch.zeros(self.num_envs,device=self.device)
                self.sums[k]+=d[k]*good
            self.depth_valid_ticks+=torch.as_tensor(self.geometry[:,8]>0,device=self.device)&active
            horizon=self.episode_length_buf>=self.max_episode_length
            timeouts=horizon&self.alive
            done=horizon|((~self.alive)&active) if self.args.mode=='train' else horizon
            self._observe()
            if self.args.mode!='train' and self.t%5==0:
                self.trace.append(dict(t=self.t*DT,body=self.robot.data.body_pose_w.torch.cpu().numpy().copy(),
                    q=self.robot.data.joint_pos.torch.cpu().numpy().copy(),command=self.previous.cpu().numpy().copy(),
                    torque=self.robot.data.applied_torque.torch.cpu().numpy().copy(),
                    good=good.cpu().numpy(),alive=self.alive.cpu().numpy(),geometry=self.geometry.copy(),
                    effort=d['normalized_torque_squared'].cpu().numpy(),support=d['rest_normal_load_bw'].cpu().numpy(),
                    foot=d['foot_load_bw'].cpu().numpy(),slip=d['max_slip_m_s'].cpu().numpy(),force=d['peak_force_N'].cpu().numpy()))
            ids=done.nonzero().flatten()
            if len(ids):
                self.finish(ids)
                if self.args.mode=='train':self._reset_rows(ids)
            return self.obs,reward,done.long(),{'time_outs':timeouts,'log':{'Rest/good':float(good.float().mean()),'Rest/effort':float(d['normalized_torque_squared'].mean())}}

    def close(self):
        write_json(self.out/'sensor_audit.json',dict(camera=self.camera_audit,depth_errors=self.depth_errors,
            actor_channels=324,privileged_actor_inputs=False,scope='level surfaces only; current measured geometry at 10 Hz, three-frame proprioceptive history'))
        write_json(self.out/'disturbances.json',self.disturbances)
        if self.trace:np.savez_compressed(self.out/'trace.npz',**{k:np.stack([x[k] for x in self.trace]) for k in self.trace[0]})
        self.raw.close()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['capture','fixed','train','evaluate'],required=True)
    p.add_argument('--bank'); p.add_argument('--checkpoint'); p.add_argument('--arm',choices=['stability','effort'],default='effort')
    p.add_argument('--num-envs',type=int,default=16); p.add_argument('--seed',type=int,default=52911)
    p.add_argument('--seconds',type=float,default=20.); p.add_argument('--iterations',type=int,default=128)
    p.add_argument('--reset-probe',action='store_true')
    p.add_argument('--effort-reference',type=float,default=.005)
    p.add_argument('--impulse-vx',type=float,default=0.)
    p.add_argument('--heights',type=float,nargs='+',default=[.25,.30,.35,.40])
    add_launcher_args(p); args=p.parse_args(); args.enable_cameras=True
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False); (out/'source').mkdir()
    sources=[Path(__file__).resolve(),ROOT/'sitting_task/src/humanoid_sitting/resting_objective.py',
             ROOT/'sitting_task/src/humanoid_sitting/resting_reward.py',MODEL]
    sources += list((ROOT/'sitting_task/src/humanoid_sitting/tasks').glob('*.py'))
    sources += [ROOT/f'sitting_task/src/humanoid_sitting/{name}.py' for name in
                ('shared_frame','support_landing_geometry','support_camera','deployment')]
    hashes={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources}
    for x in sources:shutil.copy2(x,out/'source'/x.name)
    write_json(out/'manifest.json',dict(args=vars(args),hashes=hashes,action='23 incremental joint targets, 0.01 rad/tick, physical joint/effort limits; no timed scaffold',
        actor='324 sensor-only inputs; actor and critic same',effort='all actuated joints normalized by existing peak limits; proxy, not electrical energy',
        success='full horizon alive, >=95 percent qualifying rest ticks, final2s continuously qualifying',dt=DT))
    bank=None
    if args.bank:
        meta=json.loads((Path(args.bank)/'capture.json').read_text()); assert args.num_envs==len(meta['rows'])
        geometries=meta['geometry']; bank=dict(np.load(Path(args.bank)/'candidates.npz'))
        assert bank['valid'].all(), 'Invalid candidate states cannot silently enter training'
    else:
        geometries=[dict(family='slab',depth=.5,width=.9,height=h,x=-.4,y=0.,yaw=0.) for h in args.heights]
    cfg=make_cfg(args,geometries); write_json(out/'geometry_bank.json',geometries)
    with launch_simulation(cfg,args):
        raw=gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped
        if args.mode=='capture':capture(raw,args,out); raw.close(); return
        env=RestEnv(raw,args,out,bank)
        if args.reset_probe:
            env.reset_probe(); env.close(); return
        agent=handle_deprecated_rsl_rl_cfg(FullTransitionPPORunnerCfg(),check_rsl_rl_version())
        agent.device=args.device; agent.seed=args.seed; agent.save_interval=32; agent.num_steps_per_env=64
        agent.actor.hidden_dims=[128,128]; agent.critic.hidden_dims=[128,128]
        agent.actor.obs_normalization=False; agent.critic.obs_normalization=False
        agent.actor.distribution_cfg.init_std=.15; agent.actor.distribution_cfg.std_type='log'
        agent.algorithm.learning_rate=1.e-4; agent.algorithm.schedule='fixed'; agent.algorithm.entropy_coef=.001
        agent.algorithm.num_learning_epochs=3; agent.algorithm.num_mini_batches=4
        agent.algorithm.gamma=.998; agent.algorithm.lam=.97
        agent.obs_groups={'actor':['policy'],'critic':['policy']}
        runner=create_rsl_rl_runner(env,agent,log_dir=str(out))
        if args.checkpoint:runner.load(args.checkpoint,map_location=args.device)
        else:
            layer=[x for x in runner.alg.actor.mlp.modules() if isinstance(x,torch.nn.Linear)][-1]
            with torch.no_grad():layer.weight.zero_(); layer.bias.zero_()
            with torch.inference_mode():
                initial_action=runner.get_inference_policy()(env.get_observations())
                assert torch.count_nonzero(initial_action)==0
            write_json(out/'initial_policy_check.json',dict(zero_mean_joint_increment=True,
                deterministic_initial_behavior='holds the same command as the fixed-posture baseline'))
        runner.save(str(out/'initial.pt'))
        if args.mode=='train':
            update=runner.alg.update; count=0; start=time.monotonic()
            def audited_update():
                nonlocal count
                losses=update(); count+=1
                assert all(torch.isfinite(x).all() for role in ('actor','critic') for x in getattr(runner.alg,role).parameters())
                write_json(out/'progress.json',dict(update=count,transitions=count*64*args.num_envs,seconds=time.monotonic()-start,
                    active_transitions=env.active_transitions,warmup_transitions=env.warmup_transitions,
                    losses=losses,completed=len(env.episodes),successes=sum(r['success'] for r in env.episodes)))
                return losses
            runner.alg.update=audited_update
            runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
            runner.save(str(out/'final.pt'))
            write_json(out/'verification.json',dict(updates=count,transitions=count*64*args.num_envs,finite=True,
                active_transitions=env.active_transitions,warmup_transitions=env.warmup_transitions,
                unfinished_episode_steps=env.episode_length_buf.cpu().tolist()))
        else:
            inference=runner.get_inference_policy()
            with torch.inference_mode():
                for step in range(env.max_episode_length):
                    action=torch.zeros(env.num_envs,23,device=env.device) if args.mode=='fixed' else inference(env.get_observations())
                    env.step(action)
                    if step%100==99:print('EVALUATION_SECONDS',(step+1)*DT,flush=True)
        write_json(out/'results.json',dict(episodes=env.episodes,successes=sum(r['success'] for r in env.episodes),
            observed_complete_trials=len(env.episodes),body_names=env.robot.body_names,joint_names=env.robot.joint_names,
            origins=raw.scene.env_origins.cpu().numpy().tolist()))
        env.close()


if __name__=='__main__':main()
