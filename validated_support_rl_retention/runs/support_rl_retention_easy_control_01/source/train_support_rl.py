"""Bounded sensor-feedback PPO experiment on top of a frozen motion scaffold.

PPO learns 23 joint corrections from actual onboard camera depth and simulated
encoder/torque/IMU equivalents. The frozen sensor05/contact02 controller remains
explicitly part of the action path. This is NOT from-scratch sitting learning.
Only physical_metrics() reads privileged physics for rewards and evaluation.
Complete 20 s attempts use latched failures, never a success-by-early-reset metric.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
import gymnasium as gym
from scipy.spatial.transform import Rotation
from tensordict import TensorDict
from rsl_rl.env import VecEnv
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim
from isaaclab_rl.rsl_rl import create_rsl_rl_runner, check_rsl_rl_version, handle_deprecated_rsl_rl_cfg
from humanoid_sitting.tasks.student import DeploymentEnvCfg
from humanoid_sitting.tasks.full_transition import FullTransitionPPORunnerCfg
from humanoid_sitting.tasks import furniture, full_transition, mdp
from humanoid_sitting.deployment import DeploymentStudent, BODY_JOINT_NAMES
from humanoid_sitting.sim_deployment_sensor import SimSensorInterface
from humanoid_sitting.soft_landing import ContactEstimator, SoftLandingController
from humanoid_sitting.support_objective import support_objective
from humanoid_sitting.support_depth import CurrentSupportDepth
from humanoid_sitting.shared_frame import BodyModel
from seat_depth_estimator import estimate_body_frame
import humanoid_sitting.tasks

ROOT = Path(__file__).resolve().parents[1]
BODY = ROOT/'runs/g1_kinematics_geometry_04/model.json'
ACTOR = ROOT/'runs/deployment_status_mask_05/student.pt'
CONTACT = ROOT/'runs/contact_estimator_02/contact.pt'


def physical_metrics(env):
    """Privileged training/evaluation labels. NEVER passed to actor or critic."""
    robot = env.scene['robot']; d = robot.data
    g = torch.tensor([[x[k] for k in ('depth','width','height','yaw')]
                      for x in env.cfg.geometry_bank], device=env.device)[furniture.indices(env)]
    delta = d.root_pos_w.torch-env.scene['seat'].data.root_pos_w.torch
    c, s = g[:,3].cos(), g[:,3].sin()
    local = torch.stack((c*delta[:,0]+s*delta[:,1], -s*delta[:,0]+c*delta[:,1], delta[:,2]), -1)
    # Rectangular training banks; held-out silhouettes are conservative here.
    inside = (local[:,0].abs() < g[:,0]/2-.03) & (local[:,1].abs() < g[:,1]/2-.06)
    goal = torch.stack((g[:,0]/2-.12, torch.zeros_like(c), torch.full_like(c,.1984)), -1)
    weight = d.body_mass.torch.sum(1)*9.81
    force = full_transition.seat_peak_force(env)
    unexpected = full_transition.unintended_peak_force(env)
    ids = [robot.joint_names.index(n) for n in BODY_JOINT_NAMES]
    leg_ids = [i for i,n in enumerate(BODY_JOINT_NAMES) if any(k in n for k in ('hip_','knee','ankle'))]
    tau = d.applied_torque.torch[:,ids]
    # Torque reference is an objective scale, NOT a hardware effort limit.
    scales = torch.tensor([20. if 'ankle' in n else 60. if ('hip_' in n or 'knee' in n) else 30.
                           for n in BODY_JOINT_NAMES], device=env.device)
    effort = (tau/scales).square()
    root_z = d.root_pos_w.torch[:,2]-env.scene.env_origins[:,2]
    gravity = d.projected_gravity_b.torch
    vz = d.root_lin_vel_w.torch[:,2]
    torque_bad = (d.applied_torque.torch.abs() > d.joint_effort_limits.torch+.001).any(1)
    flags = dict(fall=(root_z < .28)|(gravity[:,2] > -.65), wrong_contact=unexpected > 10.,
                 pelvis_floor=mdp.pelvis_floor_contact(env), impact=force > 2*weight,
                 fast_descent=vz < -.8, torque=torque_bad)
    failure = torch.stack(list(flags.values())).any(0)
    contact = env.scene['all_contacts']
    feet = [contact.body_names.index(f'{s}_ankle_roll_link') for s in ('left','right')]
    foot_fraction = contact.data.net_normal_forces_w.torch[:,feet,2].clamp_min(0).sum(1)/weight
    if not hasattr(env,'support_rl_pelvis_vertices'):
        env.support_rl_pelvis_vertices=torch.tensor(BodyModel(BODY).vertices['pelvis'],device=env.device,dtype=torch.float32)
    vertices=env.support_rl_pelvis_vertices[None].expand(env.num_envs,-1,-1)
    quat=d.root_quat_w.torch
    xyz=quat[:,None,:3].expand_as(vertices)
    cross=torch.cross(xyz,vertices,dim=-1)
    rotated=vertices+2*quat[:,None,3:4]*cross+2*torch.cross(xyz,cross,dim=-1)
    true_clearance=delta[:,2]-.05+rotated[:,:,2].min(1).values
    return dict(support=mdp.support_fraction(env), gravity_z=gravity[:,2], tilt2=gravity[:,:2].square().sum(1),
        speed=d.root_lin_vel_w.torch.norm(dim=1), angular_speed=d.root_ang_vel_w.torch.norm(dim=1),
        inside=inside, height_ok=(local[:,2]>.17)&(local[:,2]<.30), goal_error2=(local-goal).square().sum(1),
        peak_force_ratio=force/weight, peak_force=force, unexpected=unexpected, vz=vz,
        effort2=effort.mean(1), leg_effort2=effort[:,leg_ids].mean(1),
        leg_rms_nm=tau[:,leg_ids].square().mean(1).sqrt(), foot_fraction=foot_fraction,
        true_clearance=true_clearance,
        failure=failure, flags=flags, standing=(root_z>.68)&(gravity[:,2]<-.90)&
        (contact.data.net_normal_forces_w.torch[:,feet,2].min(1).values>10.))


class SupportVecEnv(VecEnv):
    def __init__(self, raw, args, out, controller_mount, controller_quat, live_T):
        self.raw=raw; self.cfg=raw.cfg; self.device=raw.device; self.num_envs=raw.num_envs
        self.num_actions=23; self.max_episode_length=round(args.seconds/.02)
        self.episode_length_buf=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        self.args=args; self.out=out; self.mount=controller_mount; self.quat=controller_quat; self.live_T=live_T
        self.robot=raw.scene['robot']; self.ids=[self.robot.joint_names.index(n) for n in BODY_JOINT_NAMES]
        self.policy,_=DeploymentStudent.load(ACTOR,self.device); self.contact=ContactEstimator.load(CONTACT,self.device)
        self.episodes=[]; self.episode_number=0; self.depth_times=[]; self.step_times=[]
        self.trace=[]; self.record=False
        self.reset()

    def reset(self):
        with torch.inference_mode():
            self.raw.reset(); self.t=0; self.episode_length_buf.zero_()
            q=self.robot.data.joint_pos.torch.clone()
            for _ in range(5):
                self.robot.set_joint_position_target(q)
                self.raw.scene.write_data_to_sim(); self.raw.sim.step(); self.raw.scene.update(.005)
            startup=self.raw.scene['startup_camera']; startup.update(.005,force_recompute=True)
            depth=startup.data.output['distance_to_image_plane'].torch.cpu().numpy()
            K=startup.data.intrinsic_matrices.torch.cpu().numpy()
            gravity=self.robot.data.projected_gravity_b.torch.cpu().numpy()
            estimates=[dict(valid=True,**estimate_body_frame(depth[i],K[i],self.mount,self.quat,gravity[i])) for i in range(self.num_envs)]
            self.sensor=SimSensorInterface(self.robot,estimates,self.device,delay_steps=1,seed=self.args.seed+901+self.episode_number)
            self.controller=SoftLandingController(self.policy,self.contact,mode='actions'); self.state=None
            self.previous=self.robot.data.joint_pos.torch.clone()
            live=self.raw.scene['live_camera']; live.update(.005,force_recompute=True)
            K=live.data.intrinsic_matrices.torch.cpu().numpy()
            self.observers=[CurrentSupportDepth(BODY,K[i],self.live_T,estimates[i]['surface_center'][2]) for i in range(self.num_envs)]
            self.live=torch.zeros(self.num_envs,10,device=self.device); self.live[:,-1]=1.
            self.alive=torch.ones(self.num_envs,dtype=torch.bool,device=self.device)
            self.flags={k:torch.zeros_like(self.alive) for k in ('fall','wrong_contact','pelvis_floor','impact','fast_descent','torque')}
            self.hold=torch.zeros(self.num_envs,device=self.device); self.peak_hold=self.hold.clone()
            self.peak_force=self.hold.clone(); self.total_reward=self.hold.clone()
            self.economy_sum=self.hold.clone(); self.hold_samples=self.hold.clone(); self.foot_sum=self.hold.clone()
            self.depth_count=self.hold.clone(); self.initial_standing=physical_metrics(self.raw)['standing']
            self._observe(update_depth=True)
        return self.obs

    def _observe(self, update_depth):
        """Sensor-only observation path: no simulator contacts, position or velocity."""
        if update_depth:
            started=time.perf_counter()
            camera=self.raw.scene['live_camera']; camera.update(.02,force_recompute=True)
            depth=camera.data.output['distance_to_image_plane'].torch.cpu().numpy()
            q=self.robot.data.joint_pos.torch.cpu().numpy()
            gravity=self.robot.data.projected_gravity_b.torch.cpu().numpy()
            readings=[observer.read(depth[i],q[i],gravity[i]) for i,observer in enumerate(self.observers)]
            self.live=torch.tensor(np.stack([r[0] for r in readings]),device=self.device)
            self.depth_times.append(time.perf_counter()-started)
        else:
            self.live[:,-1]=(self.live[:,-1]+.02).clamp(max=1.)
        packet=self.sensor.read(self.previous)
        nominal,self.state=self.controller(packet[:,None,:],self.state)
        self.nominal=nominal[:,0].clone()
        # Current visual features plus current proprioception reach every PPO output.
        vector=torch.cat((packet,self.nominal,self.live,self.controller.last_contact),-1)
        assert vector.shape[-1]==144 and torch.isfinite(vector).all()
        self.obs=TensorDict({'policy':vector},batch_size=[self.num_envs])

    def get_observations(self):
        return self.obs

    def step(self, actions):
        start=time.perf_counter()
        with torch.inference_mode():
            assert torch.isfinite(actions).all()
            residual=actions.clamp(-1,1)
            command=self.previous.clone()
            target=self.nominal + .08*residual
            target=target.clamp(min=self.previous[:,self.ids]-.05,max=self.previous[:,self.ids]+.05)
            limits=self.robot.data.joint_pos_limits.torch[:,self.ids]
            target=target.clamp(min=limits[...,0],max=limits[...,1])
            command[:,self.ids]=target
            delta=command[:,self.ids]-self.previous[:,self.ids]
            self.raw.step(command)
            m=physical_metrics(self.raw)
            reward,good,terms=support_objective(m,delta,residual,self.alive)
            self.alive &= ~m['failure']
            for k in self.flags:self.flags[k] |= m['flags'][k]
            self.hold=torch.where(good,self.hold+.02,torch.zeros_like(self.hold))
            self.peak_hold=torch.maximum(self.peak_hold,self.hold)
            self.peak_force=torch.maximum(self.peak_force,m['peak_force'])
            self.total_reward+=reward
            self.economy_sum+=good*m['leg_rms_nm']; self.foot_sum+=good*m['foot_fraction']; self.hold_samples+=good
            self.depth_count+=self.live[:,8]
            self.previous=command.clone(); self.t+=1; self.episode_length_buf+=1
            if self.t==76:
                self.initial_standing=m['standing'].clone()
            self._observe(update_depth=self.t%5==0)
            if self.record and self.t%5==0:
                # Evaluation-only recording; none is fed back to policy.
                self.trace.append(dict(t=self.t*.02, body=self.robot.data.body_pose_w.torch.cpu().numpy().copy(),
                    root=self.robot.data.root_pose_w.torch.cpu().numpy().copy(),
                    support=m['support'].cpu().numpy(), foot=m['foot_fraction'].cpu().numpy(),
                    torque=m['leg_rms_nm'].cpu().numpy(), force=m['peak_force'].cpu().numpy(),
                    tilt=m['tilt2'].cpu().numpy(), live=self.live.cpu().numpy().copy(),
                    true_clearance=m['true_clearance'].cpu().numpy(),
                    correction=residual.cpu().numpy().copy(), alive=self.alive.cpu().numpy().copy()))
            extras={'log':{f'Reward/{k}':v.mean().item() for k,v in terms.items()}}
            # Failed states are absorbing until this synchronized batch ends.
            done=~self.alive
            if self.t>=self.max_episode_length:
                records=self.finish_episode(); self.episodes.extend(records)
                extras['log'].update({'Trial/success':np.mean([r['success'] for r in records]),
                    'Trial/peak_hold_s':np.mean([r['hold_s'] for r in records])})
                self.episode_number+=1; self.reset()
                done=torch.ones(self.num_envs,dtype=torch.long,device=self.device)
            self.step_times.append(time.perf_counter()-start)
        return self.obs,reward,done.long(),extras

    def finish_episode(self):
        result=[]
        gi=furniture.indices(self.raw).tolist()
        for i in range(self.num_envs):
            result.append(dict(episode=self.episode_number,env=i,geometry=gi[i],
                success=bool(self.initial_standing[i] and self.alive[i] and self.peak_hold[i]>=3),
                standing=bool(self.initial_standing[i]),hold_s=float(self.peak_hold[i]),
                peak_force_N=float(self.peak_force[i]),reward=float(self.total_reward[i]),
                seated_leg_rms_Nm=float(self.economy_sum[i]/self.hold_samples[i]) if self.hold_samples[i]>0 else None,
                seated_foot_load_fraction=float(self.foot_sum[i]/self.hold_samples[i]) if self.hold_samples[i]>0 else None,
                depth_availability=float(self.depth_count[i]/max(1,self.t)),
                failures={k:bool(v[i]) for k,v in self.flags.items()}))
        return result

    def close(self):
        self.raw.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True); parser.add_argument('--mode',choices=['train','zero','evaluate'],default='train')
    parser.add_argument('--checkpoint'); parser.add_argument('--iterations',type=int,default=96)
    parser.add_argument('--num-envs',type=int,default=8); parser.add_argument('--geometry-count',type=int,default=4)
    parser.add_argument('--geometry-seed',type=int,default=51031); parser.add_argument('--seed',type=int,default=51041)
    parser.add_argument('--seconds',type=float,default=20.); parser.add_argument('--learning-rate',type=float,default=3.e-5)
    parser.add_argument('--initial-std',type=float,default=.04)
    add_launcher_args(parser); args=parser.parse_args(); args.enable_cameras=True
    out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=False)
    assert shutil.disk_usage(out).free>30*2**30
    source=out/'source'; source.mkdir()
    paths=[Path(__file__),ROOT/'sitting_task/src/humanoid_sitting/support_objective.py',ROOT/'sitting_task/src/humanoid_sitting/support_depth.py',
           ROOT/'sitting_task/src/humanoid_sitting/shared_frame.py',ROOT/'scripts/test_support_objective.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[ACTOR,CONTACT,BODY]}
    for p in paths:shutil.copy2(p,source/p.name)
    (out/'manifest.json').write_text(json.dumps(dict(arguments=vars(args),hashes=hashes,
        method='RSL-RL PPO; 23 bounded joint corrections over frozen sensor05/contact02 actions-mode scaffold',
        actor_inputs='109 sensor packet + 23 nominal targets + 10 current depth features + 2 estimated contact/load = 144',
        critic_inputs='same 144 sensor-only channels',foot_force_actor_channels=0,
        privileged_data='physical_metrics: rewards, failure flags, and offline evaluation only',
        synchronization='depth plus matching encoder/gravity samples every 100 ms; proprio packet delayed 20 ms',
        action_limits='0.08 rad residual bound; 0.05 rad/control-step total command slew; original joint/effort caps',
        scope='initially aligned rigid seat behind robot; no navigation, cushion deformation, hardware deployment, or universal surface claim'),indent=2)+'\n')
    cfg=DeploymentEnvCfg(geometry_seed=args.geometry_seed,geometry_stage=2,geometry_count=args.geometry_count,
                         include_families=True,arm_clearance=True)
    cfg.scene.num_envs=args.num_envs; cfg.sim.device=args.device; cfg.seed=args.seed
    cfg.episode_length_s=args.seconds+100.; cfg.sim.render_interval=20
    for name in ('time_out','fallen','pelvis_floor','foot_seat','seat_impact'):
        if hasattr(cfg.terminations,name):setattr(cfg.terminations,name,None)
    mount=np.array([.05,.18,.55]); forward=np.array([-1.,0.,-1.7]); forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0.,0.,1.]); right/=np.linalg.norm(right)
    quat=Rotation.from_matrix(np.stack((right,np.cross(forward,right),forward),axis=1)).as_quat()
    cfg.scene.startup_camera=CameraCfg(prim_path='{ENV_REGEX_NS}/Robot/pelvis/StartupCamera',width=320,height=240,
        data_types=['distance_to_image_plane'],update_period=1.e6,
        offset=CameraCfg.OffsetCfg(pos=tuple(mount),rot=tuple(quat),convention='ros'),
        spawn=sim.PinholeCameraCfg(focal_length=18.,horizontal_aperture=24.,clipping_range=(.05,4.)))
    live_R=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])
    live_T=np.eye(4); live_T[:3,:3]=live_R; live_T[:3,3]=[-.22,0.,.55]
    live_quat=Rotation.from_matrix(live_R).as_quat()
    cfg.scene.live_camera=CameraCfg(prim_path='{ENV_REGEX_NS}/Robot/pelvis/LiveCamera',width=128,height=96,
        data_types=['distance_to_image_plane'],update_period=.1,
        offset=CameraCfg.OffsetCfg(pos=tuple(live_T[:3,3]),rot=tuple(live_quat),convention='ros'),
        spawn=sim.PinholeCameraCfg(focal_length=18.,horizontal_aperture=24.,clipping_range=(.05,4.)))
    agent=handle_deprecated_rsl_rl_cfg(FullTransitionPPORunnerCfg(),check_rsl_rl_version())
    agent.device=args.device; agent.seed=args.seed; agent.save_interval=32; agent.num_steps_per_env=64
    agent.actor.hidden_dims=[128,128]; agent.critic.hidden_dims=[128,128]
    agent.actor.obs_normalization=True; agent.critic.obs_normalization=True
    agent.actor.distribution_cfg.init_std=args.initial_std
    agent.actor.distribution_cfg.std_type='log'
    agent.algorithm.learning_rate=args.learning_rate; agent.algorithm.schedule='fixed'
    agent.algorithm.entropy_coef=0.; agent.algorithm.num_learning_epochs=3; agent.algorithm.num_mini_batches=4
    agent.obs_groups={'actor':['policy'],'critic':['policy']}
    (out/'geometry_bank.json').write_text(json.dumps(cfg.geometry_bank,indent=2)+'\n')
    started=time.perf_counter()
    with launch_simulation(cfg,args):
        raw=gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped
        env=SupportVecEnv(raw,args,out,mount,quat,live_T)
        runner=create_rsl_rl_runner(env,agent,log_dir=str(out))
        if args.checkpoint:
            runner.load(args.checkpoint,map_location=args.device)
        else:
            # Exact zero correction initially, preserving the frozen controller.
            last=[m for m in runner.alg.actor.mlp.modules() if isinstance(m,torch.nn.Linear)][-1]
            with torch.no_grad():last.weight.zero_(); last.bias.zero_()
        runner.save(str(out/'initial.pt'))
        before={k:v.detach().clone() for k,v in runner.alg.actor.state_dict().items()}
        if args.mode=='train':
            runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
            runner.save(str(out/'final.pt'))
            params=dict(runner.alg.actor.named_parameters())
            verification=dict(finite=all(torch.isfinite(v).all().item() for v in params.values()),
                max_parameter_change=max((v-before[k]).abs().max().item() for k,v in params.items()),
                transitions=args.num_envs*agent.num_steps_per_env*args.iterations,
                ppo_iterations=args.iterations,
                optimizer_contains_actor=all(any(p is q for group in runner.alg.optimizer.param_groups for q in group['params']) for p in runner.alg.actor.parameters()))
            assert verification['finite'] and verification['max_parameter_change']>0 and verification['optimizer_contains_actor']
            (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
        else:
            inference=runner.get_inference_policy(); env.record=True
            with torch.inference_mode():
                for _ in range(env.max_episode_length):
                    action=torch.zeros(env.num_envs,23,device=env.device) if args.mode=='zero' else inference(env.get_observations())
                    env.step(action)
                    if env.t and env.t%100==0:print('EVALUATION_SECONDS',env.t*.02,flush=True)
            if env.trace:
                np.savez_compressed(out/'rollout.npz',**{k:np.stack([r[k] for r in env.trace]) for k in env.trace[0]})
                (out/'visualization_metadata.json').write_text(json.dumps(dict(body_names=env.robot.body_names,
                    origins=raw.scene.env_origins.cpu().numpy().tolist(),geometry_indices=furniture.indices(raw).tolist()),indent=2)+'\n')
        report=dict(mode=args.mode,episodes=env.episodes,elapsed_s=time.perf_counter()-started,
            observed_complete_trials=len(env.episodes),successes=sum(x['success'] for x in env.episodes),
            step_median_s=float(np.median(env.step_times)),step_p95_s=float(np.percentile(env.step_times,95)),
            depth_batch_median_s=float(np.median(env.depth_times)),depth_batch_p95_s=float(np.percentile(env.depth_times,95)),
            partial_training_episode_excluded=env.t>0)
        (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        print('SUPPORT_RL_RESULT',json.dumps({k:v for k,v in report.items() if k!='episodes'}),flush=True)
        env.close()


if __name__=='__main__':main()
