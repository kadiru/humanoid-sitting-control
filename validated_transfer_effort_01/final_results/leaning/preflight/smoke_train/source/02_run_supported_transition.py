"""Sensor-only PPO from standing to previously validated rigid support goals.

Ground truth is isolated to reset construction, rewards and evaluation labels.
Absolute joint-target actions; no target-pose/geometry-bank access in actor.
"""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import run_resting_posture as base
import run_supported_postures as support
from humanoid_sitting.supported_surface import observed_support,SupportTracker
DT=.02;FRAME=111

def write(p,v):base.write_json(p,v)

class TransitionEnv(base.VecEnv):
    def __init__(self,raw,args,out,goals,resets):
        self.raw=raw;self.cfg=raw.cfg;self.device=raw.device;self.num_envs=raw.num_envs;self.robot=raw.scene['robot']
        self.args=args;self.out=out;self.ids=[self.robot.joint_names.index(n) for n in base.BODY_JOINT_NAMES]
        self.num_actions=23;self.max_episode_length=round(args.seconds/DT);self.model=base.BodyModel(base.MODEL)
        self.goals=goals;self.resets=resets;self.reset_valid=resets.get('validated',np.ones((len(goals['q']),6),bool));self.index=base.furniture.indices(raw).cpu().numpy();self.update=0
        self.first_failures=[];self.t=0;self.active_transitions=0;self.warmup_transitions=0;self.episodes=[];self.trace=[];self.audit=[];self.camera_count=0
        self.geometry=np.zeros((self.num_envs,13),np.float32);self.geometry[:,12]=1.
        self.trackers=[SupportTracker(self.model) for _ in range(self.num_envs)];self.hist=None
        self.episode_length_buf=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.attempt=self.episode_length_buf.clone();self.warming=self.episode_length_buf.clone();self.blend=self.episode_length_buf.clone()
        self.alive=torch.ones(self.num_envs,dtype=torch.bool,device=self.device);self.good_ticks=self.episode_length_buf.clone();self.held=self.episode_length_buf.clone();self.peak_hold=self.episode_length_buf.clone()
        self.flags={k:torch.zeros_like(self.alive) for k in ['fall','forbidden','torque','impact','joint_limit']};self.sums={};self.valid_ticks=self.episode_length_buf.clone()
        # Public fixed joint ranges, independent of each surface and goal.
        meta=json.loads(base.MODEL.read_text());self.nominal=torch.tensor(meta['default_q'],device=self.device,dtype=torch.float32)[self.ids]
        self.scale=torch.tensor([1.6 if ('hip_pitch' in n or 'knee' in n) else .65 if 'ankle_pitch' in n else 1.5 if ('shoulder' in n or 'elbow' in n) else .5 for n in base.BODY_JOINT_NAMES],device=self.device)
        self.goal_q=torch.tensor(goals['q'][self.index],device=self.device)[:,self.ids]
        self.goal_root=torch.tensor(goals['root_pose'][self.index],device=self.device)
        self.reset()
    def curriculum_index(self,ids):
        if self.args.mode!='train':return np.full(len(ids),self.args.reset_index,int)
        # 32 updates per stage; final64 of224 updates are standing-only.
        stage=min(5,self.update//32);index=5-stage
        result=np.full(len(ids),index,int)
        # Every stage also exposes the policy to full standing starts.
        if stage<5:result[(ids.cpu().numpy()+self.attempt[ids].cpu().numpy())%5==0]=0
        blends=np.array([0.,.25,.5,.75,.9,1.])
        for k,i in enumerate(ids.cpu().numpy()):
            valid=np.flatnonzero(self.reset_valid[self.index[i]])
            result[k]=valid[np.argmin(abs(blends[valid]-blends[result[k]]))]
        return result
    def _reset_rows(self,ids):
        indices=self.index[ids.cpu().numpy()];blend=self.curriculum_index(ids)
        b={k:self.resets[k][indices,blend] for k in ['q','qd','command','root_pose','root_velocity']}
        tensor=lambda k:torch.tensor(b[k],device=self.device,dtype=torch.float32)
        r=self.robot;root=tensor('root_pose');root[:,:3]+=self.raw.scene.env_origins[ids]
        r.reset(ids);r.write_root_pose_to_sim_index(root_pose=root,env_ids=ids);r.write_root_velocity_to_sim_index(root_velocity=tensor('root_velocity'),env_ids=ids)
        r.write_joint_state_to_sim_index(position=tensor('q'),velocity=tensor('qd'),env_ids=ids)
        self.previous[ids]=tensor('command');r.set_joint_position_target(self.previous[ids],env_ids=ids)
        for sensor in self.raw.scene.sensors.values():sensor.reset(ids)
        self.raw.scene.write_data_to_sim()
        self.episode_length_buf[ids]=0;self.alive[ids]=True;self.warming[ids]=25;self.blend[ids]=torch.tensor(blend,device=self.device)
        for v in [self.good_ticks,self.held,self.peak_hold,self.valid_ticks,*self.flags.values(),*self.sums.values()]:v[ids]=0
        for i in ids.tolist():self.trackers[i].reset();self.geometry[i]=0;self.geometry[i,12]=1
        if self.hist is not None:
            frame=self._sensor_frame();self.hist[ids]=frame[ids,None].expand(-1,3,-1);self.obs['policy']=self.hist.flatten(1)
        self.attempt[ids]+=1
    def reset(self):
        self.raw.reset();self.previous=self.robot.data.joint_pos.torch.clone();self._reset_rows(torch.arange(self.num_envs,device=self.device))
        for _ in range(25):self.raw.step(self.previous)
        self.warming.zero_();self._observe(force=True)
        m,d,f=support.metrics(self.raw)
        data=self.robot.data;R=Rotation.from_quat(data.body_quat_w.torch[0].cpu().numpy()).as_matrix()
        delta=(data.body_com_pos_w.torch[0]-data.body_pos_w.torch[0]).cpu().numpy()
        com=np.einsum('nji,nj->ni',R,delta)
        write(self.out/'static_inertial_model.json',dict(body_names=self.robot.body_names,mass=data.body_mass.torch[0].cpu().tolist(),local_com=com.tolist(),scope='constant model properties exported offline; no runtime state estimator input'))
        write(self.out/'startup_validation.json',dict(failures={k:v.cpu().tolist() for k,v in f.items()},good=d['good'].cpu().tolist(),sensor_valid=self.geometry[:,11].tolist(),reset_index=self.blend.cpu().tolist(),global_settle_seconds=.5))
        if self.args.mode=='train':
            assert not m['failure'].any(), 'Training startup has a physical failure'
            assert bool(np.all(self.geometry[:,11])), 'Training startup lacks measured support geometry'
        root=self.robot.data.root_pose_w.torch.clone();root[:,:3]-=self.raw.scene.env_origins
        np.savez_compressed(self.out/'startup.npz',root=root.cpu().numpy(),q=self.robot.data.joint_pos.torch.cpu().numpy(),command=self.previous.cpu().numpy(),geometry=self.geometry,observation=self.obs['policy'].cpu().numpy())
        return self.obs
    def _sensor_frame(self):
        d=self.robot.data
        return torch.cat((d.joint_pos.torch[:,self.ids],d.joint_vel.torch[:,self.ids]/10.,d.applied_torque.torch[:,self.ids]/d.joint_effort_limits.torch[:,self.ids],d.projected_gravity_b.torch,d.root_ang_vel_b.torch/3.,self.previous[:,self.ids],torch.tensor(self.geometry,device=self.device)),1)
    def _observe(self,force=False):
        d=self.robot.data;q=d.joint_pos.torch.cpu().numpy();gravity=d.projected_gravity_b.torch.cpu().numpy();gyro=d.root_ang_vel_b.torch.cpu().numpy()
        measurements=[None]*self.num_envs
        if force or self.t%10==0:
            clock=self.raw.sim.get_physics_step_count();self.raw.sim.render_context.update_scene_state(clock)
            cam=self.raw.scene['rear_camera'];base.acquire_current_camera(cam);assert clock==self.raw.sim.get_physics_step_count();self.camera_count+=1
            depths=cam.data.output['distance_to_image_plane'].torch.cpu().numpy();K=cam.data.intrinsic_matrices.torch.cpu().numpy()
            for i in range(self.num_envs):
                measurements[i],diag=observed_support(self.model,K[i],support.REAR_CAMERA,depths[i],q[i],gravity[i],prior=self.trackers[i].value)
        for i in range(self.num_envs):self.geometry[i]=self.trackers[i].update(q[i],gravity[i],gyro[i],measurement=measurements[i])
        frame=self._sensor_frame();assert frame.shape==(self.num_envs,FRAME) and torch.isfinite(frame).all()
        self.hist=frame[:,None].repeat(1,3,1) if self.hist is None else torch.cat((self.hist[:,1:],frame[:,None]),1)
        self.obs=base.TensorDict({'policy':self.hist.flatten(1)},batch_size=[self.num_envs])
        # Offline truth audit; never writes actor features or selects a plane.
        if self.args.mode!='train' and (force or self.t%10==0):
            R=Rotation.from_quat(d.root_quat_w.torch.cpu().numpy()).as_matrix();root=d.root_pos_w.torch.cpu().numpy()-self.raw.scene.env_origins.cpu().numpy()
            for i,j in enumerate(self.index):
                g=self.cfg.geometry_bank[j];normal=R[i].T@np.array(g['normal']);offset=(root[i]-g['top_center'])@np.array(g['normal'])
                v=self.geometry[i];self.audit.append(dict(t=self.t*DT,case=i,valid=bool(v[11]),age=float(v[12]),normal_error_deg=float(np.rad2deg(np.arccos(np.clip(v[2:5]@normal,-1,1)))) if v[11] else None,offset_error_m=float(v[5]-offset) if v[11] else None))
    def get_observations(self):return self.obs
    def hold_action(self):return (self.previous[:,self.ids]-self.nominal)/self.scale
    def reset_probe(self):
        ids=torch.tensor([0],device=self.device);d=self.robot.data
        def snap():return [d.joint_pos.torch[1:].clone(),d.root_pose_w.torch[1:].clone(),self.previous[1:].clone(),self.obs['policy'][1:].clone()]
        before=snap();clock=self.raw.sim.get_physics_step_count();self._reset_rows(ids);after=snap()
        errors=[float((a-b).abs().max()) for a,b in zip(before,after)];assert max(errors)==0 and clock==self.raw.sim.get_physics_step_count()
        write(self.out/'reset_probe.json',dict(passed=True,other_rows_max_errors=errors,clock_unchanged=True,selected_sensor_invalid=bool(self.geometry[0,11]==0)))
    def finish(self,ids):
        rows=[]
        for i in ids.tolist():
            length=int(self.episode_length_buf[i]);good=int(self.good_ticks[i]);success=bool(self.alive[i] and length>=self.max_episode_length and self.held[i]>=250)
            row=dict(case=i,geometry_index=int(self.index[i]),attempt=int(self.attempt[i]),reset_index=int(self.blend[i]),steps=length,duration_s=length*DT,success=success,final_hold_s=int(self.held[i])*DT,peak_hold_s=int(self.peak_hold[i])*DT,good_ticks=good,failures={k:bool(v[i]) for k,v in self.flags.items()},depth_availability=int(self.valid_ticks[i])/max(length,1),effort_on_good={k:float(v[i])/good if good else None for k,v in self.sums.items()})
            rows.append(row)
        self.episodes+=rows
        with (self.out/'episodes.jsonl').open('a') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
    def step(self,actions):
        with torch.inference_mode():
            assert torch.isfinite(actions).all();warm=self.warming>0;active=~warm
            desired=self.nominal+self.scale*actions.clamp(-2.,2.)
            limits=self.robot.data.joint_pos_limits.torch[:,self.ids]
            desired=desired.clamp(min=limits[...,0]+.03,max=limits[...,1]-.03)
            delta=(.2*(desired-self.previous[:,self.ids])).clamp(-.015,.015);delta[warm]=0
            self.previous[:,self.ids]+=delta;self.raw.step(self.previous);assert not self.raw.reset_buf.any()
            self.t+=1;self.warming=(self.warming-1).clamp_min(0);m,d,flags=support.metrics(self.raw)
            before=self.alive.clone()
            for i in (m['failure']&before&active).nonzero().flatten().tolist():
                self.first_failures.append(dict(case=i,attempt=int(self.attempt[i]),t=int(self.episode_length_buf[i])*DT,flags={k:bool(v[i]) for k,v in flags.items()},root_height=float(d['root_height_m'][i]),tilt=float(d['tilt_deg'][i]),wrong_force=float(d['wrong_force_N'][i])))
            self.alive &= ~(m['failure']&active)
            good=d['good']&self.alive&active
            for k,v in flags.items():self.flags[k]|=v&active
            # Privileged dense goal shaping is used ONLY by reward computation.
            data=self.robot.data;qerr=((data.joint_pos.torch[:,self.ids]-self.goal_q)/self.scale).square().mean(1)
            localroot=data.root_pos_w.torch-self.raw.scene.env_origins;perr=(localroot-self.goal_root[:,:3]).square().sum(1)
            goal=torch.exp(-6*qerr-12*perr)
            effort=(d['normalized_torque_squared']/.01).clamp(0,1)
            reward=DT*(.5*goal+3.*good-.05*effort*good-.02*delta.square().mean(1)/.015**2)
            reward-=5.*(m['failure']&before&active);reward[warm]=0
            self.active_transitions+=int(active.sum());self.warmup_transitions+=int(warm.sum());self.episode_length_buf+=active
            self.good_ticks+=good;self.held=torch.where(good,self.held+1,torch.zeros_like(self.held));self.peak_hold=torch.maximum(self.peak_hold,self.held)
            for k in ['normalized_torque_squared','all_joint_rms_Nm','rest_normal_load_bw']:
                if k not in self.sums:self.sums[k]=torch.zeros(self.num_envs,device=self.device)
                self.sums[k]+=d[k]*good
            horizon=self.episode_length_buf>=self.max_episode_length;done=horizon|(~self.alive&active) if self.args.mode=='train' else horizon;timeouts=horizon&self.alive
            self._observe();self.valid_ticks+=torch.tensor(self.geometry[:,11]>0,device=self.device)&active
            if self.args.mode!='train' and self.t%5==0:
                self.trace.append(dict(t=self.t*DT,body=data.body_pose_w.torch.cpu().numpy().copy(),q=data.joint_pos.torch.cpu().numpy().copy(),command=self.previous.cpu().numpy().copy(),good=good.cpu().numpy(),alive=self.alive.cpu().numpy(),geometry=self.geometry.copy(),support=d['rest_normal_load_bw'].cpu().numpy(),torso=d['torso_load_bw'].cpu().numpy(),effort=d['normalized_torque_squared'].cpu().numpy()))
            ids=done.nonzero().flatten()
            if len(ids):
                self.finish(ids)
                if self.args.mode=='train':self._reset_rows(ids)
            return self.obs,reward,done.long(),{'time_outs':timeouts,'log':{'Support/good':float(good.float().mean()),'Support/goal':float(goal.mean()),'Support/depth_valid':float(np.mean(self.geometry[:,11]))}}
    def close(self):
        write(self.out/'first_failures.json',self.first_failures)
        write(self.out/'sensor_audit.json',dict(channels=333,physics_advances_during_camera=0,camera_frames=self.camera_count,actor_inputs='encoders, measured torque, IMU gravity/gyro, previous absolute joint commands, depth plane with encoder/IMU propagation',truth_validation=self.audit))
        if self.trace:np.savez_compressed(self.out/'trace.npz',**{k:np.stack([r[k] for r in self.trace]) for k in self.trace[0]})
        self.raw.close()

def initialize_hold(actor):
    layers=[x for x in actor.mlp.modules() if isinstance(x,torch.nn.Linear)]
    assert len(layers)==3
    with torch.no_grad():
        # Reserve23 positive ELU channels to exactly copy the last command.
        layers[0].weight[:23]=0;layers[0].bias[:23]=4
        for j in range(23):layers[0].weight[j,2*FRAME+75+j]=1
        layers[1].weight[:23]=0;layers[1].bias[:23]=0
        for j in range(23):layers[1].weight[j,j]=1
        layers[2].weight.zero_();layers[2].bias.zero_()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--bank',required=True);p.add_argument('--mode',choices=['probe','fixed','train','evaluate'],required=True)
    p.add_argument('--checkpoint');p.add_argument('--num-envs',type=int);p.add_argument('--seed',type=int,default=54211);p.add_argument('--seconds',type=float,default=20.);p.add_argument('--iterations',type=int,default=224);p.add_argument('--reset-index',type=int,default=0);p.add_argument('--reset-probe',action='store_true')
    base.add_launcher_args(p);args=p.parse_args();args.enable_cameras=True;args.depth_check=True
    out=Path(args.output);out.mkdir(exist_ok=False);(out/'source').mkdir()
    geometry=json.loads((Path(args.bank)/'geometry.json').read_text());goals=dict(np.load(Path(args.bank)/'candidates.npz'));resets=dict(np.load(Path(args.bank)/'reset_candidates.npz'))
    args.num_envs=args.num_envs or len(geometry);paths=[Path(__file__),Path(support.__file__),Path(base.__file__),base.MODEL,base.ROOT/'sitting_task/src/humanoid_sitting/supported_surface.py',base.ROOT/'sitting_task/src/humanoid_sitting/resting_objective.py']
    paths+=list((base.ROOT/'sitting_task/src/humanoid_sitting/tasks').glob('*.py'))
    hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
    for i,x in enumerate(paths):shutil.copy2(x,out/'source'/f'{i:02d}_{x.name}')
    write(out/'manifest.json',dict(args=vars(args),source_hashes=hashes,bank_hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in Path(args.bank).iterdir() if x.is_file()},reward='dt*(.5*exp(-6*qerr-12*rooterr)+3*valid_support-.05*clipped_effort*valid_support-.02*normalized_action_change)-5*first_failure',success='full20seconds no latched physical failure and final5seconds continuous qualifying support; starting curriculum index separately reported'))
    write(out/'geometry.json',geometry);cfg=support.make_cfg(args,geometry);cfg.scene.rest_camera=None
    with base.launch_simulation(cfg,args):
        raw=base.gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped;env=TransitionEnv(raw,args,out,goals,resets)
        if args.reset_probe:env.reset_probe();env.close();return
        if args.mode in ['probe','fixed']:
            with torch.inference_mode():
                constant=env.hold_action().clone()
                for step in range(env.max_episode_length):
                    env.step(constant)
                    if step%100==99:print('PROBE_SECONDS',(step+1)*DT,'alive',int(env.alive.sum()),flush=True)
        else:
            agent=base.handle_deprecated_rsl_rl_cfg(base.FullTransitionPPORunnerCfg(),base.check_rsl_rl_version())
            agent.device=args.device;agent.seed=args.seed;agent.save_interval=32;agent.num_steps_per_env=64
            agent.actor.hidden_dims=[128,128];agent.critic.hidden_dims=[128,128];agent.actor.obs_normalization=False;agent.critic.obs_normalization=False
            agent.actor.distribution_cfg.init_std=.05;agent.actor.distribution_cfg.std_type='log'
            agent.algorithm.learning_rate=1.e-4;agent.algorithm.schedule='fixed';agent.algorithm.entropy_coef=.001;agent.algorithm.num_learning_epochs=3;agent.algorithm.num_mini_batches=4;agent.algorithm.gamma=.998;agent.algorithm.lam=.97;agent.obs_groups={'actor':['policy'],'critic':['policy']}
            runner=base.create_rsl_rl_runner(env,agent,log_dir=str(out))
            if args.checkpoint:runner.load(args.checkpoint,map_location=args.device)
            else:
                initialize_hold(runner.alg.actor);last=[x for x in runner.alg.actor.mlp.modules() if isinstance(x,torch.nn.Linear)][-1]
                with torch.no_grad():
                    for j in range(23):last.weight[j,j]=1/env.scale[j]
                    last.bias.copy_((-4-env.nominal)/env.scale)
                with torch.inference_mode():error=float((runner.get_inference_policy()(env.get_observations())-env.hold_action()).abs().max())
                assert error<3.e-6,error
                write(out/'initial_policy_check.json',dict(max_action_error=error,holds_initial_command=True))
            runner.save(str(out/'initial.pt'))
            if args.mode=='train':
                original=runner.alg.update;count=0;start=time.monotonic()
                def update():
                    nonlocal count
                    loss=original();count+=1;env.update=count
                    assert all(torch.isfinite(x).all() for role in ['actor','critic'] for x in getattr(runner.alg,role).parameters())
                    write(out/'progress.json',dict(updates=count,transitions=count*64*env.num_envs,active=env.active_transitions,warmup=env.warmup_transitions,seconds=time.monotonic()-start,loss=loss,episodes=len(env.episodes),successes=sum(x['success'] for x in env.episodes),curriculum_index=max(0,5-count//32)))
                    return loss
                runner.alg.update=update;runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False);runner.save(str(out/'final.pt'))
                write(out/'verification.json',dict(updates=count,transitions=count*64*env.num_envs,active=env.active_transitions,warmup=env.warmup_transitions,finite=True))
            else:
                policy=runner.get_inference_policy()
                with torch.inference_mode():
                    for step in range(env.max_episode_length):env.step(policy(env.get_observations()))
        write(out/'results.json',dict(episodes=env.episodes,successes=sum(r['success'] for r in env.episodes),count=len(env.episodes),body_names=env.robot.body_names,joint_names=env.robot.joint_names,origins=raw.scene.env_origins.cpu().numpy().tolist()))
        env.close()
if __name__=='__main__':main()
