"""Direct effort PPO from supported states, with immutable command references.

New experiment only. Existing supported geometry, perception and physical labels
are imported unchanged. Truth is used for resets/reward/evaluation, not policy.
"""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import torch
import run_resting_posture as base
import run_supported_postures as support
from run_supported_transition import TransitionEnv
from humanoid_sitting.effort_rest import reference_command,posture_span,effort_reward

DT=.02
def write(p,v):base.write_json(p,v)


class EffortEnv(TransitionEnv):
    def __init__(self,raw,args,out,goals,resets):
        indices=base.furniture.indices(raw).cpu().numpy()
        self.reference=torch.tensor(goals['command'][indices],device=raw.device)
        self.full_sums={};self.late_sums={};self.late_ticks=torch.zeros(raw.num_envs,dtype=torch.long,device=raw.device)
        self.span=posture_span(base.BODY_JOINT_NAMES,raw.device)
        self.effort_reference=None
        if args.calibration:
            data=json.loads(Path(args.calibration).read_text())
            per_case=np.array([x['metrics_last10s']['total_squared_torque_Nm2'] for x in data['episodes']])
            self.effort_reference=torch.tensor(per_case[indices],device=raw.device,dtype=torch.float32)
        super().__init__(raw,args,out,goals,resets)
        self.leg_ids=[j for j,n in enumerate(self.robot.joint_names) if any(k in n for k in ['hip_','knee','ankle'])]
        self.upper_ids=[j for j in range(len(self.robot.joint_names)) if j not in self.leg_ids]
        if self.effort_reference is None:
            self.effort_reference=self.robot.data.applied_torque.torch.square().sum(1).clamp_min(1.e-6).clone()
        write(out/'effort_calibration.json',dict(reference_squared_torque_Nm2=self.effort_reference.cpu().tolist(),
            source=args.calibration or 'initial measured torque, probe/fixed only',reward_only=True))

    def curriculum_index(self,ids):
        return np.full(len(ids),5,int)

    def _augment(self):
        if self.hist is not None:
            ref=self.reference[:,self.ids,None].transpose(1,2).expand(-1,3,-1)
            self.obs['policy']=torch.cat((self.hist,ref),-1).flatten(1)
            assert self.obs['policy'].shape==(self.num_envs,402)

    def _observe(self,force=False):
        super()._observe(force)
        self._augment()

    def _reset_rows(self,ids):
        super()._reset_rows(ids)
        self.previous[ids]=self.reference[ids]
        self.late_ticks[ids]=0
        for values in [self.full_sums,self.late_sums]:
            for v in values.values():v[ids]=0
        self._augment()

    def reset(self):
        result=super().reset()
        m,d,f=support.metrics(self.raw)
        if self.args.mode=='train':
            assert bool(d['good'].all()),'Training reset does not retain support'
        limits=self.robot.data.joint_pos_limits.torch[:,self.ids]
        exact=reference_command(self.reference[:,self.ids],self.reference[:,self.ids],torch.zeros(self.num_envs,23,device=self.device),limits[...,0],limits[...,1],self.span)
        assert torch.equal(exact,self.reference[:,self.ids])
        return result

    def finish(self,ids):
        rows=[]
        for i in ids.tolist():
            ticks=int(self.episode_length_buf[i]);good=int(self.good_ticks[i]);late=int(self.late_ticks[i])
            avg=lambda values,n:{k:float(v[i])/n if n else None for k,v in values.items()}
            rows.append(dict(case=i,geometry_index=int(self.index[i]),attempt=int(self.attempt[i]),reset_index=5,
                steps=ticks,duration_s=ticks*DT,good_ticks=good,good_fraction=good/max(1,ticks),
                final_hold_s=int(self.held[i])*DT,peak_hold_s=int(self.peak_hold[i])*DT,
                success=bool(self.alive[i] and ticks>=self.max_episode_length and good>=.95*ticks and self.held[i]>=100),
                failures={k:bool(v[i]) for k,v in self.flags.items()},depth_availability=int(self.valid_ticks[i])/max(1,ticks),
                metrics_whole_trial=avg(self.full_sums,ticks),metrics_last10s=avg(self.late_sums,late),
                metrics_on_good=avg(self.sums,good),last10s_ticks=late))
        self.episodes+=rows
        with (self.out/'episodes.jsonl').open('a') as f:
            for row in rows:f.write(json.dumps(row)+'\n')

    def step(self,actions):
        with torch.inference_mode():
            warm=self.warming>0;active=~warm;data=self.robot.data
            limits=data.joint_pos_limits.torch[:,self.ids]
            command=reference_command(self.reference[:,self.ids],self.previous[:,self.ids],actions,limits[...,0],limits[...,1],self.span)
            command[warm]=self.previous[warm][:,self.ids]
            delta=command-self.previous[:,self.ids];self.previous[:,self.ids]=command
            self.raw.step(self.previous);assert not self.raw.reset_buf.any()
            self.t+=1;self.warming=(self.warming-1).clamp_min(0)
            m,d,flags=support.metrics(self.raw);before=self.alive.clone()
            first=m['failure']&before&active
            for i in first.nonzero().flatten().tolist():
                self.first_failures.append(dict(case=i,attempt=int(self.attempt[i]),t=(int(self.episode_length_buf[i])+1)*DT,
                    flags={k:bool(v[i]) for k,v in flags.items()},root_height=float(d['root_height_m'][i]),tilt=float(d['tilt_deg'][i]),wrong_force=float(d['wrong_force_N'][i])))
            self.alive &= ~(m['failure']&active)
            good=d['good']&self.alive&active
            reward,total=effort_reward(data.applied_torque.torch,self.effort_reference,good,first,delta)
            reward[warm]=0
            self.active_transitions+=int(active.sum());self.warmup_transitions+=int(warm.sum());self.episode_length_buf+=active
            for k,v in flags.items():self.flags[k]|=v&active
            self.good_ticks+=good;self.held=torch.where(good,self.held+1,torch.zeros_like(self.held));self.peak_hold=torch.maximum(self.peak_hold,self.held)
            metrics=dict(total_squared_torque_Nm2=total,total_absolute_torque_Nm=data.applied_torque.torch.abs().sum(1),
                all_joint_rms_Nm=total.div(data.applied_torque.torch.shape[1]).sqrt(),
                normalized_torque_squared=d['normalized_torque_squared'],
                leg_squared_torque_Nm2=data.applied_torque.torch[:,self.leg_ids].square().sum(1),
                upper_squared_torque_Nm2=data.applied_torque.torch[:,self.upper_ids].square().sum(1),
                support_load_bw=d['rest_normal_load_bw'],foot_load_bw=d['foot_load_bw'],
                joint_speed_rms=data.joint_vel.torch.square().mean(1).sqrt())
            late=active&(self.episode_length_buf>500);self.late_ticks+=late
            for values,mask in [(self.sums,good),(self.full_sums,active),(self.late_sums,late)]:
                for k,v in metrics.items():
                    if k not in values:values[k]=torch.zeros(self.num_envs,device=self.device)
                    values[k]+=v*mask
            horizon=self.episode_length_buf>=self.max_episode_length
            done=horizon|(~self.alive&active) if self.args.mode=='train' else horizon
            timeouts=horizon&self.alive
            self._observe();self.valid_ticks+=torch.tensor(self.geometry[:,11]>0,device=self.device)&active
            assert bool(((self.previous[:,self.ids]-self.reference[:,self.ids]).abs()<=self.span+1.e-6).all())
            if self.args.mode!='train' and self.t%5==0:
                self.trace.append(dict(t=self.t*DT,body=data.body_pose_w.torch.cpu().numpy().copy(),q=data.joint_pos.torch.cpu().numpy().copy(),
                    torque=data.applied_torque.torch.cpu().numpy().copy(),command=self.previous.cpu().numpy().copy(),
                    good=good.cpu().numpy(),alive=self.alive.cpu().numpy(),geometry=self.geometry.copy(),
                    support=d['rest_normal_load_bw'].cpu().numpy(),torso=d['torso_load_bw'].cpu().numpy(),effort=total.cpu().numpy()))
            ids=done.nonzero().flatten()
            if len(ids):
                self.finish(ids)
                if self.args.mode=='train':self._reset_rows(ids)
            return self.obs,reward,done.long(),{'time_outs':timeouts,'log':{'Rest/good':float(good.float().mean()),'Rest/effort_ratio':float((total/self.effort_reference).mean()),'Rest/physical_failure':float(first.float().mean())}}

    def close(self):
        super().close()
        audit=json.loads((self.out/'sensor_audit.json').read_text())
        audit.update(channels=402,reference='Remembered motor command outside actor; no contact/truth gating of evaluated actions')
        write(self.out/'sensor_audit.json',audit)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--bank',required=True)
    p.add_argument('--mode',choices=['fixed','probe','train','evaluate'],required=True)
    p.add_argument('--calibration');p.add_argument('--checkpoint');p.add_argument('--num-envs',type=int)
    p.add_argument('--seed',type=int,default=54501);p.add_argument('--seconds',type=float,default=20.)
    p.add_argument('--iterations',type=int,default=256);p.add_argument('--reset-probe',action='store_true')
    p.add_argument('--initial-std',type=float,default=.15)
    p.add_argument('--noise-std',type=float,default=0.);p.add_argument('--constant-action',type=float,default=0.)
    base.add_launcher_args(p);args=p.parse_args();args.enable_cameras=True;args.depth_check=True;args.reset_index=5
    if args.mode=='train':assert args.calibration,'Training must use premeasured fixed baseline calibration'
    out=Path(args.output);out.mkdir(exist_ok=False);(out/'source').mkdir()
    geometry=json.loads((Path(args.bank)/'geometry.json').read_text());goals=dict(np.load(Path(args.bank)/'candidates.npz'))
    resets={k:np.repeat(v[:,None],6,axis=1) for k,v in goals.items()};resets['validated']=np.ones((len(geometry),6),bool)
    args.num_envs=args.num_envs or len(geometry)
    paths=[Path(__file__).resolve(),Path(support.__file__).resolve(),Path(base.__file__).resolve(),
        base.ROOT/'scripts/run_supported_transition.py',base.ROOT/'sitting_task/src/humanoid_sitting/effort_rest.py',
        base.ROOT/'sitting_task/src/humanoid_sitting/supported_surface.py',base.ROOT/'sitting_task/src/humanoid_sitting/resting_objective.py',base.MODEL]
    paths+=list((base.ROOT/'sitting_task/src/humanoid_sitting/tasks').glob('*.py'))
    for i,path in enumerate(paths):shutil.copy2(path,out/'source'/f'{i:02d}_{path.name}')
    write(out/'manifest.json',dict(args=vars(args),source_hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},
        bank_hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in Path(args.bank).iterdir() if x.is_file()},
        action='remembered reference + span*tanh(action), alpha0.1, slew0.002rad/control tick, physical margin0.03; no accumulation',
        spans=posture_span(base.BODY_JOINT_NAMES).tolist(),observations=402,
        reward='dt*max(0,1/(1+sum_all_joint_tau_squared/fixed_baseline)-0.001*mean((command_delta/.002)^2))*valid_support -5*first_failure',
        success='no latched physical failure; >=95% qualifying ticks; final2seconds continuous; full20seconds',
        effort='All37 joints, physical squared Nm; torque is a proxy, not measured electrical energy'))
    write(out/'geometry.json',geometry);cfg=support.make_cfg(args,geometry);cfg.scene.rest_camera=None
    with base.launch_simulation(cfg,args):
        raw=base.gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped;env=EffortEnv(raw,args,out,goals,resets)
        if args.reset_probe:
            with torch.inference_mode():env.reset_probe()
            env.close();return
        if args.mode in ['fixed','probe']:
            generator=torch.Generator(device=env.device).manual_seed(args.seed)
            with torch.inference_mode():
                for step in range(env.max_episode_length):
                    action=torch.full((env.num_envs,23),args.constant_action,device=env.device)
                    if args.noise_std:action+=args.noise_std*torch.randn(action.shape,device=env.device,generator=generator)
                    env.step(action)
                    if step%100==99:print('SECONDS',(step+1)*DT,'alive',int(env.alive.sum()),'good',int((env.held>0).sum()),flush=True)
        else:
            agent=base.handle_deprecated_rsl_rl_cfg(base.FullTransitionPPORunnerCfg(),base.check_rsl_rl_version())
            agent.device=args.device;agent.seed=args.seed;agent.save_interval=32;agent.num_steps_per_env=64
            agent.actor.hidden_dims=[128,128];agent.critic.hidden_dims=[128,128]
            agent.actor.obs_normalization=False;agent.critic.obs_normalization=False
            agent.actor.distribution_cfg.init_std=args.initial_std;agent.actor.distribution_cfg.std_type='log'
            agent.algorithm.learning_rate=1.e-4;agent.algorithm.schedule='adaptive';agent.algorithm.desired_kl=.005
            agent.algorithm.entropy_coef=.001;agent.algorithm.num_learning_epochs=5;agent.algorithm.num_mini_batches=4
            agent.algorithm.gamma=.998;agent.algorithm.lam=.97;agent.obs_groups={'actor':['policy'],'critic':['policy']}
            runner=base.create_rsl_rl_runner(env,agent,log_dir=str(out))
            if args.checkpoint:runner.load(args.checkpoint,map_location=args.device)
            else:
                last=[x for x in runner.alg.actor.mlp.modules() if isinstance(x,torch.nn.Linear)][-1]
                with torch.no_grad():last.weight.zero_();last.bias.zero_()
                with torch.inference_mode():assert torch.count_nonzero(runner.get_inference_policy()(env.get_observations()))==0
                write(out/'initial_policy_check.json',dict(zero_action=True,reference_outside_trainable_layers=True))
            runner.save(str(out/'initial.pt'))
            if args.mode=='train':
                original=runner.alg.update;count=0;start=time.monotonic();original_obs=env.get_observations().clone()
                def update():
                    nonlocal count
                    loss=original();count+=1;env.update=count
                    assert all(torch.isfinite(x).all() for role in ['actor','critic'] for x in getattr(runner.alg,role).parameters())
                    with torch.inference_mode():initial_shift=env.span*torch.tanh(runner.get_inference_policy()(original_obs))
                    write(out/'progress.json',dict(updates=count,transitions=count*64*env.num_envs,active=env.active_transitions,warmup=env.warmup_transitions,
                        seconds=time.monotonic()-start,loss=loss,episodes=len(env.episodes),successes=sum(x['success'] for x in env.episodes),
                        latest100_successes=sum(x['success'] for x in env.episodes[-100:]),latest100_count=len(env.episodes[-100:]),
                        maximum_requested_offset_at_start_deg=float(torch.rad2deg(initial_shift.abs().max()))))
                    return loss
                runner.alg.update=update;runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False);runner.save(str(out/'final.pt'))
                write(out/'verification.json',dict(updates=count,transitions=count*64*env.num_envs,active=env.active_transitions,warmup=env.warmup_transitions,finite=True))
            else:
                policy=runner.get_inference_policy()
                with torch.inference_mode():
                    for step in range(env.max_episode_length):env.step(policy(env.get_observations()))
        write(out/'results.json',dict(episodes=env.episodes,successes=sum(r['success'] for r in env.episodes),count=len(env.episodes),
            body_names=env.robot.body_names,joint_names=env.robot.joint_names,origins=raw.scene.env_origins.cpu().numpy().tolist()))
        env.close()
if __name__=='__main__':main()
