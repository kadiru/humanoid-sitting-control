"""Bounded resting-effort PPO around a frozen standing-to-support parent.

Fork of frozen run_transfer_effort.py; previous sources/results remain unchanged.

This is not an unassisted, general sitting policy. Original environment sources
and physical labels are imported unchanged. Truth is reset/reward/evaluation only.
"""
import argparse,hashlib,json,shutil,time
from pathlib import Path
from scipy.spatial.transform import Rotation
import numpy as np
import torch
import run_resting_posture as base
import run_supported_postures as support
from run_supported_transition import TransitionEnv
from humanoid_sitting.two_surface_path import sample_path,bounded_path_command,residual_span
from humanoid_sitting.effort_learning import coordinated_actions,resting_reward,mask_learning_advantages

DT=.02
class SplitEnv(TransitionEnv):
    def __init__(self,raw,args,out,goals,resets):
        self.parent_policy=None;self.policy_enabled=True;self.learning_masks=[]
        self.paths=torch.tensor(resets['command'],device=raw.device)
        self.adapted_paths=None;self.adapted_pitch=None;self.adapted_qpaths=None
        self.pitch_paths=torch.tensor(Rotation.from_quat(resets['root_pose'][...,3:].reshape(-1,4)).as_euler('xyz')[:,1].reshape(len(resets['q']),7,1),device=raw.device,dtype=torch.float32)
        profiles=[g['control_profile'] for g in raw.cfg.geometry_bank]
        self.profile_values={k:torch.tensor([p.get(k,default) for p in profiles],device=raw.device,dtype=torch.float32) for k,default in [('kp',0.),('kd',0.),('cap',.15),('knee',0.),('knee_start_max',6.)]}
        self.ankle_ids=[raw.scene['robot'].joint_names.index(n) for n in ['left_ankle_pitch_joint','right_ankle_pitch_joint']]
        self.knee_ids=[raw.scene['robot'].joint_names.index(n) for n in ['left_knee_joint','right_knee_joint']]
        self.hip_ids=[raw.scene['robot'].joint_names.index(n) for n in ['left_hip_pitch_joint','right_hip_pitch_joint']]

        self.qpaths=torch.tensor(resets['q'],device=raw.device)
        self.span=residual_span(base.BODY_JOINT_NAMES,raw.device)
        self.full_sums={};self.late_sums={};self.last5_sums={};self.late_ticks=torch.zeros(raw.num_envs,dtype=torch.long,device=raw.device)
        self.last5_ticks=self.late_ticks.clone()
        super().__init__(raw,args,out,goals,resets)
        self.leg=[j for j,n in enumerate(self.robot.joint_names) if any(s in n for s in ['hip_','knee','ankle'])]
        self.upper=[j for j in range(len(self.robot.joint_names)) if j not in self.leg]
        self.effort_scale=torch.tensor([g['effort_scale_Nm2'] for g in raw.cfg.geometry_bank],device=self.device)[self.index]
        self.reward_reference=torch.tensor([g.get('reward_reference_Nm2',1.) for g in raw.cfg.geometry_bank],device=self.device)[self.index]
        plans=[];pitches=[];qplans=[];audit=[]
        plane_calibration=json.loads(Path(args.plane_calibration).read_text()) if args.plane_calibration else None
        for i,j in enumerate(self.index):
            profile=raw.cfg.geometry_bank[j]['control_profile']
            q=goals['q'][j];commands=resets['command'][j];roots=resets['root_pose'][j]
            assert not profile.get('adapt_plane',False), 'Leaning controller is unchanged'
            qs,cmds,rs=resets['q'][j],commands,roots
            plans.append(cmds);qplans.append(qs);pitches.append(Rotation.from_quat(rs[:,3:]).as_euler('xyz')[:,1,None])
        self.adapted_paths=torch.tensor(np.array(plans),device=self.device,dtype=torch.float32)
        self.adapted_pitch=torch.tensor(np.array(pitches),device=self.device,dtype=torch.float32)
        self.adapted_qpaths=torch.tensor(np.array(qplans),device=self.device,dtype=torch.float32)
        base.write_json(self.out/'measured_adaptation.json',audit)
        self._augment()
    def curriculum_index(self,ids):
        return np.array([self.raw.cfg.geometry_bank[self.index[i]]['start_stage'] for i in ids.cpu().tolist()])
    def phase(self):
        start=self.blend.float()
        return start if self.args.mode=='fixed' else (start+self.episode_length_buf*DT/self.args.segment_seconds).clamp_max(6)
    def reference(self):
        phase=self.phase();ref=sample_path(self.paths[self.index] if self.adapted_paths is None else self.adapted_paths,phase)
        if self.args.mode=='fixed':return ref
        # Runtime feedback uses IMU only. Desired pitch is the offline motion library.
        gravity=self.robot.data.projected_gravity_b.torch
        pitch=torch.atan2(gravity[:,0],-gravity[:,2])
        target=sample_path(self.pitch_paths[self.index] if self.adapted_pitch is None else self.adapted_pitch,phase)[:,0]
        p={k:v[self.index] for k,v in self.profile_values.items()}
        correction=(p['kp']*(pitch-target)+p['kd']*self.robot.data.root_ang_vel_b.torch[:,1]).clamp(min=-p['cap'],max=p['cap'])
        ref[:,self.ankle_ids]+=correction[:,None]
        alpha=((phase-2)/4).clamp(0,1);alpha=alpha.square()*(3-2*alpha)
        knee=p['knee']*alpha*(self.blend<=p['knee_start_max'])
        ref[:,self.knee_ids]+=knee[:,None]
        ref[:,self.hip_ids]-=.5*knee[:,None]
        ref[:,self.ankle_ids]-=.5*knee[:,None]
        return ref

    def _augment(self):
        if self.hist is not None:
            reference=self.reference()[:,self.ids];phase=self.phase()[:,None]/6
            commands=torch.cat((reference,phase),1)[:,None].expand(-1,3,-1)
            self.obs['policy']=torch.cat((self.hist,commands),-1).flatten(1)
            assert self.obs['policy'].shape==(self.num_envs,405)
    def _reset_rows(self,ids):
        super()._reset_rows(ids)
        self.late_ticks[ids]=0;self.last5_ticks[ids]=0
        for values in [self.full_sums,self.late_sums,self.last5_sums]:
            for v in values.values():v[ids]=0
        self._augment()
    def _observe(self,force=False):super()._observe(force);self._augment()
    def finish(self,ids):
        rows=[]
        for i in ids.tolist():
            ticks=int(self.episode_length_buf[i]);avg=lambda sums,n:{k:float(v[i])/n if n else None for k,v in sums.items()}
            rows.append(dict(case=i,geometry_index=int(self.index[i]),surface_id=self.raw.cfg.geometry_bank[self.index[i]]['surface_id'],profile=self.raw.cfg.geometry_bank[self.index[i]]['control_profile']['name'],attempt=int(self.attempt[i]),start_stage=int(self.blend[i]),steps=ticks,duration_s=ticks*DT,success=bool(self.alive[i] and ticks==1000 and self.held[i]>=250),good_fraction=float(self.good_ticks[i])/max(1,ticks),final_hold_s=int(self.held[i])*DT,failures={k:bool(v[i]) for k,v in self.flags.items()},depth_availability=int(self.valid_ticks[i])/max(1,ticks),metrics_whole_trial=avg(self.full_sums,ticks),metrics_last10s=avg(self.late_sums,int(self.late_ticks[i])),metrics_last5s=avg(self.last5_sums,int(self.last5_ticks[i])),late_ticks=int(self.late_ticks[i]),last5_ticks=int(self.last5_ticks[i])))
        self.episodes+=rows
        with (self.out/'episodes.jsonl').open('a') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
    def _physics_step(self,actions):
        with torch.inference_mode():
            assert torch.isfinite(actions).all();warm=self.warming>0;active=~warm
            data=self.robot.data;limits=data.joint_pos_limits.torch[:,self.ids]
            ref=self.reference();command=bounded_path_command(ref[:,self.ids],self.previous[:,self.ids],actions,limits[...,0],limits[...,1],self.span)
            command[warm]=self.previous[warm][:,self.ids]
            delta=command-self.previous[:,self.ids];self.previous[:,self.ids]=command
            self.raw.step(self.previous);assert not self.raw.reset_buf.any();self.t+=1;self.warming=(self.warming-1).clamp_min(0)
            m,d,flags=support.metrics(self.raw);first=m['failure']&self.alive&active
            for i in first.nonzero().flatten().tolist():self.first_failures.append(dict(case=i,attempt=int(self.attempt[i]),t=(int(self.episode_length_buf[i])+1)*DT,flags={k:bool(v[i]) for k,v in flags.items()},wrong_force=float(d['wrong_force_N'][i])))
            self.alive&=~(m['failure']&active)
            for k,v in flags.items():self.flags[k]|=v&active
            good=d['good']&self.alive&active;total=data.applied_torque.torch.square().sum(1)
            target=sample_path(self.qpaths[self.index] if self.adapted_qpaths is None else self.adapted_qpaths,self.phase())[:,self.ids]
            tracking=torch.exp(-20*(data.joint_pos.torch[:,self.ids]-target).square().mean(1))
            balance=torch.exp(-10*m['speed_m_s'].square()-2*m['angular_speed_rad_s'].square())
            # Privileged contact signals are reward/labels only, never observations.
            normal=(m['force_w']*m['normal_w']).sum(-1)
            tangent=(m['force_w']-normal[...,None]*m['normal_w']).norm(dim=-1)
            loaded=m['force_w'].norm(dim=-1)>.01*m['bodyweight_N'][:,None]
            utilization=torch.where(loaded,tangent/(.6*normal.clamp_min(1.e-7)),torch.zeros_like(normal)).max(1).values
            friction_cost=((utilization-.8)/.2).clamp(0,1)
            target_share=.5 if self.args.surface=='sitting' else .2
            transfer=(d['rest_vertical_load_bw']/target_share).clamp(0,1)*d['contacts_legal']*d['no_slip']*balance
            effort=total/(self.effort_scale+total)
            learning_time=(self.episode_length_buf>=600)&active
            reward=resting_reward(total,self.reward_reference,good,self.alive,learning_time,first)
            reward[warm]=0
            self.active_transitions+=int(active.sum());self.warmup_transitions+=int(warm.sum());self.episode_length_buf+=active
            self.good_ticks+=good;self.held=torch.where(good,self.held+1,torch.zeros_like(self.held));self.peak_hold=torch.maximum(self.peak_hold,self.held)
            metrics=dict(total_squared_torque_Nm2=total,total_absolute_torque_Nm=data.applied_torque.torch.abs().sum(1),all_joint_rms_Nm=(total/37).sqrt(),leg_squared_torque_Nm2=data.applied_torque.torch[:,self.leg].square().sum(1),upper_squared_torque_Nm2=data.applied_torque.torch[:,self.upper].square().sum(1),support_load_bw=d['rest_normal_load_bw'],vertical_support_load_bw=d['rest_vertical_load_bw'],foot_load_bw=d['foot_load_bw'],max_friction_utilization=utilization)
            late=active&(self.episode_length_buf>500);last5=active&(self.episode_length_buf>750);self.late_ticks+=late;self.last5_ticks+=last5
            for sums,mask in [(self.full_sums,active),(self.late_sums,late),(self.last5_sums,last5)]:
                for k,v in metrics.items():
                    if k not in sums:sums[k]=torch.zeros(self.num_envs,device=self.device)
                    sums[k]+=v*mask
            horizon=self.episode_length_buf>=1000;done=(horizon|(~self.alive&active)) if self.args.mode=='train' else horizon;timeouts=horizon&self.alive
            self._observe();self.valid_ticks+=torch.tensor(self.geometry[:,11]>0,device=self.device)&active
            if self.args.mode!='train' and self.t%5==0:
                self.trace.append(dict(t=self.t*DT,body=data.body_pose_w.torch.cpu().numpy().copy(),q=data.joint_pos.torch.cpu().numpy().copy(),torque=data.applied_torque.torch.cpu().numpy().copy(),command=self.previous.cpu().numpy().copy(),reference=ref.cpu().numpy().copy(),phase=self.phase().cpu().numpy(),good=good.cpu().numpy(),alive=self.alive.cpu().numpy(),geometry=self.geometry.copy(),support=d['rest_normal_load_bw'].cpu().numpy(),vertical_support=d['rest_vertical_load_bw'].cpu().numpy(),friction_utilization=utilization.cpu().numpy(),traction=d['traction_ok'].cpu().numpy(),no_slip=d['no_slip'].cpu().numpy(),legal=d['contacts_legal'].cpu().numpy(),root_speed=m['speed_m_s'].cpu().numpy(),gyro=m['angular_speed_rad_s'].cpu().numpy(),tilt=m['trunk_tilt_rad'].cpu().numpy(),effort=total.cpu().numpy()))
            ids=done.nonzero().flatten()
            if len(ids):
                self.finish(ids)
                if self.args.mode=='train':self._reset_rows(ids)
            return self.obs,reward,done.long(),{'time_outs':timeouts,'log':{'Path/support':float(good.float().mean()),'Path/failure':float(first.float().mean())}}
    def get_observations(self):
        if getattr(self,'parent_observation_mode',False):return self.obs
        clock=(self.episode_length_buf.float()/1000)[:,None]
        return base.TensorDict({'policy':torch.cat((self.obs['policy'],clock),1)},batch_size=[self.num_envs])
    def step(self,actions):
        assert actions.shape==(self.num_envs,2)
        self.learning_masks.append(((self.episode_length_buf>=600)&(self.warming==0)&self.policy_enabled).clone())
        reward=torch.zeros(self.num_envs,device=self.device);done=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        for _ in range(10):
            parent=self.parent_policy(self.obs) if self.parent_policy is not None else torch.zeros(self.num_envs,23,device=self.device)
            phase=((self.episode_length_buf*DT-12)/2).clamp(0,1);ramp=phase.square()*(3-2*phase)
            if not self.policy_enabled:ramp=torch.zeros_like(ramp)
            mixed=coordinated_actions(parent,actions,ramp,self.span,base.BODY_JOINT_NAMES)
            _,r,d,info=self._physics_step(mixed)
            reward+=r;done|=d
        # This is a finite20s trial. Terminal cost has no continuing-value bootstrap.
        return self.get_observations(),reward,done,{'time_outs':torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)}

    def close(self):
        base.write_json(self.out/'results.json',dict(episodes=self.episodes,successes=sum(x['success'] for x in self.episodes),count=len(self.episodes),body_names=self.robot.body_names,joint_names=self.robot.joint_names,origins=self.raw.scene.env_origins.cpu().tolist()))
        if self.args.diagnostics:
            import run_contact_transfer_probe as probe
            np.savez_compressed(self.out/'contact_diagnostics.npz',**{k:np.stack([r[k] for r in probe.records]) for k in probe.records[0]})
        super().close()
        a=json.loads((self.out/'sensor_audit.json').read_text());a.update(channels=406,actions=2,parent_updates_ms=20,child_updates_ms=200,reference='Offline joint-command/pitch motion library, control clock and IMU pitch/gyro feedback; not live simulator geometry or robot pose. Explicitly a residual/reference-based controller.');base.write_json(self.out/'sensor_audit.json',a)

def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',required=True);p.add_argument('--output',required=True);p.add_argument('--mode',choices=['reference','fixed','train','evaluate'],required=True)
    p.add_argument('--surface',choices=['sitting','leaning'],required=True);p.add_argument('--profiles',required=True);p.add_argument('--diagnostics',action='store_true');p.add_argument('--objective',choices=['effort'],default='effort');p.add_argument('--calibration');p.add_argument('--plane-calibration');p.add_argument('--inertial-model');p.add_argument('--start-stage',default='all');p.add_argument('--segment-seconds',type=float,default=2.);p.add_argument('--num-envs',type=int);p.add_argument('--seed',type=int,default=54901);p.add_argument('--iterations',type=int,default=32);p.add_argument('--checkpoint');p.add_argument('--parent-checkpoint');p.add_argument('--critic-warmup',type=int,default=8);p.add_argument('--reset-probe',action='store_true')
    base.add_launcher_args(p);a=p.parse_args();a.depth_check=True;a.enable_cameras=True;a.seconds=20.
    out=Path(a.output);out.mkdir(exist_ok=False);(out/'source').mkdir()
    geometry=json.loads((Path(a.bank)/'geometry.json').read_text());goals=dict(np.load(Path(a.bank)/'candidates.npz'));resets=dict(np.load(Path(a.bank)/'reset_candidates.npz'))
    stages=list(range(7)) if a.start_stage=='all' else [int(x) for x in a.start_stage.split(',')];profiles=json.loads(Path(a.profiles).read_text())
    cases=[(i,s,p) for p in profiles for i,g in enumerate(geometry) if g['kind']==('sit' if a.surface=='sitting' else 'lean') for s in stages];gs=[]
    for i,s,p in cases:
        g=dict(geometry[i],case=len(gs),surface_id=i,start_stage=s,control_profile=p,effort_scale_Nm2=1500. if a.surface=='sitting' else 280.);gs.append(g)
    goals={k:np.array([v[i] for i,s,p in cases]) for k,v in goals.items()};resets={k:np.array([v[i] for i,s,p in cases]) for k,v in resets.items()}
    a.num_envs=a.num_envs or len(gs)
    if a.objective=='effort':
        assert a.calibration, 'Effort reward requires a frozen corrected-reference calibration'
        cal=json.loads(Path(a.calibration).read_text())
        for g in gs:g['reward_reference_Nm2']=cal[g['label']]
    assert a.surface=='leaning'
    cfg=support.make_cfg(a,gs);cfg.scene.rest_camera=None
    paths=[base.ROOT/'sitting_task/src/humanoid_sitting/effort_learning.py',Path(__file__).resolve(),base.ROOT/'sitting_task/src/humanoid_sitting/two_surface_path.py',base.ROOT/'scripts/run_supported_transition.py',Path(support.__file__).resolve(),Path(base.__file__).resolve(),base.ROOT/'sitting_task/src/humanoid_sitting/supported_surface.py',base.ROOT/'sitting_task/src/humanoid_sitting/resting_objective.py']
    if a.plane_calibration:paths.append(Path(a.plane_calibration))
    if a.inertial_model:paths.append(Path(a.inertial_model))
    for i,x in enumerate(paths):shutil.copy2(x,out/'source'/f'{i:02d}_{x.name}')
    base.write_json(out/'manifest.json',dict(args=vars(a),sources={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},bank_hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in Path(a.bank).iterdir() if x.is_file()},observations=406,actions=2,reward='After12s only: dt*(2*valid_support-clamp(total_squared_Nm/reference,0,2)*valid_support-5*invalid_support)-5*first_physical_failure. Finite20s episodic return, coordinated2D child actions held200ms, parent updates20ms.',scope='Residual PPO around offline validated joint-command path; reference-only baseline required. Not unassisted general control.',success='Full20s, no physical failure, final5s continuous qualifying support',spans=residual_span(base.BODY_JOINT_NAMES).tolist()))
    base.write_json(out/'geometry.json',gs)
    if a.calibration:shutil.copy2(a.calibration,out/'reward_calibration.json')
    if a.diagnostics:
        import run_contact_transfer_probe as probe
    with base.launch_simulation(cfg,a):
        raw=base.gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped;env=SplitEnv(raw,a,out,goals,resets)
        env.num_actions=2
        if a.reset_probe:env.reset_probe();env.close();return
        if a.mode in ['reference','fixed']:
            with torch.inference_mode():
                for step in range(100):
                    env.step(torch.zeros(env.num_envs,2,device=env.device))
                    if step%250==249:print('SECONDS',(step+1)*DT,'alive',int(env.alive.sum()),'held5s',int((env.held>=250).sum()),flush=True)
        else:
            agent=base.handle_deprecated_rsl_rl_cfg(base.FullTransitionPPORunnerCfg(),base.check_rsl_rl_version())
            agent.device=a.device;agent.seed=a.seed;agent.save_interval=16;agent.num_steps_per_env=16
            agent.actor.hidden_dims=[64,64];agent.critic.hidden_dims=[64,64];agent.actor.obs_normalization=False;agent.critic.obs_normalization=False
            agent.actor.distribution_cfg.init_std=.3;agent.actor.distribution_cfg.std_type='log'
            agent.algorithm.learning_rate=1.e-4;agent.algorithm.schedule='adaptive';agent.algorithm.desired_kl=.01;agent.algorithm.entropy_coef=0.;agent.algorithm.num_learning_epochs=4;agent.algorithm.num_mini_batches=4;agent.algorithm.gamma=.995;agent.algorithm.lam=.98;agent.obs_groups={'actor':['policy'],'critic':['policy']}
            if a.parent_checkpoint:
                import copy
                parent_cfg=copy.deepcopy(agent);parent_cfg.actor.hidden_dims=[128,128];parent_cfg.critic.hidden_dims=[128,128]
                env.num_actions=23;env.parent_observation_mode=True
                parent_runner=base.create_rsl_rl_runner(env,parent_cfg,log_dir=None)
                parent_runner.load(a.parent_checkpoint,map_location=a.device)
                env.parent_policy=parent_runner.get_inference_policy()
                for p in parent_runner.alg.actor.parameters():p.requires_grad_(False)
                base.write_json(out/'parent_identity.json',dict(path=a.parent_checkpoint,sha256=hashlib.sha256(Path(a.parent_checkpoint).read_bytes()).hexdigest(),frozen=True))
                env.num_actions=2;env.parent_observation_mode=False
            runner=base.create_rsl_rl_runner(env,agent,log_dir=str(out))
            if a.checkpoint:
                runner.load(a.checkpoint,map_location=a.device)
                # RSL saves the just-completed zero-based iteration. Resume the
                # next index; the number of new optimizer updates is unchanged.
                if a.mode=='train':runner.current_learning_iteration+=1
            else:
                layer=[x for x in runner.alg.actor.mlp.modules() if isinstance(x,torch.nn.Linear)][-1]
                with torch.no_grad():layer.weight.zero_();layer.bias.zero_()
                with torch.inference_mode():assert torch.count_nonzero(runner.get_inference_policy()(env.get_observations()))==0
            runner.save(str(out/'initial.pt'))
            if a.mode=='train':
                original=runner.alg.update;original_returns=runner.alg.compute_returns;count=0;started=time.monotonic()
                env.policy_enabled=a.critic_warmup==0
                initial_actor={k:v.detach().clone() for k,v in runner.alg.actor.state_dict().items()}
                def compute_returns(obs):
                    original_returns(obs)
                    masks=torch.stack(env.learning_masks[-16:]).bool()
                    advantages=runner.alg.storage.advantages
                    advantages.copy_(mask_learning_advantages(advantages,masks))
                    env.learning_masks.clear()
                runner.alg.compute_returns=compute_returns
                def update():
                    nonlocal count
                    loss=original();count+=1
                    if count==a.critic_warmup:
                        assert all(torch.equal(v,initial_actor[k]) for k,v in runner.alg.actor.state_dict().items())
                        runner.save(str(out/'critic_warmup.pt'))
                    env.policy_enabled=count>=a.critic_warmup
                    assert all(torch.isfinite(x).all() for role in ['actor','critic'] for x in getattr(runner.alg,role).parameters())
                    base.write_json(out/'progress.json',dict(updates=count,actor_updates=max(0,count-a.critic_warmup),policy_decisions=count*16*env.num_envs,transitions=count*16*10*env.num_envs,active=env.active_transitions,warmup=env.warmup_transitions,seconds=time.monotonic()-started,start_stage=a.start_stage,episodes=len(env.episodes),successes=sum(x['success'] for x in env.episodes),loss=loss))
                    return loss
                runner.alg.update=update;runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=False);runner.save(str(out/'final.pt'))
                base.write_json(out/'verification.json',dict(updates=count,actor_updates=max(0,count-a.critic_warmup),policy_decisions=count*16*env.num_envs,transitions=count*16*10*env.num_envs,active=env.active_transitions,warmup=env.warmup_transitions))
            else:
                policy=runner.get_inference_policy()
                with torch.inference_mode():
                    for _ in range(100):env.step(policy(env.get_observations()))
        env.close()
if __name__=='__main__':main()
