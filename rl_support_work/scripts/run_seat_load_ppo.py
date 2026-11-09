"""New bounded whole-body PPO towards independently validated low-effort sitting.

Reference-assisted standing starts only. Existing studies are imported unchanged.
No live truth enters actor, critic, motion planning, command generation or feedback.
"""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import torch
import run_resting_posture as base
import run_supported_postures as support
import run_seat_load_supported as slow
from humanoid_sitting.two_surface_path import sample_path,bounded_path_command,residual_span
from humanoid_sitting.active_sample_ppo import install_active_update
DT=.02

class SeatLoadEnv(slow.SupportedUnload):
    def __init__(self,*args,**kwargs):
        self.policy_enabled=True
        super().__init__(*args,**kwargs)
        self.span=torch.tensor([.01 if ('hip_roll' in n or 'ankle_roll' in n or 'hip_yaw' in n or 'torso' in n) else .10 if ('shoulder' in n or 'elbow' in n) else .06 if 'knee' in n else .04 for n in base.BODY_JOINT_NAMES],device=self.device)
        base.write_json(self.out/'action_contract.json',dict(names=base.BODY_JOINT_NAMES,span=self.span.cpu().tolist(),duration_seconds=.1,clock_channel=True,learning_mask='Indexed-reset warmup only, never an actor/critic input',all_starts_standing=True))
    def get_observations(self):
        return base.TensorDict({'policy':torch.cat((self.obs['policy'],(self.episode_length_buf.float()/1000)[:,None]),1),'learn':(self.warming==0).float()[:,None]},batch_size=[self.num_envs])
    def step(self,actions):
        if not self.policy_enabled:actions=torch.zeros_like(actions)
        reward=torch.zeros(self.num_envs,device=self.device);done=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        for _ in range(5):
            _,r,d,info=self._physics_step(actions);reward+=r;done|=d
        return self.get_observations(),reward,done,{'time_outs':torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)}
    def _physics_step(self,actions):
        with torch.inference_mode():
            assert torch.isfinite(actions).all();warm=self.warming>0;active=~warm
            data=self.robot.data;limits=data.joint_pos_limits.torch[:,self.ids]
            ref=self.reference();command=bounded_path_command(ref[:,self.ids],self.previous[:,self.ids],actions,limits[...,0],limits[...,1],self.span)
            cap=self.profile_values['command_step'][self.index,None]
            desired=(ref[:,self.ids]+self.span*torch.tanh(actions)).clamp(min=limits[...,0]+.03,max=limits[...,1]-.03)
            candidate=self.previous[:,self.ids]+(.1*(desired-self.previous[:,self.ids])).clamp(min=-cap,max=cap)
            command=torch.where((cap!=.002),candidate,command)
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
            # Truth is labels/reward only, as in the unchanged contact evaluator.
            late=((self.episode_length_buf.float()*DT-10)/4).clamp(0,1)
            load=(d['rest_vertical_load_bw']/.65).clamp(0,1)
            effort=(total/150.).clamp(0,4)
            friction_cost=((utilization-.6)/.4).clamp(0,2)
            rest=(3*good+2*good*load-good*effort-2*(~good)-friction_cost)*self.alive
            reward=DT*((1-late)*(.5*tracking+.25*balance)*self.alive+late*rest-.01*(delta/.003).square().mean(1))-10*first
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
                self.trace.append(dict(estimated_root_x=self.estimated_root_x.cpu().numpy().copy() if self.estimated_root_x is not None else np.full(self.num_envs,np.nan),t=self.t*DT,body=data.body_pose_w.torch.cpu().numpy().copy(),q=data.joint_pos.torch.cpu().numpy().copy(),torque=data.applied_torque.torch.cpu().numpy().copy(),command=self.previous.cpu().numpy().copy(),reference=ref.cpu().numpy().copy(),phase=self.phase().cpu().numpy(),good=good.cpu().numpy(),alive=self.alive.cpu().numpy(),geometry=self.geometry.copy(),support=d['rest_normal_load_bw'].cpu().numpy(),vertical_support=d['rest_vertical_load_bw'].cpu().numpy(),friction_utilization=utilization.cpu().numpy(),traction=d['traction_ok'].cpu().numpy(),no_slip=d['no_slip'].cpu().numpy(),legal=d['contacts_legal'].cpu().numpy(),root_speed=m['speed_m_s'].cpu().numpy(),gyro=m['angular_speed_rad_s'].cpu().numpy(),tilt=m['trunk_tilt_rad'].cpu().numpy(),effort=total.cpu().numpy()))
            ids=done.nonzero().flatten()
            if len(ids):
                self.finish(ids)
                if self.args.mode=='train':self._reset_rows(ids)
            return self.obs,reward,done.long(),{'time_outs':timeouts,'log':{'Path/support':float(good.float().mean()),'Path/failure':float(first.float().mean())}}
    def close(self):
        super().close()
        p=self.out/'sensor_audit.json';v=json.loads(p.read_text());v.update(channels=406,actor_and_critic='405 sensor/reference channels plus episode clock; learn mask excluded from network groups',policy_step_seconds=.1);base.write_json(p,v)

def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',required=True);p.add_argument('--output',required=True);p.add_argument('--mode',choices=['reference','fixed','train','evaluate'],required=True)
    p.add_argument('--surface',choices=['sitting','leaning'],required=True);p.add_argument('--profiles',required=True);p.add_argument('--diagnostics',action='store_true');p.add_argument('--objective',choices=['legacy','effort'],default='legacy');p.add_argument('--calibration');p.add_argument('--plane-calibration');p.add_argument('--inertial-model');p.add_argument('--start-stage',default='all');p.add_argument('--segment-seconds',type=float,default=2.);p.add_argument('--num-envs',type=int);p.add_argument('--seed',type=int,default=54901);p.add_argument('--iterations',type=int,default=32);p.add_argument('--checkpoint');p.add_argument('--reset-probe',action='store_true');p.add_argument('--critic-warmup',type=int,default=8)
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
    cfg=support.make_cfg(a,gs);cfg.scene.rest_camera=None
    paths=[Path(__file__).resolve(),Path(slow.__file__).resolve(),base.ROOT/'sitting_task/src/humanoid_sitting/active_sample_ppo.py',base.ROOT/'scripts/run_contact_effort_repair.py',base.MODEL,base.ROOT/'sitting_task/src/humanoid_sitting/shared_frame.py',base.ROOT/'sitting_task/src/humanoid_sitting/contact_aware_reference.py',Path(__file__).resolve(),base.ROOT/'sitting_task/src/humanoid_sitting/two_surface_path.py',base.ROOT/'scripts/run_supported_transition.py',Path(support.__file__).resolve(),Path(base.__file__).resolve(),base.ROOT/'sitting_task/src/humanoid_sitting/supported_surface.py',base.ROOT/'sitting_task/src/humanoid_sitting/resting_objective.py']
    if a.plane_calibration:paths.append(Path(a.plane_calibration))
    if a.inertial_model:paths.append(Path(a.inertial_model))
    for i,x in enumerate(paths):shutil.copy2(x,out/'source'/f'{i:02d}_{x.name}')
    base.write_json(out/'manifest.json',dict(args=vars(a),sources={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},bank_hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in Path(a.bank).iterdir() if x.is_file()},observations=406,actions=23,study_reward='After10–14s ramp: dt*(3*good+2*good*clip(seat_vertical_load/.65,0,1)-good*clip(sum37torqueNmSquared/150,0,4)-2*invalid-clip((friction_utilization-.6)/.4,0,2)); approach .5*tracking+.25*balance; slew penalty .01; -10 first physical failure. No terminal value bootstrap. Actor excludes indexed reset warmup.',reward_legacy_unused='Legacy or effort objective is selected explicitly in args. Effort: dt*((4*good+2*good*clamp(1-total/calibration,-1,1)+.25*tracking+.25*balance+.5*transfer-.25*friction-.5*late_bad)*alive-.02*slew)-5*first_failure. Legacy: dt*((3*good+.5*tracking+.25*balance+.5*legal_slipfree_vertical_load_transfer-.25*friction_cost-.05*bounded_effort*good)*alive-.02*slew_cost)-5*first_failure; vertical targets .5BW sitting/.2BW leaning, friction cost ramps .8 to1; unchanged physical labels',scope='Residual PPO around offline validated joint-command path; reference-only baseline required. Not unassisted general control.',success='Full20s, no physical failure, final5s continuous qualifying support',spans='See action_contract.json;23 bounded residuals,14 finger commands fixed'))
    base.write_json(out/'geometry.json',gs)
    if a.calibration:shutil.copy2(a.calibration,out/'reward_calibration.json')
    if a.diagnostics:
        import run_contact_transfer_probe as probe
    with base.launch_simulation(cfg,a):
        raw=base.gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped;env=SeatLoadEnv(raw,a,out,goals,resets)
        if a.reset_probe:env.reset_probe();env.close();return
        if a.mode in ['reference','fixed']:
            with torch.inference_mode():
                for step in range(200):
                    env.step(torch.zeros(env.num_envs,23,device=env.device))
                    if step%250==249:print('SECONDS',(step+1)*DT,'alive',int(env.alive.sum()),'held5s',int((env.held>=250).sum()),flush=True)
        else:
            agent=base.handle_deprecated_rsl_rl_cfg(base.FullTransitionPPORunnerCfg(),base.check_rsl_rl_version())
            agent.device=a.device;agent.seed=a.seed;agent.save_interval=16;agent.num_steps_per_env=32
            agent.actor.hidden_dims=[64,64];agent.critic.hidden_dims=[64,64];agent.actor.obs_normalization=False;agent.critic.obs_normalization=False
            agent.actor.distribution_cfg.init_std=.25;agent.actor.distribution_cfg.std_type='log'
            agent.algorithm.learning_rate=1.e-4;agent.algorithm.schedule='fixed';agent.algorithm.desired_kl=.003;agent.algorithm.entropy_coef=0.;agent.algorithm.num_learning_epochs=4;agent.algorithm.num_mini_batches=4;agent.algorithm.gamma=.9975;agent.algorithm.lam=.98;agent.obs_groups={'actor':['policy'],'critic':['policy']}
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
                count=0;started=time.monotonic()
                env.policy_enabled=a.critic_warmup==0
                install_active_update(runner.alg,lambda:env.policy_enabled)
                original=runner.alg.update
                initial_actor={k:v.clone() for k,v in runner.alg.actor.state_dict().items()}
                def update():
                    nonlocal count
                    loss=original();count+=1
                    if count==a.critic_warmup:
                        assert all(torch.equal(v,initial_actor[k]) for k,v in runner.alg.actor.state_dict().items())
                        runner.save(str(out/'critic_warmup.pt'))
                    env.policy_enabled=count>=a.critic_warmup
                    assert all(torch.isfinite(x).all() for role in ['actor','critic'] for x in getattr(runner.alg,role).parameters())
                    base.write_json(out/'progress.json',dict(updates=count,actor_updates=max(0,count-a.critic_warmup),policy_decisions=count*32*env.num_envs,transitions=count*32*5*env.num_envs,active=env.active_transitions,warmup=env.warmup_transitions,seconds=time.monotonic()-started,start_stage=a.start_stage,episodes=len(env.episodes),successes=sum(x['success'] for x in env.episodes),loss=loss))
                    return loss
                runner.alg.update=update;runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=False);runner.save(str(out/'final.pt'))
                base.write_json(out/'verification.json',dict(updates=count,actor_updates=max(0,count-a.critic_warmup),policy_decisions=count*32*env.num_envs,transitions=count*32*5*env.num_envs,active=env.active_transitions,warmup=env.warmup_transitions))
            else:
                policy=runner.get_inference_policy()
                with torch.inference_mode():
                    for _ in range(200):env.step(policy(env.get_observations()))
        env.close()
if __name__=='__main__':main()
