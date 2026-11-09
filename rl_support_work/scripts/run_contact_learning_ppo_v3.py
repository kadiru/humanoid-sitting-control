"""Matched original/revised reward comparison; unchanged sensor-feedback controller.

This is reference-assisted control. The frozen resting parent remains visible
in the contract; the new PPO updates throughout its window, not a short pulse.
"""
import argparse,sys,time,shutil
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
p.add_argument('--reward-arm',choices=['original','revised'],required=True)
p.add_argument('--feedback-rng-state')
p.add_argument('--feedback-mode',choices=['calibrate','zero','reference','explore','train','evaluate'],required=True)
p.add_argument('--feedback-start',type=float,choices=[10.,8.,6.],default=10.)
p.add_argument('--feedback-seed',type=int,default=62111)
p.add_argument('--feedback-updates',type=int,default=1)
p.add_argument('--feedback-span',type=float,default=.004)
p.add_argument('--feedback-rate',type=float,default=.004)
p.add_argument('--feedback-calibration');p.add_argument('--feedback-checkpoint')
p.add_argument('--repair-config',required=True)
p.add_argument('--landing-config',required=True)
p.add_argument('--repair-kind',choices=['timing','phase','original'],default='timing')
options,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_positive_slope_ppo_v4 as old
from feedback_posture_ppo import FeedbackPPO,slew,complete_returns
from contact_reward_v1 import tick_reward, reward_to_go
from contact_reward_v3 import LandingLabels,episode_reward
from positive_slope_common_v5 import read,write,sha
from positive_slope_runtime_v2 import parent_decision_steps,material_audit
from posture_episode_ppo import PosturePPO
base=old.base;parent_harness=old.parent_harness
repair=read(options.repair_config)
if options.repair_kind!='original':
    if options.repair_kind=='timing':from low_positive_timing import plan
    else:from low_positive_planner import plan
    parent_harness.traction.progressive.rolling_plan=lambda *a,**kw:plan(*a,**kw,inertial=parent_harness.traction.bridge.inertial,**repair)
parent_harness.traction.bridge.bridge_args.learning_start=options.feedback_start

class FeedbackEnv(old.PositiveEnv):
    def _physics_step(self,actions):
        result=super()._physics_step(actions)
        # Applied motor torque is also available as a sensor; full37-joint
        # integral is used only for the reward audit here.
        self.contact_records[-1]['total_squared_torque_Nm2']=self.robot.data.applied_torque.torch.square().sum(1).cpu().numpy()
        m,d,_=old.support.metrics(self.raw)
        self.contact_records[-1].update(root_speed=m['speed_m_s'].cpu().numpy(),
            gyro=m['angular_speed_rad_s'].cpu().numpy(),good=d['good'].cpu().numpy(),
            alive=self.alive.cpu().numpy().copy())
        return result

# The same measured-plane normal-landing controller as physical best09.
from low_seat_normal_timing_controller import install as install_landing
install_landing(sys.modules[__name__],read(options.landing_config))
from positive_slope_common_v4 import pulse_gain

def main():
    a=parent_harness.arguments();assert a.checkpoint
    assert 0<options.feedback_span<=.004 and options.feedback_rate>0
    out=Path(a.output);out.mkdir(exist_ok=False);(out/'source').mkdir()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    cfg,goals,resets=parent_harness.build(a,out)
    cal=read(options.feedback_calibration) if options.feedback_calibration else None
    if options.feedback_mode in ['train','evaluate']:assert cal is not None
    manifest=read(out/'manifest.json')
    matched_calibration=cal is not None and cal['bank_sha256']==manifest['bank_hashes'] and cal['repair_sha256']==sha(options.repair_config)
    if options.feedback_mode in ['train','explore'] and cal is not None:assert matched_calibration,'Reward centering requires the identical development bank and repair'
    manifest.update(feedback_args=vars(options),repair=repair,repair_sha256=sha(options.repair_config),
        parent_checkpoint_sha256=sha(a.checkpoint),feedback_checkpoint_sha256=sha(options.feedback_checkpoint) if options.feedback_checkpoint else None)
    for name in ['run_contact_learning_ppo_v3.py','contact_reward_v3.py','contact_reward_v1.py','feedback_slip_reward.py','feedback_posture_ppo.py','run_positive_slope_ppo_v4.py','low_positive_timing.py','low_positive_planner.py','posture_episode_ppo.py']:
        f=Path(__file__).with_name(name);manifest['sources'][str(f.resolve())]=sha(f);shutil.copy2(f,out/'source'/name)
    write(out/'manifest.json',manifest)
    with base.launch_simulation(cfg,a):
        raw=base.gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped
        env=FeedbackEnv(raw,a,out,goals,resets);write(out/'material_audit.json',material_audit(env))
        assert not any(x.any() for x in env.flags.values()) and (env.geometry[:,11]>0).all()
        parent=PosturePPO(np.zeros(414),np.ones(414),0,env.device);parent.load(a.checkpoint)
        for m in [parent.actor,parent.critic]:
            for w in m.parameters():w.requires_grad_(False)
        parent.save(out/'loaded_parent.pt')
        assert old.early.early_checkpoint and old.early.early_calibration
        early_parent=PosturePPO(np.zeros(414),np.ones(414),0,env.device)
        early_parent.load(old.early.early_checkpoint)
        early_norm=read(old.early.early_calibration)['stages']['6.0']
        early_parent.mean=torch.tensor(early_norm['mean'],device=env.device,dtype=torch.float32)
        early_parent.scale=torch.tensor(early_norm['scale'],device=env.device,dtype=torch.float32)
        for m in [early_parent.actor,early_parent.critic]:
            for w in m.parameters():w.requires_grad_(False)
        early_parent.save(out/'loaded_early_parent.pt')
        parent_hashes={p:sha(p) for p in [a.checkpoint,old.early.early_checkpoint]}
        write(out/'frozen_parents.json',parent_hashes)
        norm=cal['normalization'] if cal else dict(mean=np.zeros(417),scale=np.ones(417))
        learner=FeedbackPPO(norm['mean'],norm['scale'],options.feedback_seed,env.device)
        if options.feedback_checkpoint:
            learner.load(options.feedback_checkpoint);shutil.copy2(options.feedback_checkpoint,out/'loaded_feedback.pt')
            assert torch.equal(learner.mean,torch.tensor(norm['mean'],dtype=torch.float32,device=env.device))
            assert torch.equal(learner.scale,torch.tensor(norm['scale'],dtype=torch.float32,device=env.device))
        learner.save(out/'initial.pt')
        if options.feedback_rng_state:
            rng=torch.load(options.feedback_rng_state,map_location='cpu',weights_only=False)
            assert rng['updates']==learner.updates
            torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state_all(rng['cuda']);np.random.set_state(rng['numpy'])
        initial_updates=learner.updates
        label_geometry=LandingLabels(env.robot.body_names,
            [raw.cfg.geometry_bank[int(i)] for i in env.index.cpu().tolist()],raw.scene.env_origins.cpu().numpy())
        geometry_audits=[];reward_audits=[]
        count=options.feedback_updates if options.feedback_mode in ['calibrate','zero','explore','train'] else 1
        first_step=round(10*options.feedback_start);decision_steps=list(range(first_step,200,5))
        audits=[];batches=[];applied=[];sensors=[];parent_decisions=[];started=time.monotonic()
        for iteration in range(count):
            if iteration:
                with torch.inference_mode():
                    env._reset_rows(torch.arange(env.num_envs,device=env.device))
                    for _ in range(5):env.step(torch.zeros(env.num_envs,23,device=env.device))
                    assert not env.warming.any() and not env.episode_length_buf.any()
            before=len(env.episodes);residual=torch.zeros(env.num_envs,3,device=env.device)
            target=residual.clone();parent_target=residual.clone();early_target=residual.clone();handover=None
            po_saved=torch.zeros(env.num_envs,414,device=env.device);pa_saved=residual.clone()
            observations=[];actions=[];logps=[];means=[];cost=[];physical_cost=[];all_sensors=[];geometry_labels=[]
            for step in range(200):
                with torch.no_grad():
                    sensor=env.get_observations()['policy'].detach().clone()
                    assert sensor.shape==(env.num_envs,414)
                    all_sensors.append(sensor.cpu().numpy())
                    if step in decision_steps:
                        obs=torch.cat([sensor,residual/options.feedback_span],-1)
                        choice=learner.choose(obs,options.feedback_mode in ['explore','train'])
                        macro=choice[0]
                        if options.feedback_mode in ['zero','reference','calibrate']:macro=torch.zeros_like(macro)
                        target=options.feedback_span*torch.tanh(macro)
                        observations.append(obs);actions.append(macro);logps.append(choice[1]);means.append(choice[3])
                    if step==60:early_target=.001*torch.tanh(early_parent.choose(sensor,False)[0])
                    if step==110:handover=parent_decision_steps(env)
                    if step in [110,150]:
                        due=handover==step;pa=parent.choose(sensor,False)[0]
                        po_saved[due]=sensor[due];pa_saved[due]=pa[due]
                        if options.feedback_mode!='reference':parent_target[due]=.008*torch.tanh(pa[due])
                with torch.inference_mode():
                    if step>=first_step:residual=slew(residual,target,rate=options.feedback_rate,span=options.feedback_span)
                    frozen_target=early_target*pulse_gain(step/10,6.,1.)
                    if handover is not None:frozen_target=torch.where((step>=handover)[:,None],parent_target,frozen_target)
                    offset=residual+frozen_target
                    assert abs(offset).max()<=.01200001
                    applied.append(np.stack([residual.cpu().numpy().copy(),frozen_target.cpu().numpy().copy(),offset.cpu().numpy().copy()]))
                    env.step(old.mapped(env,offset))
                    geometry_labels.append(label_geometry(env.robot.data.body_pose_w.torch.cpu().numpy()))
                    cost.append(np.stack([r['total_squared_torque_Nm2'] for r in env.contact_records[-5:]]).sum(0)*.02)
                    physical_cost.append(np.stack([tick_reward(r,(step*5+j+1)*.02) for j,r in enumerate(env.contact_records[-5:])]).sum(0))
            rows=env.episodes[before:];assert len(rows)==env.num_envs and all(r['steps']==1000 for r in rows)
            integrals=torch.tensor(np.array(cost),device=env.device)
            baseline=torch.tensor(cal['torque_integrals'],device=env.device) if matched_calibration else torch.zeros_like(integrals)
            baseline_margin=cal['margin_ticks'] if matched_calibration else [0]*len(rows)
            penalty=torch.tensor([20*any(r['failures'].values())+10*(not r['success'])+5*(r['slip_ticks']>0)+5*(r['illegal_ticks']>0)+2*(r['traction_bad_ticks']>baseline_margin[i]) for i,r in enumerate(rows)],device=env.device)
            dense=torch.tensor(np.array(physical_cost),device=env.device)
            returns=reward_to_go(integrals,baseline,penalty,decision_steps,dense)
            original_returns=returns
            labels={k:np.array([v[k] for v in geometry_labels]) for k in geometry_labels[0]}
            records={k:np.array([r[k] for r in env.contact_records[-1000:]]) for k in env.contact_records[-1]}
            revised,control,reward_audit=episode_reward(records,labels,rows,baseline_margin,decision_steps)
            revised_returns=torch.tensor(revised,dtype=torch.float32,device=env.device)
            if options.reward_arm=='revised':returns=revised_returns
            geometry_audits.append(labels);reward_audits.append(reward_audit)
            obs=torch.stack(observations);act=torch.stack(actions);lp=torch.stack(logps);mu=torch.stack(means)
            batch=dict(obs=obs.cpu(),actions=act.cpu(),logp=lp.cpu(),old_mean=mu.cpu(),reward=returns.cpu(),torque_integrals=integrals.cpu(),baseline_integrals=baseline.cpu(),terminal_penalty=penalty.cpu(),physical_cost=dense.cpu(),decision_steps=decision_steps)
            batch.update(original_reward=original_returns.cpu(),revised_reward=revised_returns.cpu(),revised_control=torch.tensor(control),reward_arm=options.reward_arm)
            stats=learner.update(obs.flatten(0,1),act.flatten(0,1),lp.flatten(),mu.flatten(0,1),returns.flatten()) if options.feedback_mode=='train' else {}
            audits.append(dict(iteration=iteration,updates=learner.updates,active=env.active_transitions,warmup=env.warmup_transitions,decisions=len(decision_steps)*env.num_envs,stable=sum(r['success'] for r in rows),clean=sum(r['success'] and not any(r['failures'].values()) and not r['slip_ticks'] and not r['illegal_ticks'] for r in rows),slip=sum(r['slip_ticks']>0 for r in rows),illegal=sum(r['illegal_ticks']>0 for r in rows),physical=sum(any(r['failures'].values()) for r in rows),optimizer=stats))
            batches.append(batch);sensors.append(np.array(all_sensors));parent_decisions.append(dict(observation=po_saved.cpu().numpy(),action=pa_saved.cpu().numpy(),handover=handover.cpu().numpy()))
            write(out/'learning_audit.json',audits);write(out/'progress.json',dict(completed=iteration+1,updates=learner.updates,seconds=time.monotonic()-started,last=audits[-1]))
            if options.feedback_mode=='train':
                learner.save(out/f'model_{learner.updates:03d}.pt')
                torch.save(batch,out/f'batch_{learner.updates:03d}.pt')
                # Atomic status written only after complete checkpoint and batch.
                write(out/'weights_ready.json',dict(updates=learner.updates,checkpoint=f'model_{learner.updates:03d}.pt'))
            write(out/'reward_audit.json',reward_audits)
            if options.feedback_mode=='calibrate' and iteration==0:
                x=np.array(all_sensors)[60:].reshape(-1,414)
                write(out/'feedback_calibration.json',dict(normalization=dict(mean=np.r_[x.mean(0),np.zeros(3)].tolist(),scale=np.r_[np.maximum(x.std(0),.05),np.ones(3)].tolist()),torque_integrals=integrals.cpu().tolist(),margin_ticks=[r['traction_bad_ticks'] for r in rows],parent_sha256=sha(a.checkpoint),bank_sha256=manifest['bank_hashes'],repair_sha256=sha(options.repair_config),source='Development zero-residual full trajectory only; no holdout observations'))
        assert all(sha(p)==h for p,h in parent_hashes.items())
        learner.save(out/'final.pt');torch.save(batches,out/'decision_batches.pt')
        torch.save(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),updates=learner.updates),out/'training_rng_state.pt')
        np.savez_compressed(out/'reward_geometry.npz',**{k:np.array([x[k] for x in geometry_audits]) for k in geometry_audits[0]})
        np.savez_compressed(out/'applied_offsets.npz',offsets=np.array(applied))
        np.savez_compressed(out/'sensor_inputs.npz',sensors=np.array(sensors),parent_observation=np.array([x['observation'] for x in parent_decisions]),parent_action=np.array([x['action'] for x in parent_decisions]),handover=np.array([x['handover'] for x in parent_decisions]))
        env.close();assert sha(a.checkpoint)==manifest['parent_checkpoint_sha256']
        for name in ['manifest.json','sensor_audit.json']:
            f=out/name;value=read(f);value.update(observations=417,channels=417,base_sensor_channels=414,controller_memory_channels=3,feedback_contract='Actual actor/critic: 414 sensor/reference inputs plus three previous normalized learned residuals. No contact/pose truth.');write(f,value)
        write(out/'feedback_contract.json',dict(mode=options.feedback_mode,start_s=options.feedback_start,inputs=417,actions=3,decision_period_s=.5,command_period_s=.1,span_rad=options.feedback_span,slew_rad_s=options.feedback_rate,reference_assisted=True,parent_frozen=True,parent_sha256=sha(a.checkpoint),parent_enabled=options.feedback_mode!='reference',initial_updates=initial_updates,final_updates=learner.updates,local_updates=learner.updates-initial_updates,active=env.active_transitions,warmup=env.warmup_transitions,total=env.active_transitions+env.warmup_transitions,initial_settling_excluded=25*env.num_envs,decisions=count*len(decision_steps)*env.num_envs,feedback_window_ticks=count*env.num_envs*(1000-first_step*5),teacher_prefix_ticks=count*env.num_envs*first_step*5,reward='undiscounted complete-episode reward-to-go; all37 torque squared; physical terminal penalties; no bootstrap'))
        contract=read(out/'feedback_contract.json');contract.update(reward=('Revised v3: clean completion 50, bounded shaping, causal first-event costs, scale 50; exact equations in frozen contact_reward_v3.py' if options.reward_arm=='revised' else 'Original v1 reward unchanged'), reward_arm=options.reward_arm, revised_reward_source_sha256=sha(Path(__file__).with_name('contact_reward_v3.py')), original_reward_description='Contact reward v1: legal stationary support saturating at0.2BW and good-rest duration from8s; effort coefficient0.25 on all37 torque squared; inherited dense loaded-slip, wrong-contact and terminal penalties; complete undiscounted reward-to-go; no bootstrap',learning_scope='Known failing development cases are included in optimization. Exploration failures are recorded, not a physical safety certificate. Selection must preserve baseline clean cases; explicit parent fallback.');write(out/'feedback_contract.json',contract)
        print('CONTACT_LEARNING_COMPLETE',audits[-1],flush=True)

if __name__=='__main__':main()
