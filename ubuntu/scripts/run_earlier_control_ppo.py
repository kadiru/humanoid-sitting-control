"""Train a separate early action while keeping the resting policy immutable.

The reference remains feedback-controlled. Simulator truth is labels only.
"""
import argparse,sys,json,time,shutil
from pathlib import Path
import numpy as np
import torch
extra=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
extra.add_argument('--early-mode',choices=['calibrate','zero','explore','train','evaluate'],required=True)
extra.add_argument('--early-start',type=float,choices=[10.,9.,8.,6.],default=10.)
extra.add_argument('--early-seed',type=int,required=True)
extra.add_argument('--early-span',type=float,default=.001)
extra.add_argument('--early-updates',type=int,default=1)
extra.add_argument('--early-calibration');extra.add_argument('--early-checkpoint')
early,rest=extra.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_posture_episode_ppo as parent_harness
from posture_episode_ppo import PosturePPO
from earlier_control_common import STAGES,sha,write,read
base=parent_harness.base;support=parent_harness.support
# The inherited parser accepts11; the new explicit contract defines earlier times.
parent_harness.traction.bridge.bridge_args.learning_start=early.early_start

def reward(rows,baseline):
    values=[]
    for i,row in enumerate(rows):
        cost=(row['metrics_whole_trial']['total_squared_torque_Nm2']-baseline['whole20_Nm2'][i])*20/(9*150)
        penalty=20*any(row['failures'].values())+10*(not row['success'])+5*(row['slip_ticks']>0)+5*(row['illegal_ticks']>0)
        penalty+=2*(row['traction_bad_ticks']>baseline['margin_ticks'][i])
        values.append(-cost-penalty)
    return values

def mapped(env,offset):
    action=torch.zeros(env.num_envs,23,device=env.device)
    for j,joint in enumerate(['hip_pitch','knee','ankle_pitch']):
        for side in ['left','right']:
            i=base.BODY_JOINT_NAMES.index(side+'_'+joint+'_joint')
            action[:,i]=torch.atanh((offset[:,j]/env.span[i]).clamp(-.999999,.999999))
    return action

def main():
    a=parent_harness.arguments();assert a.checkpoint
    out=Path(a.output);out.mkdir(exist_ok=False);(out/'source').mkdir()
    cfg,goals,resets=parent_harness.build(a,out)
    cal=read(early.early_calibration) if early.early_calibration else None
    manifest=read(out/'manifest.json');manifest.update(early_args=vars(early),parent_checkpoint_sha256=sha(a.checkpoint),early_checkpoint_sha256=sha(early.early_checkpoint) if early.early_checkpoint else None)
    for name in ['run_earlier_control_ppo.py','earlier_control_common.py']:
        p=Path(__file__).with_name(name);manifest['sources'][str(p.resolve())]=sha(p);shutil.copy2(p,out/'source'/name)
    write(out/'manifest.json',manifest)
    with base.launch_simulation(cfg,a):
        raw=base.gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped
        env=parent_harness.EpisodeEnv(raw,a,out,goals,resets)
        assert not any(x.any() for x in env.flags.values()) and (env.geometry[:,11]>0).all()
        parent=PosturePPO(np.zeros(414),np.ones(414),0,env.device);parent.load(a.checkpoint)
        for model in [parent.actor,parent.critic]:
            for p in model.parameters():p.requires_grad_(False)
        parent.save(out/'loaded_parent.pt')
        norm=cal['stages'][str(early.early_start)] if cal else dict(mean=np.zeros(414),scale=np.ones(414))
        learner=PosturePPO(norm['mean'],norm['scale'],early.early_seed,env.device)
        if early.early_checkpoint:
            learner.load(early.early_checkpoint);shutil.copy2(early.early_checkpoint,out/'loaded_early.pt')
            learner.mean=torch.tensor(norm['mean'],device=env.device,dtype=torch.float32);learner.scale=torch.tensor(norm['scale'],device=env.device,dtype=torch.float32)
        learner.save(out/'initial.pt')
        count=early.early_updates if early.early_mode in ['calibrate','explore','train'] else 1
        records=[];history=[];stage_obs={str(t):[] for t in STAGES};early_obs=[];early_actions=[];parent_obs=[];parent_actions=[]
        initial_updates=learner.updates;started=time.monotonic()
        for iteration in range(count):
            if iteration:
                with torch.inference_mode():
                    env._reset_rows(torch.arange(env.num_envs,device=env.device))
                    for _ in range(5):env.step(torch.zeros(env.num_envs,23,device=env.device))
                    assert not env.warming.any() and not env.episode_length_buf.any()
            before=len(env.episodes);offset=torch.zeros(env.num_envs,3,device=env.device)
            for step in range(200):
                with torch.no_grad():
                    if step in [int(t*10) for t in STAGES]:
                        observation=env.get_observations()['policy'].detach().clone();stage_obs[str(step/10)].append(observation.cpu().numpy())
                    if step==int(early.early_start*10):
                        obs=env.get_observations()['policy'].detach().clone();assert obs.shape==(env.num_envs,414)
                        choice=learner.choose(obs,early.early_mode in ['train','explore']);macro=choice[0]
                        if early.early_mode in ['zero','calibrate']:macro=torch.zeros_like(macro)
                        offset=early.early_span*torch.tanh(macro);early_obs.append(obs.cpu().numpy());early_actions.append(macro.cpu().numpy())
                    if step==110:
                        po=env.get_observations()['policy'].detach().clone();pa=parent.choose(po,False)[0]
                        offset=.008*torch.tanh(pa);parent_obs.append(po.cpu().numpy());parent_actions.append(pa.cpu().numpy())
                with torch.inference_mode():env.step(mapped(env,offset))
            rows=env.episodes[before:];assert len(rows)==env.num_envs and all(x['steps']==1000 for x in rows)
            baseline=cal['baseline'] if cal and len(cal['baseline']['whole20_Nm2'])==len(rows) else dict(whole20_Nm2=[0.]*len(rows),margin_ticks=[0]*len(rows))
            returns=torch.tensor(reward(rows,baseline),device=env.device,dtype=torch.float32)
            stats=learner.update(obs,macro,choice[1],choice[3],returns) if early.early_mode=='train' else {}
            records.append(dict(iteration=iteration,updates=learner.updates,active=env.active_transitions,warmup=env.warmup_transitions,stable=sum(x['success'] for x in rows),strict=sum(x['strict_transfer'] for x in rows),slip=sum(x['slip_ticks']>0 for x in rows),illegal=sum(x['illegal_ticks']>0 for x in rows),physical=sum(any(x['failures'].values()) for x in rows),offsets_rad=(early.early_span*torch.tanh(macro)).cpu().tolist(),optimizer=stats))
            history.append(dict(obs=obs.cpu(),actions=macro.cpu(),logp=choice[1].cpu(),old_mean=choice[3].cpu(),reward=returns.cpu()))
            write(out/'learning_audit.json',records);write(out/'progress.json',dict(completed=iteration+1,global_updates=learner.updates,seconds=time.monotonic()-started,last=records[-1]))
            if early.early_mode=='train':learner.save(out/f'model_{iteration+1:02d}.pt')
            if early.early_mode=='calibrate' and iteration==0:
                calibration=dict(stages={key:dict(mean=values[0].mean(0).tolist(),scale=np.maximum(values[0].std(0),.05).tolist()) for key,values in stage_obs.items()},baseline=dict(whole20_Nm2=[x['metrics_whole_trial']['total_squared_torque_Nm2'] for x in rows],margin_ticks=[x['traction_bad_ticks'] for x in rows]),parent_sha256=sha(a.checkpoint),runtime='414 sensor/reference inputs only; physical baselines are reward labels only')
                write(out/'early_calibration.json',calibration)
        learner.save(out/'final.pt');torch.save(history,out/'decision_batches.pt')
        np.savez_compressed(out/'episode_decisions.npz',early_observation=np.array(early_obs),early_action=np.array(early_actions),parent_observation=np.array(parent_obs),parent_action=np.array(parent_actions))
        np.savez_compressed(out/'stage_observations.npz',**{key:np.array(values) for key,values in stage_obs.items()})
        env.close();assert sha(a.checkpoint)==manifest['parent_checkpoint_sha256']
        write(out/'early_contract.json',dict(mode=early.early_mode,start_s=early.early_start,handover_s=11.,early_span=early.early_span,parent_span=.008,inputs=414,early_actions=3,parent_actions=3,parent_frozen=True,parent_checkpoint=a.checkpoint,parent_sha256=sha(a.checkpoint),calibration=early.early_calibration,initial_updates=initial_updates,final_updates=learner.updates,local_updates=learner.updates-initial_updates,episodes=len(env.episodes),active=env.active_transitions,warmup=env.warmup_transitions,total=env.active_transitions+env.warmup_transitions,initial_settle_excluded=25*env.num_envs,early_optimizer_actor_steps=sum(x['optimizer'].get('actor_steps',0) for x in records),early_optimizer_critic_steps=sum(x['optimizer'].get('critic_steps',0) for x in records),fixed_fingers_and_other_body_residuals=True))
        print('EARLIER_CONTROL_COMPLETE',json.dumps(records[-1]),flush=True)
if __name__=='__main__':main()
