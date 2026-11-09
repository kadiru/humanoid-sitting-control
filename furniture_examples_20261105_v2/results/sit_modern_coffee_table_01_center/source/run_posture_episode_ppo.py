"""Standing-reference motion followed by one sensor-conditioned posture choice.

Imports the completed traction controller unchanged. This separate experiment
uses whole-episode PPO credit, small symmetric sagittal offsets, and a direct
unclipped all-joint torque objective. Physical labels never enter the networks.
"""
import argparse,hashlib,json,sys,time,shutil
from pathlib import Path
import numpy as np
import torch

extra=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
extra.add_argument('--episode-mode',choices=['calibrate','zero','explore','train','evaluate'],required=True)
extra.add_argument('--episode-updates',type=int,default=16)
extra.add_argument('--episode-calibration')
extra.add_argument('--episode-span',type=float,default=.008)
option,rest=extra.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_traction_transfer_ppo as traction
from posture_episode_ppo import PosturePPO,episode_return
base=traction.progressive.base;support=traction.progressive.support
ROOT=Path('/data/humanoid-sitting-perception/posture_credit_01')
OLD=ROOT.parent/'traction_transfer_01'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,x):base.write_json(path,x)

class EpisodeEnv(traction.TractionEnv):
    def __init__(self,*args,**kwargs):
        self.post11_margin=None
        super().__init__(*args,**kwargs)
        self.post11_margin=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
    def _reset_rows(self,ids):
        if self.post11_margin is not None:self.post11_margin[ids]=0
        super()._reset_rows(ids)
    def _physics_step(self,actions):
        active=(self.warming==0)&(self.motion_time()>=11)
        result=super()._physics_step(actions)
        if self.post11_margin is not None:
            # Metrics are reward/report labels only, not controller inputs.
            d=support.metrics(self.raw)[1]
            self.post11_margin+=(~d['traction_ok'])&active
        return result

def arguments():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--bank',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',required=True);p.add_argument('--surface',default='sitting')
    p.add_argument('--profiles',required=True);p.add_argument('--plane-calibration',required=True)
    p.add_argument('--seed',type=int,default=59101);p.add_argument('--start-stage',default='0')
    p.add_argument('--num-envs',type=int,default=18);p.add_argument('--curriculum-stage',type=int,default=0)
    p.add_argument('--reset-library');p.add_argument('--checkpoint');base.add_launcher_args(p)
    a=p.parse_args();a.invocation_mode=a.mode;a.mode='evaluate'
    a.depth_check=True;a.enable_cameras=True;a.seconds=20.;a.objective='legacy';a.inertial_model=None
    a.segment_seconds=2.;a.diagnostics=False;a.reset_index=0
    assert a.curriculum_stage==0 and a.start_stage=='0'
    return a

def build(a,out):
    geometry=json.loads((Path(a.bank)/'geometry.json').read_text());profiles=json.loads(Path(a.profiles).read_text())
    original_goals=dict(np.load(Path(a.bank)/'candidates.npz'));original_resets=dict(np.load(Path(a.bank)/'reset_candidates.npz'))
    cases=[(i,p) for p in profiles for i,g in enumerate(geometry) if g['kind']=='sit'];gs=[]
    for i,p in cases:gs.append(dict(geometry[i],case=len(gs),surface_id=i,start_stage=0,control_profile=p,effort_scale_Nm2=1500.))
    goals={k:np.array([v[i] for i,p in cases]) for k,v in original_goals.items()}
    resets={k:np.array([v[i] for i,p in cases]) for k,v in original_resets.items()}
    cfg=support.make_cfg(a,gs);cfg.scene.rest_camera=None
    write(out/'geometry.json',gs)
    paths=[Path(__file__),Path(__file__).with_name('posture_episode_ppo.py')]
    old=json.loads((OLD/'ppo_01/protocol.json').read_text())
    sources={**old['hashes'],**{str(p.resolve()):sha(p) for p in paths}}
    for p in paths:shutil.copy2(p,out/'source'/p.name)
    write(out/'manifest.json',dict(args=vars(a),episode_args=vars(option),sources=sources,observations=414,actions=3,
          old_protocol_sha256=sha(OLD/'ppo_01/protocol.json'),bank_hashes={p.name:sha(p) for p in Path(a.bank).iterdir() if p.is_file()}))
    return cfg,goals,resets

def mapped(env,macro):
    action=torch.zeros(env.num_envs,23,device=env.device)
    offset=option.episode_span*torch.tanh(macro)
    for j,joint in enumerate(['hip_pitch','knee','ankle_pitch']):
        for s in ['left','right']:
            i=base.BODY_JOINT_NAMES.index(s+'_'+joint+'_joint')
            action[:,i]=torch.atanh((offset[:,j]/env.span[i]).clamp(-.999999,.999999))
    return action,offset

def main():
    a=arguments();out=Path(a.output);out.mkdir(exist_ok=False);(out/'source').mkdir()
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    cfg,goals,resets=build(a,out)
    with base.launch_simulation(cfg,a):
        raw=base.gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped
        env=EpisodeEnv(raw,a,out,goals,resets)
        assert not any(x.any() for x in env.flags.values()) and (env.geometry[:,11]>0).all()
        cal=json.loads(Path(option.episode_calibration).read_text()) if option.episode_calibration else None
        learner=PosturePPO(cal['mean'] if cal else np.zeros(414),cal['scale'] if cal else np.ones(414),a.seed,env.device)
        if a.checkpoint:learner.load(a.checkpoint)
        learner.save(out/'initial.pt')
        full=option.episode_mode in ['train','explore'];count=option.episode_updates if full else 1
        records=[];started=time.monotonic();history=[];decision_inputs=[];actions=[]
        for iteration in range(count):
            if iteration:
                with torch.inference_mode():
                    env._reset_rows(torch.arange(env.num_envs,device=env.device))
                    for _ in range(5):env.step(torch.zeros(env.num_envs,23,device=env.device))
                    assert not env.warming.any() and not env.episode_length_buf.any()
            before=len(env.episodes);macro=torch.zeros(env.num_envs,3,device=env.device)
            obs=None;choice=None;applied=None;offset=None
            for step in range(200):
                with torch.no_grad():
                    if step==110:
                        assert (env.episode_length_buf==550).all()
                        obs=env.get_observations()['policy'].detach().clone()
                        assert obs.shape==(env.num_envs,414)
                        if option.episode_mode in ['train','explore','evaluate']:
                            choice=learner.choose(obs,option.episode_mode!='evaluate');macro=choice[0]
                        else:choice=learner.choose(obs,False);macro.zero_()
                        applied,offset=mapped(env,macro);decision_inputs.append(obs.cpu().numpy());actions.append(macro.cpu().numpy())
                with torch.inference_mode():
                    env.step(applied if step>=110 else torch.zeros(env.num_envs,23,device=env.device))
            rows=env.episodes[before:];assert len(rows)==env.num_envs
            for i,row in enumerate(rows):row['post11_margin_ticks']=int(env.post11_margin[i])
            assert all(row['steps']==1000 for row in rows)
            if option.episode_mode=='calibrate':
                x=obs.cpu().numpy();baseline=[row['metrics_whole_trial']['total_squared_torque_Nm2'] for row in rows]
                write(out/'episode_calibration.json',dict(mean=x.mean(0).tolist(),scale=np.maximum(x.std(0),.05).tolist(),baseline_whole20_Nm2=baseline,
                    sensor_channels=414,source='Development reference at11s; reward baseline never supplied to actor/critic',reference_results_sha256=None))
            baseline=cal['baseline_whole20_Nm2'] if cal and len(cal['baseline_whole20_Nm2'])==len(rows) else [0.]*len(rows)
            reward=torch.tensor(episode_return(rows,baseline),device=env.device,dtype=torch.float32)
            stats=learner.update(obs,macro,choice[1],choice[3],reward) if option.episode_mode=='train' else {}
            records.append(dict(iteration=iteration,physics_transitions=env.active_transitions+env.warmup_transitions,
                active=env.active_transitions,warmup=env.warmup_transitions,stable=sum(x['success'] for x in rows),
                strict=sum(x['strict_transfer'] for x in rows),slip=sum(x['slip_ticks']>0 for x in rows),
                illegal=sum(x['illegal_ticks']>0 for x in rows),physical=sum(any(x['failures'].values()) for x in rows),
                post11_margin=sum(x['post11_margin_ticks'] for x in rows),
                mean_whole20_torque2=float(np.mean([x['metrics_whole_trial']['total_squared_torque_Nm2'] for x in rows])),
                mean_last5_torque2=float(np.mean([x['metrics_last5s']['total_squared_torque_Nm2'] for x in rows])),
                offsets_rad=offset.cpu().tolist(),macro_actions=macro.cpu().tolist(),episode_return=reward.cpu().tolist(),optimizer=stats))
            history.append(dict(obs=obs.cpu(),actions=macro.cpu(),logp=choice[1].cpu(),old_mean=choice[3].cpu(),reward=reward.cpu()))
            write(out/'progress.json',dict(completed_episodes_per_case=iteration+1,updates=learner.updates,seconds=time.monotonic()-started,last=records[-1]))
            write(out/'learning_audit.json',records)
            if option.episode_mode=='train':learner.save(out/f'model_{iteration+1:02d}.pt')
            # Stop a preflight exploration on real physical instability. Main
            # training retains all failures and completes its frozen budget.
            if option.episode_mode=='explore' and any(x['slip_ticks'] or x['illegal_ticks'] or any(x['failures'].values()) or not x['success'] or x['post11_margin_ticks'] for x in rows):break
        learner.save(out/'final.pt');torch.save(history,out/'decision_batches.pt')
        np.savez_compressed(out/'episode_decisions.npz',observation=np.array(decision_inputs),action=np.array(actions))
        env.close()
        write(out/'episode_contract.json',dict(options=vars(option),sensor_inputs=414,macro_actions=3,names=['symmetric_hip_pitch','symmetric_knee','symmetric_ankle_pitch'],
          span_rad=option.episode_span,decision_time_s=11.,held_until_s=20.,initial_settle_excluded=25*env.num_envs,
          all_starts_standing=True,fingers_and_other17_body_residuals_zero=True,baseline_reward_only=True,updates=learner.updates,
          active=env.active_transitions,warmup=env.warmup_transitions,transitions=env.active_transitions+env.warmup_transitions,
          episodes=len(env.episodes),optimizer_actor_steps=sum(x['optimizer'].get('actor_steps',0) for x in records),
          optimizer_critic_steps=sum(x['optimizer'].get('critic_steps',0) for x in records)))
        if option.episode_mode=='calibrate':
            f=out/'episode_calibration.json';x=json.loads(f.read_text());x['reference_results_sha256']=sha(out/'results.json');write(f,x)
        print('EPISODE_STUDY_COMPLETE',json.dumps(records[-1]),flush=True)

if __name__=='__main__':main()
