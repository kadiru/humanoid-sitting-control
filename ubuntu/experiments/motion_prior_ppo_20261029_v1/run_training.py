"""Full standing-start evaluation and PPO, one bounded4090 process."""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
from common import ROOT,OBS,read,write,sha
sys.path.insert(1,str(ROOT/'scripts'))
import run_resting_posture as base
import run_supported_postures as support
from environment import FullBodyEnv
from policy import Learner,advantages

def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['evaluate','train'],required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--demonstrations');p.add_argument('--seed',type=int,required=True);p.add_argument('--updates',type=int,default=256)
    base.add_launcher_args(p);a=p.parse_args();assert 1<=a.updates<=256
    torch.manual_seed(a.seed);np.random.seed(a.seed);a.enable_cameras=True;a.depth_check=True
    bankpath=Path(a.bank);meta=read(bankpath/'metadata.json');geometry=read(bankpath/'geometry.json')
    with np.load(bankpath/'bank.npz',allow_pickle=False) as z:bank={k:z[k] for k in z.files}
    a.num_envs=72;assert meta['cases']==72
    out=Path(a.output);out.mkdir(exist_ok=False)
    write(out/'manifest.json',dict(args=vars(a),bank_sha256=sha(bankpath/'bank.npz'),metadata=meta,
        action_limits_unchanged=True,physical_criteria_unchanged=True,controller='sensor-selected demonstration motion prior plus23 learned residuals'))
    cfg=support.make_cfg(a,geometry);cfg.scene.rest_camera=None;started=time.monotonic()
    with base.launch_simulation(cfg,a):
        raw=base.gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped
        env=FullBodyEnv(raw,a,out,bank,meta)
        try:
            write(out/'startup.json',env.startup)
            assert not any(env.startup['warm_failure']) and all(env.startup['sensor_valid'])
            agent=Learner(np.zeros(OBS),np.ones(OBS),a.seed,a.device);agent.load(a.checkpoint)
            with torch.no_grad():
                _,choice,dist=agent.prior(env.observation(),True)
            write(out/'initial_selection.json',dict(library_indices=choice.cpu().tolist(),distances=dist.cpu().tolist(),
                fixed_by_latched_initial_sensors=True,case_ids_used=False))
            if a.mode=='evaluate':
                with torch.inference_mode():
                    for tick in range(1000):
                        env.step(agent.choose(env.observation(),False)[0],record=True)
                        if tick%100==99:print(dict(tick=tick+1,alive=int(env.alive.sum())),flush=True)
                env.save_records(out/'trajectory.npz');return
            assert a.demonstrations and agent.updates==0
            with np.load(a.demonstrations,allow_pickle=False) as z:agent.bind_demonstrations(z['obs'])
            agent.save(out/'initial.pt');stalled=0
            for update in range(a.updates):
                b={k:[] for k in ['obs','action','logp','mean','reward','value','next','done','learn']}
                with torch.no_grad():
                    for _ in range(64):
                        obs=env.observation();action,logp,value,mean=agent.choose(obs)
                        nxt,reward,done,learn,horizon=env.step(action)
                        nv=agent.choose(nxt,False)[2]
                        for k,v in dict(obs=obs,action=action,logp=logp,mean=mean,reward=reward,value=value,next=nv,done=done,learn=learn).items():b[k].append(v.clone())
                        if horizon or not bool(env.alive.any()):
                            if not horizon:
                                with (out/'aborted_training_attempts.jsonl').open('a') as f:f.write(__import__('json').dumps(dict(attempt=env.attempt,steps=int(env.episode_length_buf[0]),reason='all_failed',clean=False))+'\n')
                            assert env.attempt<128,'RESET_BUDGET'
                            check=env.reset(0);assert not any(check['warm_failure']) and all(check['sensor_valid'])
                b={k:torch.stack(v) for k,v in b.items()};adv,target=advantages(b['reward'],b['value'],b['next'],b['done'])
                select=b['learn'];assert int(select.sum())>=2
                m=agent.update(b['obs'][select],b['action'][select],b['logp'][select],b['mean'][select],adv[select],target[select])
                stalled=stalled+1 if m['actor_steps']==0 else 0
                m.update(update=update+1,consecutive_no_policy_change=stalled,live_samples=int(select.sum()),mean_rollout_reward=float(b['reward'].mean()),
                    seconds=time.monotonic()-started,complete_episodes=len(env.episodes),active_transitions=env.active_transitions)
                with (out/'updates.jsonl').open('a') as f:f.write(__import__('json').dumps(m)+'\n')
                write(out/'status.json',dict(state='training',**m));print({k:v for k,v in m.items() if k!='candidate_tests'},flush=True)
                if update==0:
                    agent.save(out/'first_update.pt');torch.save({k:v.cpu() for k,v in b.items()},out/'first_rollout.pt')
                    from audit import weights
                    weights(out);write(out/'weights_ready.json',dict(checkpoint='first_update.pt'))
                if update+1 in [16,64,128,256]:agent.save(out/f'update_{update+1:03d}.pt')
                assert stalled<8,'POLICY_UPDATE_STALLED_EXPLICITLY_REPORTED'
            agent.save(out/'final.pt')
            from audit import weights
            weights(out,'final.pt')
            write(out/'status.json',dict(state='complete',updates=agent.updates,actor_steps=agent.actor_steps,critic_steps=agent.critic_steps,seconds=time.monotonic()-started))
        finally:env.close()

if __name__=='__main__':main()
