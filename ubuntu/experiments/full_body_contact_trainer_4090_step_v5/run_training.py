"""Bounded simulator entry point, launched by the single finite coordinator."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from common import ROOT, read, write, sha
sys.path.insert(1,str(ROOT/'scripts'))
import run_resting_posture as base
import run_supported_postures as support
from environment import FullBodyEnv
from policy import Learner, advantages


def curriculum(update):
    return 3 if update<32 else 2 if update<64 else 1 if update<128 else 0


def main():
    p=argparse.ArgumentParser(); p.add_argument('--bank',required=True); p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['preflight','teacher','evaluate','train'],required=True)
    p.add_argument('--checkpoint'); p.add_argument('--reset-audit'); p.add_argument('--seed',type=int,required=True)
    p.add_argument('--updates',type=int,default=256); base.add_launcher_args(p); a=p.parse_args()
    assert 1<=a.updates<=256; np.random.seed(a.seed); torch.manual_seed(a.seed)
    a.enable_cameras=True; a.depth_check=True; out=Path(a.output); out.mkdir(exist_ok=False)
    bankpath=Path(a.bank); meta=read(bankpath/'metadata.json'); geometry=read(bankpath/'geometry.json')
    with np.load(bankpath/'bank.npz',allow_pickle=False) as z:bank={k:z[k] for k in z.files}
    a.num_envs=meta['cases']; assert a.num_envs in [18,72]
    reset_valid=None
    if a.reset_audit:
        reset=read(a.reset_audit); assert reset['passed'] and reset['bank_sha256']==sha(bankpath/'bank.npz')
        reset_valid=np.array(reset['valid'],bool)
    if a.mode=='train':assert reset_valid is not None
    cfg=support.make_cfg(a,geometry); cfg.scene.rest_camera=None
    write(out/'manifest.json',dict(args=vars(a),bank_sha256=sha(bankpath/'bank.npz'),metadata=meta,
          controller='23 independent filtered/rate-limited joint increments; fixed fingers',success_criteria_unchanged=True))
    started=time.monotonic()
    with base.launch_simulation(cfg,a):
        raw=base.gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped
        env=FullBodyEnv(raw,a,out,bank,meta,reset_valid)
        try:
            write(out/'startup.json',env.startup)
            assert not any(env.startup['warm_failure']), 'STANDING_RESET_PHYSICAL_FAILURE'
            assert all(env.startup['sensor_valid']), 'STANDING_RESET_SENSOR_INVALID'
            if a.mode=='preflight':
                # Validate every eligible recorded reset; failed rows fall back
                # to the separately validated original standing state.
                valid=np.zeros((a.num_envs,4),bool); valid[:,0]=True; checks=[env.startup]
                for slot in [1,2,3]:
                    env.reset_valid[:,slot]=env.admitted
                    check=env.reset(slot); checks.append(check)
                    valid[:,slot]=env.admitted&~np.array(check['warm_failure'])&np.array(check['sensor_valid'])
                assert all(valid[env.admitted,1:].any(0)), 'NO_VALID_RESET_FOR_A_CURRICULUM_STAGE'
                write(out/'reset_audit.json',dict(passed=True,valid=valid.tolist(),checks=checks,
                      bank_sha256=sha(bankpath/'bank.npz'),true_velocities=True,interpolated_states=False,
                      failed_reference_resets_used=False,full_start_always_available=True))
                return
            agent=Learner(np.zeros(336),np.ones(336),a.seed,a.device)
            if a.checkpoint:agent.load(a.checkpoint)
            if a.mode in ['train','evaluate']:assert a.checkpoint
            if a.mode in ['teacher','evaluate']:
                with torch.inference_mode():
                    for tick in range(1000):
                        action=agent.choose(env.observation(),False)[0]
                        teacher=torch.as_tensor(bank['command'][tick],device=env.device) if a.mode=='teacher' else None
                        env.step(action,teacher_command=teacher,record=True)
                        if tick%100==99:print(json.dumps(dict(tick=tick+1,alive=int(env.alive.sum()))),flush=True)
                env.save_records(out/'trajectory.npz'); return
            # Initial checkpoint is independently compared with actual updates.
            agent.save(out/'initial.pt'); update0=agent.updates; assert update0==0
            first=env.reset(curriculum(0)); assert not any(first['warm_failure'])
            for update in range(a.updates):
                batch={k:[] for k in ['obs','action','logp','mean','reward','value','next','done','learn']}
                with torch.no_grad():
                    for _ in range(64):
                        obs=env.observation(); action,logp,value,mean=agent.choose(obs)
                        next_obs,reward,done,learn,horizon=env.step(action)
                        next_value=agent.choose(next_obs,False)[2]
                        for k,v in dict(obs=obs,action=action,logp=logp,mean=mean,reward=reward,value=value,next=next_value,done=done,learn=learn).items():batch[k].append(v.clone())
                        if horizon or not bool(env.alive.any()):
                            # Early all-dead training attempts are failures, not
                            # successful full-duration evaluations. Reset them
                            # rather than optimizing batches of dead frames.
                            if not horizon:
                                with (out/'aborted_training_attempts.jsonl').open('a') as f:
                                    f.write(json.dumps(dict(attempt=env.attempt,steps=int(env.episode_length_buf[0]),reason='all_rows_failed',clean=False))+'\n')
                            assert env.attempt<128, 'TRAINING_RESET_BUDGET_EXHAUSTED'
                            check=env.reset(curriculum(update)); assert not any(check['warm_failure']), 'VALIDATED_RESET_CHANGED'
                b={k:torch.stack(v) for k,v in batch.items()}; adv,target=advantages(b['reward'],b['value'],b['next'],b['done'])
                select=b['learn']; assert int(select.sum())>=2, 'INSUFFICIENT_LIVE_TRAINING_SAMPLES'
                metrics=agent.update(b['obs'][select],b['action'][select],b['logp'][select],b['mean'][select],adv[select],target[select])
                metrics.update(update=update+1,mean_rollout_reward=float(b['reward'].mean()),live_samples=int(select.sum()),seconds=time.monotonic()-started,
                    active_transitions=env.active_transitions,warmup_transitions=env.warmup_transitions,complete_episodes=len(env.episodes),curriculum_slot=curriculum(update))
                with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
                write(out/'status.json',dict(state='training',**metrics)); print(json.dumps(metrics),flush=True)
                if update==0:
                    agent.save(out/'first_update.pt')
                    torch.save({k:v.cpu() for k,v in b.items()},out/'first_rollout.pt')
                    assert metrics['actor_steps']>0
                    from audit import weights
                    weights(out)
                    write(out/'weights_ready.json',dict(checkpoint='first_update.pt',independent_audit='first_update_audit.json'))
                if update+1 in [64,128,256]:agent.save(out/f'update_{update+1:03d}.pt')
            agent.save(out/'final.pt')
            from audit import weights
            weights(out,'final.pt')
            write(out/'status.json',dict(state='complete',updates=agent.updates,active_transitions=env.active_transitions,
                  warmup_transitions=env.warmup_transitions,seconds=time.monotonic()-started,weight_updates_verified=True))
        finally:env.close()


if __name__=='__main__':main()
