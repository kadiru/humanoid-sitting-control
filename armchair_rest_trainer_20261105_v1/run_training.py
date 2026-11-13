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
    p.add_argument('--mesh',action='store_true');p.add_argument('--demonstrations');p.add_argument('--seed',type=int,required=True);p.add_argument('--updates',type=int,default=128)
    p.add_argument('--hold-effort-reward', action='store_true')
    p.add_argument('--candidate-library')
    base.add_launcher_args(p);a=p.parse_args();assert 1<=a.updates<=128
    torch.manual_seed(a.seed);np.random.seed(a.seed);a.enable_cameras=True;a.depth_check=True
    bankpath=Path(a.bank);meta=read(bankpath/'metadata.json');geometry=None if a.mesh else read(bankpath/'geometry.json')
    with np.load(bankpath/'bank.npz',allow_pickle=False) as z:bank={k:z[k] for k in z.files}
    a.num_envs=meta['cases'];assert 1<=a.num_envs<=72
    import os
    assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
    assert torch.cuda.device_count()==1 and '4090' in torch.cuda.get_device_name(0)
    out=Path(a.output);out.mkdir(exist_ok=False)
    write(out/'manifest.json',dict(args=vars(a),bank_sha256=sha(bankpath/'bank.npz'),metadata=meta,
        action_limits_unchanged=True,physical_criteria_unchanged=True,controller='sensor-selected demonstration motion prior plus23 learned residuals'))
    adapter=None
    if a.mesh:
        sys.path[1:1]=[str(ROOT/'experiments/furniture_perception_repair_20261029_v1'),str(ROOT/'experiments/furniture_test_20261028_v1'),str(ROOT/'experiments/furniture_validation_20261028_v2')]
        from capture import Observer
        from mesh_batch import MeshMetrics,make_cfg
        import run_supported_transition as transition
        case=next(c for c in read(ROOT/'experiments/furniture_test_20261028_v1/cases.json') if c['id']=='myd1WSucAz_rear_back')
        observer=Observer('back');transition.observed_support=observer
        adapter=MeshMetrics(case,support.metrics,support,record=False);support.metrics=adapter
        cfg=make_cfg(a,case)
    else:cfg=support.make_cfg(a,geometry)
    cfg.scene.rest_camera=None;started=time.monotonic()
    with base.launch_simulation(cfg,a):
        raw=base.gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped
        env=FullBodyEnv(raw,a,out,bank,meta)
        try:
            write(out/'startup.json',env.startup)
            assert not any(env.startup['warm_failure']) and all(env.startup['sensor_valid'])
            agent=Learner(np.zeros(OBS),np.ones(OBS),a.seed,a.device);agent.load(a.checkpoint)
            if a.candidate_library:
                assert a.mode == 'evaluate' and meta['task'] == 'sitting'
                with np.load(a.candidate_library, allow_pickle=False) as z:
                    agent.bind_sitting({**agent.sitting, **dict(z)})
            with torch.no_grad():
                _,choice,dist=agent.prior(env.observation(),True)
            write(out/'initial_selection.json',dict(library_indices=choice.cpu().tolist(),distances=dist.cpu().tolist(),
                fixed_by_latched_initial_sensors=True,case_ids_used=False))
            if a.mode=='evaluate':
                with torch.inference_mode():
                    for tick in range(1000):
                        obs=env.observation();desired=agent.choose(obs,False)[0];previous=env.previous.cpu().numpy().copy()
                        env.step(desired,record=True)
                        env.episode_records[-1].update(obs=obs.cpu().numpy(),previous=previous,desired=desired.cpu().numpy())
                        if tick%100==99:print(dict(tick=tick+1,alive=int(env.alive.sum())),flush=True)
                env.save_records(out/'trajectory.npz')
                if adapter:write(out/'contact_mapping.json',dict(validated=adapter.validated.tolist(),calls=adapter.calls))
                return
            assert a.demonstrations and agent.updates==0
            with np.load(a.demonstrations,allow_pickle=False) as z:agent.bind_demonstrations(z['obs'])
            agent.save(out/'initial.pt');stalled=0
            approach_controls = 0

            @torch.no_grad()
            def execute_frozen_approach():
                if 'residual_start_tick' not in agent.library:
                    return 0
                last_start = int(agent.library['residual_start_tick'].max())
                for tick in range(last_start+1):
                    obs = env.observation()
                    if (agent.residual_mask(obs)[:, 0].bool() & env.alive).any():
                        return tick
                    assert env.alive.any(), 'FROZEN_APPROACH_FAILED'
                    if tick < last_start:
                        env.step(agent.choose(obs, False)[0])
                raise RuntimeError('NO_LIVE_SUPPORTED_LEAN_REACHED_LEARNING_PHASE')

            approach_controls += execute_frozen_approach()
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
                            if adapter:adapter.reset_recording(env.num_envs)
                            check=env.reset(0);assert not any(check['warm_failure']) and all(check['sensor_valid'])
                            approach_controls += execute_frozen_approach()
                b={k:torch.stack(v) for k,v in b.items()};adv,target=advantages(b['reward'],b['value'],b['next'],b['done'])
                select=b['learn'];assert int(select.sum())>=2
                m=agent.update(b['obs'][select],b['action'][select],b['logp'][select],b['mean'][select],adv[select],target[select])
                stalled=stalled+1 if m['actor_steps']==0 else 0
                m.update(update=update+1,consecutive_no_policy_change=stalled,live_samples=int(select.sum()),mean_rollout_reward=float(b['reward'].mean()),
                    seconds=time.monotonic()-started,complete_episodes=len(env.episodes),active_transitions=env.active_transitions,
                    frozen_approach_controls=approach_controls)
                with (out/'updates.jsonl').open('a') as f:f.write(__import__('json').dumps(m)+'\n')
                write(out/'status.json',dict(state='training',**m));print({k:v for k,v in m.items() if k!='candidate_tests'},flush=True)
                if update==0:
                    agent.save(out/'first_update.pt');torch.save({k:v.cpu() for k,v in b.items()},out/'first_rollout.pt')
                    from audit import weights
                    weights(out);write(out/'weights_ready.json',dict(checkpoint='first_update.pt'))
                if update+1 in [16,64,128]:agent.save(out/f'update_{update+1:03d}.pt')
                assert stalled<8,'POLICY_UPDATE_STALLED_EXPLICITLY_REPORTED'
            agent.save(out/'final.pt')
            from audit import weights
            weights(out,'final.pt')
            write(out/'status.json',dict(state='complete',updates=agent.updates,actor_steps=agent.actor_steps,critic_steps=agent.critic_steps,seconds=time.monotonic()-started))
        finally:env.close()

if __name__=='__main__':main()
