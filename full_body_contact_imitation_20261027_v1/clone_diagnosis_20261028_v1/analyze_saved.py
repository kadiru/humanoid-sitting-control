"""Read-only, CPU-only analysis of completed files; no simulator imports."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import json, sys, hashlib
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
R=Path('/home/kadir/humanoid-sitting')
sys.path.insert(0,str(R/'experiments/full_body_contact_trainer_4090_step_v5'))
from policy import Learner, command
v4=R/'outputs/full_body_contact_imitation_v4'
v5=R/'outputs/full_body_contact_imitation_v5'
bank=np.load(v4/'dataset/sitting/bank.npz')
meta=json.loads((v4/'dataset/sitting/metadata.json').read_text())
positive=bank['admitted']; ids=meta['joint_ids']
obs=bank['obs']; target=bank['bc_target']; previous=bank['previous'][...,:23]
teacher=bank['command'][...,:23]; qteacher=bank['q_post'][...,:23]
out={'scope':'Saved-array arithmetic only; no new physical trial or training modification', 'evaluations':{},'offline_fit':{},'training':{}}
def stats(x):
    x=np.asarray(x);return {'min':float(x.min()),'median':float(np.median(x)),'max':float(x.max()),'mean':float(x.mean())}
lower=np.array(meta['lower'])+.03; upper=np.array(meta['upper'])-.03
# Exact envelope of current filtered action map, not only rate and box tests.
reachable_lo=previous+.1*(np.maximum(previous-.03,lower)-previous)
reachable_hi=previous+.1*(np.minimum(previous+.03,upper)-previous)
violation=np.maximum(np.maximum(reachable_lo-teacher,teacher-reachable_hi),0)
out['action_map_reachability']={'positive_max_violation_rad':float(violation[:,positive].max()),'positive_channels_above_2e_7':int((violation[:,positive]>2e-7).sum())}
for seed in [64111,64117]:
    agent=Learner(np.zeros(336),np.ones(336),seed);agent.load(v4/f'bc_sitting_{seed}/initial.pt')
    flat=obs.reshape(-1,336); mus=[]
    with torch.no_grad():
        for start in range(0,len(flat),2048):mus.append(agent.choose(torch.from_numpy(flat[start:start+2048]),False)[0].numpy())
    mu=np.concatenate(mus).reshape(1000,72,23)
    applied=command(torch.from_numpy(previous),torch.from_numpy(mu),torch.tensor(meta['lower']),torch.tensor(meta['upper'])).numpy()
    err=applied-teacher
    # Oracle-state sum is an accumulation diagnostic, not a rollout prediction.
    accumulated=np.cumsum(err,axis=0)
    perjoint=np.sqrt(np.mean(err[:,positive]**2,axis=(0,1)))
    rec={'normalized_target_mse_positive':float(((np.tanh(mu)-target)[:,positive]**2).mean()),
         'applied_command_rmse_rad':float(np.sqrt((err[:,positive]**2).mean())),
         'oracle_state_integrated_error_6s_rad':stats(abs(accumulated[299,positive])),
         'oracle_state_integrated_error_10s_rad':stats(abs(accumulated[499,positive])),
         'worst_joint_rmse':[{'joint':meta['joint_names'][j],'rmse_rad':float(perjoint[j])} for j in np.argsort(perjoint)[-6:][::-1]],
         'normalization_scale_min':float(agent.scale.min())}
    # Same saved teacher inputs through final PPO policy: how much imitation is retained?
    cp=v5/f'train_sitting_{seed}/final.pt'
    if cp.exists():
        agent.load(cp); pred=[]
        with torch.no_grad():
            for start in range(0,len(flat),2048):pred.append(agent.choose(torch.from_numpy(flat[start:start+2048]),False)[0].numpy())
        rec['final_PPO_target_mse_positive']=float(((np.tanh(np.concatenate(pred).reshape(1000,72,23))-target)[:,positive]**2).mean())
    out['offline_fit'][str(seed)]=rec

for path in sorted(v5.glob('eval_sitting_*')):
    if not path.is_dir() or not (path/'independent_audit.json').exists():continue
    audit=json.loads((path/'independent_audit.json').read_text()); data=np.load(path/'trajectory.npz')
    alive=data['alive']; flags=data['flags']; first=np.where((~alive).any(0),np.argmax(~alive,axis=0),1000)
    failures=[]
    for i,t in enumerate(first):
        if t<1000:failures.append({'id':meta['ids'][i],'first_failure_s':float((t+1)*.02),'flags':[n for n,b in zip(['fall','forbidden','torque','impact','joint_limit'],flags[t,i]) if b]})
    early={}
    for t in [0,49,149,299,399,499]:
        early[str((t+1)*.02)]={'alive':int(alive[t].sum()),'q_rmse_vs_teacher_rad':float(np.sqrt(((data['q'][t,positive,:23]-qteacher[t,positive])**2).mean())), 'command_rmse_vs_teacher_rad':float(np.sqrt(((data['command'][t,positive,:23]-teacher[t,positive])**2).mean()))}
    cumulative={name:int(flags[-1,:,j].sum()) for j,name in enumerate(['fall','forbidden','torque','impact','joint_limit'])}
    result=json.loads((path/'results.json').read_text())
    qfirst=data['q'][np.minimum(first,999),np.arange(72),:23]
    joint_bad=((qfirst<=np.array(meta['lower'])+.005)|(qfirst>=np.array(meta['upper'])-.005))&(first<1000)[:,None]
    out['evaluations'][path.name]={'clean':int(sum(audit['clean'])),'stable':int(sum(audit['stable'])),
        'repairs':int((np.asarray(audit['clean'])&~positive).sum()),'regressions':int((~np.asarray(audit['clean'])&positive).sum()),
        'survival_seconds_censored_at_20':stats(np.minimum(first+1,1000)*.02),
        'mean_total_reward_per_case':float(data['reward'].sum(0).mean()),'mean_live_reward':float(data['reward'][data['learn']].mean()),
        'total_good_ticks':int(data['good'].sum()),'max_final_hold_s':max(r['final_hold_s'] for r in result['episodes']),
        'cumulative_flags':cumulative,'time_alignment_diagnostics':early,'first_failures':failures,
        'joint_limits_at_first_failure':{meta['joint_names'][j]:int(v) for j,v in enumerate(joint_bad.sum(0)) if v},
        'trajectory_sha256':audit['files']['trajectory.npz']}
    data.close()

for seed in [64111,64117]:
    p=v5/f'train_sitting_{seed}';f=p/'updates.jsonl'
    if not f.exists():continue
    rows=[json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    phases={}
    for slot in [3,2,1,0]:
        rr=[r for r in rows if r['curriculum_slot']==slot]
        if rr:phases[str(slot)]={'updates':len(rr),'mean_reward_first8':float(np.mean([r['mean_rollout_reward'] for r in rr[:8]])), 'mean_reward_last8':float(np.mean([r['mean_rollout_reward'] for r in rr[-8:]])), 'actor_steps':sum(r['actor_steps'] for r in rr),'zero_actor_updates':sum(r['actor_steps']==0 for r in rr),'mean_live_samples':float(np.mean([r['live_samples'] for r in rr]))}
    ep=p/'episodes.jsonl';episodes=[json.loads(x) for x in ep.read_text().splitlines()] if ep.exists() else []
    out['training'][str(seed)]={'latest_update':rows[-1]['update'],'phases':phases,'complete_episode_rows':len(episodes),'clean_training_episodes':sum(x['clean'] for x in episodes),'stable_training_episodes':sum(x['stable'] for x in episodes),'latest':rows[-1]}
print(json.dumps(out,indent=2,allow_nan=False))
