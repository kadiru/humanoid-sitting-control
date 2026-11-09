"""CPU-only, read-only analysis; no simulator, optimizer step, or GPU access."""
import collections
import datetime
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
ROOT = Path('/home/kadir/humanoid-sitting')
OUT = ROOT/'outputs/recovery_imitation_sitting_20261028_v3'
OLD = ROOT/'outputs/recovery_imitation_sitting_20261028_v2'
SRC = ROOT/'experiments/recovery_imitation_sitting_20261028_v3'
sys.path.insert(0, str(SRC))
from policy import Learner

def read(p):
    return json.loads(p.read_text())

def lines(p):
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

def stats(a):
    a = np.asarray(a)
    return dict(min=float(a.min()), median=float(np.median(a)), max=float(a.max()), mean=float(a.mean())) if a.size else None

train = OUT/'train_sitting_64111'
manifest = read(train/'manifest.json')
bankpath = Path(manifest['args']['bank'])
meta = read(bankpath/'metadata.json')
with np.load(bankpath/'bank.npz', allow_pickle=False) as z:
    protected = z['admitted'].astype(bool)
    bankq = z['q_post'][:, :, :23]
    bankcommand = z['command'][:, :, :23]
    bankprevious = z['previous'][:, :, :23]
report = dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              scope='Saved first-seed arithmetic and CPU network inference only; no simulator, training, or GPU work.',
              provenance=dict(manifest=manifest, source_sha256={n:hashlib.sha256((SRC/n).read_bytes()).hexdigest() for n in ['policy.py','environment.py','run_training.py','rewards.py','clone.py']}))
u = lines(train/'updates.jsonl')
report['optimizer'] = dict(updates=len(u), actor_steps=sum(x['actor_steps'] for x in u),
    critic_steps=sum(x['critic_steps'] for x in u), updates_with_actor_change=[x['update'] for x in u if x['actor_steps']],
    updates_without_actor_change=sum(not x['actor_steps'] for x in u),
    rejected_proposals=sum(x['rejected_kl_steps'] for x in u),
    rejected_last_candidate_KL_under_limit=sum(x['rejected_kl_steps'] and x['last_candidate_kl']<=.02 for x in u),
    final_anchor=u[-1]['anchor_loss'], anchor_ceiling=u[-1]['anchor_ceiling'],
    final_anchor_headroom_fraction=1-u[-1]['anchor_loss']/u[-1]['anchor_ceiling'],
    rejection_reason_limit='Logging combines objective/KL/anchor failures as rejected_kl_steps; rejected candidate anchor/objective values are not saved, so exact reason counts cannot be recovered.',
    stages={})
for start,end in [(1,32),(33,64),(65,128),(129,256)]:
    part=[x for x in u if start<=x['update']<=end]
    report['optimizer']['stages'][f'{start}-{end}']=dict(actor_steps=sum(x['actor_steps'] for x in part),
        live_fraction=sum(x['live_samples'] for x in part)/(len(part)*64*72),
        rollout_reward_mean=float(np.mean([x['mean_rollout_reward'] for x in part])),
        imitation_loss=stats([x['imitation_loss'] for x in part]),
        candidate_kl=stats([x['last_candidate_kl'] for x in part]))
episodes=lines(train/'episodes.jsonl')
report['training_episodes']={str(t):dict(count=len(a), clean=sum(x['clean'] for x in a),stable=sum(x['stable'] for x in a),
    final_hold_s=stats([x['final_hold_s'] for x in a])) for t in sorted({x['start_time_s'] for x in episodes})
    for a in [[x for x in episodes if x['start_time_s']==t]]}
report['aborted_training_attempts']=lines(train/'aborted_training_attempts.jsonl')
agents={}
for name,filename in [('initial','initial.pt'),('64','update_064.pt'),('128','update_128.pt'),('256','update_256.pt')]:
    a=Learner(np.zeros(336),np.ones(336),64111,'cpu')
    a.load(train/filename)
    a.actor.eval()
    agents[name]=a
report['weight_differences']={}
for left,right in [('initial','64'),('64','128'),('128','256')]:
    ds=[(p-q).double().square().sum().item() for p,q in zip(agents[left].actor.parameters(),agents[right].actor.parameters())]
    report['weight_differences'][left+'->'+right]=dict(actor_l2=sum(ds)**.5,actor_bit_identical=all(torch.equal(p,q) for p,q in zip(agents[left].actor.parameters(),agents[right].actor.parameters())))
with np.load(manifest['args']['demonstrations'],allow_pickle=False) as z:
    dx=z['obs'];dy=z['target']
report['demonstration_fit']={}
with torch.inference_mode():
    for name in ['initial','256']:
        agent=agents[name]; errors=[];changes=[];command_errors=[]
        for i in range(0,len(dx),4096):
            x=torch.from_numpy(dx[i:i+4096]);y=dy[i:i+4096]
            pred=agent.actor(agent.normalize(x)).numpy()
            prior=dx[i:i+4096,222+75:222+98]
            errors.append(pred-y)
            command_errors.append(np.clip(.1*(pred-prior),-.003,.003)-np.clip(.1*(y-prior),-.003,.003))
        error=np.concatenate(errors);command_error=np.concatenate(command_errors)
        report['demonstration_fit'][name]=dict(samples=len(dx),desired_rmse_rad=float(np.sqrt(np.mean(error.astype('float64')**2))),
            desired_max_error_rad=float(abs(error).max()),applied_command_rmse_rad=float(np.sqrt(np.mean(command_error.astype('float64')**2))),
            normalized_imitation_mse=float(np.mean((error.astype('float64')/.03)**2)),
            by_source={})
        for label,start,end in [('original',0,64000),('recovery0',64000,127000),('recovery1',127000,len(dx))]:
            e=error[start:end]
            report['demonstration_fit'][name]['by_source'][label]=dict(samples=len(e),rmse_rad=float(np.sqrt(np.mean(e.astype('float64')**2))))
report['recovery_collection']=[]
for roundn in [0,1]:
    p=OLD/f'recovery_64111_{roundn}'
    with np.load(p/'trajectory.npz',allow_pickle=False) as z:
        learner=z['recovery_learner'][:,0]
        longest=current=0
        for bit in learner:
            current=current+1 if bit else 0
            longest=max(longest,current)
        report['recovery_collection'].append(dict(round=roundn,learner_ticks=int(learner.sum()),total_ticks=len(learner),
            teacher_fraction=float((~learner).mean()),max_uninterrupted_learner_s=longest*.02))

report['evaluations']={}
for label,p in [('clone',OLD/'clone_evaluation_64111'),('128',OUT/'eval_sitting_64111_128'),('256',OUT/'eval_sitting_64111_256')]:
    rows=read(p/'results.json')['episodes']
    audit=read(p/'independent_audit.json');assert audit['passed']
    names=['first_failure','flags','good','held_ticks','learn','alive','legal','no_slip','phase','q_error','speed','gyro','loaded_slip','wrong_force','peak_bw','normalized_command_delta','support_bw','torque_squared','clean_completion','stable_completion','reward','q','command','force','physical_inside_patch','physical_permitted','physical_trunk_tilt_rad']
    with np.load(p/'trajectory.npz',allow_pickle=False) as z:
        a={k:z[k] for k in names}
    cases=[];firsttimes=[];categories=collections.Counter();contacts=collections.Counter()
    for i,row in enumerate(rows):
        hits=np.flatnonzero(a['first_failure'][:,i]);t=int(hits[0]) if len(hits) else None
        reasons=[]
        if t is not None:
            firsttimes.append((t+1)*.02)
            reasons=[n for n,v in zip(['fall','forbidden_contact','torque','impact','joint_margin'],a['flags'][t,i]) if v]
            categories.update(reasons)
            if 'impact' in reasons:contacts.update([str(int(np.linalg.norm(a['force'][t,i],axis=-1).argmax()))])
        cases.append(dict(id=row['id'],protected=bool(protected[i]),clean=row['clean'],stable=row['stable'],
            first_failure_s=(t+1)*.02 if t is not None else None,first_reasons=reasons,
            peak_bw_at_first_failure=float(a['peak_bw'][t,i]) if t is not None else None,
            wrong_N_at_first_failure=float(a['wrong_force'][t,i]) if t is not None else None,
            maximum_good_hold_s=float(a['held_ticks'][:,i].max()*.02),final_hold_s=row['final_hold_s'],
            first_slip_s=(np.flatnonzero(~a['no_slip'][:,i])[0]+1)*.02 if (~a['no_slip'][:,i]).any() else None))
    safe=a['alive']&a['legal'];phase=a['phase'];good=safe&a['good'];live=a['learn']
    terms=dict(imitation=np.where(phase==0,2.,.5)*np.exp(-a['q_error'])*safe,
        contact=(phase>=1)*safe*a['no_slip']*np.exp(-(a['speed']/.1)**2-(a['gyro']/.25)**2)*np.clip(a['support_bw'],0,.2)/.2,
        settling=(phase>=2)*3*good,hold=(phase==3)*2*good,
        effort=.2*(phase==3)*good*(a['held_ticks']>=250)/(1+a['torque_squared']/1500),
        slip_cost=-2*(a['loaded_slip']/.02)**2,wrong_cost=-2*(a['wrong_force']/10)**2,
        impact_cost=-2*np.maximum(a['peak_bw']-1,0)**2,
        smooth_cost=-.02*np.mean(a['normalized_command_delta']**2,axis=-1))
    returns={k:.02*v*live for k,v in terms.items()}
    returns['terminal']=(400*a['clean_completion']+20*a['stable_completion']-20*a['first_failure'])*live
    reconstructed=sum(returns.values())
    err=abs(reconstructed-a['reward'])
    drift={}
    for seconds in [.02,1,2,4,6,8]:
        t=max(0,int(round(seconds/.02))-1)
        mask=protected&live[t]
        qe=a['q'][t,:,:23]-bankq[t];ce=a['command'][t,:,:23]-bankcommand[t]
        drift[str(seconds)]=dict(alive_protected_count=int(mask.sum()),
            pose_rmse_rad=float(np.sqrt(np.mean(qe[mask].astype('float64')**2))) if mask.any() else None,
            command_rmse_rad=float(np.sqrt(np.mean(ce[mask].astype('float64')**2))) if mask.any() else None,
            biggest_command_errors=[dict(joint=meta['joint_names'][j],rmse_rad=float(np.sqrt(np.mean(ce[mask,j].astype('float64')**2)))) for j in np.argsort(np.mean(ce[mask]**2,axis=0))[-4:][::-1]] if mask.any() else [] )
    ev=dict(clean=sum(r['clean'] for r in rows),stable=sum(r['stable'] for r in rows),
        physical_failures=len(firsttimes),first_failure_seconds=stats(firsttimes),first_failure_categories_overlap=dict(categories),
        impact_peak_channel=dict(contacts),good_ticks=int(a['good'].sum()),
        any_good_cases=int(a['good'].any(0).sum()),maximum_good_hold_s=float(a['held_ticks'].max()*.02),
        live_command_rate_saturation_fraction=float((abs(a['normalized_command_delta'][live])>=.9999).mean()),
        drift_before_failure=drift,mean_return=float(a['reward'].sum(0).mean()),
        reward_components_mean_return={k:float(v.sum(0).mean()) for k,v in returns.items()},
        reward_reconstruction_max_error=float(err.max()),cases=cases,
        sensor_summary={})
    auditframes=read(p/'sensor_audit.json')['truth_validation']
    before=[f for f in auditframes if a['learn'][min(999,max(0,round(f['t']/.02)-1)),f['case']]]
    for field in ['offset_error_m','normal_error_deg']:
        ev['sensor_summary'][field]=stats([abs(f[field]) for f in before if f[field] is not None])
    ev['sensor_summary']['invalid_before_failure']=sum(not f['valid'] for f in before)
    report['evaluations'][label]=ev

report['latest_second_run']=read(OUT/'train_sitting_64117/status.json')
print(json.dumps(report,indent=2,allow_nan=False))
