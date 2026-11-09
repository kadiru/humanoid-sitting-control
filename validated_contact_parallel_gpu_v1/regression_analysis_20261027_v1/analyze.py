"""CPU-only analysis of immutable completed PPO trajectories; never trains or simulates."""
import os
os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import sys, json, time, hashlib, datetime
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(1)
ROOT = Path('/home/kadir/humanoid-sitting')
sys.path.insert(0, str(ROOT/'scripts'))
from feedback_posture_ppo import FeedbackPPO
D = Path('/data/humanoid-sitting-perception/low_seat_contact_01')
M = D/'contact_reward_comparison_128_v4'
OUT = D/'contact_parallel_gpu_v1/regression_analysis_20261027_v1'
start = time.monotonic()
bindings = {}

def bind(p):
    p = Path(p)
    if str(p) not in bindings:
        bindings[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    assert time.monotonic()-start < 180, 'Analysis wall cap'
    return p

def read(p):
    return json.loads(bind(p).read_text())

def clean(r):
    return bool(r['success'] and not any(r['failures'].values()) and not r['slip_ticks'] and not r['illegal_ticks'])

def first(mask, dt=.02):
    x = np.flatnonzero(mask)
    return float((x[0]+1)*dt) if len(x) else None

def stats(values):
    x = np.asarray(values, dtype=float)
    return dict(n=x.size, mean=float(x.mean()), min=float(x.min()), max=float(x.max()), std=float(x.std())) if x.size else dict(n=0)

def casekey(bank, case):
    return f'{bank}/{int(case)}'

selection = read(M/'training_bank/selection.json')['cases']
train_keys = [casekey(c['bank'],2*c['index']+j) for c in selection for j in [0,1]]
base = {}
for bank in ['known_old','known_transfer','known_recent']:
    for r in read(D/'current_retention_09'/bank/'candidate/results.json')['episodes']:
        base[casekey(bank,r['case'])] = r
evaluations = {}
actions = {}
for arm in ['original','revised']:
    for stage in [32,64]:
        tag = f'{arm}_{stage}'
        evaluations[tag] = {}
        action_list = []
        for bank in ['known_old','known_transfer','known_recent']:
            p = M/f'eval_{arm}_63111_{stage:03d}_{bank}'
            assert read(p/'independent_audit.json')['passed']
            rows = read(p/'results.json')['episodes']
            reward = read(p/'reward_audit.json')[0]
            batch = torch.load(bind(p/'decision_batches.pt'),map_location='cpu',weights_only=False)[0]
            with np.load(bind(p/'every_tick.npz')) as x:
                ticks = {k:x[k] for k in ['force','weight','slip','no_slip','good','failure','legal','wrong','traction']}
            with np.load(bind(p/'applied_offsets.npz')) as x:offsets=x['offsets'].copy()
            with np.load(bind(p/'reward_geometry.npz')) as x:geometry={k:x[k][0] for k in x.files}
            loaded = np.linalg.norm(ticks['force'],axis=-1) > .01*ticks['weight'][...,None]
            speed = np.where(loaded,ticks['slip'],0).max(-1)
            original_slip_cost = .02*500*np.maximum(speed-.02,0).sum(0)
            for j,r in enumerate(rows):
                key=casekey(bank,r['case']); comps={k:float(v[j]) for k,v in reward['component_totals_raw'].items()};events={k:float(v[j]) for k,v in reward['event_totals_raw'].items()}
                full=float(reward['full_episode_return'][j]);total=(sum(comps.values())+sum(events.values())+reward['terminal_raw'][j])/50
                assert abs(full-total)<2e-7,(tag,key,full,total)
                assert int((~ticks['no_slip'][:,j]).sum())==r['slip_ticks']
                slip_first=first(~ticks['no_slip'][:,j]); peak=int(np.argmax(speed[:,j]))
                i=max(0,min(199,int((slip_first or 12.0)*10)-1))
                evaluations[tag][key]=dict(clean=clean(r),baseline_clean=clean(base[key]),in_training=key in train_keys,profile=r['profile'],success=r['success'],failures=r['failures'],slip_ticks=r['slip_ticks'],illegal_ticks=r['illegal_ticks'],traction_bad_ticks=r['traction_bad_ticks'],first_slip_s=slip_first,first_physical_failure_s=first(ticks['failure'][:,j]),first_good_s=first(ticks['good'][:,j]),final_hold_s=r['final_hold_s'],peak_loaded_tangent_speed_m_s=float(speed[:,j].max()),peak_speed_time_s=float((peak+1)*.02),loaded_speed_clipped_ticks_after6=int((speed[300:,j]>=.02).sum()),original_dense_slip_cost=float(original_slip_cost[j]),revised_full_return=full,original_first_return=float(batch['original_reward'][0,j]),revised_first_return=float(batch['revised_reward'][0,j]),components_raw=comps,events_raw=events,terminal_raw=float(reward['terminal_raw'][j]),residual_peak_rad=float(np.abs(offsets[:,0,j]).max()),residual_at_first_slip_or12s_rad=offsets[i,0,j].tolist(),residual_at_10s_rad=offsets[99,0,j].tolist(),housing_min_after6_m=float(geometry['housing_vertex_clearance_m'][60:,j].min()),effort_whole=r['metrics_whole_trial']['total_squared_torque_Nm2'])
            action_list.append(offsets[100:130,0])
        a=np.concatenate(action_list,axis=1)
        actions[tag]=dict(landing_10to13s_max_abs_rad=float(np.abs(a).max()),mean_same_time_case_std_rad=float(a.std(1).mean()),max_same_time_case_range_rad=float(np.ptp(a,axis=1).max()),mean_channels_rad=a.mean((0,1)).tolist())

regressions={tag:sorted(k for k,r in v.items() if r['baseline_clean'] and not r['clean']) for tag,v in evaluations.items()}
repaired={tag:sorted(k for k,r in v.items() if not r['baseline_clean'] and r['clean']) for tag,v in evaluations.items()}
changed={arm:dict(new_regressions=sorted(set(regressions[arm+'_64'])-set(regressions[arm+'_32'])),restored_cases=sorted(set(regressions[arm+'_32'])-set(regressions[arm+'_64'])),persistent_regressions=sorted(set(regressions[arm+'_32'])&set(regressions[arm+'_64']))) for arm in ['original','revised']}
focus=sorted(set(regressions['revised_64'])|set(regressions['original_64']))
training={};learning=[]
for arm in ['original','revised']:
    for stage in [16,32,64]:
        p=M/f'train_{arm}_63111_{stage:03d}'
        assert read(p/'independent_audit.json')['passed']
        rows=read(p/'results.json')['episodes'];rewards=read(p/'reward_audit.json');logs=read(p/'learning_audit.json')
        assert len(rows)==24*len(rewards)==24*len(logs)
        block={}
        for j,key in enumerate(train_keys):
            rr=rows[j::24]
            assert all(int(r['case'])==j for r in rr)
            full=[x['full_episode_return'][j] for x in rewards]
            block[key]=dict(n=len(rr),clean=sum(clean(x) for x in rr),slip_ticks=stats([x['slip_ticks'] for x in rr]),revised_return=stats(full),failure_returns=stats([v for v,r in zip(full,rr) if not clean(r)]),clean_returns=stats([v for v,r in zip(full,rr) if clean(r)]),terminal_raw=stats([x['terminal_raw'][j] for x in rewards]),component_raw={k:stats([x['component_totals_raw'][k][j] for x in rewards]) for k in rewards[0]['component_totals_raw']})
        training[f'{arm}_{stage}']=dict(cases=block,mean_clean_training_24=float(np.mean([r['clean'] for r in logs])),mean_own_reward=float(np.mean([r['optimizer']['reward_mean'] for r in logs])),mean_actor_steps=float(np.mean([r['optimizer']['actor_steps'] for r in logs])))
        if stage!=64:continue
        # Inference only on the stored batches and preceding CPU-loaded critics.
        learner=FeedbackPPO(torch.zeros(417),torch.ones(417),0,'cpu')
        for it,log in enumerate(logs):
            update=log['updates'];prior=p/'initial.pt' if it==0 else p/f'model_{update-1:03d}.pt'
            learner.load(bind(prior));b=torch.load(bind(p/f'batch_{update:03d}.pt'),map_location='cpu',weights_only=False)
            with torch.inference_mode():value=learner.critic(learner.inputs(b['obs'].reshape(-1,417))).reshape(28,24)
            target=b['reward'];raw=target-value;adv=(raw-raw.mean())/(raw.std(unbiased=False)+1e-8)
            ev=1-float((target-value).var(unbiased=False)/(target.var(unbiased=False)+1e-12))
            for j,key in enumerate(train_keys):
                learning.append(dict(arm=arm,update=update,key=key,clean=clean(rows[it*24+j]),slip_ticks=rows[it*24+j]['slip_ticks'],raw_advantage_prelanding_mean=float(raw[:10,j].mean()),normalized_advantage_prelanding_mean=float(adv[:10,j].mean()),positive_advantage_fraction_prelanding=float((adv[:10,j]>0).float().mean()),normalized_advantage_late_mean=float(adv[14:,j].mean()),revised_first_return=float(b['revised_reward'][0,j]),original_first_return=float(b['original_reward'][0,j]),critic_explained_variance_in_batch=ev))

signals={}
for arm in ['original','revised']:
    signals[arm]={}
    for key in train_keys:
        rr=[x for x in learning if x['arm']==arm and x['key']==key]
        signals[arm][key]={'all':{k:stats([x[k] for x in rr]) for k in ['normalized_advantage_prelanding_mean','positive_advantage_fraction_prelanding','normalized_advantage_late_mean']},'nonclean':{k:stats([x[k] for x in rr if not x['clean']]) for k in ['normalized_advantage_prelanding_mean','positive_advantage_fraction_prelanding','normalized_advantage_late_mean']},'clean':{k:stats([x[k] for x in rr if x['clean']]) for k in ['normalized_advantage_prelanding_mean','positive_advantage_fraction_prelanding','normalized_advantage_late_mean']}}

result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),classification='Saved trajectory and component arithmetic plus frozen-critic CPU inference; no simulation, optimizer steps, source changes, GPU work or interruption',scope=dict(seed=63111,evaluation_episodes=288,training_episodes=3072,critic_inference_batches=64),training_case_keys=train_keys,regressions=regressions,repairs=repaired,case_churn=changed,focus_case_comparison={k:{tag:v[k] for tag,v in evaluations.items()} for k in focus},evaluations=evaluations,action_case_spread=actions,training=training,advantage_signals=signals,all_preupdate_learning_rows=learning,seconds=time.monotonic()-start,bindings=bindings,limits=dict(cpu_threads=1,wall_cap_s=180,no_live_rollout_reads=True),physical_repair_claimed=False)
assert len(base)==72 and len(train_keys)==24
for tag,v in evaluations.items():
    cs=[x['revised_full_return'] for x in v.values() if x['clean']];fs=[x['revised_full_return'] for x in v.values() if not x['clean']]
    assert min(cs)>max(fs)
OUT.mkdir(exist_ok=True)
with (OUT/'saved_analysis.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'seconds':result['seconds'],'scope':result['scope'],'regressions':regressions,'case_churn':changed,'focus':{k:{tag:{n:r[n] for n in ['in_training','clean','slip_ticks','first_slip_s','peak_loaded_tangent_speed_m_s','revised_full_return','original_dense_slip_cost','residual_peak_rad']} for tag,r in v.items()} for k,v in result['focus_case_comparison'].items()},'actions':actions}))
