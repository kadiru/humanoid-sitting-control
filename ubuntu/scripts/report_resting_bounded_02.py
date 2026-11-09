"""Matched fixed/bounded-PPO/hold comparison; audit before writing conclusions."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('study');args=p.parse_args()
root=Path(args.study);out=root/'results'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mean(v):return float(np.mean(v)) if len(v) else None
protocol=read(root/'protocol.json');banks=('starts','heights','impulse','novel')
seeds=protocol['seeds'];variants=protocol['runtime_variants']
assert all(sha(Path(p))==digest for p,digest in read(root/'frozen/manifest.json').items())

def get(name,n):
    rows=read(root/name/'results.json')['episodes']
    assert len(rows)==n and [r['case'] for r in rows]==list(range(n))
    assert all(r['steps']==1000 and r['duration_s']==20 for r in rows)
    return rows

def summary(rows):
    good=[r for r in rows if r['success']]
    return dict(successes=len(good),total=len(rows),mean_good_fraction=mean([r['good_fraction'] for r in rows]),
        failures={k:sum(r['failures'][k] for r in rows) for k in rows[0]['failures']},
        successful_metrics={k:mean([r['metrics_on_good'][k] for r in good]) for k in rows[0]['metrics_on_good']})

def paired(a,b):
    assert len(a)==len(b)
    common=[(x,y) for x,y in zip(a,b) if x['success'] and y['success']]
    metrics={}
    for key in ('normalized_torque_squared','all_joint_rms_Nm','leg_rms_Nm','torso_arm_hand_rms_Nm','foot_load_bw','rest_normal_load_bw'):
        old=mean([x['metrics_on_good'][key] for x,y in common]);new=mean([y['metrics_on_good'][key] for x,y in common])
        metrics[key]=dict(reference=old,candidate=new,percent_change=100*(new/old-1) if old else None)
    return dict(common_successes=len(common),gained=sum(not x['success'] and y['success'] for x,y in zip(a,b)),
        lost=sum(x['success'] and not y['success'] for x,y in zip(a,b)),metrics=metrics,
        common_cases_at_least_10percent_effort_reduction=sum(y['metrics_on_good']['normalized_torque_squared']<=.9*x['metrics_on_good']['normalized_torque_squared'] for x,y in common))

fixed={b:get(f'fixed_{b}',12 if b in ('heights','impulse') else 16) for b in banks}
for b in banks:
    with np.load(root/f'fixed_{b}/trace.npz') as z:
        assert np.array_equal(z['command'],np.broadcast_to(z['command'][0],z['command'].shape))
budgets=[];parity=[];data={};hold_audit=[];command_audit=[];initial_states={}
for seed in seeds:
    train=root/f'train_{seed}';v=read(train/'verification.json')
    assert v['updates']==64 and v['transitions']==65536 and v['finite']
    assert v['active_transitions']+v['warmup_transitions']==65536
    rows=[json.loads(s) for s in (train/'episodes.jsonl').read_text().splitlines()]
    assert sum(r['steps'] for r in rows)+sum(v['unfinished_episode_steps'])==v['active_transitions']
    initial=torch.load(train/'initial.pt',map_location='cpu',weights_only=False)
    final=torch.load(train/'final.pt',map_location='cpu',weights_only=False)
    initial_states[seed]=initial
    assert read(train/'initial_policy_check.json')['zero_mean_joint_increment']
    change=max(float(abs(x-final['actor_state_dict'][k]).max()) for k,x in initial['actor_state_dict'].items())
    assert change>0 and all(torch.isfinite(x).all() for role in ('actor_state_dict','critic_state_dict') for x in final[role].values())
    budgets.append(dict(seed=seed,actor_max_change=change,final_sha256=sha(train/'final.pt'),**v))
    for variant in variants:
        for bank in banks:
            name=f'{bank}_{seed}_{variant}';path=root/name
            rows=get(name,len(fixed[bank]));data[(seed,variant,bank)]=rows
            with np.load(root/f'fixed_{bank}/startup.npz') as a,np.load(path/'startup.npz') as b:
                assert set(a.files)==set(b.files)
                error=max(float(np.max(abs(a[k]-b[k]))) for k in a.files)
                assert error<=protocol['startup_tolerance'],(name,error)
            assert read(path/'geometry_bank.json')==read(root/f'fixed_{bank}/geometry_bank.json')
            assert read(path/'disturbances.json')==read(root/f'fixed_{bank}/disturbances.json')
            loaded=torch.load(path/'initial.pt',map_location='cpu',weights_only=False)
            assert all(torch.equal(x,loaded[role][k]) for role in ('actor_state_dict','critic_state_dict') for k,x in final[role].items())
            parity.append(dict(run=name,max_startup_error=error,same_final_checkpoint=True,matched_geometry_and_disturbances=True))
            info=read(path/'hold_goal.json')
            assert info['enabled']==(variant=='ppo_hold')
            with np.load(path/'trace.npz') as z:
                with np.load(path/'startup.npz') as startup:
                    reference=startup['command']
                names=read(path/'results.json')['joint_names']
                spans=np.array([.025 if 'roll_joint' in joint and ('hip' in joint or 'ankle' in joint) else .05 for joint in names])
                drift=np.abs(z['command']-reference[None])
                assert np.all(drift<=spans[None,None]+1.e-6),(name,float(drift.max()))
                command_audit.append(dict(run=name,max_reference_offset_rad=float(drift.max()),bounded=True))
                trace=info['trace'];held=np.array([r['held'] for r in trace],dtype=bool)
                assert held.shape==z['good'].shape
                live=held&z['alive'].astype(bool)
                transitions=held&~np.vstack([np.zeros_like(held[:1]),held[:-1]])
                hold_audit.append(dict(run=name,entries=sum(info['entries']),releases=sum(info['releases']),
                    held_sample_fraction=float(held.mean()),live_held_samples=int(live.sum()),
                    physically_good_live_held_samples=int((live&z['good'].astype(bool)).sum()),
                    entry_samples=int(transitions.sum()),good_entry_samples=int((transitions&z['good'].astype(bool)).sum())))

comparisons=[];by_height=[]
for seed in seeds:
    for bank in (*banks,'known','unperturbed','all'):
        include=banks if bank=='all' else ('starts','heights','impulse') if bank=='known' else ('starts','heights','novel') if bank=='unperturbed' else (bank,)
        baseline=sum((fixed[b] for b in include),[])
        rows={v:sum((data[(seed,v,b)] for b in include),[]) for v in variants}
        comparisons.append(dict(seed=seed,bank=bank,fixed=summary(baseline),**{v:summary(r) for v,r in rows.items()},
            ppo_vs_fixed=paired(baseline,rows['ppo']),hold_vs_fixed=paired(baseline,rows['ppo_hold']),hold_vs_ppo=paired(rows['ppo'],rows['ppo_hold'])))
    for bank in ('starts','heights','novel','impulse'):
        for h in sorted({r['height_m'] for r in fixed[bank]}):
            f=[r for r in fixed[bank] if r['height_m']==h]
            plain=[r for r in data[(seed,'ppo',bank)] if r['height_m']==h]
            held=[r for r in data[(seed,'ppo_hold',bank)] if r['height_m']==h]
            by_height.append(dict(seed=seed,bank=bank,height_m=h,fixed=summary(f),ppo=summary(plain),ppo_hold=summary(held),
                ppo_vs_fixed=paired(f,plain),hold_vs_fixed=paired(f,held),hold_vs_ppo=paired(plain,held)))
report=dict(protocol=protocol,budgets=budgets,parity=parity,hold_audit=hold_audit,command_audit=command_audit,comparisons=comparisons,by_height=by_height,
    frozen_sources_verified=True,promoted=False,cases={f'{s}_{v}_{b}':rows for (s,v,b),rows in data.items()},fixed_cases=fixed,
    caution='Two training seeds repeat44 unique reset states plus12 impulse repeats. Known40 conditions are regression tests;16 new states form the fresh holdout. Effort is a torque proxy, not electrical energy. Nm averages instantaneous joint RMS. Only shared full-duration successes support paired effort claims.')
out.mkdir(exist_ok=False)
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
shutil.copy2(root/'protocol.json',out/'protocol.json')
shutil.copy2(root/'retention_validation.json',out/'retention_validation.json')
fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
labels=['Fixed']+[f'{v}\n{s}' for s in seeds for v in variants]
groups=[sum(fixed.values(),[])]+[sum((data[(s,v,b)] for b in banks),[]) for s in seeds for v in variants]
colors=['#6f7981','#3985ab','#2a8a72','#3985ab','#2a8a72']
bars=axes[0].bar(labels,[sum(r['success'] for r in rows) for rows in groups],color=colors)
axes[0].bar_label(bars,padding=2);axes[0].set(ylim=(0,62),ylabel='Successful 20-second trials /56',title='Stability before effort')
common=[i for i in range(56) if all(rows[i]['success'] for rows in groups)]
if common:
    values=[mean([rows[i]['metrics_on_good']['normalized_torque_squared'] for i in common]) for rows in groups]
    axes[1].bar(labels,[100*x/values[0] for x in values],color=colors)
    axes[1].axhline(90,color='#a84840',linestyle='--',label='10% saving target');axes[1].legend()
    axes[1].set_ylabel('Normalized squared torque, fixed =100%')
else:axes[1].text(.5,.5,'No successes shared by all variants',ha='center',transform=axes[1].transAxes)
axes[1].set_title(f'Comparable effort: {len(common)} common successes')
for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)
rows=[]
for r in comparisons:
    a=r['ppo_vs_fixed']['metrics']['normalized_torque_squared']['percent_change']
    b=r['hold_vs_fixed']['metrics']['normalized_torque_squared']['percent_change']
    rows.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [r['seed'],r['bank'],
        f"{r['fixed']['successes']}/{r['fixed']['total']}",r['ppo']['successes'],r['ppo_hold']['successes'],
        f'{a:+.2f}%' if a is not None else 'n/a',f'{b:+.2f}%' if b is not None else 'n/a'])+'</tr>')
page='''<!doctype html><meta charset="utf-8"><title>Bounded resting PPO</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:40px auto;color:#213b43}table{border-collapse:collapse;width:100%}th,td{padding:8px;border-bottom:1px solid #ccd5da;text-align:left}img{width:100%}</style><h1>Bounded resting PPO versus fixed posture</h1><p>The previous whole-body-effort reward is unchanged. Two fresh64-update learners use bounded reference offsets. The optional measured-effort hold rule is tested on the same final checkpoints, so its effect is separated from training.</p><img src="comparison.png" alt="Full-duration success and comparable normalized joint effort"><table><tr><th>Seed</th><th>Bank</th><th>Fixed success</th><th>PPO success</th><th>PPO+hold success</th><th>PPO effort vs fixed</th><th>PPO+hold effort vs fixed</th></tr>'''
page+=''.join(rows)+'</table><p>'+html.escape(report['caution'])+'</p><p>Budgets including warmup, finite changed weights, evaluation checkpoint matching, startup matching, fixed commands and source preservation verified. Per-height and disturbance comparisons, leg/torso effort, hold-entry physical validity and failure categories are in results.json. No policy promoted.</p>'
(out/'report.html').write_text(page)
print(json.dumps([r for r in comparisons if r['bank'] in ('all','novel')],indent=2))
