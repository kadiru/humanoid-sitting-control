"""Matched full-duration resting evaluation, with explicit conditional effort."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(); p.add_argument('study'); args=p.parse_args(); root=Path(args.study)
out=root/'results'; out.mkdir(exist_ok=False)
protocol=json.loads((root/'protocol.json').read_text())
for path,digest in json.loads((root/'frozen/manifest.json').read_text()).items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path


def get(name):
    rows=json.loads((root/name/'results.json').read_text())['episodes']
    assert all(r['steps']==1000 and r['duration_s']==20. for r in rows)
    return rows


def mean(values):return float(np.mean(values)) if len(values) else None


def summary(rows):
    good=[r for r in rows if r['success']]
    return dict(successes=len(good),total=len(rows),
        mean_good_fraction=mean([r['good_fraction'] for r in rows]),
        successful_whole_body_rms_Nm=mean([r['metrics_on_good']['all_joint_rms_Nm'] for r in good]),
        successful_effort_proxy=mean([r['metrics_on_good']['normalized_torque_squared'] for r in good]),
        force_max_N=max(r['peak_force_N'] for r in rows),
        failures={key:sum(r['failures'][key] for r in rows) for key in rows[0]['failures']})


def paired(a,b):
    assert len(a)==len(b)
    shared=[(x,y) for x,y in zip(a,b) if x['success'] and y['success']]
    result=dict(common_successes=len(shared),
        gained=sum(not x['success'] and y['success'] for x,y in zip(a,b)),
        lost=sum(x['success'] and not y['success'] for x,y in zip(a,b)))
    for key in ('normalized_torque_squared','all_joint_rms_Nm','leg_rms_Nm','torso_arm_hand_rms_Nm','foot_load_bw'):
        old=mean([x['metrics_on_good'][key] for x,y in shared]); new=mean([y['metrics_on_good'][key] for x,y in shared])
        result[key]=dict(reference=old,candidate=new,percent_change=(100*(new/old-1) if old else None))
    return result


labels=('starts','heights','impulse'); seeds=(52911,52917); arms=('stability','effort')
baseline={label:get(f'fixed_{label}') for label in labels}
data={}; parity=[]; budgets=[]
for seed in seeds:
    states={}
    for arm in arms:
        train=root/f'train_{seed}_{arm}'
        v=json.loads((train/'verification.json').read_text())
        assert v['finite'] and v['updates']==128 and v['transitions']==131072
        assert v['active_transitions']+v['warmup_transitions']==v['transitions']
        episodes=[json.loads(x) for x in (train/'episodes.jsonl').read_text().splitlines()]
        assert sum(x['steps'] for x in episodes)+sum(v['unfinished_episode_steps'])==v['active_transitions']
        initial=torch.load(train/'initial.pt',map_location='cpu',weights_only=False)
        final=torch.load(train/'final.pt',map_location='cpu',weights_only=False)
        states[arm]=initial
        change=max(float((final['actor_state_dict'][k]-value).abs().max()) for k,value in initial['actor_state_dict'].items())
        assert change>0 and all(torch.isfinite(x).all() for role in ('actor_state_dict','critic_state_dict') for x in final[role].values())
        budgets.append(dict(seed=seed,arm=arm,actor_max_change=change,**v))
        for label in labels:
            name=f'{label}_{seed}_{arm}'; rows=get(name); data[(seed,arm,label)]=rows
            assert len(rows)==(16 if label=='starts' else 12)
            a=np.load(root/f'fixed_{label}'/'startup.npz'); b=np.load(root/name/'startup.npz')
            errors={key:float(np.max(np.abs(a[key]-b[key]))) for key in a.files}
            assert max(errors.values())<=protocol['startup_parity_absolute_tolerance'],(name,errors)
            assert json.loads((root/f'fixed_{label}'/'disturbances.json').read_text())==json.loads((root/name/'disturbances.json').read_text())
            assert json.loads((root/f'fixed_{label}'/'geometry_bank.json').read_text())==json.loads((root/name/'geometry_bank.json').read_text())
            parity.append(dict(run=name,max_startup_error=max(errors.values()),matched_disturbances=True))
    for role in ('actor_state_dict','critic_state_dict'):
        assert all(torch.equal(v,states['effort'][role][k]) for k,v in states['stability'][role].items()),(seed,role)

# A fixed-target baseline must actually have issued a constant command.
for label in labels:
    trace=np.load(root/f'fixed_{label}'/'trace.npz')
    assert np.array_equal(trace['command'],np.broadcast_to(trace['command'][0],trace['command'].shape))

comparisons=[]
for seed in seeds:
    for label in (*labels,'unperturbed_all','all_conditions'):
        include=('starts','heights') if label=='unperturbed_all' else labels if label=='all_conditions' else (label,)
        fixed=sum((baseline[k] for k in include),[])
        stable=sum((data[(seed,'stability',k)] for k in include),[])
        effort=sum((data[(seed,'effort',k)] for k in include),[])
        comparisons.append(dict(seed=seed,bank=label,fixed=summary(fixed),stability=summary(stable),effort=summary(effort),
            effort_vs_fixed=paired(fixed,effort),stability_vs_fixed=paired(fixed,stable),effort_vs_stability=paired(stable,effort)))
height_rows=[]
for seed in seeds:
    for arm in ('fixed',*arms):
        rows=sum((baseline[k] if arm=='fixed' else data[(seed,arm,k)] for k in ('starts','heights')),[])
        for height in sorted({r['height_m'] for r in rows}):
            height_rows.append(dict(seed=seed,arm=arm,height_m=height,**summary([r for r in rows if r['height_m']==height])))
report=dict(protocol=protocol,budgets=budgets,parity=parity,comparisons=comparisons,by_height=height_rows,
    source_hashes_verified=True,initial_arm_weights_equal=True,
    caution='40 conditions per policy use28 unique captured states;12 states repeat with perturbations. Two training seeds are not broad statistical evidence. Effort comparisons are conditional on shared successes.')
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
fig,axs=plt.subplots(1,2,figsize=(12,4.5))
names=['Fixed','Stability\n52911','Effort\n52911','Stability\n52917','Effort\n52917']
groups=[sum(baseline.values(),[])]+[sum((data[(s,a,k)] for k in labels),[]) for s in seeds for a in arms]
axs[0].bar(names,[sum(r['success'] for r in rows) for rows in groups],color=['#777777','#9db7ce','#227c8b','#9db7ce','#227c8b'])
axs[0].set(ylim=(0,40),ylabel='Successful full20-second trials /40',title='Stability, including impulse trials')
common=[i for i in range(40) if all(rows[i]['success'] for rows in groups)]
values=[mean([rows[i]['metrics_on_good']['all_joint_rms_Nm'] for i in common]) for rows in groups]
if common:axs[1].bar(names,values,color=['#777777','#9db7ce','#227c8b','#9db7ce','#227c8b'])
else:axs[1].text(.5,.5,'No successes shared by all policies',ha='center',transform=axs[1].transAxes)
axs[1].set(ylabel='Whole-body torque RMS (Nm)',title=f'Comparable effort: {len(common)} shared successes')
for ax in axs:ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(out/'comparison.png',dpi=180); plt.close(fig)
table=[]
for r in comparisons:
    c=r['effort_vs_fixed']; change=c['all_joint_rms_Nm']['percent_change']
    table.append('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in
        [r['seed'],r['bank'],f"{r['fixed']['successes']}/{r['fixed']['total']}",
         r['stability']['successes'],r['effort']['successes'],c['common_successes'],
         f'{change:+.2f}%' if change is not None else 'n/a'])+'</tr>')
page='''<!doctype html><meta charset="utf-8"><title>Resting PPO comparison</title>
<style>body{font:16px system-ui;max-width:1150px;margin:40px auto;color:#19313d}table{border-collapse:collapse;width:100%}th,td{padding:9px;border-bottom:1px solid #ccd5da;text-align:left}img{width:100%}p{line-height:1.6}</style>
<h1>Resting PPO versus holding a fixed posture</h1>
<p>All policies start in physically settled contact. Compare full-duration stability first; lower effort among a selected subset alone is not a win.</p>
<img src="comparison.png" alt="Full-duration success and matched whole-body effort">
<table><tr><th>Seed</th><th>Bank</th><th>Fixed success</th><th>Stability PPO</th><th>Effort PPO</th><th>Shared fixed/effort successes</th><th>Whole-body RMS change vs fixed</th></tr>'''
page+=''.join(table)+'</table><p>'+html.escape(report['caution'])+'</p><p>Budgets, initial weight equality, source hashes, geometry, disturbances and startup matching verified. Raw metrics and failure counts are in results.json. No automatic deployment or policy promotion.</p>'
(out/'report.html').write_text(page)
print(json.dumps(dict(report=str(out/'report.html'),comparisons=[r for r in comparisons if r['bank']=='all_conditions']),indent=2))
