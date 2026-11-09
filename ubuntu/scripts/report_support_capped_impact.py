"""Two-seed, matched final-policy comparison, with transparent timer correction."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--study',required=True);a=p.parse_args()
study=Path(a.study);store=study.parent;out=study/'results';out.mkdir(exist_ok=False)
seeds=(52813,52817);banks=('regression','fresh','starts','novel','unseen');arms=('legacy','capped')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mean(v):return float(np.mean(v)) if len(v) else None
def corrected(rows):
    result=[]
    for row in rows:
        r=dict(row);ticks=r.get('hold_ticks',round(r['hold_s']/.02))
        assert abs(r['hold_s']-ticks*.02)<.001
        r.setdefault('legacy_float_success',r['success']);r['hold_ticks']=ticks;r['hold_s']=ticks*.02
        r['success']=bool(r['standing'] and r['standing_checked'] and r['startup_ready'] and
                          not any(r['failures'].values()) and ticks>=150)
        if 'hold_ticks' in row:assert r['success']==row['success']
        result.append(r)
    return result
def summary(rows):
    return dict(successes=sum(r['success'] for r in rows),attempts=len(rows),
        mean_peak_force_N=mean([r['peak_force_N'] for r in rows]),
        p95_peak_force_N=float(np.quantile([r['peak_force_N'] for r in rows],.95)),
        max_force_N=max(r['peak_force_N'] for r in rows),mean_hold_s=mean([r['hold_s'] for r in rows]),
        failures={k:sum(r['failures'][k] for r in rows) for k in rows[0]['failures']})
def comparison(before,after):
    pairs=list(zip(before,after));common=[(x,y) for x,y in pairs if x['success'] and y['success']]
    return dict(gained=sum(not x['success'] and y['success'] for x,y in pairs),
        lost=sum(x['success'] and not y['success'] for x,y in pairs),common_successes=len(common),
        common_mean_peak_force_change_N=mean([y['peak_force_N']-x['peak_force_N'] for x,y in common]),
        common_mean_leg_change_Nm=mean([y['seated_leg_rms_Nm']-x['seated_leg_rms_Nm'] for x,y in common]),
        common_mean_foot_change=mean([y['seated_foot_load_fraction']-x['seated_foot_load_fraction'] for x,y in common]))

report=dict(training={},banks={},seeds={},timer_validation=read(study/'timer_validation.json'),promoted=False)
parent=torch.load(store/'support_split_ppo_train_01/final.pt',map_location='cpu',weights_only=False)
for seed in seeds:
    for arm in arms:
        name=f'train_{seed}_{arm}';d=study/name;v=read(d/'verification.json');account=read(d/'attempt_accounting.json')
        assert v['transitions']==32768 and v['ppo_iterations']==32 and v['finite']
        assert account['completed_transitions']+account['unfinished_transitions']==32768
        assert len((d/'study_updates.jsonl').read_text().splitlines())==32
        initial=torch.load(d/'initial.pt',map_location='cpu',weights_only=False)
        final=torch.load(d/'final.pt',map_location='cpu',weights_only=False)
        for role in ('actor_state_dict','critic_state_dict'):
            assert all(torch.equal(x,initial[role][k]) for k,x in parent[role].items())
            assert all(torch.equal(x,final[role][k]) for k,x in parent[role].items() if k.startswith('obs_normalizer.'))
        report['training'][name]=dict(verification=v,accounting=account,sha256=sha(d/'final.pt'))
        dest=out/name;dest.mkdir()
        for f in ('verification.json','attempt_accounting.json','manifest.json','study_updates.jsonl'):
            shutil.copy2(d/f,dest/f)

for bank in banks:
    initial=(study/'unseen_initial' if bank=='unseen' else
        store/'support_reward_study_01/novel_initial' if bank=='novel' else store/f'support_split_{bank}_trained_01')
    paths={'initial':initial,**{f'{seed}_{arm}':study/f'{bank}_{seed}_{arm}' for seed in seeds for arm in arms}}
    data={key:corrected(read(d/'results.json')['episodes']) for key,d in paths.items()}
    for key,rows in data.items():
        assert len(rows)==16 and all(r['steps']==1000 and r['duration_s']==20 for r in rows)
        assert read(paths[key]/'geometry_bank.json')==read(initial/'geometry_bank.json')
        with np.load(paths[key]/'startup.npz') as actual,np.load(initial/'startup.npz') as reference:
            assert set(actual.files)==set(reference.files)
            assert all(np.array_equal(actual[k],reference[k],equal_nan=True) for k in actual.files),(bank,key)
        dest=out/bank/key;dest.mkdir(parents=True)
        for name in ('results.json','manifest.json','geometry_bank.json','attempt_accounting.json'):
            shutil.copy2(paths[key]/name,dest/name)
    report['banks'][bank]=dict(cases=data,summaries={k:summary(v) for k,v in data.items()},startup_bitwise_equal=True)

for seed in seeds:
    rows={arm:[r for bank in banks for r in report['banks'][bank]['cases'][f'{seed}_{arm}']] for arm in arms}
    initial=[r for bank in banks for r in report['banks'][bank]['cases']['initial']]
    report['seeds'][str(seed)]=dict(initial=summary(initial),**{arm:summary(rows[arm]) for arm in arms},
        capped_vs_legacy=comparison(rows['legacy'],rows['capped']),
        capped_vs_initial=comparison(initial,rows['capped']),
        legacy_vs_initial=comparison(initial,rows['legacy']))
pooled={arm:[r for seed in seeds for bank in banks for r in report['banks'][bank]['cases'][f'{seed}_{arm}']] for arm in arms}
report['pooled']=dict(**{arm:summary(rows) for arm,rows in pooled.items()},
    capped_vs_legacy=comparison(pooled['legacy'],pooled['capped']),
    note='160 trials per objective across two trained seeds, on the same80 placements; not160 unique cases.')
root=Path(__file__).resolve().parents[1]
assert all(sha(root/rel)==h for rel,h in read(study/'frozen/manifest.json').items())
report['frozen_files_unchanged']=True
shutil.copy2(study/'timer_validation.json',out/'timer_validation.json')
shutil.copy2(study/'archived_cost_audit.json',out/'archived_cost_audit.json')
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')

fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
for ax,seed in zip(axes,seeds):
    x=np.arange(5)
    for j,(role,label,color) in enumerate([('initial','Starting policy','#788695'),(f'{seed}_legacy','Old reward','#3982b4'),(f'{seed}_capped','Capped impact','#22865b')]):
        ax.bar(x+(j-1)*.25,[report['banks'][b]['summaries'][role]['successes'] for b in banks],.25,label=label,color=color)
    ax.set_title(f'Training seed{seed}');ax.set_xticks(x,['Known','Fresh¹','Starts','Novel¹','Unseen'],rotation=20)
    ax.set_ylim(0,19);ax.set_ylabel('Successful sits /16');ax.spines[['top','right']].set_visible(False)
axes[0].legend(fontsize=8);fig.suptitle('32 updates per arm and seed · exact150-tick hold scoring\n¹Previously tested; only Unseen is a new holdout')
fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
lines=['# Capped-impact-only PPO results','',
    'All four runs completed32 updates /32,768 transitions each. Final policies evaluated on the same80 cases per seed; every attempt ran20 seconds. No policy promoted.','',
    '| Training seed | Starting policy | Old reward | Capped impact |','|---|---:|---:|---:|']
for seed,e in report['seeds'].items():lines.append(f"|{seed}|{e['initial']['successes']}/80|{e['legacy']['successes']}/80|{e['capped']['successes']}/80|")
lines += ['','![Matched successes](comparison.png)','',
    'The timer regression reproduced all archived motion/sensor/command arrays bitwise and corrected only the known150-tick case. Existing reference results were transparently rescored using ticks. All evaluation startup archives and geometries match.','',
    'The first64 placements are known regression sets; the last16 are a fresh holdout. Results are reported by seed, not by selecting the best seed. Pooled outcomes repeat placements across two different learned policies. Two seeds remain exploratory evidence.','']
for seed,e in report['seeds'].items():
    c=e['capped_vs_legacy']
    lines += [f"## Seed{seed}",'',f"Capped versus old reward: gained{c['gained']}, lost{c['lost']}. Among{c['common_successes']} common successes, mean peak-force change{c['common_mean_peak_force_change_N']}N; leg RMS change{c['common_mean_leg_change_Nm']}Nm.",'']
lines += ['Full failure types, force tails, per-case outcomes, training accounting and source verification are in results.json. Neither actor nor critic gained privileged runtime inputs. Effort reward, perception and motor limits stayed fixed.','']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(report['seeds'],indent=2))
