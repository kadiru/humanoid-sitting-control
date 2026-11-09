"""Full-horizon physical comparison, not comparison of unlike reward scales."""
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

p=argparse.ArgumentParser(); p.add_argument('--study',required=True); args=p.parse_args()
study=Path(args.study); store=study.parent; out=study/'results'; out.mkdir(exist_ok=False)
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def summarize(rows):
    force=[r['peak_force_N'] for r in rows]
    return dict(successes=sum(r['success'] for r in rows),trials=len(rows),
        max_force_N=max(force),p95_force_N=float(np.quantile(force,.95)),
        mean_hold_s=float(np.mean([r['hold_s'] for r in rows])),
        failures={k:sum(r['failures'][k] for r in rows) for k in rows[0]['failures']})

report=dict(training={},banks={},promoted=False)
starting=torch.load(store/'support_split_ppo_train_01/final.pt',map_location='cpu',weights_only=False)
for arm in ('gentle','legacy'):
    path=study/f'train_{arm}'; v=read(path/'verification.json'); accounting=read(path/'attempt_accounting.json')
    assert v['transitions']==128*64*16 and v['ppo_iterations']==128 and v['finite']
    assert accounting['completed_transitions']+accounting['unfinished_transitions']==v['transitions']
    updates=[json.loads(s) for s in (path/'study_updates.jsonl').read_text().splitlines()]
    assert len(updates)==128
    initial=torch.load(path/'initial.pt',map_location='cpu',weights_only=False)
    final=torch.load(path/'final.pt',map_location='cpu',weights_only=False)
    for role in ('actor_state_dict','critic_state_dict'):
        assert all(torch.equal(v,initial[role][k]) for k,v in starting[role].items())
        assert all(torch.equal(v,final[role][k]) for k,v in starting[role].items() if k.startswith('obs_normalizer.'))
    report['training'][arm]=dict(verification=v,accounting=accounting,checkpoint_sha256=sha(path/'final.pt'),
        exact_initial_parameters=True,frozen_normalization=True)
    for name in ('verification.json','attempt_accounting.json','manifest.json','study_updates.jsonl'):
        dest=out/f'train_{arm}'; dest.mkdir(exist_ok=True); shutil.copy2(path/name,dest/name)

for bank in ('regression','fresh','starts','novel'):
    paths={arm:study/f'{bank}_{arm}' for arm in ('gentle','legacy')}
    paths['initial']=study/'novel_initial' if bank=='novel' else store/f'support_split_{bank}_trained_01'
    rows={arm:read(path/'results.json')['episodes'] for arm,path in paths.items()}
    for arm, rr in rows.items():
        assert len(rr)==16 and all(r['steps']==1000 and abs(r['duration_s']-20.)<1e-6 for r in rr)
        assert read(paths[arm]/'geometry_bank.json')==read(paths['initial']/'geometry_bank.json')
        with np.load(paths[arm]/'startup.npz') as actual,np.load(paths['initial']/'startup.npz') as reference:
            assert set(actual.files)==set(reference.files)
            assert all(np.array_equal(actual[k],reference[k],equal_nan=True) for k in actual.files), (bank,arm)
    entry={arm:summarize(rr) for arm,rr in rows.items()}
    entry['paired_startup_equal']=True; entry['comparisons']={}
    for arm in ('legacy','gentle'):
        common=[(a,b) for a,b in zip(rows['initial'],rows[arm]) if a['success'] and b['success']]
        entry['comparisons'][arm]=dict(gained=[b['env'] for a,b in zip(rows['initial'],rows[arm]) if not a['success'] and b['success']],
            lost=[a['env'] for a,b in zip(rows['initial'],rows[arm]) if a['success'] and not b['success']],
            common_successes=len(common),
            mean_common_leg_change_Nm=float(np.mean([b['seated_leg_rms_Nm']-a['seated_leg_rms_Nm'] for a,b in common])) if common else None,
            mean_common_foot_load_change=float(np.mean([b['seated_foot_load_fraction']-a['seated_foot_load_fraction'] for a,b in common])) if common else None)
    entry['cases']=rows; report['banks'][bank]=entry
    for arm,path in paths.items():
        dest=out/bank/arm; dest.mkdir(parents=True)
        for name in ('results.json','manifest.json','attempt_accounting.json','geometry_bank.json'):
            shutil.copy2(path/name,dest/name)
hashes=read(study/'frozen/manifest.json'); root=Path(__file__).resolve().parents[1]
report['frozen_files_unchanged']=all(sha(root/relative)==digest for relative,digest in hashes.items())
assert report['frozen_files_unchanged']
report['original_controller_first48']=sum(read(store/f'support_split_{bank}_parent_01/results.json')['successes'] for bank in ('regression','fresh','starts'))
report['totals']={arm:sum(e[arm]['successes'] for e in report['banks'].values()) for arm in ('initial','legacy','gentle')}
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained'); x=np.arange(4)
for i,(arm,label,color) in enumerate([('initial','Starting PPO','#777777'),('legacy','More PPO / old reward','#3975ad'),('gentle','More PPO / new reward','#22865b')]):
    for ax,key in zip(axes,('successes','p95_force_N','mean_hold_s')):
        ax.bar(x+(i-1)*.24,[e[arm][key] for e in report['banks'].values()],.24,label=label,color=color)
for ax,title in zip(axes,('Successful sits /16','95th percentile seat force (N)','Mean longest stable hold (s)')):
    ax.set_title(title); ax.set_xticks(x,['Known','Fresh','New starts','New bank']); ax.spines[['top','right']].set_visible(False)
axes[0].set_ylim(0,19); axes[0].legend(fontsize=8)
fig.suptitle('128 PPO updates per arm · paired 20-second attempts · all failures retained')
fig.savefig(out/'comparison.png',dpi=160); plt.close(fig)
lines=['# Longer PPO reward experiment','',
    'Both arms completed128 updates (131,072 transitions each) from the same151-input checkpoint. Perception, motor limits and success criteria stayed fixed. No policy was promoted.','',
    '| Bank | Starting PPO | Old reward | New reward |','|---|---:|---:|---:|']
for bank,e in report['banks'].items(): lines.append(f"| {bank} | {e['initial']['successes']}/16 | {e['legacy']['successes']}/16 | {e['gentle']['successes']}/16 |")
lines += ['',f"Totals out of64: {report['totals']}. Original controller scored35/48 on the first three banks; the starting PPO scored37/48.",'',
    '![Comparison](comparison.png)','',
    'Full case outcomes, force maxima, failures, gained/lost successes and leg/foot-load changes on common successes are in results.json. All144 new evaluation attempts ran the full20 seconds; startup sensor archives and geometry matched within each bank. Final checkpoints were evaluated without selection.','',
    'This is one training seed and initially aligned rigid seats. The two reward changes are tested together. It does not establish general sitting, hardware readiness, or a causal benefit from the distance channels. Tracking imperfections remain part of the experiment. Training returns use different objectives and should not be compared directly.','']
for bank,e in report['banks'].items():
    lines += [f"## {bank}",'',f"Worst forces (N): starting {e['initial']['max_force_N']:.1f}, old reward {e['legacy']['max_force_N']:.1f}, new reward {e['gentle']['max_force_N']:.1f}.",'']
    for arm in ('legacy','gentle'):
        c=e['comparisons'][arm]; leg=c['mean_common_leg_change_Nm']
        lines.append(f"{arm}: gained {c['gained']}; lost {c['lost']}; leg RMS change among {c['common_successes']} common successes: {leg} Nm.")
    lines.append('')
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(report['totals']))
