"""Known-set diagnostic with identical learned weights and neutralized pace."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];store=Path('/data/humanoid-sitting-perception')
suite=store/'support_rl_pace_results_01';out=suite/'pace_ablation';out.mkdir(exist_ok=False)
shutil.copy2(__file__,out/'report_source.py')
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
names=['support_rl_pace_easy_prior_01','support_rl_pace_easy_candidate_01','support_rl_pace_easy_neutral_01']
loaded=[]
for name in names:
 p=store/name;m=read(p/'manifest.json');r=read(p/'results.json');z=dict(np.load(p/'rollout.npz'))
 assert len(r['episodes'])==16 and all(x['steps']==1000 and x['duration_s']==20 for x in r['episodes'])
 assert len(z['t'])==200 and np.allclose(z['t'],np.arange(1,201)*.1)
 for k,h in m['hashes'].items():
  assert digest(root/k)==h
  if k.endswith('.py'):assert digest(p/'source'/Path(k).name)==h
 loaded.append(dict(path=p,manifest=m,result=r,trace=z,bank=read(p/'geometry_bank.json')))
assert all(x['bank']==loaded[0]['bank'] for x in loaded)
assert all(x['manifest']['arguments']['seed']==51407 for x in loaded)
assert np.all(loaded[2]['trace']['command_pace']==1.)
a=torch.load(loaded[1]['path']/'initial.pt',map_location='cpu',weights_only=False)
b=torch.load(loaded[2]['path']/'initial.pt',map_location='cpu',weights_only=False)
assert all(torch.equal(v,b[group][k]) for group in ('actor_state_dict','critic_state_dict') for k,v in a[group].items())
rows=[x['result']['episodes'] for x in loaded]
summary=[dict(run=n,successes=sum(x['success'] for x in r),worst_impact_N=max(x['peak_force_N'] for x in r),
 failures={k:sum(x['failures'][k] for x in r) for k in r[0]['failures']},
 failed_cases=[x['env'] for x in r if not x['success']]) for n,r in zip(names,rows)]
report=dict(summary=summary,identical_actor_and_critic_weights=True,full_horizon_attempts=48,
 source_hashes_verified=True,pace_exactly_one=True,extra_evaluation_attempts=16,
 regained_by_disabling_pace=[i for i in range(16) if not rows[1][i]['success'] and rows[2][i]['success']],
 lost_by_disabling_pace=[i for i in range(16) if rows[1][i]['success'] and not rows[2][i]['success']],
 still_lost_vs_previous=[i for i in range(16) if rows[0][i]['success'] and not rows[2][i]['success']],
 limitations='Diagnostic on one known regression bank; not a new holdout, additional training, or promotion decision. Fixing pace also fixes its correctly observed actuator state.')
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
labels=['Previous PPO','New PPO\npace enabled','Same new PPO\npace disabled'];colors=['#64748b','#c2410c','#047857']
for j,r in enumerate(summary):
 axes[0].bar(j,r['successes'],color=colors[j]);axes[0].text(j,r['successes']+.3,f"{r['successes']}/16",ha='center')
 axes[1].bar(j,r['worst_impact_N'],color=colors[j])
axes[0].set(ylim=(0,18),ylabel='Successful complete attempts')
axes[1].set(ylabel='Worst impact, all attempts (N)');axes[1].axhline(632.527832,color='#b91c1c',ls='--')
for ax in axes:ax.set_xticks(range(3),labels);ax.grid(axis='y',alpha=.2)
fig.suptitle('Known easier bank: isolate the runtime pace adjustment')
fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
lines=['# Pace-disabled diagnostic','',report['limitations'],'',
 '| Controller | Full-attempt success | Worst impact |','|---|---:|---:|']
for label,r in zip(['Previous PPO','New PPO, pace enabled','Same new PPO, pace disabled'],summary):
 lines.append(f"| {label} | {r['successes']}/16 | {r['worst_impact_N']:.1f} N |")
lines+=['',f"Disabling pace regains cases {report['regained_by_disabling_pace']} and loses cases {report['lost_by_disabling_pace']}.",
 f"Cases still lost relative to the previous PPO: {report['still_lost_vs_previous']}.",
 'The candidate and diagnostic have exactly identical actor and critic weights. Both run for the full 20 seconds with latched failures. No checkpoint was retrained or selected using this test.','']
(out/'report.md').write_text('\n'.join(lines))
# Retain the original main report verbatim before linking the diagnostic.
shutil.copy2(suite/'report.md',suite/'report_before_ablation.md')
with (suite/'report.md').open('a') as f:
 f.write('\n## Diagnostic after the known-set regression\n\n'+
  f"With the same final learned weights, fixing pace to its original value changes known easier-set success from {summary[1]['successes']}/16 to {summary[2]['successes']}/16; the previous PPO scored {summary[0]['successes']}/16. "+
  'See [the pace-disabled comparison](pace_ablation/report.md). This is a known-set diagnostic, not a new holdout.\n')
print(json.dumps(report,indent=2))
