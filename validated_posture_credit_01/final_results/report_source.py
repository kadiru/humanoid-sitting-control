"""Package validated results with physical outcomes before paired effort."""
import argparse,json,shutil,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from render_posture_credit_replay import export

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--verified',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
 read=lambda p:json.loads(p.read_text());v=read(a.verified/'verification.json');assert v['passed'];protocol=read(a.study/'protocol.json')
 names={'reference':'Repaired reference','previous_58311':'Previous PPO58311','previous_58317':'Previous PPO58317',**{str(s):f'Posture PPO{s}' for s in protocol['seeds']}}
 labels=list(names);fig,axes=plt.subplots(2,3,figsize=(15,9));tables=[]
 for row,bank in enumerate(['development','holdout']):
  paths={k:Path(p) for k,p in protocol['development_baselines'].items()} if bank=='development' else {k:a.study/('holdout_'+k) for k in protocol['development_baselines']}
  paths.update({str(s):a.study/(f'seed_{s}/development_final' if bank=='development' else f'holdout_policy_{s}') for s in protocol['seeds']})
  episodes={k:read(p/'results.json')['episodes'] for k,p in paths.items()};count=18 if bank=='development' else 16
  common=[i for i in range(count) if all(e[i]['success'] for e in episodes.values())];x=np.arange(len(labels));summaries=[v['evaluations'][bank+'_'+k] for k in labels]
  ax=axes[row,0];ax.bar(x-.2,[s['successes'] for s in summaries],.4,label='Stable');ax.bar(x+.2,[s['physical_failures'] for s in summaries],.4,label='Physical failures',color='#bb514a');ax.set_ylim(0,count+1);ax.legend(fontsize=8);ax.set_title(f'{bank.capitalize()} · {count} trials')
  for ax,window,title in zip(axes[row,1:],['metrics_whole_trial','metrics_last5s'],['Whole20s torque²','Final5s torque²']):
   values=[np.mean([episodes[k][i][window]['total_squared_torque_Nm2'] for i in common]) if common else np.nan for k in labels];ax.bar(x,values,color=['#8493a5']*3+['#1a877b','#46af9d']);ax.set_title(f'{title} · {len(common)} shared successes');ax.set_ylabel('Nm²')
  for ax in axes[row]:ax.set_xticks(x,[names[k] for k in labels],rotation=35,ha='right');ax.grid(axis='y',alpha=.15)
  rows=''.join(f'<tr><td>{names[k]}</td><td>{s["successes"]}/{count}</td><td>{s["strict_successes"]}/{count}</td><td>{s["physical_failures"]}</td><td>{len(s["slip_cases"])}</td></tr>' for k,s in zip(labels,summaries))
  tables.append(f'<h2>{bank.capitalize()}</h2><table><tr><th>Controller</th><th>Stable</th><th>Strict clean</th><th>Physical failures</th><th>Slip cases</th></tr>{rows}</table>')
  cells=[]
  for seed in protocol['seeds']:
   for b in protocol['development_baselines']:
    c=v['comparisons'][f'{bank}_{seed}'][b];g=c['grouped']['all'];values=[g[t]['total_squared_torque_Nm2']['change_percent'] for t in ['metrics_whole_trial','metrics_last10s','metrics_last5s']]
    cells.append(f'<tr><td>{seed}</td><td>{names[b]}</td><td>{len(c["common_success_cases"])}</td>'+''.join(f'<td>{z:+.3f}%</td>' if z is not None else '<td>no common successes</td>' for z in values)+'</tr>')
  tables.append('<table><tr><th>New PPO</th><th>Baseline</th><th>Common successes</th><th>Whole20s torque²</th><th>Final10s</th><th>Final5s</th></tr>'+''.join(cells)+'</table>')
  for name,path in paths.items():
   dest=a.output/'evidence'/bank/name;dest.mkdir(parents=True)
   for fn in ['results.json','startup.npz','traction_startup.npz','episode_decisions.npz','manifest.json','geometry.json','first_failures.json','sensor_audit.json','traction_contract.json','traction_config.json','episode_contract.json']:
    if (path/fn).exists():shutil.copy2(path/fn,dest/fn)
 fig.suptitle('Persistent posture PPO: stability and whole-body effort');fig.tight_layout();fig.savefig(a.output/'comparison.png',dpi=150);plt.close(fig)
 description='Two fresh seeds,10 complete-episode PPO updates each,368100 total physics transitions including indexed warmup. All resets start standing. The repaired reference controls the first11s, then the policy chooses three bounded symmetric posture offsets held until20s. Nine development surfaces and eight reserved surfaces, repeated twice. Direct unclipped all37-joint squared-torque reward; complete outcome credit. This bundles learning changes, not a single-factor ablation. No policy is promoted.'
 (a.output/'findings.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Persistent posture PPO comparison</title><style>body{font:17px/1.5 system-ui;max-width:1400px;margin:30px auto;padding:0 24px;color:#23394b}table{border-collapse:collapse;width:100%;margin:18px 0}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}img{width:100%}a{color:#137d71}</style><h1>Complete-outcome posture learning</h1><p>'+description+'</p>'+''.join(tables)+'<p>Negative effort changes mean savings. Each pair uses its own full-duration common-success set; no savings from failed fragments. Strict clean adds zero all-tick slipping, unwanted contacts and conservative traction-margin flags. Prefix flags remain visible. Absolute/RMS torque, leg/upper-body contributions, all surface groups and every-tick contacts are in the verification file.</p><img src="comparison.png"><p>Torque is an effort proxy, not energy or a global optimum. Two seeds and nearby rigid slabs do not establish arbitrary-furniture generalization. Ground truth is used for rewards/evaluation only, not runtime actor or critic inputs.</p><p><a href="motion_replay.html">Matched replay</a> · <a href="verification.json">Complete independent verification and comparisons</a> · <a href="protocol.json">Frozen protocol</a></p>')
 for src,name in [(a.verified/'verification.json','verification.json'),(a.study/'protocol.json','protocol.json'),(a.study/'holdout_barrier.json','holdout_barrier.json')]:shutil.copy2(src,a.output/name)
 for seed in protocol['seeds']:
  dest=a.output/'training'/str(seed);dest.mkdir(parents=True)
  for name in ['initial.pt','final.pt','episode_contract.json','learning_audit.json','decision_batches.pt','episode_decisions.npz','progress.json','manifest.json','control_accounting.json']:shutil.copy2(a.study/f'seed_{seed}/train'/name,dest/name)
 export(a.study,a.output/'motion_replay.html');shutil.copy2(__file__,a.output/'report_source.py');shutil.copy2(a.verified/'verification_source.py',a.output/'verification_source.py')
 (a.output/'SHA256.json').write_text(json.dumps({str(p.relative_to(a.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output.rglob('*') if p.is_file()},indent=2));print(a.output)
if __name__=='__main__':main()
