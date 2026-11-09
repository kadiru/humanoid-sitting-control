"""Export verified complete results without hiding failures or failed fragments."""
import argparse,json,shutil,hashlib,html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from render_traction_replay import export
p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--verified',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
read=lambda p:json.loads(Path(p).read_text())
v=read(a.verified/'verification.json');assert v['passed'];protocol=read(a.study/'protocol.json');labels=list(protocol['development_baselines'])+[str(s) for s in protocol['seeds']]
names={'current_reference':'Current reference','current_56211':'PPO 56211','current_56217':'PPO 56217','bridge_reference':'Previous bridge','bridge_58111':'PPO 58111','bridge_58117':'PPO 58117','new_reference':'Traction repair',**{str(s):f'New PPO {s}' for s in protocol['seeds']}}
fig,axes=plt.subplots(2,3,figsize=(17,10));tables=[]
for row,bank in enumerate(['development','holdout']):
    paths={k:Path(path) for k,path in protocol['development_baselines'].items()} if bank=='development' else {k:a.study/('holdout_'+k) for k in protocol['development_baselines']}
    paths.update({str(s):a.study/(f'seed_{s}/development_final' if bank=='development' else f'holdout_policy_{s}') for s in protocol['seeds']})
    episodes={k:read(p/'results.json')['episodes'] for k,p in paths.items()};count=18 if bank=='development' else 16
    common=[i for i in range(count) if all(e[i]['success'] for e in episodes.values())];x=np.arange(len(labels))
    summaries=[v['evaluations'][f'{bank}_{k}'] for k in labels]
    ax=axes[row,0];ax.bar(x-.2,[s['successes'] for s in summaries],.4,label='Stable');ax.bar(x+.2,[s['physical_failures'] for s in summaries],.4,label='Physical failure',color='#c04c43');ax.set_ylim(0,count+1);ax.set_title(bank.capitalize()+f' • {count} trials');ax.legend(fontsize=8)
    for ax,window,title in zip(axes[row,1:],['metrics_whole_trial','metrics_last5s'],['Whole20s torque²','Final5s torque²']):
        values=[np.mean([episodes[k][i][window]['total_squared_torque_Nm2'] for i in common]) if common else 0 for k in labels]
        ax.bar(x,values,color=['#7b8ca6']*(len(labels)-2)+['#18847e','#39a89a']);ax.set_title(f'{title} • {len(common)} common successes');ax.set_ylabel('Nm²')
    for ax in axes[row]:ax.set_xticks(x,[names[k] for k in labels],rotation=45,ha='right');ax.grid(axis='y',alpha=.15)
    rows=''.join(f'<tr><td>{html.escape(names[k])}</td><td>{s["successes"]}/{count}</td><td>{s.get("strict_successes","not logged")}</td><td>{s["physical_failures"]}</td></tr>' for k,s in zip(labels,summaries))
    tables.append(f'<h2>{bank.capitalize()}</h2><table><tr><th>Controller</th><th>Stable full20s</th><th>Strict clean</th><th>Physical failures</th></tr>{rows}</table>')
    for name,path in paths.items():
        dest=a.output/'evidence'/bank/name;dest.mkdir(parents=True)
        for fn in ['results.json','startup.npz','traction_startup.npz','bridge_startup.npz','manifest.json','geometry.json','first_failures.json','sensor_audit.json','traction_contract.json','traction_config.json']:
            if (path/fn).exists():shutil.copy2(path/fn,dest/fn)
fig.suptitle('Traction-safe sitting: physical outcomes first; torque on shared successes only');fig.tight_layout();fig.savefig(a.output/'comparison.png',dpi=150);plt.close(fig)
description='Two fresh PPO seeds,64 scheduled rollouts each,368640 physics transitions including teacher and reset-warmup ticks. Every episode starts standing; learned adjustments begin at11s. Reference repair and incremental PPO gains are separate. Nine development geometries and eight withheld geometries, each repeated at two placements, are limited evidence. Torque is an effort proxy, not energy or proof of an optimum. No policy is promoted.'
(a.output/'findings.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Traction-safe sitting comparison</title><style>body{font:17px/1.5 system-ui;color:#213749;max-width:1400px;margin:30px auto;padding:0 24px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}img{width:100%}a{color:#167a78}</style><h1>Weight-transfer PPO comparison</h1><p>'+description+'</p>'+''.join(tables)+'<img src="comparison.png"><p>Strict clean adds zero slipping, unwanted contacts and grip-margin violations over the entire20s. Missing older strict counts are not zero. Effort summaries exclude failed fragments.</p><p><a href="motion_replay.html">Inspect the matched motion</a> · <a href="verification.json">Verified results, budgets, paired effort by height/slope and geometry errors</a> · <a href="protocol.json">Frozen protocol</a></p>')
for src,name in [(a.verified/'verification.json','verification.json'),(a.study/'protocol.json','protocol.json'),(a.study/'holdout_barrier.json','holdout_barrier.json')]:shutil.copy2(src,a.output/name)
for s in protocol['seeds']:
    dest=a.output/'training'/str(s);dest.mkdir(parents=True)
    for name in ['verification.json','control_accounting.json','progress.json','manifest.json','initial.pt','critic_warmup.pt','final.pt']:
        shutil.copy2(a.study/f'seed_{s}/train'/name,dest/name)
export(a.study,a.output/'motion_replay.html')
shutil.copy2(__file__,a.output/'report_source.py');(a.output/'SHA256.json').write_text(json.dumps({str(p.relative_to(a.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output.rglob('*') if p.is_file()},indent=2))
print(a.output)
