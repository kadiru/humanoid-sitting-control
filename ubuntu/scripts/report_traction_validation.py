"""Report physical preflight evidence; never interprets failed fragments as savings."""
import argparse,json,hashlib,html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path('/data/humanoid-sitting-perception/traction_transfer_01');out=Path(a.output);out.mkdir(exist_ok=False)
read=lambda p:json.loads(Path(p).read_text())
baselines=read(root.parent/'seat_load_transfer_01/ppo_01/protocol.json')['development_baselines'];baselines.pop('new_reference')
baselines['previous bridge']=str(root.parent/'crouch_rest_bridge_01/safe_env_startup_reference_01')
for seed in [58111,58117]:
    state=read(root.parent/f'crouch_rest_bridge_01/ppo_safe_surfaces_01/training_finished_{seed}.json')
    baselines[f'previous PPO {seed}']=state['final_development_evaluation']
baselines['traction repair']=str(root/a.candidate)
rows={k:read(Path(v)/'results.json')['episodes'] for k,v in baselines.items()}
candidate=rows['traction repair'];comparisons={}
for name,es in rows.items():
    assert len(es)==18 and all(e['steps']==1000 for e in es)
    pairs=[(x,y) for x,y in zip(candidate,es) if x['success'] and y['success']]
    comparisons[name]=dict(trials=len(es),stable=sum(e['success'] for e in es),strict=sum(e.get('strict_transfer',False) for e in es) if 'strict_transfer' in es[0] else None,physical=sum(any(e['failures'].values()) for e in es),common_successes=len(pairs),paired={window:{metric:dict(candidate=float(np.mean([x[window][metric] for x,y in pairs])),baseline=float(np.mean([y[window][metric] for x,y in pairs]))) for metric in ['total_squared_torque_Nm2','total_absolute_torque_Nm','all_joint_rms_Nm','leg_squared_torque_Nm2','upper_squared_torque_Nm2']} for window in ['metrics_whole_trial','metrics_last10s','metrics_last5s']} if pairs else {})
common=[i for i in range(18) if all(es[i]['success'] for es in rows.values())]
fig,axs=plt.subplots(1,3,figsize=(16,6));names=list(rows);colors=['#73839c']*(len(names)-1)+['#148a83']
axs[0].barh(names,[comparisons[n]['stable'] for n in names],color=colors);axs[0].set_xlim(0,18);axs[0].set_xlabel('Stable full 20-second trials / 18')
for ax,window,title in zip(axs[1:],['metrics_whole_trial','metrics_last5s'],['Whole motion','Final 5 seconds']):
    values=[np.mean([es[i][window]['total_squared_torque_Nm2'] for i in common]) if common else 0 for es in rows.values()]
    ax.bar(range(len(names)),values,color=colors);ax.set_xticks(range(len(names)),names,rotation=30,ha='right');ax.set_title(title);ax.set_ylabel('Mean sum of squared joint torque (Nm²)')
fig.suptitle(f'Traction repair preflight • {len(common)} common full-duration successes for effort');fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
summary=dict(candidate=str(root/a.candidate),comparisons=comparisons,common_cases=common,scope='Development only: nine rigid seat geometries repeated at two placements. Torque is an effort proxy, not energy. Reference/controller repair gains are not learned PPO gains. Failed fragments excluded from effort comparison. No held-out evaluation yet.')
(out/'summary.json').write_text(json.dumps(summary,indent=2))
table=''.join('<tr><td>'+html.escape(n)+'</td><td>'+str(v['stable'])+'/18</td><td>'+str(v['strict'])+'</td><td>'+str(v['physical'])+'</td></tr>' for n,v in comparisons.items())
(out/'findings.html').write_text('<!doctype html><meta charset="utf-8"><title>Traction transfer validation</title><style>body{font:17px system-ui;max-width:1200px;margin:36px auto;background:#f4f6f9;color:#203046}td,th{padding:10px 20px;text-align:left}img{width:100%}</style><h1>Weight-transfer validation</h1><p>'+html.escape(summary['scope'])+'</p><table><tr><th>Controller</th><th>Stable</th><th>Strict clean</th><th>Physical failures</th></tr>'+table+'</table><img src="comparison.png"><p>Strict clean means no slipping, friction-margin violation or unwanted contact during any active tick, with stable final support. Missing strict counts for older harnesses are not zero.</p>')
(out/'SHA256.json').write_text(json.dumps({str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*') if f.is_file()},indent=2))
print(json.dumps({n:{k:v[k] for k in ['stable','strict','physical','common_successes']} for n,v in comparisons.items()},indent=2))
