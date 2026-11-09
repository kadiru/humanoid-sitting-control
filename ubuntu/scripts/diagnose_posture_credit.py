"""Audit learning limitations and sustained-pose potential on development only."""
import json,hashlib,math
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from verify_seat_load_study import evaluation,starts,compare
from verify_progressive_seat_study import quality
from posture_credit_common import DATA,OLD,read,sha
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 out=DATA/'diagnosis/verified_01';out.mkdir(exist_ok=False)
 ref=OLD/'env_reference_01';rs,rr,rg=evaluation(ref,18);quality(rs,rr)
 trace=np.load(ref/'trace.npz');times=trace['t'];total=trace['effort']
 result={'probes':{},'old_learning':{},'reference_effort_clip_fraction':{},'rollout_seconds':3.2,'gae_efold_seconds':-.1/math.log(.9975*.98),
         'scope':'Development only. Probe gains are prescribed motion, not learned PPO. Causes are supported hypotheses, not a single-factor ablation.'}
 for label,mask in [('post11',times>11),('transfer11to14',(times>11)&(times<=14)),('rest15to20',times>15)]:
  result['reference_effort_clip_fraction'][label]=float((total[mask]>=600).mean())
 names=read(ref/'results.json')['joint_names'];sagittal=[i for i,n in enumerate(names) if any(x in n for x in ['hip_pitch','knee','ankle_pitch'])]
 for seed in [58311,58317]:
  folder=OLD/f'ppo_01/seed_{seed}/train';events=EventAccumulator(str(folder));events.Reload()
  loss=events.Scalars('Loss/value');used=events.Scalars('Loss/active_samples')
  ft=np.load(OLD/f'ppo_01/seed_{seed}/development_final/trace.npz');delta=ft['command'][times>15][:,:,sagittal]-trace['command'][times>15][:,:,sagittal]
  result['old_learning'][str(seed)]=dict(zero_sample_rollouts=sum(x.value==0 for x in used),total_rollouts=len(used),last_value_mse=next(x.value for x in reversed(loss) if x.value),
    resting_sagittal_command_change_abs_mean_rad=float(abs(delta).mean()),resting_sagittal_command_change_abs_max_rad=float(abs(delta).max()),
    training_control_ticks=read(folder/'control_accounting.json')['control_ticks'])
 for p in sorted((DATA/'probes').iterdir()):
  if not p.is_dir():continue
  summary,rows,geometry=evaluation(p,18);quality(summary,rows);startup=starts(ref,p);assert startup['exact']
  t=np.load(p/'trace.npz');mask=times<=11
  for k in ['q','body','command','reference','torque','geometry']:assert np.array_equal(trace[k][mask],t[k][mask]),(p,k)
  result['probes'][p.name]=dict(summary=summary,paired=compare(rows,rr,geometry),prefix_exact=True,contract=read(p/'probe_contract.json'))
 (out/'diagnosis.json').write_text(json.dumps(result,indent=2))
 labels=list(result['probes']);changes=[result['probes'][k]['paired']['grouped']['all']['metrics_last5s']['total_squared_torque_Nm2']['change_percent'] for k in labels]
 fig,ax=plt.subplots(figsize=(11,5));ax.barh([x.replace('_pitch','').replace('_',' ') for x in labels],changes,color=['#168477' if x<0 else '#b35a53' for x in changes]);ax.axvline(0,color='#263b49',lw=1);ax.set_xlabel('Resting all-joint torque² change vs repaired reference (%)');ax.set_title('Sustained ±0.008 rad adjustments: development-only physical probes')
 for i,x in enumerate(changes):ax.text(x+(.3 if x>=0 else -.3),i,f'{x:+.2f}%',va='center',ha='left' if x>=0 else 'right')
 ax.set_xlim(min(changes)-4,max(changes)+4);fig.tight_layout();fig.savefig(out/'diagnosis.png',dpi=150);plt.close(fig)
 rows=''.join(f'<tr><td>{k}</td><td>{v["summary"]["successes"]}/18</td><td>{v["summary"]["physical_failures"]}</td><td>{len(v["summary"]["slip_cases"])}</td><td>{v["paired"]["grouped"]["all"]["metrics_last5s"]["total_squared_torque_Nm2"]["change_percent"]:+.2f}%</td></tr>' for k,v in result['probes'].items())
 (out/'findings.html').write_text('''<!doctype html><meta charset="utf-8"><title>Why persistent posture credit</title><style>body{font:17px/1.5 system-ui;max-width:1100px;margin:32px auto;color:#203745;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}img{width:100%}</style><h1>Stable lower-effort poses exist inside the old limits</h1><p>These are prescribed development probes, not new PPO results. They preserve the entire first11 seconds of the repaired reference exactly. The two existing conservative touchdown margin flags remain reported.</p><table><tr><th>Persistent adjustment</th><th>Stable</th><th>Physical failures</th><th>Slip cases</th><th>Final5s torque²</th></tr>'''+rows+'''</table><img src="diagnosis.png"><p>The old learner used23 independent100ms actions,3.2-second rollouts and inaccurate bootstrapped values. Sustained small adjustments expose a clear effort signal. The new experiment tests three persistent symmetric posture offsets and full-episode direct effort credit, retaining the physical controller and stability criteria. This bundled change cannot isolate a single cause.</p><p><a href="diagnosis.json">Complete quantitative audit</a></p>''')
 (out/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in out.iterdir() if p.is_file()},indent=2))
 print(json.dumps({k:v for k,v in result.items() if k!='probes'},indent=2));print(dict(zip(labels,changes)))
if __name__=='__main__':main()
