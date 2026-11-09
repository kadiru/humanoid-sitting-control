"""Aggregate actual outcomes, per-family metrics, and residual-policy ablations."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];out=root/'runs/furniture_progress_01';out.mkdir(exist_ok=True)
rows=[];families=[];residuals=[]
for path in sorted((root/'runs').glob('furniture_*/results.json')):
 r=json.loads(path.read_text());row=dict(run=path.parent.name,trials=r['num_trials'],success_rate=r['success_rate'],fall_rate=r['fall_rate'],unintended_contact_rate=r['unintended_contact_rate'],impact_violation_rate=r['excessive_impact_rate'],peak_seat_force_N=r['maximum_seat_impact_N'],max_descent_m_s=r['maximum_descent_m_s'],min_hold_s=r['minimum_seated_hold_s'],median_hold_s=r['median_seated_hold_s'],geometry_seed=r['geometry_seed'],initialization_seeds=r['initialization_seeds'],reference_mode=r['reference_mode'],arm_clearance=r.get('arm_clearance',False),torque_violation_rate=r.get('joint_torque_violation_rate'))
 rows.append(row)
 for family in sorted(set(g['family'] for g in r['geometry_bank'])):
  outcomes=[o for o in r['outcomes'] if r['geometry_bank'][o['geometry_index']]['family']==family]
  families.append(dict(run=row['run'],family=family,trials=len(outcomes),success_rate=sum(o['success'] for o in outcomes)/len(outcomes),fall_rate=sum(o['fall'] for o in outcomes)/len(outcomes),unintended_contact_rate=sum(o['unintended_contact'] for o in outcomes)/len(outcomes),peak_seat_force_N=max(o['max_seat_impact_N'] for o in outcomes),max_descent_m_s=max(o['max_descent_m_s'] for o in outcomes),min_hold_s=min(o['seated_hold_s'] for o in outcomes)))
 if r['checkpoint'] and (path.parent/'rollouts.npz').exists():
  with np.load(path.parent/'rollouts.npz') as z:
   delta=.08*np.clip(z['actions'],-1,1)
   residuals.append(dict(run=row['run'],rms_joint_residual_rad=float(np.sqrt(np.mean(delta**2))),peak_joint_residual_rad=float(abs(delta).max())))
(out/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'families.json').write_text(json.dumps(families,indent=2)+'\n');(out/'residuals.json').write_text(json.dumps(residuals,indent=2)+'\n')
with (out/'metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
md=['# Rigid furniture progress','', 'Actual whole-trial evaluations; safety failures are retained. Force is the resultant of seat normal forces sampled every 5 ms. Every trial lasts 20 seconds. Hold duration can remain long in a failed trial because any earlier failure permanently vetoes success.','', '| Run | Trials | Success | Falls | Unintended contact | Peak force N | Descent m/s | Minimum hold s |','|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:md.append(f"| {r['run']} | {r['trials']} | {r['success_rate']:.1%} | {r['fall_rate']:.1%} | {r['unintended_contact_rate']:.1%} | {r['peak_seat_force_N']:.1f} | {r['max_descent_m_s']:.3f} | {r['min_hold_s']:.2f} |")
(out/'report.md').write_text('\n'.join(md)+'\n')
selected=[r for r in rows if 'video' not in r['run'] and 'ablation' not in r['run']]
fig,axes=plt.subplots(1,2,figsize=(16,max(5,len(selected)*.4)))
y=np.arange(len(selected));labels=[r['run'].replace('furniture_','') for r in selected]
axes[0].barh(y,[r['success_rate']*100 for r in selected],color=['seagreen' if r['success_rate']==1 else 'darkorange' for r in selected]);axes[0].set_yticks(y,labels);axes[0].set_xlabel('Whole-trial success (%)');axes[0].set_xlim(0,105)
axes[1].barh(y,[r['peak_seat_force_N'] for r in selected],color='steelblue');axes[1].set_yticks(y,labels);axes[1].set_xlabel('Maximum seat normal resultant (N)');axes[1].axvline(632.5278,color='red',linestyle='--')
for ax in axes:ax.invert_yaxis();ax.grid(axis='x',alpha=.3)
fig.tight_layout();fig.savefig(out/'progress.png',dpi=140)
print(json.dumps(rows[-3:],indent=2))
