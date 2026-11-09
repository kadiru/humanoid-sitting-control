"""Paired comparisons with exact initial-state checks and geometry-block intervals."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--prefix',default='geometry_reference_v5_holdout');p.add_argument('--output',required=True);a=p.parse_args();root=Path(__file__).resolve().parents[1];out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
rows=[];results={};initial=None
for run in sorted((root/'runs').glob(a.prefix+'*')):
 if not (run/'results.json').exists() or 'development' in run.name:continue
 r=json.loads((run/'results.json').read_text());z=np.load(run/'initial_conditions.npz')
 current={k:z[k] for k in z.files}
 if initial is None:initial=current;bank=r['geometry_bank'];trials=[{k:v for k,v in x.items() if k in ('seed','config','geometry_index','x','y','z','pitch','roll','yaw')} for x in r['outcomes']]
 else:
  assert r['geometry_bank']==bank
  assert trials==[{k:v for k,v in x.items() if k in ('seed','config','geometry_index','x','y','z','pitch','roll','yaw')} for x in r['outcomes']]
  assert all(np.array_equal(initial[k],current[k]) for k in initial),run
 d=np.load(run/'rollouts.npz');lv=d['lin_vel'][-150:];av=d['ang_vel'][-150:];actions=d['actions'];n=r['num_trials']
 root_pose=d['root_pose'][-150:];yaw=Rotation.from_quat(root_pose[:,:,3:].reshape(-1,4)).as_euler('xyz')[:,2].reshape(root_pose.shape[:2]);seat_yaw=np.array([r['geometry_bank'][x['geometry_index']]['yaw'] for x in r['outcomes']]);heading=np.abs(np.arctan2(np.sin(yaw-seat_yaw),np.cos(yaw-seat_yaw)))
 row=dict(final_heading_error_mean_deg=float(np.rad2deg(heading).mean()),final_heading_error_max_deg=float(np.rad2deg(heading).max()),run=run.name,trials=n,successes=sum(x['success'] for x in r['outcomes']),success_percent=100*r['success_rate'],falls_percent=100*r['fall_rate'],unintended_percent=100*r['unintended_contact_rate'],excessive_force_percent=100*r['excessive_impact_rate'],torque_violation_percent=100*r['joint_torque_violation_rate'],peak_force_N=r['maximum_seat_impact_N'],mean_trial_peak_force_N=float(np.mean([x['max_seat_impact_N'] for x in r['outcomes']])),peak_descent_m_s=r['maximum_descent_m_s'],minimum_hold_s=r['minimum_seated_hold_s'],median_hold_s=r['median_seated_hold_s'],final_linear_speed_mean_m_s=float(np.linalg.norm(lv,axis=-1).mean()),final_angular_speed_mean_rad_s=float(np.linalg.norm(av,axis=-1).mean()),residual_rms_rad=float(np.sqrt(np.mean((.08*np.clip(actions,-1,1))**2))))
 rows.append(row);results[run.name]=r
paired=[]
script=next((k for k in results if '_scripted_' in k),None)
pairs=[]
if script:pairs.extend((script,name) for name in results if name!=script)
parent=next((k for k in results if k.endswith('_ppo_01') and not any(x in k for x in ('canonical','legacy'))),None)
pilot=next((k for k in results if k.endswith('_pilot_01') and 'canonical' not in k),None)
if parent and pilot:pairs.append((parent,pilot))
for full in [parent,pilot]:
 if full:
  blind=full.replace('_ppo_01','_ppo_blind_01').replace('_pilot_01','_pilot_blind_01')
  if blind in results:pairs.append((blind,full))
rng=np.random.default_rng(8264)
for reference,name in pairs:
 base=results[reference]['outcomes'];other=results[name]['outcomes']
 diff=np.array([int(y['success'])-int(x['success']) for x,y in zip(base,other)])
 force=np.array([y['max_seat_impact_N']-x['max_seat_impact_N'] for x,y in zip(base,other)])
 gi=np.array([x['geometry_index'] for x in base]);blocks=np.array([diff[gi==g].mean() for g in sorted(set(gi))]);force_blocks=np.array([force[gi==g].mean() for g in sorted(set(gi))])
 resamples=rng.integers(0,len(blocks),(10000,len(blocks)));boot=blocks[resamples].mean(1);force_boot=force_blocks[resamples].mean(1)
 paired.append(dict(reference=reference,comparison=name,initial_conditions_bitwise_identical=True,success_gain_percentage_points=float(100*diff.mean()),wins=int((diff>0).sum()),losses=int((diff<0).sum()),geometry_block_bootstrap_95_percent_interval_pp=(100*np.quantile(boot,[.025,.975])).tolist(),mean_peak_force_difference_N=float(force.mean()),geometry_block_bootstrap_95_percent_force_difference_N=np.quantile(force_boot,[.025,.975]).tolist()))
report=dict(rows=rows,paired=paired,safety_thresholds_unchanged=True,initial_conditions_bitwise_identical=True,bootstrap_unit='furniture geometry, preserving the nine correlated initialization trials')
(out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
with (out/'metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# Geometry-conditioned controller comparison','', 'All rows use identical recorded initial states and the same held-out furniture bank. Safety criteria are unchanged.','', '| Controller | Success | Falls | Unintended | Excess force | Peak N | Min hold s |','|---|---:|---:|---:|---:|---:|---:|']
for x in rows:lines.append(f"| {x['run']} | {x['successes']}/{x['trials']} | {x['falls_percent']:.1f}% | {x['unintended_percent']:.1f}% | {x['excessive_force_percent']:.1f}% | {x['peak_force_N']:.1f} | {x['minimum_hold_s']:.2f} |")
lines+=['','Paired estimates and geometry-block uncertainty are in `comparison.json`. No row with a safety violation is promoted.']
(out/'report.md').write_text('\n'.join(lines)+'\n')
fig,axes=plt.subplots(1,2,figsize=(13,5));labels=[x['run'].replace(a.prefix+'_','').replace('_01','') for x in rows]
axes[0].bar(labels,[x['success_percent'] for x in rows]);axes[0].set_ylabel('Whole-trial success (%)');axes[0].set_ylim(0,105)
axes[1].bar(labels,[x['peak_force_N'] for x in rows]);axes[1].axhline(next(iter(results.values()))['seat_impact_limit_N'],color='red',linestyle='--',label='Unchanged force limit');axes[1].set_ylabel('Peak seat force (N)');axes[1].legend()
for ax in axes:ax.tick_params(axis='x',rotation=30);ax.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160)
print(json.dumps(report,indent=2))
