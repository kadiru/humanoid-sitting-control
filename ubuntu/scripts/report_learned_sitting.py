"""Collect paired whole-trial results with geometry-clustered uncertainty."""
import json,csv,argparse
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=True,parents=True)
root=out.resolve().parents[1]
paths={
 'teacher_A':'geometry_reference_v12_holdout_scripted_01','teacher_B':'geometry_reference_v12_confirmation_scripted_01',
 'old_residual_A':'geometry_reference_v12_holdout_ppo_01','old_residual_B':'geometry_reference_v12_confirmation_ppo_01',
 'first_MLP_A':'student_bc_holdout_01',
 'BC_small_A':'student_structured_holdout_01','BC_small_B':'student_structured_confirmation_01',
 'RL_small_A':'student_rl_holdout_01','RL_small_B':'student_rl_confirmation_01',
 'RL_small_no_clock_A':'student_rl_no_clock_01',
 'BC_expanded_A':'student_expanded_holdout_01','BC_expanded_B':'student_expanded_confirmation_01',
 'RL_expanded_A':'student_expanded_rl_holdout_01','RL_expanded_B':'student_expanded_rl_confirmation_01',
 'BC_expanded_fresh':'student_expanded_fresh_01','teacher_fresh':'student_fresh_teacher_01',
 'BC_expanded_blind_A':'student_expanded_blind_01','BC_expanded_no_clock_A':'student_expanded_no_clock_01','BC_expanded_no_balance_A':'student_expanded_no_balance_01',
 'DAgger_A':'student_dagger_holdout_01','DAgger_B':'student_dagger_confirmation_01','DAgger_fresh':'student_dagger_fresh_01',
 'DAgger_RL_A':'student_dagger_rl_holdout_01','DAgger_RL_B':'student_dagger_rl_confirmation_01','DAgger_RL_fresh':'student_dagger_rl_fresh_01',
 'DAgger_blind_A':'student_dagger_blind_01','DAgger_no_clock_A':'student_dagger_no_clock_01','DAgger_no_balance_A':'student_dagger_no_balance_01',
 'old_PPO_no_reference_A':'old_ppo_no_reference_01'}
rows=[];loaded={}
for label,dirname in paths.items():
 folder=root/'runs'/dirname
 if not (folder/'results.json').exists():continue
 r=json.loads((folder/'results.json').read_text());loaded[label]=(folder,r)
 metrics=dict(label=label,path=str(folder.relative_to(root)),geometry_seed=r['geometry_seed'],trials=r['num_trials'],successful=sum(o['success'] for o in r['outcomes']),fall_rate=r['fall_rate'],unintended_contact_rate=r['unintended_contact_rate'],excessive_impact_rate=r['excessive_impact_rate'],joint_torque_violation_rate=r['joint_torque_violation_rate'],peak_seat_force_N=r['maximum_seat_impact_N'],maximum_descent_m_s=r['maximum_descent_m_s'],minimum_hold_s=r['minimum_seated_hold_s'],median_hold_s=r['median_seated_hold_s'])
 metrics['success_rate']=metrics['successful']/metrics['trials'];metrics['families']={}
 for family in ('chair','bench','box','rigid_couch'):
  subset=[o for o in r['outcomes'] if r['geometry_bank'][o['geometry_index']]['family']==family]
  metrics['families'][family]=dict(trials=len(subset),successful=sum(o['success'] for o in subset),peak_seat_force_N=max(o['max_seat_impact_N'] for o in subset))
 rows.append(metrics)
paired=[]
for bank in ('A','B'):
 teacher=loaded.get('teacher_'+bank)
 if not teacher:continue
 for name in ('BC_small','RL_small','BC_expanded','RL_expanded','DAgger','DAgger_RL','old_residual'):
  key=name+'_'+bank
  if key not in loaded:continue
  folder,r=loaded[key];tf,tr=teacher
  assert r['geometry_bank']==tr['geometry_bank']
  z=np.load(folder/'initial_conditions.npz');tz=np.load(tf/'initial_conditions.npz')
  for field in ('root_pose','joint_pos'):assert np.array_equal(z[field],tz[field]),(key,field)
  assert [(x['seed'],x['config'],x['geometry_index']) for x in r['outcomes']]==[(x['seed'],x['config'],x['geometry_index']) for x in tr['outcomes']]
  delta=np.array([x['max_seat_impact_N']-y['max_seat_impact_N'] for x,y in zip(r['outcomes'],tr['outcomes'])]);group=np.array([x['geometry_index'] for x in r['outcomes']]);means=np.array([delta[group==i].mean() for i in np.unique(group)]);rng=np.random.default_rng(77);boot=means[rng.integers(0,len(means),(5000,len(means)))].mean(1)
  paired.append(dict(controller=key,reference='teacher_'+bank,exact_initial_conditions=True,mean_trial_peak_force_difference_N=float(delta.mean()),geometry_bootstrap_95CI_N=np.quantile(boot,[.025,.975]).tolist()))
with open(out/'metrics.csv','w') as f:
 flat=[{k:v for k,v in r.items() if k!='families'} for r in rows];w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
(out/'comparison.json').write_text(json.dumps(dict(metrics=rows,paired=paired),indent=2)+'\n')
fig,axs=plt.subplots(3,1,figsize=(12,10));labels=[r['label'] for r in rows];x=np.arange(len(rows))
axs[0].bar(x,[r['success_rate']*100 for r in rows]);axs[0].set_ylabel('Success %');axs[0].set_ylim(0,105)
axs[1].bar(x,[r['peak_seat_force_N'] for r in rows]);axs[1].axhline(632.5278,c='red',ls='--');axs[1].set_ylabel('Peak force N');axs[1].set_yscale('log')
axs[2].bar(x,[r['minimum_hold_s'] for r in rows]);axs[2].axhline(3,c='red',ls='--');axs[2].set_ylabel('Minimum hold s');axs[2].set_xticks(x,labels,rotation=60,ha='right')
for ax in axs[:2]:ax.set_xticks(x,[])
fig.tight_layout();fig.savefig(out/'comparison.png',dpi=140);plt.close(fig)
print(json.dumps(rows,indent=2))
