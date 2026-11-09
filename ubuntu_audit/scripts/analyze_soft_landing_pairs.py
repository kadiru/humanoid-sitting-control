"""Matched initial-condition audits and geometry-cluster bootstrap of ablation."""
import json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];out=root/'runs/soft_landing_01'
pairs=[('soft_landing_actions_clean_01','soft_landing_no_contact_clean_01'),('soft_landing_actions_noisy_01','soft_landing_limiter_confirmation_noise_01'),('soft_landing_actions_fresh_clean_01','soft_landing_limiter_fresh_clean_01'),('soft_landing_actions_fresh_noisy_01','soft_landing_limiter_fresh_noise_01'),('soft_landing_contact_novel_01','soft_landing_limiter_novel_01')]
checks=[];clusters={};all_force=[];all_success=[]
for contact,limiter in pairs:
 a=root/'runs'/contact;b=root/'runs'/limiter;ra=json.loads((a/'results.json').read_text());rb=json.loads((b/'results.json').read_text())
 za=np.load(a/'initial_conditions.npz');zb=np.load(b/'initial_conditions.npz');assert all(np.array_equal(za[k],zb[k]) for k in za.files);assert ra['geometry_bank']==rb['geometry_bank'];assert ra['seat_impact_limit_N']==rb['seat_impact_limit_N']
 for x,y in zip(ra['outcomes'],rb['outcomes']):
  assert (x['seed'],x['config'],x['geometry_index'])==(y['seed'],y['config'],y['geometry_index'])
  key=(ra['geometry_seed'],x['geometry_index']);d=x['max_seat_impact_N']-y['max_seat_impact_N'];s=int(x['success'])-int(y['success']);clusters.setdefault(key,[]).append((d,s));all_force.append(d);all_success.append(s)
 checks.append({'contact':contact,'limiter':limiter,'exact_initial_root_and_joints_match':True,'geometry_match':True,'unchanged_force_limit':True,'trials':ra['num_trials']})
keys=list(clusters);sums=np.array([np.array(clusters[k]).sum(0) for k in keys]);counts=np.array([len(clusters[k]) for k in keys]);rng=np.random.default_rng(517);sample=rng.integers(len(keys),size=(10000,len(keys)));bootstrap=sums[sample].sum(1)/counts[sample].sum(1)[:,None]
result={'paired_audits':checks,'episodes':len(all_force),'geometry_clusters':len(keys),'contact_minus_fixed_limiter_mean_peak_force_N':float(np.mean(all_force)),'geometry_cluster_bootstrap_95CI_force_N':np.quantile(bootstrap[:,0],[.025,.975]).tolist(),'contact_minus_fixed_limiter_success_fraction':float(np.mean(all_success)),'geometry_cluster_bootstrap_95CI_success_fraction':np.quantile(bootstrap[:,1],[.025,.975]).tolist(),'note':'Exploratory correlated simulator trials, not independent real-robot evidence. Negative force difference favors contact feedback.'}
# Also verify baseline-vs-controller matched banks and paired video starts.
for a,b in [('deployment_status_fresh_05','soft_landing_actions_clean_01'),('deployment_status_fresh_noisy_05','soft_landing_actions_noisy_01'),('soft_landing_baseline_fresh_clean_01','soft_landing_actions_fresh_clean_01'),('soft_landing_baseline_fresh_noisy_01','soft_landing_actions_fresh_noisy_01'),('soft_landing_baseline_video_01','soft_landing_contact_video_01')]:
 za=np.load(root/'runs'/a/'initial_conditions.npz');zb=np.load(root/'runs'/b/'initial_conditions.npz');assert all(np.array_equal(za[k],zb[k]) for k in za.files);checks.append({'baseline':a,'controller':b,'exact_initial_conditions_match':True})
(out/'paired_analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
