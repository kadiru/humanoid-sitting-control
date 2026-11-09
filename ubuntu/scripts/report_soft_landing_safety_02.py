"""Matched-bank, whole-trial safety report, including rejected candidates."""
import json,hashlib,shutil
from pathlib import Path
import numpy as np
out=Path('runs/soft_landing_02')
names=['soft_landing_actions_clean_01','soft_landing_actions_noisy_01','soft_landing_actions_fresh_clean_01','soft_landing_actions_fresh_noisy_01','soft_landing_smooth_clean_02','soft_landing_damping_clean_02','soft_landing_conservative_clean_02','soft_landing_conservative_noisy_02','soft_landing_conservative_fresh_clean_02','soft_landing_conservative_fresh_noisy_02','soft_landing_causal_height_02','soft_landing_causal_delay_02','soft_landing_causal_command_02']
rows=[]
for name in names:
 p=Path('runs')/name
 if not (p/'results.json').exists():continue
 r=json.loads((p/'results.json').read_text());z=np.load(p/'rollouts.npz');peak=z['seat_force_peak'].max(0);truth=np.concatenate((np.zeros_like(z['support'][:1]),z['support'][:-1]))>.05
 row={'run':name,'n':len(peak),'success':sum(o['success'] for o in r['outcomes']),'falls':sum(o['fall'] for o in r['outcomes']),'wrong_contacts':sum(o['unintended_contact'] for o in r['outcomes']),'impact_failures':sum(o['excessive_impact'] for o in r['outcomes']),'peak_force_N':dict(zip(['p50','p95','p99','max'],map(float,np.quantile(peak,[.5,.95,.99,1])))),'maximum_descent_m_s':r['maximum_descent_m_s'],'max_effort_Nm':r['maximum_observed_absolute_joint_torque_Nm'],'minimum_hold_s':r['minimum_seated_hold_s'],'max_final_linear_rms_m_s':max(o['final_3s_linear_rms_m_s'] for o in r['outcomes'])}
 if 'joint_torque' in z:row['absolute_effort_sample_p95_p99_Nm']=list(map(float,np.quantile(abs(z['joint_torque']),[.95,.99])))
 if 'estimated_contact' in z:
  pred=z['estimated_contact'][...,0]>.65;tp=(pred&truth).sum();fp=(pred&~truth).sum();fn=(~pred&truth).sum();early=[];lat=[]
  for i in range(len(peak)):
   if not truth[:,i].any():continue
   t=int(truth[:,i].argmax());before=np.flatnonzero(pred[:t,i]);after=np.flatnonzero(pred[t:,i])
   if len(before):early.append((before[0]-t)*.02)
   if len(after):lat.append(after[0]*.02)
  row.update(precision=float(tp/max(1,tp+fp)),recall=float(tp/max(1,tp+fn)),early_false_episodes_over_200ms=int(sum(v<-.2 for v in early)),earliest_false_relative_to_load_s=min(early) if early else None,median_false_relative_to_load_s=float(np.median(early)) if early else None,post_onset_detection_latency_p50_p95_max_s=list(map(float,np.quantile(lat,[.5,.95,1]))) if lat else [])
 rows.append(row)
pairs=[]
for condition in ['clean','noisy']:
 for fresh in [False,True]:
  old='soft_landing_actions_'+('fresh_' if fresh else '')+condition+'_01';new='soft_landing_conservative_'+('fresh_' if fresh else '')+condition+'_02';a=Path('runs')/old;b=Path('runs')/new
  if not (b/'results.json').exists():continue
  za=np.load(a/'initial_conditions.npz');zb=np.load(b/'initial_conditions.npz');assert za.files==zb.files
  for k in za.files:np.testing.assert_array_equal(za[k],zb[k])
  ra=json.loads((a/'results.json').read_text());rb=json.loads((b/'results.json').read_text());assert ra['geometry_bank']==rb['geometry_bank'];assert ra['seat_impact_limit_N']==rb['seat_impact_limit_N']
  pairs.append({'baseline':old,'candidate':new,'initial_conditions_geometry_ceiling_identical':True})
(out/'metrics.json').write_text(json.dumps({'rows':rows,'matched_pairs':pairs,'promotion':False,'force_ceiling_N':632.527832,'detection_label':'sensor-time previous support > 0.05mg; not any-force touchdown; onset latency reported separately from premature positives','joint_effort_quantile_note':'Across all joints/timesteps incl parked hands; absolute max and original joint caps also retained in results.json'},indent=2))
checks=json.loads((out/'preservation.json').read_text());assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in checks['sha256'].items())
artifacts=[Path('runs')/n for n in names if n.endswith('_02')]+[out,Path('runs/contact_estimator_conservative_05'),Path('baselines/sensor-soft-landing-155-of-168')]
size=sum(f.stat().st_size for p in artifacts if p.exists() for f in p.rglob('*') if f.is_file())
size+=sum((Path('logs')/(n+'.log')).stat().st_size for n in names if n.endswith('_02') and (Path('logs')/(n+'.log')).exists())
size+=(Path('logs/contact_estimator_conservative_05.log')).stat().st_size
free=shutil.disk_usage('.').free/2**30;assert size<250*2**20 and free>30
(out/'integrity.json').write_text(json.dumps({'preserved_checkpoint_hashes_verified':True,'new_artifact_MiB':size/2**20,'free_GiB':free,'budget_MiB':250},indent=2));print(json.dumps(rows,indent=2))
