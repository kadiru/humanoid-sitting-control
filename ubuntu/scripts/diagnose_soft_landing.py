"""Offline force-label diagnostics; never imported by the deployed controller."""
import json
from pathlib import Path
import numpy as np
BANKS=['soft_landing_actions_clean_01','soft_landing_actions_noisy_01','soft_landing_actions_fresh_clean_01','soft_landing_actions_fresh_noisy_01','soft_landing_contact_novel_01']
rows=[];cases=[]
for name in BANKS:
 p=Path('runs')/name;r=json.loads((p/'results.json').read_text());z=np.load(p/'rollouts.npz');force=z['seat_force_peak']; peaks=force.max(0);support=z['support'];truth=np.concatenate((np.zeros_like(support[:1]),support[:-1]))>.05
 row={'run':name,'n':len(peaks),'successes':sum(o['success'] for o in r['outcomes']),'impact_N':dict(zip(['p50','p95','p99','max'],map(float,np.quantile(peaks,[.5,.95,.99,1])))),'maximum_joint_effort_Nm':r['maximum_observed_absolute_joint_torque_Nm'],'minimum_hold_s':r['minimum_seated_hold_s']}
 if 'estimated_contact' in z:
  pred=z['estimated_contact'][...,0]>.65;tp=(pred&truth).sum();fp=(pred&~truth).sum();fn=(~pred&truth).sum();row.update(precision=float(tp/max(1,tp+fp)),recall=float(tp/max(1,tp+fn)))
  onset=[];false_times=[]
  for i in range(len(peaks)):
   if truth[:,i].any():
    t=np.flatnonzero(truth[:,i])[0];before=np.flatnonzero(pred[:t,i]);after=np.flatnonzero(pred[t:,i]);
    if len(before):false_times.append(float((before[0]-t)*.02))
    if len(after):onset.append(float(after[0]*.02))
  row['first_false_detection_relative_to_contact_s']=list(map(float,np.quantile(false_times,[0,.5,1]))) if false_times else []
  row['post_onset_detection_latency_s']=list(map(float,np.quantile(onset,[.5,.95,1]))) if onset else []
 for i,o in enumerate(r['outcomes']):
  if not o['excessive_impact']:continue
  peak=int(force[:,i].argmax());touch=int(np.flatnonzero(support[:,i]>.05)[0]) if (support[:,i]>.05).any() else None
  c={'run':name,'trial':i,'geometry':o['geometry_index'],'config':o['config'],'peak_N':float(peaks[i]),'peak_time_s':peak*.02,'first_load_time_s':None if touch is None else touch*.02,'phase':'first_contact' if touch is not None and peak-touch<=10 else 'weight_transfer_or_hold'}
  if 'lin_vel' in z:c['pre_peak_root_vz_m_s']=float(z['lin_vel'][max(0,peak-1),i,2])
  if 'targets' in z:
   s=slice(max(0,peak-5),min(len(force),peak+2));c['local_max_target_step_rad']=float(abs(np.diff(z['targets'][s,i],axis=0)).max());c['local_max_effort_Nm']=float(abs(z['joint_torque'][s,i]).max());c['local_max_tracking_error_rad']=float(abs(z['targets'][s,i]-z['joint_pos'][s,i]).max())
  if 'estimated_contact' in z:
   d=np.flatnonzero(z['estimated_contact'][:,i,0]>.65);c['first_detection_relative_to_load_s']=float((d[0]-touch)*.02) if len(d) and touch is not None else None
  cases.append(c)
 rows.append(row)
out=Path('runs/soft_landing_02');(out/'baseline_diagnostics.json').write_text(json.dumps({'banks':rows,'failing_cases':cases,'causal_limit':'Associations only; geometry errors, delays and gain effects require matched interventions. First-load label is support >5% body weight, not infinitesimal touch.'},indent=2));print(json.dumps(rows,indent=2));print('Failure phases',[(c['run'],c['trial'],c['phase']) for c in cases])
