"""Post-hoc diagnosis only; never replace the predeclared all-frame scores."""
import json
from pathlib import Path
import numpy as np
store=Path('/data/humanoid-sitting-perception');out=store/'support_landing_failure_audit_01';out.mkdir(exist_ok=False);report={}
for name in ['support_landing_validation_capture_01','support_landing_regression_capture_01','support_landing_fresh_capture_01']:
 cap=store/name
 with np.load(cap/'evaluation_labels.npz') as f:labels={k:f[k] for k in ['t','failure','true_clearance']}
 with np.load(cap/'proprioception.npz') as f:g=f['gravity']
 with np.load(cap/'shadow_score/tracking.npz') as f:z={k:f[k] for k in f.files}
 rows=json.loads((cap/'results.json').read_text())['episodes'];cases=[];pre=np.zeros_like(z['valid']);successful=np.zeros_like(pre)
 for i,row in enumerate(rows):
  failed=np.flatnonzero(labels['failure'][:,i]);first=int(failed[0]) if len(failed) else len(pre)
  pre[:first,i]=True
  if row['success']:successful[:,i]=True
  missing=~z['valid'][:,i];starts=np.flatnonzero(missing&np.r_[True,~missing[:-1]]);ends=np.flatnonzero(missing&np.r_[~missing[1:],True])
  longest=max(zip(starts,ends),key=lambda p:p[1]-p[0],default=None)
  item=dict(case=i,startup_ready=row['startup_ready'],motion_success=row['success'],final_failure_flags=row['failures'],first_physical_failure_s=float(z['t'][first]) if first<len(pre) else None)
  if longest:
   a,b=longest;item.update(longest_gap_start_s=float(z['t'][a]),longest_gap_s=(b-a+1)*.02,
       gap_start_tilt_deg=float(np.degrees(np.arccos(np.clip(-g[a,i,2]/np.linalg.norm(g[a,i]),-1,1)))),gap_starts_after_failure=bool(a>=first))
  cases.append(item)
 def metric(mask):
  valid=mask&z['valid'];e=abs(z['clearance']-labels['true_clearance'])*100
  return dict(frames=int(mask.sum()),accepted=int(valid.sum()),availability=float(valid.sum()/max(mask.sum(),1)),max_error_cm=float(e[valid].max()) if valid.any() else None)
 report[name]=dict(cases=cases,before_first_physical_failure=metric(pre),successful_trials_only=metric(successful),note='Conditional diagnostic; unsuccessful trials and post-failure frames remain in the primary validation scores. This does not change the failed gate.')
(out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
