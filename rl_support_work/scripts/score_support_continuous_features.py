"""Evaluate transported sensor points against simulator motion, offline only."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--capture',required=True);a=p.parse_args();cap=Path(a.capture)
with np.load(cap/'evaluation_labels.npz') as data:times=data['t'];poses=data['root_pose']
rows=[];errors=[];predicted=[]
for line in (cap/'tracking_feature_audit.jsonl').open():
    r=json.loads(line)
    if not r['usable']:continue
    i=int(round(r['t']/.02))-1;j=int(round(r['anchor_stamp']/.02))-1;case=r['case']
    assert abs(times[i]-r['t'])<1e-6 and abs(times[j]-r['anchor_stamp'])<1e-6
    pa=poses[j,case];pc=poses[i,case]
    Ra=Rotation.from_quat(pa[3:7]).as_matrix();Rc=Rotation.from_quat(pc[3:7]).as_matrix()
    truth=(np.array(r['anchor_probes_P'])@Ra.T+pa[:3]-pc[:3])@Rc
    error=float(np.linalg.norm(np.array(r['current_probes_P'])-truth,axis=1).max())
    errors.append(error)
    if r['patch_age_s']>0:predicted.append(error)
    if error>=.02:rows.append(dict(t=r['t'],case=case,error_m=error,source=r['reason']))
report=dict(samples=len(errors),predicted_samples=len(predicted),max_error_m=max(errors,default=None),
    predicted_max_error_m=max(predicted,default=None),predicted_p95_error_m=float(np.quantile(predicted,.95)) if predicted else None,
    errors_over_2cm=rows,scope='Rigid transport error of originally observed probe points; not absolute hidden furniture extent accuracy',
    truth_used='Offline relative pelvis transforms only; absent from runtime transport')
(cap/'patch_transport_score.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
