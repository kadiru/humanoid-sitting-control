"""Separate truth-based scoring of frozen sensor-only tracking output."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--tracking',required=True);a=p.parse_args();out=Path(a.tracking)
def load(path):
    with np.load(path) as archive:return {k:archive[k] for k in archive.files}
m=json.loads((out/'manifest.json').read_text());cap=Path(m['capture']);z=load(out/'tracking.npz')
truth=load(cap/'evaluation_labels.npz');depth=load(cap/'depth.npz');roll=load(cap/'rollout.npz')
np.testing.assert_array_equal(z['t'],truth['t']);error=abs(z['clearance']-truth['true_clearance'])*100
valid=z['valid'];near=np.zeros_like(valid);events={}
for i in range(valid.shape[1]):
    touch=np.flatnonzero(roll['support'][:,i]>.25)
    if len(touch):
        t=float(roll['t'][touch[0]]);events[str(i)]=t;near[:,i]=(z['t']>=t-1.-1e-6)&(z['t']<=t+.5+1e-6)
legacy_valid=[];new_valid=[];legacy_error=[];new_error=[];near_camera=[]
for stamp,case,old in zip(depth['t'],depth['case'],depth['legacy']):
    i=int(case);tick=int(round(stamp/.02))-1
    legacy_valid.append(bool(old[8]));new_valid.append(bool(valid[tick,i]));near_camera.append(bool(near[tick,i]))
    legacy_error.append(abs(old[0]-truth['true_clearance'][tick,i])*100 if old[8] else np.nan)
    new_error.append(error[tick,i] if valid[tick,i] else np.nan)
legacy_valid=np.array(legacy_valid);new_valid=np.array(new_valid);near_camera=np.array(near_camera)
def metrics(mask):
    selected=mask&valid;e=error[selected]
    return dict(frames=int(mask.sum()),accepted=int(selected.sum()),availability=float(selected.sum()/max(mask.sum(),1)),
        p95_error_cm=float(np.quantile(e,.95)) if len(e) else None,max_error_cm=float(e.max()) if len(e) else None,
        errors_above_2cm=int((e>2).sum()))
by_source={source:metrics(z['source']==i) for i,source in enumerate(m['sources']) if i}
bad=np.argwhere(valid&(error>2));bad_rows=[dict(case=int(i),t=float(z['t'][k]),error_cm=float(error[k,i]),source=m['sources'][z['source'][k,i]]) for k,i in bad]
def longest_gap(mask):
    longest=0;current=0
    for missing in mask:
        current=current+1 if missing else 0;longest=max(longest,current)
    return longest*.02
per_case={str(i):dict(accepted=int(valid[:,i].sum()),frames=len(valid),
    near_contact_accepted=int((valid[:,i]&near[:,i]).sum()),near_contact_frames=int(near[:,i].sum()),
    longest_gap_s=longest_gap(~valid[:,i]),longest_near_contact_gap_s=longest_gap(near[:,i]&~valid[:,i]),
    max_error_cm=float(error[valid[:,i],i].max()) if valid[:,i].any() else None) for i in range(valid.shape[1])}
report=dict(per_case=per_case,all=metrics(np.ones_like(valid)),near_contact=metrics(near),by_source=by_source,
    at_depth_frames=dict(frames=len(legacy_valid),legacy_accepted=int(legacy_valid.sum()),new_accepted=int(new_valid.sum()),
        legacy_max_error_cm=float(np.nanmax(legacy_error)),new_max_error_cm=float(np.nanmax(new_error)),
        near_contact_frames=int(near_camera.sum()),legacy_near_contact_accepted=int((near_camera&legacy_valid).sum()),
        new_near_contact_accepted=int((near_camera&new_valid).sum())),
    event_cases={i:dict(t=t,valid=bool(valid[int(round(t/.02))-1,int(i)]),
        error_cm=float(error[int(round(t/.02))-1,int(i)]) if valid[int(round(t/.02))-1,int(i)] else None) for i,t in events.items()},
    errors_above_2cm=bad_rows,source_counts=dict(Counter(m['sources'][i] for i in z['source'].ravel())),
    source_sha256=m['tracker_sha256'],fault=m['fault'])
(out/'score.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('errors_above_2cm','event_cases','per_case')},indent=2))
