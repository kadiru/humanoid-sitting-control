"""Score geometry actually consumed by the actor, separately from tracker logs."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--capture',required=True);a=p.parse_args();cap=Path(a.capture)
def load(name):
    with np.load(cap/name) as f:return {k:f[k] for k in f.files}
c=load('controller.npz');truth=load('evaluation_labels.npz');roll=load('rollout.npz')
t=c['t'];v=c['policy'][:,:,132:142];valid=v[:,:,8]>.5
assert not valid[0].any(),'Warm-up command must not invent geometry'
np.testing.assert_allclose(t[1:],truth['t'][:-1],atol=1e-7)
error=np.full(valid.shape,np.nan);error[1:]=abs(v[1:,:,0]-truth['true_clearance'][:-1])*100
near=np.zeros_like(valid)
for case in range(valid.shape[1]):
    contact=np.flatnonzero(roll['support'][:,case]>.25)
    if len(contact):
        event=roll['t'][contact[0]];near[:,case]=(t>=event-1-1e-6)&(t<=event+.5+1e-6)
def score(mask):
    accepted=mask&valid;e=error[accepted]
    return dict(frames=int(mask.sum()),accepted=int(accepted.sum()),availability=float(accepted.sum()/max(1,mask.sum())),
        max_error_cm=float(e.max()) if len(e) else None,p95_error_cm=float(np.quantile(e,.95)) if len(e) else None,
        errors_above_2cm=int((e>2).sum()))
bad=np.argwhere(valid&(error>2))
r=dict(all=score(np.ones_like(valid)),near_contact=score(near),
    bad_frames=[dict(t=float(t[i]),case=int(j),error_cm=float(error[i,j])) for i,j in bad],
    scope='Actual 144-channel actor packets, all 1000 commands including initial unavailable warm-up',
    truth_inputs='Evaluation only; no runtime controller import')
(cap/'actor_geometry_score.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
