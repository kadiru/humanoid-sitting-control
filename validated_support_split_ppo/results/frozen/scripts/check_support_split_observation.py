"""Prospective gates for the isolated observation-interface pilot."""
import argparse
import json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--reference');p.add_argument('--gate',action='store_true')
a=p.parse_args();run=Path(a.run)
with np.load(run/'controller.npz') as f:obs=f['policy'];stamp=f['t']
with np.load(run/'evaluation_labels.npz') as f:
    truth=f['true_clearance'];failure=f['failure'];support=f['support']
valid=obs[:,:,148]>0;patch=obs[:,:,140]>0
error=abs(obs[:,:,144]-truth)
before=np.ones_like(valid);landing=np.zeros_like(valid)
for case in range(valid.shape[1]):
    fail=np.flatnonzero(failure[:,case])
    if len(fail):before[fail[0]:,case]=False
    contact=np.flatnonzero(support[:,case]>.25)
    if len(contact):landing[:,case]=(stamp>=stamp[contact[0]]-1)&(stamp<=stamp[contact[0]]+.5)
def score(mask):
    accepted=valid & mask
    return dict(frames=int(mask.sum()),distance_availability=float(valid[mask].mean()) if mask.any() else None,
        patch_availability=float(patch[mask].mean()) if mask.any() else None,
        recovered_distance_frames=int((valid & ~patch & mask).sum()),
        max_accepted_error_m=float(error[accepted].max()) if accepted.any() else None)
metrics=json.loads((run/'results.json').read_text())
report=dict(all_frames=score(np.ones_like(valid)),before_first_failure=score(before),landing=score(landing),
    successes=metrics['successes'],trials=metrics['observed_complete_trials'])
if a.reference:
    ref=Path(a.reference)
    with np.load(ref/'rollout.npz') as old, np.load(run/'rollout.npz') as new:
        report['trajectory_parity']={key:bool(np.array_equal(old[key],new[key])) for key in old.files}
    with np.load(ref/'controller.npz') as old, np.load(run/'controller.npz') as new:
        report['old_policy_inputs_bitwise_equal']=bool(np.array_equal(old['policy'],new['policy'][:,:,:144]))
        report['actions_bitwise_equal']=bool(np.array_equal(old['actions'],new['actions']))
checks=dict(distance_accuracy=report['before_first_failure']['max_accepted_error_m']<.02,
    distance_coverage=report['before_first_failure']['distance_availability']>=.99,
    finite_observations=bool(np.isfinite(obs).all()),missing_distance_zero=bool((obs[:,:,:][~valid][:,144:149]==0).all()))
if a.reference:
    checks.update(known_success=metrics['successes']==16,action_parity=report['actions_bitwise_equal'],
                  trajectory_parity=all(report['trajectory_parity'].values()))
report['checks']=checks;report['passed']=all(checks.values())
(run/'split_observation_score.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
if a.gate:assert report['passed'],report
