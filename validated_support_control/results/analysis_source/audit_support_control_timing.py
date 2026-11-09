"""Post-hoc matched controller diagnostics; never imported by control code."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--store',default='/data/humanoid-sitting-perception');p.add_argument('--output',required=True)
a=p.parse_args();root=Path(a.store);out=Path(a.output);out.mkdir(exist_ok=False)
old=np.load(root/'support_rl_retention_easy_parent_01/rollout.npz')
report=dict(training=False,case=14,runs={},comparisons={})
data={}
for scope in ('none','startup','live','both'):
    folder=root/f'support_control_timing_{scope}_02'
    result=json.loads((folder/'results.json').read_text());roll=np.load(folder/'rollout.npz')
    control=np.load(folder/'controller.npz');data[scope]={k:control[k] for k in control.files}
    report['runs'][scope]=dict(successes=result['successes'],case14=result['episodes'][14],
        max_force_N=max(e['peak_force_N'] for e in result['episodes']),
        failed_cases=[e['env'] for e in result['episodes'] if not e['success']],
        startup_geometry=json.loads((folder/'admission_geometry.json').read_text())[0]['geometry'][14],
        original_trace_exact={k:bool(np.array_equal(old[k],roll[k])) for k in old.files})
baseline=data['none']
for scope,current in data.items():
    diff={}
    for key,lo,hi in [('proprio',0,75),('previous_targets',75,98),('startup_geometry',98,109),('nominal',109,132),('live_depth',132,142),('learned_contact',142,144)]:
        delta=np.abs(current['policy'][:,14,lo:hi]-baseline['policy'][:,14,lo:hi])
        changed=np.where(np.max(delta,axis=1)>1e-7)[0]
        diff[key]=dict(first_difference_s=float(current['t'][changed[0]]) if len(changed) else None,max_abs=float(delta.max()))
    for key in ('actions','command_target'):
        delta=np.abs(current[key][:,14]-baseline[key][:,14]);changed=np.where(delta.max(1)>1e-7)[0]
        diff[key]=dict(first_difference_s=float(current['t'][changed[0]]) if len(changed) else None,max_abs=float(delta.max()))
    report['comparisons'][scope]=diff
(out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
