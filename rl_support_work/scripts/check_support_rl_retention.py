"""Predeclared evaluation gates; exit 2 means a valid experiment failed its gate."""
import argparse
import json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('gate',choices=['parity','easy','wide']);p.add_argument('run');a=p.parse_args()
store=Path('/data/humanoid-sitting-perception');run=store/a.run
rows=json.loads((run/'results.json').read_text())['episodes']
assert len(rows)==16 and all(r['steps']==1000 and r['duration_s']==20. for r in rows)
result=dict(gate=a.gate,run=a.run,successes=sum(r['success'] for r in rows),
            worst_impact_N=max(r['peak_force_N'] for r in rows))
if a.gate=='parity':
    old=store/'support_rl_pace_easy_prior_01'
    x=np.load(old/'rollout.npz');y=np.load(run/'rollout.npz')
    keys=('t','body','root','support','foot','torque','force','tilt','live','true_clearance','correction','alive')
    result['trace_bitwise_equal']={k:np.array_equal(x[k],y[k]) for k in keys}
    result['passed']=all(result['trace_bitwise_equal'].values()) and result['successes']==16
elif a.gate=='easy':
    result['passed']=result['successes']==16
else:
    parent=json.loads((store/'support_rl_pace_wide_prior_01/results.json').read_text())['episodes']
    result['lost_cases']=[i for i,(x,y) in enumerate(zip(parent,rows)) if x['success'] and not y['success']]
    result['gained_cases']=[i for i,(x,y) in enumerate(zip(parent,rows)) if not x['success'] and y['success']]
    result['passed']=(not result['lost_cases'] and bool(result['gained_cases']) and
                      result['worst_impact_N']<=max(r['peak_force_N'] for r in parent))
(run/'gate.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
raise SystemExit(0 if result['passed'] else 2)
