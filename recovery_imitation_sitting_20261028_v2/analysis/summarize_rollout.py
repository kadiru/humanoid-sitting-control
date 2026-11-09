"""Read-only explanation of complete, independently audited sitting trials."""
import argparse,json
from pathlib import Path
import numpy as np

def summarize(root):
    root=Path(root)
    audit=json.loads((root/'independent_audit.json').read_text());assert audit['passed']
    rows=json.loads((root/'results.json').read_text())['episodes']
    assert len(rows)==72 and all(r['steps']==1000 and r['full_standing_start'] for r in rows)
    with np.load(root/'trajectory.npz',allow_pickle=False) as z:
        failure=z['failure'];flags=z['flags'];good=z['good'];no_slip=z['no_slip'];legal=z['legal']
        data=[]
        for i,row in enumerate(rows):
            hits=np.flatnonzero(failure[:,i]);t=int(hits[0]) if len(hits) else None
            first=None if t is None else dict(time_s=(t+1)*.02,reasons=[n for n,v in zip(['fall','forbidden_contact','torque','impact','joint_margin'],flags[t,i]) if v])
            data.append(dict(id=row['id'],clean=row['clean'],stable=row['stable'],first_failure=first,
                good_ticks=int(good[:,i].sum()),slip_ticks=int((~no_slip[:,i]).sum()),illegal_ticks=int((~legal[:,i]).sum()),final_hold_s=row['final_hold_s']))
        recovery='recovery_obs' in z.files
        additions={}
        if recovery:
            prior=z['recovery_prior'];teacher=prior+np.clip(.1*(z['recovery_target']-prior),-.003,.003)
            delta=abs(z['command'][:,:,:23]-teacher).max(-1)
            learner=z['recovery_learner']
            additions=dict(learner_ticks_per_case=learner.sum(0).tolist(),max_learner_command_deviation_rad=float(delta[learner].max()),
                clean_with_actual_disturbance=int((np.asarray(audit['clean'])&((delta>2e-7)&learner).any(0)).sum()))
        return dict(root=str(root),scope='teacher-assisted recovery' if recovery else 'autonomous policy',
            clean=audit['clean_count'],stable=int(sum(audit['stable'])),cases=data,**additions)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();print(json.dumps(summarize(a.root),indent=2))
