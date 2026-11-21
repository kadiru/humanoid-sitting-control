"""Check replay alarm times against the trajectories and measure when support force first becomes available."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np


def first(mask):
    hits=np.flatnonzero(mask)
    return int(hits[0]) if len(hits) else None


def verify(archive, experiment):
    report=json.loads((experiment/'report.json').read_text())
    with (experiment/'paired_warnings.csv').open(newline='') as f:
        cases=list(csv.DictReader(f))
    with np.load(experiment/'scores.npz') as z:
        ticks=z['ticks']
        assert (np.diff(ticks)==5).all()
        for policy in ['timer','sensor']:
            values=z[policy]
            cal=[i for i,r in enumerate(cases) if r['split']=='calibration' and r['clean']=='True']
            threshold=max(min(p[j],p[j-1]) for p in values[cal] for j in range(1,len(ticks)))
            for r,p in zip(cases,values,strict=True):
                alarm=next((int(ticks[j]) for j in range(1,len(ticks)) if min(p[j],p[j-1])>threshold),-1)
                assert alarm==int(r[policy])
    banks=defaultdict(list)
    for row in cases:banks[row['bank']].append(row)
    totals=defaultdict(Counter)
    details=[]
    for name,group in banks.items():
        path=archive/'rollouts'/name/'evaluation'
        episodes=json.loads((path/'results.json').read_text())['episodes']
        with np.load(path/'trajectory.npz') as d:
            f,n,bw=(d[k] for k in ('physical_force_w','physical_normal_w','physical_bodyweight_N'))
            active=np.linalg.norm(f,axis=-1)>.01*bw[...,None]
            rest,inside,permitted=(d[k] for k in ('physical_rest_patch','physical_inside_patch','physical_permitted'))
            normal=(f*n).sum(-1)
            legal_load=(active&rest&inside&permitted&(normal>0)).any(-1)
            any_load=(active&rest).any(-1)
            slip=active&(d['physical_slip_m_s']>.02)
            forbidden=(d['wrong_force']>10)|(active&~inside).any(-1)
            np.testing.assert_array_equal(slip.any(-1),~d['no_slip'])
            np.testing.assert_array_equal(np.logical_or.accumulate(forbidden,axis=0),d['flags'][:,:,1])
        for i,(r,ep) in enumerate(zip(group,episodes,strict=True)):
            assert r['id']==ep['id'] and (r['clean']=='True')==ep['clean']
            event=first(slip[:,i].any(-1)|forbidden[:,i])
            assert int(r['onset'])==(-1 if event is None else event)
            if event is None:continue
            legal=first(legal_load[:,i])
            any_rest=first(any_load[:,i])
            # The post-step force at sample k is first usable by the controller at k+1.
            lead=None if legal is None else event-legal-1
            any_lead=None if any_rest is None else event-any_rest-1
            patch=rest[event,i]
            feet=bool((slip[event,i]&~patch).any())
            seat=bool((slip[event,i]&patch).any())
            new=dict(bank=name,id=r['id'],split=r['split'],first_event_tick=event,
                forbidden_at_onset=bool(forbidden[event,i]),feet_slipping_at_onset=feet,
                rest_patch_slipping_at_onset=seat,first_legal_rest_force_tick=legal,
                legal_load_lead_ticks=lead,any_rest_load_lead_ticks=any_lead,
                sensor_alarm_tick=int(r['sensor']),sensor_alarm_outcome=r['sensor_result'])
            details.append(new)
            total=totals[r['split']]
            total.update(events=1,legal_load_by_100ms=int(lead is not None and lead>=5),
                         any_rest_load_by_100ms=int(any_lead is not None and any_lead>=5),
                         legal_load_by_20ms=int(lead is not None and lead>=1),
                         feet_slipping_at_first_event=int(feet),rest_slipping_at_first_event=int(seat),
                         forbidden_at_first_event=int(forbidden[event,i]))
    for split,s in totals.items():assert s['events']==report['results'][split]['sensor']['events']
    with (experiment/'contact_timing.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=details[0]);w.writeheader();w.writerows(details)
    (experiment/'contact_timing.json').write_text(json.dumps(totals,indent=2)+'\n')
    print(json.dumps(totals,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path);p.add_argument('experiment',type=Path)
    args=p.parse_args()
    verify(args.archive,args.experiment)
