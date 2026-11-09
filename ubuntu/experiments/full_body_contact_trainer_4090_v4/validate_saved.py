"""Validate observable reward components on existing, untouched trajectories.

Full new reward replay awaits aligned50Hz capture: missing values are not
interpolated or presented as measurements.
"""
import argparse
from pathlib import Path
import numpy as np
from common import read, write, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    rows=[];sources={}
    for name in ['known_old','known_recent','known_transfer']:
        root=Path(a.inputs)/name
        with np.load(root/'every_tick.npz') as z:d={k:z[k] for k in z.files}
        with np.load(root/'trace.npz') as z:t={k:z[k] for k in z.files}
        episodes=read(root/'results.json')['episodes']
        loaded=np.linalg.norm(d['force'],axis=-1)>.01*d['weight'][...,None]
        noslip=~(loaded&(d['slip']>.02)).any(-1)
        np.testing.assert_array_equal(noslip,d['no_slip'])
        slip=np.where(loaded,d['slip'],0).max(-1);cost=.02*2*(slip/.02)**2
        assert np.all(.02*2*((slip+.001)/.02)**2>cost)
        wrong=.02*2*(d['wrong']/10)**2
        assert np.all(.02*2*((d['wrong']+.1)/10)**2>wrong)
        # Only evaluate components whose exact sampled fields are present.
        safe=t['alive']&t['legal'];good=safe&t['good'];phase=t['t'][:,None]>=15
        hold=2*good*phase
        assert not np.any(hold[~safe]) and not np.any(hold[~t['good']])
        for i,e in enumerate(episodes):
            assert e['slip_ticks']==int((~noslip[:,i]).sum())
            clean=bool(e['success'] and noslip[:,i].all() and d['legal'][:,i].all())
            rows.append(dict(id=f'{name}/{i}',clean=clean,stable=bool(e['success']),
                slip_ticks=int((~noslip[:,i]).sum()),slip_cost_total=float(cost[:,i].sum()),
                wrong_contact_cost_total=float(wrong[:,i].sum()),sampled_hold_positive_ticks=int((hold[:,i]>0).sum())))
        sources.update({str(root/n):sha(root/n) for n in ['trace.npz','every_tick.npz','results.json']})
    assert len(rows)==72 and sum(r['clean'] for r in rows)==64
    write(a.output,dict(passed=True,scope='Measured full-rate slip/contact costs and sampled hold gates only',
        full_new_reward_replay_complete=False,missing_full_rate_fields=['q and sensor/action alignment','peak impact history','continuous good-hold labels'],
        no_values_fabricated=True,rows=rows,sources=sources,physical_criteria_unchanged=True))


if __name__=='__main__':main()
