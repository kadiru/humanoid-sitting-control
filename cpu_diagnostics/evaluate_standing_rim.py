"""Paired standing-rim scan evaluations, gated on the original per-case physical scores."""
import argparse
from pathlib import Path
import numpy as np

from compare_touchdown_ppo import verify_audit
from evaluate_seat_contact_continuity import placement
from evaluate_seat_plane_approach import evaluate
from evaluate_seat_slip import reserve
from study_live_sitting import read, sha, write


def perception(folder):
    with np.load(folder/'standing_rims.npz') as z:
        rows=z['values']
    with np.load(folder/'standing_scan.npz') as z:
        np.testing.assert_array_equal(z['ticks'],[0,10,20,30,40])
    assert rows.shape[0]==5
    error=1000*(rows[:,:,0]-rows[:,:,1])
    direct=rows[:,:,6]==1
    count=direct.sum(0)
    finite=np.abs(error[np.isfinite(error)&direct])
    return dict(cases=len(count),frames=int(direct.size),detected_frames=int(direct.sum()),
                cases_with_four_direct_frames=int((count>=4).sum()),
                detections_per_case=count.tolist(),audited_frames=len(finite),
                p90_abs_error_mm=float(np.percentile(finite,90)) if len(finite) else None,
                max_abs_error_mm=float(finite.max()) if len(finite) else None)


def trial(out,ref,reservation,bank=None):
    evaluate(out,ref,bank=bank,braking='standing_rim',inputs=[Path(__file__),reservation])
    old,new=ref/'evaluation',out/'evaluation'
    manifests=[read(p/'manifest.json') for p in (old,new)]
    assert manifests[0]['bank_sha256']==manifests[1]['bank_sha256']
    assert manifests[0]['args']['seed']==manifests[1]['args']['seed']
    assert manifests[1]['action_limits_unchanged'] and manifests[1]['physical_criteria_unchanged']
    rows=[verify_audit(p)[1] for p in (old,new)]
    counts=[]
    for group in rows:
        assert all(r['steps']==1000 and r['full_standing_start'] for r in group)
        counts.append(dict(clean=sum(r['clean'] for r in group),stable=sum(r['stable'] for r in group),
                           slip=sum(bool(r['slip_ticks']) for r in group),
                           forbidden=sum(bool(r['physical_failures'][1]) for r in group)))
    with np.load(old/'trajectory.npz') as a,np.load(new/'trajectory.npz') as b:
        cmds=b['command']
        assert np.max(abs(cmds-b['previous']))<=.0030003
        difference=abs(cmds-a['command']).max(axis=(0,2))
    bad=[];repairs=[];details=[]
    for i,(a,b) in enumerate(zip(*rows,strict=True)):
        assert a['id']==b['id']
        worse=(a['clean'] and not b['clean'] or a['stable'] and not b['stable'] or
               not a['slip_ticks'] and b['slip_ticks'] or not a['illegal_ticks'] and b['illegal_ticks'] or
               np.any(np.logical_and(np.logical_not(a['physical_failures']),b['physical_failures'])))
        if worse: bad.append(a['id'])
        if b['clean'] and not a['clean']: repairs.append(b['id'])
        details.append(dict(id=a['id'],before=a,after=b,command_diff_rad=float(difference[i])))
    write(out/'comparison.json',details)
    return dict(total=len(details),before=counts[0],after=counts[1],repairs=repairs,regressions=bad,
                commands_changed=int((difference>0).sum()),perception=perception(new))


def run(out,previous):
    refs={name:previous/'rollouts'/name for name in read(previous/'summary.json')['known']}
    matched=previous/'rollouts/fresh'
    refs['previous_fresh']=matched
    banks=[Path(read(p/'evaluation/manifest.json')['args']['bank']) for p in refs.values()]
    shapes=[g for b in banks for g in read(b/'geometry.json')]
    assert len(shapes)==476
    bank=reserve(out,previous,seed=9262187,prefix='standing_rim_holdout')
    fresh=[placement(g) for g in read(bank/'geometry.json')]
    assert len(set(fresh))==24 and set(fresh).isdisjoint(map(placement,shapes))
    reservation=out/'reservation.json'
    stages=out/'rollouts'
    matched_result=trial(stages/'matched',matched,reservation)
    improved=matched_result['after']['clean']>matched_result['before']['clean'] and not matched_result['regressions']
    summary=dict(matched=matched_result,qualified=bool(improved))
    write(out/'summary.json',summary)
    print('matched',matched_result,flush=True)
    if improved:
        summary['known']={}
        for name,ref in refs.items():
            summary['known'][name]=trial(stages/name,ref,reservation)
            write(out/'summary.json',summary)
        assert sum(r['total'] for r in summary['known'].values())==476
        assert all(sha(p)==digest for p,digest in read(reservation).items())
        evaluate(stages/'fresh_reference',matched,bank=bank,inputs=[reservation])
        summary['fresh']=trial(stages/'fresh',stages/'fresh_reference',reservation,bank)
        assert summary['fresh']['total']==24
        write(out/'summary.json',summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('previous',type=Path)
    args=parser.parse_args()
    run(args.output.resolve(),args.previous.resolve())
