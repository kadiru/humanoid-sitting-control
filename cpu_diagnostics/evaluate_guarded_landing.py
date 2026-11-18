"""Check fallback decisions on the two regression banks and audited standing-start rollouts."""
import argparse
from pathlib import Path

import numpy as np

from compare_touchdown_ppo import verify_audit
from evaluate_seat_contact_continuity import placement
from evaluate_seat_plane_approach import compare, evaluate
from evaluate_seat_slip import reserve
from study_live_sitting import read, sha, write


TARGETS = {
    'shear_holdout/dining_chair/1': 'clean',
    'shear_holdout/padded_side_chair/1': 'clean',
    'landing_holdout/ladder_back/1': 'clean',
    'landing_holdout/two_seat_sofa/1': 'safe_contact',
}


def replay(reference):
    """Derive the first decision from the incumbent's saved pre-intervention measurements."""
    with np.load(reference/'velocity_brake.npz') as z:
        motion = z['trace']
    with np.load(reference/'plane_demonstration.npz') as z:
        indices = z['index']
        distances = z['worst_error_m']
    with np.load(reference/'seat_shear_feedback.npz') as z:
        loads = z['first_load_tick']
    with np.load(reference/'edge_depth_samples.npz') as z:
        planes = z['plane'][0]
    with np.load(reference/'seat_plane_approach.npz') as z:
        usable = np.isin(np.arange(len(loads)),z['active'])
    usable &= (planes[:,11]>0) & (planes[:,12]==0) & (abs(planes[:,4])>.80)
    established = usable & (distances<=.0005)
    offers = np.full(len(loads),-1)
    starts = offers.copy()
    speeds = np.full(len(loads),np.nan)
    for row in range(len(loads)):
        if not motion.size or indices[row]>=0:
            continue
        samples = motion[motion[:,1]==row]
        for j,(tick,_,gap,speed,*_) in enumerate(samples):
            if loads[row]>=0 and tick>=loads[row]:
                break
            if not (.005<gap<=.025 and speed>.020):
                continue
            offers[row] = int(tick)
            if j>0:
                last_tick,_,last_gap = samples[j-1,:3]
                speeds[row] = (last_gap-gap)/((tick-last_tick)*.02)
                if not established[row] and speeds[row] > .009+.005*.10/.02:
                    starts[row] = int(tick)
            break
    with np.load(reference/'seat_normal_landing.npz') as z:
        np.testing.assert_array_equal(offers,z['start'])
    return offers, starts, speeds, established


def trial(out,reference,reservation,bank=None):
    evaluate(out,reference,bank=bank,braking='guard_landing',inputs=[Path(__file__),reservation])
    old_dir = reference/'evaluation'
    new_dir = out/'evaluation'
    result = compare(old_dir,new_dir,out)
    offers, starts, speeds, established = replay(old_dir)
    with np.load(new_dir/'seat_normal_landing.npz') as z:
        np.testing.assert_array_equal(z['decision_tick'],offers)
        np.testing.assert_array_equal(z['established'],established)
        np.testing.assert_array_equal(z['start'],starts)
        np.testing.assert_allclose(z['closing_speed_m_s'],speeds,rtol=1e-12,equal_nan=True)
    with np.load(old_dir/'plane_demonstration.npz') as a,np.load(new_dir/'plane_demonstration.npz') as b:
        np.testing.assert_array_equal(a['index'],b['index'])
    veto = (offers>=0) & (starts<0)
    with np.load(old_dir/'trajectory.npz') as a,np.load(new_dir/'trajectory.npz') as b:
        old, new = a['command'],b['command']
        ticks = np.arange(len(old))[:,None]
        protected = (~veto | (ticks<offers)) & (ticks<750)
        np.testing.assert_array_equal(old[protected],new[protected])
    _, previous = verify_audit(old_dir)
    _, rows = verify_audit(new_dir)
    result.pop('cases')
    pairs = list(zip(previous,rows,strict=True))
    result.update(
        lost_stable=[b['id'] for a,b in pairs if a['stable'] and not b['stable']],
        new_slips=[b['id'] for a,b in pairs if not a['slip_ticks'] and b['slip_ticks']],
        new_illegal=[b['id'] for a,b in pairs if not a['illegal_ticks'] and b['illegal_ticks']],
        new_failure_modes={b['id']:np.flatnonzero(np.logical_and(~np.array(a['physical_failures']),b['physical_failures'])).tolist()
                           for a,b in pairs if np.any(np.logical_and(~np.array(a['physical_failures']),b['physical_failures']))},
        vetoed={rows[i]['id']:dict(tick=int(offers[i]),established=bool(established[i]),
                                   closing_m_s=float(speeds[i]) if np.isfinite(speeds[i]) else None)
                for i in np.flatnonzero(veto)},
        targets={r['id']:r for r in rows if r['id'] in TARGETS})
    print(out.name,result['counts'],flush=True)
    return result


def qualified(result):
    return (result['counts']['new']['clean']>result['counts']['old']['clean']
            and not any(result[k] for k in ('regressions','lost_stable','new_slips','new_illegal','new_failure_modes')))


def run(out,previous):
    bank = reserve(out,previous,seed=890321,prefix='guarded_landing_holdout')
    reservation = out/'reservation.json'
    refs = {name:previous/'rollouts'/name for name in read(previous/'summary.json')['known']}
    refs['known_last_holdout'] = previous/'rollouts/fresh'
    geometry = []
    for ref in refs.values():
        baseline = read(ref/'evaluation/manifest.json')['args']['bank']
        geometry.extend(read(Path(baseline)/'geometry.json'))
    assert len(geometry)==452 and set(map(placement,geometry)).isdisjoint(map(placement,read(bank/'geometry.json')))
    stages = out/'rollouts'
    screens = {}
    for name,ref in [('screen_known',refs['known_fresh']),('screen_latest',refs['known_last_holdout'])]:
        screens[name] = trial(stages/name,ref,reservation)
    targets = {key:r for result in screens.values() for key,r in result['targets'].items()}
    assert targets.keys()==TARGETS.keys()
    repaired = all(targets[key]['clean'] if criterion=='clean' else
                   targets[key]['stable'] and not targets[key]['physical_failures'][1] and not targets[key]['illegal_ticks']
                   for key,criterion in TARGETS.items())
    summary = dict(screens=screens,qualified=repaired and all(map(qualified,screens.values())))
    write(out/'summary.json',summary)
    if not summary['qualified']:
        return
    summary['known'] = {}
    for name,ref in refs.items():
        summary['known'][name] = trial(stages/name,ref,reservation)
        write(out/'summary.json',summary)
    assert sum(r['total'] for r in summary['known'].values())==452
    assert all(sha(p)==digest for p,digest in read(reservation).items())
    evaluate(stages/'fresh_reference',refs['known_last_holdout'],bank=bank,inputs=[reservation])
    summary['fresh'] = trial(stages/'fresh',stages/'fresh_reference',reservation,bank)
    assert summary['fresh']['total']==24
    write(out/'summary.json',summary)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('previous',type=Path)
    args = parser.parse_args()
    run(args.output.resolve(),args.previous.resolve())
