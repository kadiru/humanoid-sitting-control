"""Matched standing-start trials with an independently scored conditional holdout."""
import argparse
from pathlib import Path

import numpy as np

from compare_touchdown_ppo import verify_audit
from seat_plane_demonstration import plane_features
from evaluate_seat_contact_continuity import placement
from evaluate_seat_plane_approach import compare, evaluate
from evaluate_seat_slip import reserve
from study_live_sitting import read, sha, write


def build_library(refs,out):
    entries, features, paths, inputs = [],[],[],{}
    for ref in refs:
        e = ref/'evaluation'
        _, rows = verify_audit(e)
        with np.load(e/'seat_plane_approach.npz') as z:
            active = z['active']
        clean = [i for i in active if rows[i]['clean']]
        if not clean:
            continue
        for name in ('trajectory.npz','edge_depth_samples.npz','results.json','seat_plane_approach.npz'):
            inputs[str(e/name)] = sha(e/name)
        with np.load(e/'edge_depth_samples.npz') as z:
            assert z['tick'][0] == 0
            plane, valid = plane_features(z['plane'][0])
        with np.load(e/'trajectory.npz') as z:
            desired = z['desired']
            for i in clean:
                if valid[i]:
                    entries.append(rows[i]['id'])
                    features.append(plane[i])
                    paths.append(desired[:,i])
    library = out/'demonstrations.npz'
    assert entries and len(entries) == len(set(entries))
    np.savez_compressed(library,features=features,desired=paths,ids=entries)
    write(out/'demonstrations.json',dict(sources=inputs,ids=entries,sha256=sha(library)))
    return library


def score(out, reference, library):
    result = compare(reference/'evaluation',out/'evaluation',out)
    with np.load(reference/'evaluation/edge_depth_samples.npz') as z:
        plane, valid = plane_features(z['plane'][0])
    with np.load(reference/'evaluation/seat_plane_approach.npz') as z:
        valid &= np.isin(np.arange(len(valid)),z['active'])
    with np.load(library) as z:
        donors = z['features']
    # Recompute the four corner distances independently of the worker's closed form.
    corners = np.array([[1,x,y] for x in (-.1,.1) for y in (-.125,.125)])
    delta = plane[:,None] - donors
    error = np.max(np.abs(delta @ corners.T),axis=-1)
    error[plane[:,None,2]*donors[:,2] <= 0.] = np.inf
    closest = error.argmin(1)
    row = np.arange(len(valid))
    distance = error[row,closest]
    active = valid & (distance>.0005) & (distance<=.010) & (np.abs(delta[row,closest,0])<=.005)
    with np.load(out/'evaluation/plane_demonstration.npz') as z:
        np.testing.assert_array_equal(z['index'],np.where(active,closest,-1))
        selected = dict(zip(z['rows'].astype(int),z['selected_ids'],strict=True))
    with np.load(reference/'evaluation/velocity_brake.npz') as z:
        motion = z['trace']
    with np.load(reference/'evaluation/seat_shear_feedback.npz') as z:
        load = z['first_load_tick']
    fallback = np.full(len(valid),-1)
    for tick,i,gap,speed,*_ in motion:
        i = int(i)
        if (not active[i] and fallback[i]<0 and (load[i]<0 or tick<load[i])
                and .005<gap<=.025 and speed>.020):
            fallback[i] = int(tick)
    with np.load(out/'evaluation/seat_normal_landing.npz') as z:
        np.testing.assert_array_equal(z['start'],fallback)
    mask = ~active & (fallback<0)
    with np.load(reference/'evaluation/trajectory.npz') as a, np.load(out/'evaluation/trajectory.npz') as b:
        old_commands, new_commands = a['command'],b['command']
        np.testing.assert_array_equal(old_commands[:221],new_commands[:221])
        np.testing.assert_array_equal(old_commands[:750,mask],new_commands[:750,mask])
        result['inactive_post_ppo_command_difference_rad'] = float(np.max(
            np.abs(old_commands[750:,mask]-new_commands[750:,mask]),initial=0.))
    _, old = verify_audit(reference/'evaluation')
    _, new = verify_audit(out/'evaluation')
    pairs = list(zip(old,new,strict=True))
    result.pop('cases')
    result.update(lost_stable=[a['id'] for a,b in pairs if a['stable'] and not b['stable']],
                  new_slips=[a['id'] for a,b in pairs if not a['slip_ticks'] and b['slip_ticks']],
                  new_forbidden=[a['id'] for a,b in pairs if not a['physical_failures'][1] and b['physical_failures'][1]],
                  new_illegal=[a['id'] for a,b in pairs if not a['illegal_ticks'] and b['illegal_ticks']],
                  new_failure_modes={a['id']:np.flatnonzero(np.logical_and(np.logical_not(a['physical_failures']),b['physical_failures'])).tolist()
                                     for a,b in pairs if np.any(np.logical_and(np.logical_not(a['physical_failures']),b['physical_failures']))},
                  donors={new[i]['id']:str(name) for i,name in selected.items()})
    return result


def run(out,previous):
    fresh = reserve(out,previous,seed=1728399,prefix='landing_holdout')
    stages = out/'rollouts'
    refs = {'known_'+n:previous/'rollouts'/('known_'+n) for n in read(previous/'summary.json')['known']}
    matched = previous/'rollouts/fresh'
    library = build_library(refs.values(),out)
    refs['known_latest'] = matched
    bank = []
    for reference in refs.values():
        source = read(reference/'evaluation/manifest.json')['args']['bank']
        bank.extend(read(Path(source)/'geometry.json'))
    assert len(bank)==428 and set(map(placement,bank)).isdisjoint(map(placement,read(fresh/'geometry.json')))
    def trial(name,reference,bank=None):
        evaluate(stages/name,reference,bank=bank,plane_demos=library,inputs=[Path(__file__),out/'demonstrations.json',out/'reservation.json'])
        result = score(stages/name,reference,library)
        print(name,result['counts'],flush=True)
        return result
    check = trial('matched',matched)
    gate = check['counts']['new']['clean'] > check['counts']['old']['clean'] and not any(
        check[k] for k in ('regressions','lost_stable','new_slips','new_forbidden','new_illegal','new_failure_modes'))
    summary = dict(matched=check,qualified=gate)
    write(out/'summary.json',summary)
    if not gate:
        return
    summary['known'] = {}
    for name,reference in refs.items():
        summary['known'][name] = trial(name,reference)
        write(out/'summary.json',summary)
    assert sum(r['total'] for r in summary['known'].values())==428
    assert all(sha(path)==digest for path,digest in read(out/'reservation.json').items())
    evaluate(stages/'fresh_reference',matched,bank=fresh,inputs=[out/'reservation.json'])
    summary['fresh'] = trial('fresh',stages/'fresh_reference',fresh)
    write(out/'summary.json',summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('previous',type=Path)
    args = parser.parse_args()
    run(args.output.resolve(),args.previous.resolve())
