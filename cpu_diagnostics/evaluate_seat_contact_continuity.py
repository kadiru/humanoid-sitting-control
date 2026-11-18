"""Test grip release on the previous fresh chairs, then check retention and generalization."""

import argparse
from pathlib import Path

import numpy as np

from compare_touchdown_ppo import verify_audit
from evaluate_seat_plane_approach import evaluate
from evaluate_seat_slip import paired, reserve
from study_live_sitting import read, sha, write


def known_groups(previous):
    groups = ['retain_' + group for group in read(previous / 'summary.json')['retention']]
    return [*groups, 'matched', 'fresh']


def placement(row):
    return tuple(row['top_center'] + row['rotation_xyzw'])


def prepare(out, previous):
    fresh = reserve(out, previous, seed=461593, prefix='continuity_holdout')
    seen = set()
    count = 0
    for name in known_groups(previous):
        manifest = read(previous / name / 'evaluation' / 'manifest.json')
        rows = read(Path(manifest['args']['bank']) / 'geometry.json')
        count += len(rows)
        seen.update(map(placement, rows))
    reserved = {placement(row) for row in read(fresh / 'geometry.json')}
    assert count == 404 and len(reserved) == 24 and seen.isdisjoint(reserved)
    return fresh


def trial(out, reference, bank=None):
    result = paired(out, reference, 'verify_contact', bank, inputs=[Path(__file__)])
    _, before = verify_audit(reference / 'evaluation')
    _, after = verify_audit(out / 'evaluation')
    result['new_illegal'] = [b['id'] for a, b in zip(before, after, strict=True)
                             if not a['illegal_ticks'] and b['illegal_ticks']]
    with np.load(out / 'evaluation/seat_shear_feedback.npz') as z:
        result['released_contacts'] = {
            after[i]['id']: dict(tick=int(t), travel_m=float(z['patch_travel_m'][i]))
            for i, t in enumerate(z['release_tick']) if t >= 0}
    return result, after


def run(out, previous):
    fresh = prepare(out, previous)
    stages = out / 'rollouts'
    matched, rows = trial(stages / 'matched', previous / 'fresh')
    by_id = {row['id']: row for row in rows}
    stool = by_id['shear_holdout/round_stool/2']
    chair = by_id['shear_holdout/dining_chair/2']
    no_new_failures = not any(matched[k] for k in (
        'regressions', 'lost_stable', 'new_slips', 'new_forbidden', 'new_illegal'))
    more_clean = matched['counts']['new']['clean'] > matched['counts']['old']['clean']
    qualifies = bool(stool['clean'] and not chair['physical_failures'][1] and more_clean and no_new_failures)
    summary = dict(matched=matched, qualified=qualifies, stool=stool, dining_chair=chair)
    write(out / 'summary.json', summary)
    print('matched', matched['counts'], 'qualified', qualifies, flush=True)
    if not qualifies:
        return
    summary['known'] = {}
    for name in known_groups(previous):
        result, _ = trial(stages / ('known_' + name), previous / name)
        summary['known'][name] = result
        write(out / 'summary.json', summary)
        print(name, result['counts'], flush=True)
    assert sum(item['total'] for item in summary['known'].values()) == 404
    assert all(sha(path) == digest for path, digest in read(out / 'reservation.json').items())
    evaluate(stages / 'fresh_reference', previous / 'fresh', bank=fresh,
             inputs=[out / 'reservation.json'])
    result, _ = trial(stages / 'fresh', stages / 'fresh_reference', fresh)
    assert result['total'] == 24
    summary['fresh'] = result
    write(out / 'summary.json', summary)
    print('fresh', result['counts'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('previous', type=Path)
    args = parser.parse_args()
    run(args.output.resolve(), args.previous.resolve())
