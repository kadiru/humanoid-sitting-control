"""Run the original twelve paired starts and audit both feet during descent."""
import argparse
import csv
from pathlib import Path
import shutil

import numpy as np

from run import DISK_RESERVE, P, PREPARED, read, sha, stage, write
import sitting_load_effort_study as score_module

BASE = P / 'sitting_load_effort_01' / 'low_shear'
STUDY = P / 'planted_descent_01'
SOURCES = ('planted_descent.py', 'planted_descent_worker.py', 'planted_descent_study.py')
NAMES = ('bank.npz', 'metadata.json', 'geometry.json')
ASSETS = {'Chair_Wire': dict(setback=.013, pitch=.105, roll=-.018), 'Anora': dict(pitch=.105)}
WIRE_HOLD = dict(recenter=True, setback=-.003, hold_start=579, hold_load=.25,
                 hold_gyro=.20, hold_tilt=.684, hold_samples=3)
TREATMENTS = {
    '1': dict(WIRE_HOLD, brake_start=572, brake_ticks=28, hold_start=602),
    '2': WIRE_HOLD,
    '3': dict(hold_start=580, hold_load=.32),
    '7': dict(press_start=601, press_depth_m=.0022),
    '8': dict(press_start=601, press_depth_m=.0022),
    '10': dict(lateral_m=.026, roll=-.047),
    '11': dict(hold_start=550, hold_load=.29),
}


def prepare():
    from rescan_sitting import points, shoe_frame
    from stance23_planner import model

    STUDY.mkdir()
    base_config = read(BASE / 'step_config.json')
    existing = P / 'post_step_rescan_01' / 'library_inputs.json'
    refs = read(existing)
    library = {}
    # Reload the same native, independently audited demonstrations for all four assets.
    demos = [(P / item['run'], item['index'], item['asset']) for item in refs]
    demos.append((P / 'protected_hip_landing_0' / 'evaluation', 14, 'Whitecliff_Ottoman'))
    for ev, row, name in demos:
        audit = read(ev / 'independent_audit.json')
        result = read(ev / 'results.json')
        assert audit['passed'] and audit['strict_predicates_exact'] and not audit['physical_tolerances_changed']
        assert sha(ev / 'trajectory.npz') == audit['files']['trajectory.npz']
        assert result['episodes'][row]['id'] == 'sit/' + name
        ids = [result['body_names'].index(n) for n in ('pelvis', 'left_ankle_roll_link', 'right_ankle_roll_link')]
        with np.load(ev / 'standing_scan.npz') as s, np.load(ev / 'trajectory.npz') as z:
            assert z['good'][-250:, row].all()
            _, mid, axes = shoe_frame(model, s['q'][0, row], s['gravity'][0, row])
            rotation = s['rotation'][0, row]
            body = z['body'][:, row, ids].astype(float)
            body[..., :3] -= result['origins'][row]
            load = z['physical_force_w'][:, row, :5, 2].sum(1) / z['physical_bodyweight_N'][:, row]
            cloud = points(s['depth'][0, row], s['intrinsics'][0, row], s['mount'][0], s['q'][0, row],
                           s['gravity'][0, row], model)
            fields = dict(origin=s['root'][0, row] + rotation @ mid, frame=rotation @ axes, cloud=cloud,
                          body=body, q=z['q'][:, row], desired=z['desired'][:, row],
                          force=z['physical_force_w'][:, row, 5:], loaded=int(np.flatnonzero(load > .05)[0]))
            library.update({name + '_' + k: v for k, v in fields.items()})
        if name == 'Whitecliff_Ottoman':
            refs.append(dict(run=str(ev), index=row, asset=name, trace_sha256=audit['files']['trajectory.npz'],
                             scan_sha256=sha(ev / 'standing_scan.npz'), contact_tick=fields['loaded']))
    assert set(base_config['templates_by_row']) == {d[2] for d in demos}
    np.savez_compressed(STUDY / 'native_sits.npz', **library)
    write(STUDY / 'references.json', refs)


def run(name, enabled, treatments):
    frozen = read(BASE / 'evaluation.freeze.json')
    for filename, digest in frozen['inputs'].items():
        assert sha(filename) == digest, filename
    target = STUDY / name
    target.mkdir()
    bank = target / 'bank'
    shutil.copytree(BASE / 'bank', bank)
    shutil.copyfile(BASE / 'landing_profile.json', target / 'landing_profile.json')
    config = read(BASE / 'step_config.json')
    extra = []
    if enabled:
        library = STUDY / 'native_sits.npz'
        config['planted_descent'] = dict(rows=enabled, library=str(library), assets=ASSETS,
                                        treatments=treatments)
        extra = [library, STUDY / 'references.json', *(P / s for s in SOURCES)]
    write(target / 'step_config.json', config)
    write(target / 'reservation.json', dict(files={str(f): sha(f) for f in bank.iterdir()}))
    cmd = frozen['command']
    stage(target, 'evaluation', bank, PREPARED / 'library.npz',
          worker=P / ('planted_descent_worker.py' if enabled else 'sitting_load_effort_worker.py'),
          extra=cmd[3:cmd.index('--mode')],
          inputs=[*map(Path, frozen['inputs']), *extra, target / 'landing_profile.json', target / 'step_config.json'],
          disk_reserve=DISK_RESERVE)
    if enabled:
        checked = read(target / 'evaluation' / 'planted_descent.json')
        assert {row['row'] for row in checked} == set(enabled)
    score(name)


def score(name):
    score_module.STUDY = STUDY
    rows = score_module.score(name)
    root = STUDY / name
    with np.load(root / 'evaluation' / 'trajectory.npz') as z:
        f = z['physical_force_w']
        bw = z['physical_bodyweight_N']
        ground = f[:, :, 5:, 2] / bw[:, :, None]
        active = np.linalg.norm(f[:, :, 5:], axis=-1) > .01 * bw[:, :, None]
        slips = active & (z['physical_slip_m_s'][:, :, 5:] > .02)
        flags = z['flags']
        for i, row in enumerate(rows):
            row['descent_min_foot_load_pct'] = round(float(ground[:, i].min()) * 100, 2)
            row['descent_feet_planted'] = bool((ground[:, i] > .05).all() and not slips[:, i].any())
            bad = np.flatnonzero(flags[:, i].any(1) | slips[:, i].any(1))
            row['first_flag_or_floor_slip_tick'] = int(bad[0]) if len(bad) else None
    write(root / 'measures.json', rows)


def paired(name, baseline):
    before, after = STUDY / baseline, STUDY / name
    for n in NAMES:
        assert sha(before / 'bank' / n) == sha(after / 'bank' / n) == sha(BASE / 'bank' / n)
    a, b = [read(d / 'measures.json') for d in (before, after)]
    old = read(BASE / 'measures.json')
    if baseline == 'registered':
        # The first run reached the simulator before the new hook was installed.
        # Use it as a baseline repeat only if its commands and gate histories
        # exactly reproduce the independently audited incumbent.
        assert not (before / 'evaluation' / 'planted_descent.json').exists()
        with np.load(before / 'evaluation' / 'trajectory.npz') as repeat, np.load(
                BASE / 'evaluation' / 'trajectory.npz') as incumbent:
            for key in ('command', 'good', 'flags', 'no_slip'):
                np.testing.assert_array_equal(repeat[key], incumbent[key])
    assert len(a) == len(b) == len(old) == 12
    protected = {row['case'] for row in old if row['supported']}
    assert len(protected) == 4
    for x, y in zip(a, b, strict=True):
        assert x['case'] == y['case']
        x['comparison'], y['comparison'] = 'baseline', 'candidate'
    kept = {row['case'] for row in b if row['supported'] and row['descent_feet_planted']}
    passed = {row['case'] for row in a if row['supported'] and row['descent_feet_planted']}
    summary = dict(baseline=baseline, before=len(passed), after=len(kept), repaired=sorted(kept - passed),
                   baseline_regressions=sorted(passed - kept), lost_protected=sorted(protected - kept),
                   remaining=[x['case'] for x in b if x['case'] not in kept])
    with (STUDY / (name + '_paired.csv')).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in a+b for k in row)))
        writer.writeheader()
        writer.writerows(a+b)
    write(STUDY / (name + '_summary.json'), summary)
    print(summary, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('command', choices=('prepare', 'run', 'score', 'paired'))
    cli.add_argument('--name', default='wire_brake')
    cli.add_argument('--baseline', default='registered')
    cli.add_argument('--rows', type=int, nargs='*', default=[1, 2, 3, 5, 7, 8, 10, 11])
    args = cli.parse_args()
    if args.command == 'prepare':
        prepare()
    if args.command == 'run':
        run(args.name, [] if args.name == 'baseline' else args.rows, TREATMENTS)
    if args.command == 'score':
        score(args.name)
    if args.command == 'paired':
        paired(args.name, args.baseline)
