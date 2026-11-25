"""Freeze and evaluate twenty-four fresh sitting starts and the original twelve."""
import argparse
import csv
import itertools
import os
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

import planted_descent_study as scorer
from run import DISK_RESERVE, P, PREPARED, read, sha, stage, write

OLD = P / 'sitting_skills_01'
STUDY = P / 'fresh_sitting_skills_24_01'
ARMS = {'staged': 'handoff_eval', 'shared': 'shared_eval', 'direct': 'direct_eval'}
BANK_FILES = ('bank.npz', 'metadata.json', 'geometry.json')
GPU = 'GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
SEED = 471166
SCALE = np.array([.04, .034, .10])
ENVELOPE = [.009, .013, .035]
SOURCE = ('fresh_sitting_skills_study.py', 'fresh_sitting_skills_worker.py')


def archived(name):
    folder = OLD / name
    ev = folder / 'evaluation'
    audit = read(ev / 'independent_audit.json')
    assert audit['passed'] and audit['strict_predicates_exact'] and not audit['physical_tolerances_changed']
    for f in ('trajectory.npz', 'results.json'):
        assert sha(ev / f) == audit['files'][f]
    return folder


def bank_digest(bank):
    return {f: sha(bank / f) for f in BANK_FILES}


def verified_inputs(entries, snapshots, names):
    paths = []
    for path, digest in entries.items():
        src = Path(path)
        if sha(src) != digest:
            assert src.name in names, path
            src = snapshots / src.name
            assert sha(src) == digest, src
        paths.append(src)
    return paths


def preparation_sources(original):
    eval_folder = original / 'evaluation'
    return [Path(__file__).resolve(), *(original / 'bank' / f for f in BANK_FILES),
            *(OLD / 'training_bank' / f for f in BANK_FILES),
            *(eval_folder / f for f in ('standing_scan.npz', 'single_step.npz', 'skill_decisions.npz',
                                        'independent_audit.json')),
            original / 'step_config.json', original / 'skills.json', original / 'measures.json',
            Path(read(original / 'step_config.json')['planted_descent']['library'])]


def prepare(excluded_banks=(), label='fresh'):
    from rescan_sitting import match, points
    from stance23_planner import model

    original = archived(ARMS['staged'])
    base = original / 'bank'
    config = read(original / 'step_config.json')
    skills = read(original / 'skills.json')
    incumbent_measures = read(original / 'measures.json')
    assert len(incumbent_measures) == 12
    assert all(r['supported'] and r['descent_feet_planted'] for r in incumbent_measures)
    assert all(bank_digest(archived(name) / 'bank') == bank_digest(base) for name in ARMS.values())
    geometry = read(base / 'geometry.json')
    meta = read(base / 'metadata.json')
    with np.load(base / 'bank.npz', allow_pickle=False) as z:
        src = {k: z[k] for k in z.files}
    with np.load(OLD / 'training_bank' / 'bank.npz', allow_pickle=False) as z:
        trained = z['reset_root_pose']
    excluded = []
    for bank in excluded_banks:
        with np.load(bank / 'bank.npz', allow_pickle=False) as z:
            excluded.append((read(bank / 'geometry.json'), z['reset_root_pose']))
    with np.load(original / 'evaluation' / 'skill_decisions.npz', allow_pickle=False) as record:
        accepted_options = {int(i): {} for i in skills['verified_rows']}
        for i, phase, a, enabled in zip(*(record[k] for k in ('row', 'phase', 'action', 'enabled')), strict=True):
            if i in accepted_options and a and enabled:
                accepted_options[int(i)][str(phase)] = int(a)

    bank = STUDY / 'bank'
    bank.mkdir(parents=True)
    routes = {}
    setup = config['planted_descent']
    with np.load(setup['library'], allow_pickle=False) as demos, np.load(
            original / 'evaluation' / 'standing_scan.npz', allow_pickle=False) as scan, np.load(
            original / 'evaluation' / 'single_step.npz', allow_pickle=False) as step:
        for start in range(0, 12, 3):
            asset = config['templates_by_row'][start]
            routes[asset] = []
            for i, family in ((start + 1, 'lateral'), (start + 2, 'angled')):
                assert step['options'][i] == 0
                cloud = points(scan['depth'][0,i], scan['intrinsics'][0,i], scan['mount'][0],
                               scan['q'][0,i], scan['gravity'][0,i], model)
                matched = match(demos[asset+'_cloud'], cloud)
                assert matched is not None, (asset, family)
                pose, error = matched
                assert np.isfinite(pose).all()
                routes[asset].append(dict(anchor_row=i, family=family, pose=pose.tolist(),
                                          registration_median_m=error,
                                          planted=i in setup['rows'],
                                          treatment=setup['treatments'].get(str(i), {}),
                                          options=accepted_options.get(i, {})))

    rng = np.random.default_rng(SEED)
    new_geom, attempts, poses, indices = [], [], [], []
    for k in range(4):
        start = 3*k
        template = start + 1
        anchor = src['reset_root_pose'][template].copy()
        anchor[1] = src['reset_root_pose'][start+2,1]
        ref = np.vstack((src['reset_root_pose'][start:start+3], trained[start:start+3]))
        asset = geometry[template]['name']
        others = [pose for geoms, poses in excluded for geom, pose in zip(geoms, poses, strict=True)
                  if geom['name'] == asset]
        if others:
            ref = np.vstack((ref, others))
        previous = [np.r_[p[:2]-anchor[:2], Rotation.from_quat(p[3:]).as_euler('xyz')[2]]
                    for p in ref]
        for family in ('lateral','angled'):
            signs = (1,-1,1) if (k + (family=='angled'))%2==0 else (-1,1,-1)
            for j, sign in enumerate(signs):
                for _ in range(1000):
                    x = rng.uniform(-.008,.008)
                    if family == 'lateral':
                        y = sign*rng.uniform(.024,.049)
                        yaw = rng.uniform(-.012,.012)
                    else:
                        y = rng.uniform(-.005,.005)
                        yaw = sign*np.deg2rad(rng.uniform(3.8,8.0))
                    delta = np.array([x,y,yaw])
                    distance = np.linalg.norm((np.stack(previous)-delta)*[1.,1.,.25],axis=1).min()
                    if distance >= .007:
                        break
                else:
                    raise RuntimeError('Could not construct a disjoint start')
                previous.append(delta)
                i = len(poses)
                name = asset
                case = f'sit/{name}/{label}_{family}_{j+1:02d}'
                row_pose = anchor.copy()
                row_pose[:2] += delta[:2]
                row_pose[3:] = (Rotation.from_rotvec([0.,0.,yaw])*Rotation.from_quat(anchor[3:])).as_quat()
                indices.append(template)
                poses.append(row_pose)
                new_geom.append(dict(geometry[template], case=i, surface_id=i, label=case))
                attempts.append(dict(case=case, asset=name, family=family, x_cm=x*100, y_cm=y*100,
                                     yaw_deg=np.rad2deg(yaw).item(), nearest_previous_m=float(distance)))
    assert len(poses)==24 and sum(a['family']=='lateral' for a in attempts)==12
    arrays = {key: np.take(data, indices, axis=int(key=='q_post')) for key,data in src.items()}
    arrays['reset_root_pose'] = np.asarray(poses,dtype=src['reset_root_pose'].dtype)
    arrays['reference'] = np.arange(len(poses))
    np.savez_compressed(bank / 'bank.npz', **arrays)
    write(bank / 'geometry.json', new_geom)
    write(bank / 'metadata.json', dict(meta, cases=len(poses), ids=[a['case'] for a in attempts]))
    write(STUDY / 'routes.json', dict(templates=routes, family_scale=SCALE.tolist(), envelope=ENVELOPE))
    write(STUDY / 'protocol.json', dict(seed=SEED, trials=attempts, bank=bank_digest(bank),
                                      route_sha256=sha(STUDY / 'routes.json'),
                                      preparation_sources={str(f): sha(f) for f in (
                                          *preparation_sources(original),
                                          *(bank / f for bank in excluded_banks for f in BANK_FILES))}))
    print('frozen', len(attempts), 'fresh starts, bank SHA256', sha(bank / 'bank.npz'), flush=True)


def rollout(arm):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPU
    known = arm == 'retention'
    original = archived(ARMS['staged' if known else arm])
    fresh = read(STUDY / 'protocol.json')
    source = original / 'bank' if known else STUDY / 'bank'
    assert known or bank_digest(source) == fresh['bank']
    start = read(original / 'evaluation.freeze.json')
    prior = verified_inputs(start['inputs'], OLD / 'matched_raw_source',
                            {'sitting_skills.py', 'sitting_skills_study.py'})
    prepared = verified_inputs(fresh['preparation_sources'], STUDY / 'initial_source',
                               {'fresh_sitting_skills_study.py'})
    out = STUDY / arm
    out.mkdir()
    bank = out / 'bank'
    shutil.copytree(source, bank)
    shutil.copyfile(original / 'landing_profile.json', out / 'landing_profile.json')
    config = read(original / 'step_config.json')
    skills = read(original / 'skills.json')
    worker = P / 'sitting_skills_worker.py' if known else P / SOURCE[1]
    extras = [*prepared, STUDY / 'protocol.json', Path(__file__).resolve()]
    if not known:
        config['templates_by_row'] = [a['asset'] for a in fresh['trials']]
        config['planted_descent']['rows'] = []
        config['planted_descent']['treatments'] = {}
        skills['choices'] = [[0]*4 for _ in fresh['trials']]
        skills['hold_rows'] = [i for i,a in enumerate(fresh['trials']) if a['asset'] in ('Chair_Wire','Basic')]
        skills['split'] = 'fresh'
        skills['route_spec'] = str(STUDY / 'routes.json')
        skills['implementation'] = str(next(f for f in prior if f.name == 'sitting_skills.py'))
        if arm == 'staged':
            skills['verified_rows'] = []
        assert sha(STUDY / 'routes.json') == fresh['route_sha256']
        extras.extend([STUDY / 'routes.json', worker])
    write(out / 'skills.json', skills)
    write(out / 'step_config.json', config)
    write(out / 'reservation.json', dict(files={str(f): sha(f) for f in bank.iterdir()}))
    cmd = start['command']
    stage(out, 'evaluation', bank, PREPARED / 'library.npz', worker=worker,
          extra=cmd[3:cmd.index('--mode')],
          inputs=[*prior, *extras, out / 'skills.json',
                  out / 'step_config.json', out / 'landing_profile.json'],
          disk_reserve=DISK_RESERVE)
    scorer.STUDY = STUDY
    scorer.score(arm)


def stats(rows):
    passing = [x for x in rows if x['supported'] and x['descent_feet_planted']]
    return dict(total=len(rows), passed=len(passing), strict=sum(x['strict'] for x in passing),
                feet_planted=sum(x['descent_feet_planted'] for x in rows),
                five_second_terminal_holds=sum(x['hold_s']>=5. for x in rows),
                all_starts_mean_hold_s=round(float(np.mean([x['hold_s'] for x in rows])),3),
                passing_means={key: round(float(np.mean([x[key] for x in passing])),3) if passing else None
                               for key in ('hold_s','hold_seat_normal_pct','hold_mechanical_power_W')})


def report():
    results = {arm: read(STUDY / arm / 'measures.json') for arm in (*ARMS,'retention')}
    orig = read(OLD / ARMS['staged'] / 'measures.json')
    assert [x['case'] for x in orig] == [x['case'] for x in results['retention']]
    protocol = read(STUDY / 'protocol.json')
    ids = [x['case'] for x in protocol['trials']]
    records = []
    scopes = {}
    for arm in ARMS:
        target = STUDY / arm
        assert bank_digest(target / 'bank') == protocol['bank']
        assert [r['case'] for r in results[arm]] == ids
        ev = target / 'evaluation'
        routed = read(ev / 'camera_routes.json') if (ev / 'camera_routes.json').exists() else []
        route = {x['row']: x for x in routed}
        with np.load(ev / 'skill_decisions.npz', allow_pickle=False) as choice:
            counts = {p: int(((choice['phase']==p) & (choice['action']!=0) & choice['enabled']).sum())
                      for p in range(4)}
        scopes[arm] = dict(overall=stats(results[arm]), active_choices=counts,
                           routed=len([x for x in routed if x['registration_ok']]),
                           eligible_for_guard=sum(x['in_envelope'] for x in routed),
                           by_family={f: stats([r for c,r in zip(protocol['trials'],results[arm],strict=True)
                                                if c['family']==f]) for f in ('lateral','angled')})
        for i,(trial,record) in enumerate(zip(protocol['trials'],results[arm],strict=True)):
            used = route.get(i,{})
            records.append(dict(trial, **{k:v for k,v in record.items() if k!='case'}, arm=arm,
                                routed_family=used.get('family'), in_guard_envelope=used.get('in_envelope')))
    with (STUDY / 'fresh_comparison.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in records for k in r)))
        w.writeheader()
        w.writerows(records)
    with (STUDY / 'retention.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in results['retention'] for k in r)))
        w.writeheader()
        w.writerows(results['retention'])
    common = []
    for a,b in itertools.combinations(ARMS,2):
        both = [(x,y) for x,y in zip(results[a],results[b],strict=True)
                if all(r['supported'] and r['descent_feet_planted'] for r in (x,y))]
        common.append(dict(arms=[a,b], count=len(both), a_minus_b_means={
            k: round(float(np.mean([x[k]-y[k] for x,y in both])),3) if both else None
            for k in ('hold_s','hold_seat_normal_pct','hold_mechanical_power_W')}))
    s = dict(gpu=GPU, fresh=scopes, retention=stats(results['retention']),
             lost_protected=[r['case'] for r in results['retention'] if not (r['supported'] and r['descent_feet_planted'])],
             shared_passes=common)
    write(STUDY / 'summary.json',s)
    print(s,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('prepare','run','report'))
    parser.add_argument('--arm', choices=(*ARMS,'retention'))
    options = parser.parse_args()
    if options.operation == 'prepare':
        prepare()
    elif options.operation == 'run':
        rollout(options.arm)
    else:
        report()
