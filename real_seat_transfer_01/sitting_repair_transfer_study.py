"""Paired transfer test of the repaired sitting controller and two frozen actors."""
import argparse
import csv
import os
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

import fresh_sitting_skills_study as fresh
import planted_descent_study as scorer
from lateral_sitting_repair_study import audited, passed, previous_sources
from lateral_sitting_repair_worker import choose_camera_route
from run import DISK_RESERVE, P, PREPARED, read, sha, stage, write


STUDY = P / 'sitting_repair_transfer_24_01'
PREVIOUS = fresh.STUDY
REPAIRED = P / 'lateral_sitting_repair_01' / 'supported_handoff'
ARMS = ('repaired', 'shared', 'direct')
SEED = 82219
METRICS = ('hold_s', 'hold_seat_normal_pct', 'hold_mechanical_power_W')


def freeze_camera_selector():
    measures = audited(REPAIRED)
    assert len(measures) == 24 and all(map(passed, measures))
    source = REPAIRED / 'evaluation' / 'camera_routes.json'
    camera = read(source)
    repaired = read(REPAIRED / 'repair.json')
    prior = read(PREVIOUS / 'protocol.json')
    assert fresh.bank_digest(REPAIRED/'bank') == prior['bank']
    templates = {}
    for row, (obs, result, trial) in enumerate(zip(camera, measures, prior['trials'], strict=True)):
        assert obs['row'] == row and obs['registration_ok'] and obs['standing']
        assert obs['asset'] == trial['asset'] and result['case'] == trial['case']
        pose = np.asarray(obs['relative_pose'])
        axis = 1 if trial['family'] == 'lateral' else 2
        assert (abs(pose[1]) >= .016) == (axis == 1) and np.isfinite(pose).all()
        prototypes = templates.setdefault(obs['asset'], [])
        prototypes.append(dict(case=result['case'], pose=pose.tolist(), axis=axis,
                               repair=repaired['routes'].get(str(row)), shared=row in repaired['use_shared']))
    selector = dict(templates=templates, lateral_boundary_m=.016, scale=fresh.SCALE.tolist(), max_distance=1.)
    for asset, prototypes in templates.items():
        for obs in prototypes:
            selected, error = choose_camera_route(asset, obs['pose'], selector)
            assert selected == obs and error == 0.
    return selector, [source, REPAIRED/'repair.json', REPAIRED/'measures.json', PREVIOUS/'protocol.json',
                      REPAIRED/'evaluation'/'independent_audit.json']


def check_separation(bank):
    def poses(folder):
        with np.load(folder/'bank.npz', allow_pickle=False) as z:
            rows = z['reset_root_pose']
        yaw = Rotation.from_quat(rows[:, 3:]).as_euler('xyz')[:, 2]
        return [x['name'] for x in read(folder/'geometry.json')], np.c_[rows[:, :2], yaw]

    names, current = poses(bank)
    previous = [poses(p) for p in (PREVIOUS/'bank', PREVIOUS/'retention'/'bank', fresh.OLD/'training_bank')]
    distance = []
    for i, (asset, row) in enumerate(zip(names, current, strict=True)):
        same_asset = [v for pool, values in previous for name, v in zip(pool, values, strict=True) if name == asset]
        same_asset.extend(v for j, (name, v) in enumerate(zip(names, current, strict=True)) if i != j and name == asset)
        delta = np.asarray(same_asset) - row
        delta[:,2] = np.arctan2(np.sin(delta[:,2]), np.cos(delta[:,2])) * .25
        distance.append(float(np.linalg.norm(delta, axis=1).min()))
    assert min(distance) >= .007 - 1e-7, min(distance)
    return distance


def prepare():
    selector, sources = freeze_camera_selector()
    STUDY.mkdir()
    write(STUDY/'camera_selector.json', selector)
    fresh.STUDY, fresh.SEED = STUDY, SEED
    fresh.prepare(excluded_banks=(PREVIOUS/'bank',), label='transfer2')
    protocol = read(STUDY/'protocol.json')
    protocol.update(selector_sha256=sha(STUDY/'camera_selector.json'),
                    selector_sources={str(p):sha(p) for p in sources},
                    separation_m=check_separation(STUDY/'bank'))
    assert sha(STUDY/'routes.json') == read(PREVIOUS/'protocol.json')['route_sha256']
    write(STUDY/'protocol.json', protocol)
    print('reserved disjoint matched bank; minimum separation:', min(protocol['separation_m']), flush=True)


def rollout(arm):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == fresh.GPU
    source = PREVIOUS / ('staged' if arm == 'repaired' else arm)
    assert len(audited(source)) == 24
    frozen = read(source/'evaluation.freeze.json')
    old_sources = previous_sources(frozen)
    protocol = read(STUDY/'protocol.json')
    assert all(sha(path) == h for path,h in protocol['selector_sources'].items())
    assert all(sha(path) == h for path,h in protocol['preparation_sources'].items())
    assert sha(STUDY/'routes.json') == protocol['route_sha256']
    assert sha(STUDY/'camera_selector.json') == protocol['selector_sha256']
    out = STUDY/arm
    out.mkdir()
    bank = out/'bank'
    shutil.copytree(STUDY/'bank', bank)
    assert fresh.bank_digest(bank) == protocol['bank']
    shutil.copyfile(source/'landing_profile.json', out/'landing_profile.json')
    config = read(source/'step_config.json')
    skills = read(source/'skills.json')
    assets = [t['asset'] for t in protocol['trials']]
    config['templates_by_row'] = assets
    config['planted_descent']['rows'] = []
    config['planted_descent']['treatments'] = {}
    skills['choices'] = [[0]*4 for _ in assets]
    skills['hold_rows'] = [i for i,a in enumerate(assets) if a in ('Chair_Wire','Basic')]
    skills['route_spec'] = str(STUDY/'routes.json')
    extras = [Path(__file__).resolve(), STUDY/'protocol.json', STUDY/'routes.json',
              *map(Path, protocol['preparation_sources']), *map(Path, protocol['selector_sources'])]
    if arm == 'repaired':
        worker = P/'lateral_sitting_repair_worker.py'
        repair = read(REPAIRED/'repair.json')
        shared_weights = Path(repair['shared']['weights'])
        assert sha(shared_weights) in read((PREVIOUS/'shared')/'evaluation.freeze.json')['inputs'].values()
        repair['shared']['route_spec'] = str(STUDY/'routes.json')
        repair['shared']['hold_rows'] = skills['hold_rows']
        repair.update(routes={}, use_shared=[], camera_selector=read(STUDY/'camera_selector.json'))
        write(out/'repair.json', repair)
        extras.extend([out/'repair.json', STUDY/'camera_selector.json', shared_weights,
                       Path(repair['shared']['implementation'])])
    else:
        worker = P/'fresh_sitting_skills_worker.py'
        assert sha(worker) in frozen['inputs'].values()
    write(out/'skills.json', skills)
    write(out/'step_config.json', config)
    write(out/'reservation.json', dict(files={str(p):sha(p) for p in bank.iterdir()}))
    cmd = frozen['command']
    stage(out, 'evaluation', bank, PREPARED/'library.npz', worker=worker,
          extra=cmd[3:cmd.index('--mode')], disk_reserve=DISK_RESERVE,
          inputs=[*old_sources, *extras, worker, out/'skills.json', out/'step_config.json', out/'landing_profile.json'])
    scorer.STUDY = STUDY
    scorer.score(arm)


def report():
    protocol = read(STUDY/'protocol.json')
    scopes, results, combined = {}, {}, []
    for arm in ARMS:
        out = STUDY/arm
        measures = audited(out)
        assert fresh.bank_digest(out/'bank') == protocol['bank']
        assert read(out/'evaluation.freeze.json')['gpu'] == fresh.GPU
        ev = out/'evaluation'
        assert read(ev/'manifest.json')['action_limits_unchanged']
        assert read(ev/'results.json')['success_criteria_unchanged']
        with np.load(ev/'trajectory.npz') as z:
            cap = float(np.abs(np.diff(z['command'][:,:,:23],axis=0)).max())
            forbidden = z['flags'][:,:,1].any(axis=0)
            assert cap <= .0030001
            ground = z['physical_force_w'][:,:,5:]
            bw = z['physical_bodyweight_N'][:,:,None]
            slipping = (np.linalg.norm(ground,axis=-1) > .01*bw) & (z['physical_slip_m_s'][:,:,5:] > .02)
            planted = (ground[...,2] > .05*bw).all(axis=(0,2)) & ~slipping.any(axis=(0,2))
            assert planted.tolist() == [r['descent_feet_planted'] for r in measures]
        with np.load(ev/'single_step.npz') as z:
            steps = z['options']
        selected = read(ev/'signed_camera_selection.json') if arm == 'repaired' else [{} for _ in measures]
        if arm == 'repaired':
            assert (steps == 0).all()
        for c,r,opt,hit,choice in zip(protocol['trials'],measures,steps,forbidden,selected,strict=True):
            assert c['case'] == r['case']
            combined.append(dict(c, **{k:v for k,v in r.items() if k!='case'}, arm=arm, passed=passed(r),
                                 pre_descent_step=int(opt), forbidden_contact=bool(hit),
                                 selected_anchor=choice.get('anchor'), selected_mode=choice.get('mode')))
        scopes[arm] = dict(overall=fresh.stats(measures), max_command_step_rad=cap,
                           pre_descent_steps=int((steps>0).sum()), forbidden_contacts=int(forbidden.sum()),
                           families={f:fresh.stats([r for c,r in zip(protocol['trials'],measures,strict=True)
                                                    if c['family']==f]) for f in ('lateral','angled')})
        results[arm] = measures
    paired = {}
    for arm in ('shared','direct'):
        both = [(a,b) for a,b in zip(results['repaired'],results[arm],strict=True) if passed(a) and passed(b)]
        paired[arm] = dict(both_pass=len(both), mean_repaired_minus_baseline={
            k:round(float(np.mean([a[k]-b[k] for a,b in both])),3) if both else None for k in METRICS})
    with (STUDY/'comparison.csv').open('w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in combined for k in r)))
        w.writeheader()
        w.writerows(combined)
    summary = dict(gpu=fresh.GPU, matched_starts=24, scopes=scopes, paired=paired)
    write(STUDY/'summary.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('prepare','run','report'))
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    if args.operation == 'prepare':
        prepare()
    elif args.operation == 'run':
        rollout(args.arm)
    else:
        report()
