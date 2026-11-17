"""Test the preload brake on matched, retention and fresh chair cases."""

import argparse
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from compare_touchdown_ppo import verify_audit
from evaluate_chair_shapes import CHAIRS, pieces
from evaluate_seat_plane_approach import BYTES_PER_TRIAL, DISK_RESERVE, compare, evaluate
from pelvis_velocity_brake import INTERVAL, PelvisVelocityBrake
from preload_brake import FIRST_TICK, GAP_INTERVAL, MAX_DELAY, PreloadBrake
from seat_plane_approach import MEASURE_UNTIL, SENSOR_FRAME, SeatPlaneApproach
from study_live_sitting import MODEL, bank_from_geometry, read, sha, write
from humanoid_sitting.shared_frame import BodyModel


def saved_replay(baseline, out):
    folder = baseline/'evaluation'
    audit, rows = verify_audit(folder)
    bank = Path(read(folder/'manifest.json')['args']['bank'])
    metadata = read(bank/'metadata.json')
    approach = SeatPlaneApproach(BodyModel(MODEL), metadata, len(rows))
    with np.load(folder/'trajectory.npz', allow_pickle=False) as z:
        frames = z['obs'][:, :, SENSOR_FRAME]
        load = z['support_bw']
        wrong = z['wrong_force']
    for tick in range(0, MEASURE_UNTIL + 1, 10):
        approach.observe(tick, frames[tick, :, -13:], frames[tick])
    approach.freeze()
    brake = PreloadBrake(approach)
    assert len(brake.active) == 46
    for tick in range(FIRST_TICK, len(frames)):
        observed = None
        if tick % GAP_INTERVAL == 0:
            previous = np.rint(brake.clock).astype(int)
            observed = frames[previous, brake.active]
        brake.update(tick, observed)
    clock = np.asarray(brake.trace)[:, :, 0]
    assert np.all(np.diff(clock, axis=0) > 0)
    assert np.max(np.arange(FIRST_TICK, len(frames))[:, None] - clock) <= MAX_DELAY
    cases = []
    for j, i in enumerate(brake.active):
        event = np.flatnonzero((load[:, i] > .05) | (wrong[:, i] > 10))[0]
        hit = int(np.searchsorted(clock[:, j], event))
        assert 0 < hit < 750 - FIRST_TICK
        loads_first = bool(load[event, i] > .05 and wrong[event, i] <= 10)
        if loads_first:
            assert 0 <= brake.started[j] - FIRST_TICK < hit
        cases.append(dict(id=rows[i]['id'], first_load_or_hard_contact_source_tick=int(event),
                          first_brake_tick=int(brake.started[j]), replay_event_tick=FIRST_TICK + hit,
                          loads_first=loads_first,
                          clock_rate_at_event=float(clock[hit, j] - clock[hit-1, j])))
    assert max(r['clock_rate_at_event'] for r in cases if r['loads_first']) < .90
    write(out/'saved_replay.json', dict(preflight_only=True, trace_sha256=audit['files']['trajectory.npz'],
                                       cases=cases))


def reserve_holdout(out, baseline):
    bank = Path(read(baseline/'evaluation/manifest.json')['args']['bank'])
    source = read(bank/'geometry.json')[0]
    cases = []
    for i, (family, depth, width, shape, back, arms, base) in enumerate(CHAIRS):
        for j in range(2):
            height = (.361, .374, .389, .407, .424, .437)[(i + 2*j) % 6]
            pitch = (13.6, 15.8, 17.9, 19.1)[(3*i + 2*j) % 4]
            dx = (-.024, -.011, .008, .019, .026)[(2*i+j) % 5]
            dy = (-.034, -.021, .014, .023, .039)[(i+3*j) % 5]
            yaw = (-1, 1)[j] * (8., 11., 14., 16.)[(i+j) % 4]
            rotation = Rotation.from_euler('ZY', [yaw, pitch], degrees=True)
            forward, _, normal = rotation.as_matrix().T
            front = np.array([-.4+dx, dy, height]) + .25*Rotation.from_euler('y', pitch, degrees=True).as_matrix()[:, 0]
            top = front - .5*depth*forward
            label = f'brake_holdout/{family}/{j+1}'
            cases.append(dict(source, label=label, family=family, shape=shape, depth=depth, width=width,
                              height=float(top[2]), pitch_deg=pitch, yaw_deg=yaw, yaw=np.deg2rad(yaw),
                              x=float(top[0]), y=float(top[1]), top_center=top.tolist(),
                              rotation_xyzw=rotation.as_quat().tolist(), normal=normal.tolist(),
                              reference_height_cm=100*height, fore_aft_offset_cm=100*dx,
                              lateral_offset_cm=100*dy, parts=pieces(depth,width,top[2],rotation,back,arms,base)))
    holdout = out/'holdout_bank'
    bank_from_geometry(holdout, bank, cases)
    write(out/'holdout_reservation.json', {str(p): sha(p) for p in holdout.iterdir()})
    return holdout


def saved_velocity_witness(baseline, out):
    folder = baseline/'evaluation'
    audit, rows = verify_audit(folder)
    bank = Path(read(folder/'manifest.json')['args']['bank'])
    approach = SeatPlaneApproach(BodyModel(MODEL), read(bank/'metadata.json'), len(rows))
    with np.load(folder/'trajectory.npz', allow_pickle=False) as z:
        frames = z['obs'][:, :, SENSOR_FRAME]
        desired = z['desired']
        load = np.linalg.norm(z['physical_force_w'][:, :, 0], axis=-1)/z['physical_bodyweight_N']
        wrong = z['wrong_force']
    for tick in range(0, MEASURE_UNTIL + 1, 10):
        approach.observe(tick, frames[tick, :, -13:], frames[tick])
    approach.freeze()
    brake = PelvisVelocityBrake(approach)
    assert len(brake.active) == 46
    targets = {}
    for j, i in enumerate(brake.active):
        event = np.flatnonzero(load[:, i] > .01)
        if event.size and not np.any(wrong[:event[0]+1, i] > 10):
            tick = int(event[0])
            targets[j] = (max(0, ((tick-1)//INTERVAL)*INTERVAL), tick)
    cases = []
    for tick in range(FIRST_TICK, max(t for t, _ in targets.values()) + 1, INTERVAL):
        brake.update(tick, frames[tick, brake.active], desired[tick, brake.active])
        for i, (capture, event) in targets.items():
            if tick == capture:
                cases.append(dict(id=rows[brake.active[i]]['id'], first_pelvis_load_tick=event,
                                  first_brake_tick=int(brake.started[i]), forward_m=float(brake.forward[i]),
                                  delta_max_rad=float(np.max(np.abs(brake.delta[i])))))
    assert len(cases) >= 41
    assert sum(r['forward_m'] >= .005 and r['delta_max_rad'] > 0 for r in cases) >= 35
    assert not np.any(brake.rejected)
    write(out/'saved_velocity_witness.json', dict(preflight_only=True, cases=cases,
                                                 trace_sha256=audit['files']['trajectory.npz']))


def paired(out, baseline, method, **options):
    label = options.get('label', 'evaluation')
    evaluate(out, baseline, braking=method, inputs=[Path(__file__)], **options)
    return compare(baseline/label, out/'evaluation', out)


def run(out, previous, retention, method):
    assert shutil.disk_usage(out.parent).free > DISK_RESERVE + (96 + 356 + 48)*BYTES_PER_TRIAL
    out.mkdir()
    if method in ('clearance', 'loaded_hold'):
        audit, rows = verify_audit(previous/'fixed_yaw/evaluation')
        assert len(rows) == 72 and sum(r['physical_failures'][1] for r in rows) == 33
        write(out/'baseline_trace.json', audit['files'])
    else:
        check = saved_velocity_witness if method == 'velocity' else saved_replay
        check(previous/'fixed_yaw', out)
    holdout = reserve_holdout(out, previous/'fixed_yaw')
    summary = {}
    for name, count in (('fixed_yaw', 72), ('shifted', 24)):
        result = paired(out/name, previous/name, method)
        assert result['total'] == count
        summary[name] = {k:v for k,v in result.items() if k != 'cases'}
        write(out/'summary.json', summary)
        print(name, summary[name], flush=True)
    regressions = [i for cohort in summary.values() for i in cohort['regressions']]
    improved = any(cohort['repairs'] for cohort in summary.values())
    summary['retention_and_holdout_authorized'] = improved and not regressions
    write(out/'summary.json', summary)
    if not summary['retention_and_holdout_authorized']:
        return

    groups = read(retention/'retention.json')['banks']
    retained = {}
    summary['retention'] = retained
    for name in groups:
        label = 'evaluation' if name == 'shifted' else name
        result = paired(out/f'retain_{name}', retention, method, label=label)
        assert result['total'] == groups[name]['total']
        retained[name] = {k:v for k,v in result.items() if k != 'cases'}
        write(out/'summary.json', summary)
        print('retained', name, result['counts'], flush=True)
    assert sum(value['total'] for value in retained.values()) == 356

    for path,digest in read(out/'holdout_reservation.json').items():
        assert sha(path) == digest
    reference = out/'holdout_reference'
    evaluate(reference, previous/'fixed_yaw', bank=holdout, inputs=[out/'holdout_reservation.json'])
    result = paired(out/'holdout', reference, method, bank=holdout)
    assert result['total'] == 24
    summary['holdout'] = {k:v for k,v in result.items() if k != 'cases'}
    write(out/'summary.json', summary)
    print('holdout', summary['holdout'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('previous', type=Path)
    parser.add_argument('retention', type=Path)
    parser.add_argument('--method', choices=('clock', 'velocity', 'clearance', 'loaded_hold'), required=True)
    args = parser.parse_args()
    run(args.output, args.previous, args.retention, args.method)
