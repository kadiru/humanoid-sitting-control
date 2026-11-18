"""Paired evaluations of a sensor-derived pelvis roll on the RTX 4090."""

import argparse
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from compare_touchdown_ppo import HOLD_TICK, verify_audit
from seat_plane_approach import MEASURE_UNTIL, ROLL_START, SENSOR_FRAME, SeatPlaneApproach, stance_axes
from study_live_sitting import MODEL, PREPARED, read, sha, stage, write
from humanoid_sitting.shared_frame import BodyModel


HERE = Path(__file__).resolve().parent
APPROACH = (HERE/'live_sitting_edge.py', HERE/'seat_plane_approach.py', Path(__file__).resolve())
# The separate NVMe volume needs 6 GB free throughout. Prior complete
# evaluations used about 3.2 MB per case, including traces and logs.
DISK_RESERVE = 6_000_000_000
BYTES_PER_TRIAL = 5_000_000


def saved_witness(bank, baseline, out):
    """Check the finite command construction against measured touchdown poses."""
    metadata = read(bank/'metadata.json')
    model = BodyModel(MODEL)
    with np.load(baseline/'evaluation/trajectory.npz', allow_pickle=False) as z:
        obs = z['obs'][:, :, SENSOR_FRAME]
        wrong = z['wrong_force']
    with np.load(baseline/'evaluation/edge_depth_samples.npz', allow_pickle=False) as z:
        standing_q = z['q'][0]
    approach = SeatPlaneApproach(model, metadata, len(metadata['ids']))
    for tick in range(0, MEASURE_UNTIL + 1, 10):
        approach.observe(tick, obs[tick, :, -13:], obs[tick])
    approach.freeze()
    geometry = read(bank/'geometry.json')
    turned = np.flatnonzero([abs(g['yaw_deg']) > 5 for g in geometry])
    assert approach.active.size and set(approach.active) <= set(turned)
    rows = []
    for i in approach.active:
        tick = int(np.flatnonzero(wrong[:, i] > 10)[0])
        frame = obs[tick, i]
        joints = frame[:23]
        base = standing_q[i].copy()
        base[metadata['joint_ids']] = joints
        delta, error = approach.correct(joints, frame, approach.normal[i], approach.roll[i])
        corrected = base.copy()
        corrected[metadata['joint_ids']] += delta
        before, after = model.fk(base), model.fk(corrected)
        # Recover the corrected pelvis transform from the unchanged left foot.
        foot = 'left_ankle_roll_link'
        transform = before[foot] @ np.linalg.inv(after[foot])
        normal = stance_axes(approach.feet(joints), frame[69:72]) @ approach.normal[i]
        gaps = []
        for poses, root in ((before, np.eye(4)), (after, transform)):
            def vertices(link):
                pose = root @ poses[link]
                return model.vertices[link] @ pose[:3, :3].T + pose[:3, 3]
            pelvis = np.min(vertices('pelvis') @ normal)
            hip = min(np.min(vertices(f'{side}_hip_pitch_link') @ normal) for side in ('left', 'right'))
            gaps.append(hip - pelvis)
        rows.append(dict(id=metadata['ids'][i], tick=tick, relative_clearance_before_m=gaps[0],
                         relative_clearance_after_m=gaps[1], foot_position_error_m=error[0],
                         foot_angle_error_rad=error[1], roll_rad=approach.roll[i],
                         correction_max_rad=np.max(np.abs(delta))))
    assert all(r['relative_clearance_after_m'] - r['relative_clearance_before_m'] > .004 for r in rows)
    write(out/'saved_witness.json', dict(
        cases=rows, pose_only=True, ineligible_turned=[metadata['ids'][i] for i in np.setdiff1d(turned, approach.active)],
        trajectory_sha256=sha(baseline/'evaluation/trajectory.npz')))


def evaluate(out, baseline, *, label='evaluation', bank=None, braking=None, inputs=(), plane_demos=None):
    source = read(baseline/f'{label}.freeze.json')
    command = source['command']
    grip = command.index('evaluate', 3)
    options = command[grip:command.index('--mode')]
    option_inputs = [Path(x) for x in options if x.startswith('/')]
    for path in option_inputs:
        assert sha(path) == source['inputs'][str(path)], path
    mode = command.index('--mode')
    assert command[command.index('--seed', mode) + 1] == '65011'
    if bank is None:
        bank = Path(command[command.index('--bank', mode) + 1])
        for path in bank.iterdir():
            assert sha(path) == source['inputs'][str(path)], path
    count = len(read(bank/'metadata.json')['ids'])
    assert shutil.disk_usage(out.parent).free > DISK_RESERVE + count * BYTES_PER_TRIAL
    out.mkdir()
    write(out/'reservation.json', dict(files={str(p): sha(p) for p in bank.iterdir()}))
    launcher = HERE/'grip_hold_ppo.py'
    worker = launcher
    if '--worker' in command[:grip]:
        worker = HERE/'evaluate_chair_shapes.py'
        options = ['--worker', str(launcher), *options]
    if '--seat-plane-approach' not in options:
        options += ['--seat-plane-approach']
    if plane_demos is not None:
        assert '--verify-grip-contact' in options
        options += ['--plane-demonstrations',str(plane_demos)]
        inputs = [plane_demos,*inputs]
    if '--plane-demonstrations' in options:
        inputs = [HERE/'seat_plane_demonstration.py',HERE/'seat_normal_landing.py',*inputs]
    if braking == 'guard_landing':
        assert '--plane-demonstrations' in options and '--guard-normal-landing' not in options
        options += ['--guard-normal-landing']
    if braking == 'clock':
        options += ['--preload-braking']
        inputs = [HERE/'preload_brake.py', *inputs]
    if braking in ('velocity', 'clearance', 'loaded_hold') and '--preload-velocity-brake' not in options:
        options += ['--preload-velocity-brake']
    if '--preload-velocity-brake' in options:
        inputs = [HERE/'pelvis_velocity_brake.py', *inputs]
    if braking == 'clearance':
        options += ['--housing-clearance']
        inputs = [HERE/'housing_clearance.py', *inputs]
    if braking == 'loaded_hold':
        options += ['--loaded-clearance-hold']
    if '--loaded-clearance-hold' in options:
        inputs = [HERE/'loaded_clearance_hold.py', *inputs]
    if braking in ('shear','normal_transfer','follow_descent'):
        assert '--loaded-clearance-hold' in options and '--preload-velocity-brake' in options
        options += ['--seat-shear-feedback']
        if braking in ('normal_transfer','follow_descent'):
            options += ['--normal-load-transfer']
        if braking == 'follow_descent':
            options += ['--follow-normal-descent']
    if braking == 'verify_contact':
        assert '--follow-normal-descent' in options
        options += ['--verify-grip-contact']
    if '--seat-shear-feedback' in options:
        inputs = [HERE/'seat_shear_feedback.py', *inputs]
    return stage(out, 'evaluation', bank, PREPARED/'library.npz', worker=worker, extra=options,
                 inputs=[*APPROACH, launcher, HERE/'touchdown_feedback.py', HERE/'prepare_sitting_repairs.py',
                         baseline/f'{label}.freeze.json', *option_inputs, *inputs], disk_reserve=DISK_RESERVE)


def compare(before, after, out):
    base_audit, base_rows = verify_audit(before)
    new_audit, new_rows = verify_audit(after)
    meta, new_meta = (read(folder/'manifest.json') for folder in (before, after))
    assert meta['bank_sha256'] == new_meta['bank_sha256'] and meta['args']['seed'] == new_meta['args']['seed']
    assert new_meta['action_limits_unchanged'] and new_meta['physical_criteria_unchanged']
    with np.load(before/'trajectory.npz', allow_pickle=False) as old, \
            np.load(after/'trajectory.npz', allow_pickle=False) as new:
        a, b = old['command'], new['command']
        assert np.max(np.abs(b - new['previous'])) <= .0030003
        assert np.array_equal(a[:ROLL_START], b[:ROLL_START])
        command_difference = np.max(np.abs(a - b), axis=(0, 2))
        approach_difference = np.max(np.abs(a[:HOLD_TICK] - b[:HOLD_TICK]), axis=(0, 2))
    with np.load(after/'seat_plane_approach.npz', allow_pickle=False) as z:
        eligible = z['active']
        assert np.all(z['accepted'][eligible] > 0)
    hold = after/'loaded_clearance_hold.npz'
    if hold.exists():
        with np.load(hold, allow_pickle=False) as z:
            eligible = np.union1d(eligible,z['active'])
    inactive = np.setdiff1d(np.arange(len(base_rows)), eligible)
    assert not np.any(approach_difference[inactive])
    rows = []
    for i, (a, b) in enumerate(zip(base_rows, new_rows, strict=True)):
        assert a['id'] == b['id'] and a['full_standing_start'] and b['full_standing_start']
        assert a['steps'] == b['steps'] == 1000
        def metrics(row, audit):
            return dict(clean=row['clean'], stable=row['stable'], hold_s=row['final_hold_s'],
                        slip=bool(row['slip_ticks']), forbidden=row['physical_failures'][1],
                        power_w=audit['final_absolute_mechanical_power_W'][i])
        rows.append(dict(id=a['id'], old=metrics(a, base_audit), new=metrics(b, new_audit),
                         max_command_difference_rad=float(command_difference[i])))
    counts = {side: {key: sum(row[side][key] for row in rows) for key in ('clean', 'stable', 'slip', 'forbidden')}
              for side in ('old', 'new')}
    score = dict(total=len(rows), activated=len(eligible), counts=counts,
                 max_inactive_command_difference_rad=float(command_difference[inactive].max(initial=0.)),
                 repairs=[r['id'] for r in rows if r['new']['clean'] and not r['old']['clean']],
                 regressions=[r['id'] for r in rows if r['old']['clean'] and not r['new']['clean']],
                 cases=rows)
    brake = after/'preload_brake.npz'
    if brake.exists():
        with np.load(brake, allow_pickle=False) as z:
            trace = z['trace']
            delay = 0.
            if trace.size:
                clock = trace[:, :, 0]
                delay = float((z['first_tick'] + np.arange(len(clock))[:, None] - clock).max())
            score['brake'] = dict(eligible=len(z['active']), started=int((z['started'] >= 0).sum()),
                                  maximum_delay_ticks=delay)
    velocity = after/'velocity_brake.npz'
    if velocity.exists():
        with np.load(velocity, allow_pickle=False) as z:
            score['brake'] = dict(eligible=len(z['active']), started=int((z['started'] >= 0).sum()),
                                  maximum_forward_m=float(z['forward_m'].max(initial=0.)),
                                  rejected_pose_updates=int(z['rejected'].sum()))
    write(out/'comparison.json', score)
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('yaw', type=Path)
    parser.add_argument('shifted', type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    saved_witness(args.yaw/'bank', args.yaw, args.output)
    for name, baseline in (('fixed_yaw', args.yaw), ('shifted', args.shifted)):
        condition = args.output/name
        evaluate(condition, baseline)
        compare(baseline/'evaluation', condition/'evaluation', condition)


if __name__ == '__main__':
    main()
