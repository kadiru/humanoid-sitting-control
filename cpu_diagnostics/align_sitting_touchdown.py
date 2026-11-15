"""Align the pelvis along the seat before transferring normal load."""

import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from prepare_sitting_repairs import BodyModel, HOUSINGS, encode, leg_fk, smoothstep
from repair_fresh_sitting import checked_trace
from repair_live_touchdown import HERE, MODEL, ROOT, applied_path, read, screen, sha, write
from repair_sitting_touchdown import retarget


PROBE = ROOT/'outputs/live_touchdown_20261115_screen2'
PREVIOUS = ROOT/'outputs/live_touchdown_20261115_full'
TARGETS = {'matched/01/02': ('candidate', ('probe/08',)),
           'live_holdout/01': ('fresh_once', ('probe/24', 'probe/26'))}
HOLD = 750
DT = .02


def align(path, initial, pose, q, applied, load, bad, geometry, model, fk, legs, limits,
          clearance, anchor_offset, gain, pitch=0.):
    rotation = Rotation.from_quat(pose[:, 3:]).as_matrix()
    normal = np.asarray(geometry['normal'])
    tangent = Rotation.from_quat(geometry['rotation_xyzw']).as_matrix()[:, 0]
    center = np.asarray(geometry['top_center'])
    t = np.arange(HOLD)
    first = int(np.flatnonzero(load > .025)[0])
    last = int(bad[-1])
    start, end = first-110, min(last+95, HOLD-1)
    ready, release = first-18, last+16
    assert start < ready < first <= last < release < end
    measured = pose[:HOLD, :3]@tangent
    # Use the approach trend for the reference. The brief collision oscillation
    # in the recorded pose would require a correction faster than the motors.
    positions = gaussian_filter1d(measured, sigma=9)
    anchor = float(np.median(positions[release:release+30])) + anchor_offset
    alignment = smoothstep((t-start)/(ready-start))*(1-smoothstep((t-release)/(end-release)))
    # Keep the measured height profile unless a separate clearance ramp is
    # selected. This separates along-seat alignment from lowering to the seat.
    lowering = first-34
    lift = clearance*smoothstep((t-start)/(lowering-start))*(1-smoothstep((t-lowering)/(first+48-lowering)))
    translations = gain*((anchor-positions)*alignment)[:, None]*tangent + lift[:, None]*normal
    desired_rotation = rotation
    if pitch:
        start, end = first-370, HOLD-1
        lean = smoothstep((t-start)/(ready-start))*(1-smoothstep((t-first-12)/(end-first-12)))
        lateral = np.cross(normal, tangent)
        desired_rotation = Rotation.from_rotvec(pitch*lean[:, None]*lateral).as_matrix()@rotation[:HOLD]
        pelvis = model.vertices['pelvis']
        gap_before = (pelvis@rotation[:HOLD].transpose(0, 2, 1)@normal).min(1)
        gap_after = (pelvis@desired_rotation.transpose(0, 2, 1)@normal).min(1)
        translations += (gap_before-gap_after)[:, None]*normal
    knots = np.unique(np.r_[np.arange(start, end, 4), ready, first, last, release, end])
    delta = []
    fit_error = 0.
    for tick in knots:
        joint_position = q[tick].astype(float).copy()
        target = fk(joint_position)
        new_frame = desired_rotation[tick].T
        target[:, :3, 3] = (target[:, :3, 3]@rotation[tick].T-translations[tick])@new_frame.T
        target[:, :3, :3] = new_frame@rotation[tick]@target[:, :3, :3]

        def residual(angles):
            joint_position[legs] = angles
            feet = fk(joint_position)
            twist = Rotation.from_matrix(feet[:, :3, :3]@target[:, :3, :3].transpose(0, 2, 1)).as_rotvec()
            return np.c_[feet[:, :3, 3]-target[:, :3, 3], .2*twist].ravel()

        lower, upper = limits[0][legs]+.03, limits[1][legs]-.03
        guess = np.clip(q[tick, legs], lower, upper)
        fit = least_squares(residual, guess, bounds=(lower, upper),
                            max_nfev=24, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        fit_error = max(fit_error, float(np.abs(fit.fun).max()))
        delta.append(fit.x-q[tick, legs])
    assert fit_error < 1e-6, fit_error
    delta = PchipInterpolator(knots, delta, axis=0)(t[start:end+1])
    target = applied[:HOLD, :23].astype(float).copy()
    target[start:end+1, legs] += delta
    latent, decoded = encode(target, initial, *limits)
    error = np.abs(decoded-target)
    if error.max() > .004:
        tick, joint = np.unravel_index(error.argmax(), error.shape)
        raise ValueError(f'command encoding {error[tick,joint]:.5f} rad at tick {tick}, joint {joint}')
    # Reject modeled loss of housing clearance and report changes in gap.
    # The unchanged independent simulation scorer decides physical success.
    min_gap = np.inf
    baseline_gap = np.inf
    for tick in range(ready, release+1, 4):
        modified = q[tick].astype(float).copy()
        modified[legs] += decoded[tick, legs]-target[tick, legs]+delta[tick-start]
        for coordinates, translation, orientation in (
                (modified, translations[tick], desired_rotation[tick]),
                (q[tick], np.zeros(3), rotation[tick])):
            body = model.fk(coordinates)
            margins = []
            for name in HOUSINGS:
                housing = body[name]
                local = model.vertices[name]@housing[:3, :3].T+housing[:3, 3]
                world = local@orientation.T+pose[tick, :3]+translation
                margins.append(float(((world-center)@normal).min()))
            gap = min(margins)
            if coordinates is modified:
                min_gap = min(min_gap, gap)
            else:
                baseline_gap = min(baseline_gap, gap)
    assert min_gap > 0., min_gap
    result = path.copy()
    result[start:HOLD] = latent[start:HOLD]
    desired_position = positions+translations@tangent
    return result, dict(tick_load=first, final_bad_tick=last, alignment_complete_tick=ready,
                        reference_anchor_m=anchor, plateau_end_tick=release,
                        tangent_speed_m_s_before=float(np.abs(np.diff(positions[ready:release+1])).max()/DT),
                        tangent_speed_m_s_target=float(np.abs(np.diff(desired_position[ready:release+1])).max()/DT),
                        housing_gap_m=min_gap, baseline_housing_gap_m=baseline_gap,
                        foot_fit_max=fit_error, encoding_error_rad=float(error.max()),
                        max_translation_m=float(np.linalg.norm(translations, axis=1).max()))


def counterbalance(path, initial, load, joints, limits, shoulder, elbow):
    """Bring both arms forward before closing the remaining pelvis-seat gap."""
    first = int(np.flatnonzero(load > .025)[0])
    t = np.arange(len(path))
    reach = smoothstep((t-18)/(first-45-18))
    recover = 1-smoothstep((t-first-24)/(999-first-24))
    weight = reach*recover
    target = applied_path(path, initial, *limits).astype(float)
    for joint, displacement in (('shoulder_pitch', shoulder), ('elbow_pitch', elbow)):
        columns = [joints.index(f'{side}_{joint}_joint') for side in ('left', 'right')]
        target[:, columns] -= displacement*weight[:, None]
    latent, decoded = encode(target, initial, *limits)
    error = float(np.abs(decoded-target).max())
    assert error < .004, error
    return latent, dict(shoulder_rad=shoulder, elbow_rad=elbow,
                        full_forward_at_tick=first-45, arm_release_starts_tick=first+24,
                        arm_encoding_error_rad=error)


def prepare(out, use_counterbalance=False, forward_lean=False, transfer=None):
    inputs = {str(p): sha(p) for p in (Path(__file__), MODEL, HERE/'prepare_sitting_repairs.py',
                                      HERE/'repair_live_touchdown.py')}
    source = read(PROBE/'screen_results.json')
    freeze = read(PROBE/'screen.freeze.json')['inputs']
    paths_file = PROBE/'paths.npy'
    assert sha(paths_file) == freeze[str(paths_file)]
    for p in (paths_file, PROBE/'screen_results.json', PROBE/'screen.freeze.json'):
        inputs[str(p)] = sha(p)
    result, geometry, trace = checked_trace(PROBE/'screen', inputs)
    reference = Path(read(PROBE/'screen/manifest.json')['args']['bank'])
    meta = read(reference/'metadata.json')
    limits = tuple(np.asarray(meta[k]) for k in ('lower', 'upper'))
    model = BodyModel(MODEL)
    fk, legs = leg_fk(model, meta['joint_names'])
    paths = np.load(paths_file, allow_pickle=False)
    rows, plans, resets, geoms, rejected = [], [], [], [], []

    def admit(name, choice, checks, path, geo, reset):
        j = len(rows)
        rows.append(dict(id=f'probe/{j+1:02d}', case=name, choice=choice, checks=checks))
        plans.append(path)
        resets.append(reset)
        geoms.append(dict(geo, case=j, surface_id=j))

    targets = dict(TARGETS)
    if transfer:
        targets.pop('live_holdout/01')
        transfer_result, transfer_geo, transfer_trace = checked_trace(transfer/'screen', inputs)
        transfer_rows = read(transfer/'screen_results.json')
        transfer_paths = np.load(transfer/'paths.npy', allow_pickle=False)
        for file in ('screen_results.json', 'paths.npy', 'screen.freeze.json'):
            inputs[str(transfer/file)] = sha(transfer/file)
        assert inputs[str(transfer/'paths.npy')] == read(transfer/'screen.freeze.json')['inputs'][str(transfer/'paths.npy')]
    for name, (bank_name, anchors) in targets.items():
        bank = Path(read(PREVIOUS/bank_name/'manifest.json')['args']['bank'])
        i = read(bank/'metadata.json')['ids'].index(name)
        target_geo = read(bank/'geometry.json')[i]
        for file in bank.iterdir():
            inputs[str(file)] = sha(file)
        with np.load(bank/'bank.npz', allow_pickle=False) as z:
            reset = {k: z[k][i] for k in z.files if k.startswith('reset_')}
        if transfer:
            for j,row in enumerate(transfer_rows):
                if row['case']!='live_holdout/01' or not row['clean']:
                    continue
                assert transfer_result['episodes'][j]['clean']
                pose = transfer_trace['body'][:, j, transfer_result['body_names'].index('pelvis')].copy()
                pose[:, :3] -= np.asarray(transfer_result['origins'][j])
                initial = transfer_trace['previous'][0,j]
                np.testing.assert_array_equal(initial, trace['previous'][0,7])
                base_path = transfer_paths[j]
                applied = applied_path(base_path, initial, *limits)
                motion = dict(body=pose, q=transfer_trace['q'][:,j], load=transfer_trace['support_bw'][:,j])
                admit(name, dict(method='verified_replay', probe=row['id']), {}, base_path,target_geo,reset)
                for shift in (0., -.002, .002):
                    choice = dict(method='verified_transfer', probe=row['id'], fore_aft_m=shift)
                    try:
                        goal, checks = retarget(motion, transfer_geo[j], target_geo, limits, applied,
                                                model, fk, legs, fore_aft_m=shift)
                        command, actual = encode(goal, initial, *limits)
                        error = float(np.abs(actual-goal).max())
                        assert error <= .004, error
                        checks['encoding_error_rad'] = error
                    except (AssertionError, ValueError) as exc:
                        rejected.append(dict(case=name, choice=choice, reason=str(exc)))
                        continue
                    admit(name,choice,checks,command,target_geo,reset)
        for anchor in anchors:
            idx = next(i for i, row in enumerate(source) if row['id'] == anchor and row['case'] == name)
            for field in ('top_center', 'rotation_xyzw', 'width', 'depth'):
                assert geometry[idx][field] == target_geo[field]
            assert not any(result['episodes'][idx]['physical_failures'])
            pose = trace['body'][:, idx, result['body_names'].index('pelvis')].copy()
            pose[:, :3] -= np.asarray(result['origins'][idx])
            if use_counterbalance:
                variants = [(0., 0., gain, shoulder, elbow, 0.)
                            for gain in (0., .32) for shoulder, elbow in ((.32, .33), (.45, .42), (.53, .49))]
            elif forward_lean:
                variants = [(0., 0., gain, 0., 0., pitch)
                            for gain in (0., .25) for pitch in (.08, .14, .20)]
            else:
                variants = [(clearance, offset, gain, 0., 0., 0.) for clearance, offset, gain in
                            ((0., 0., 1.), (.003, 0., 1.), (.006, 0., 1.),
                             (.003, .003, 1.), (.003, -.003, 1.), (.003, 0., .65))]
            method = 'counterbalance' if use_counterbalance else 'forward_lean' if forward_lean else 'tangential_alignment'
            for clearance, offset, gain, shoulder, elbow, pitch in variants:
                choice = dict(method=method, probe=anchor,
                              clearance_m=clearance, anchor_offset_m=offset, alignment_fraction=gain)
                if use_counterbalance:
                    choice.update(shoulder_rad=shoulder, elbow_rad=elbow)
                if forward_lean:
                    choice['forward_pitch_rad'] = pitch
                try:
                    initial = trace['previous'][0, idx]
                    load = trace['support_bw'][:, idx]
                    path, checks = paths[idx], {}
                    if gain or pitch:
                        path, checks = align(path, initial, pose, trace['q'][:, idx], trace['command'][:, idx],
                                             load, np.flatnonzero(~trace['no_slip'][:, idx]), target_geo,
                                             model, fk, legs, limits, clearance, offset, gain, pitch)
                    if use_counterbalance:
                        path, arm_checks = counterbalance(path, initial, load, meta['joint_names'], limits,
                                                          shoulder, elbow)
                        checks.update(arm_checks)
                except (ValueError, AssertionError) as exc:
                    rejected.append(dict(case=name, choice=choice, reason=str(exc)))
                    continue
                admit(name,choice,checks,path,target_geo,reset)
    assert set(targets) == {row['case'] for row in rows}, rejected
    out.mkdir()
    bank = out/'bank'
    bank.mkdir()
    n = len(rows)
    states = {k: np.stack([r[k] for r in resets]) for k in resets[0]}
    states.update(reference=np.arange(n), admitted=np.ones(n, bool),
                  q_post=np.broadcast_to(states['reset_q'], (1000, n, 37)),
                  phase_times=np.full((n, 2), 15.))
    np.savez_compressed(bank/'bank.npz', **states)
    np.save(out/'paths.npy', np.stack(plans).astype(np.float32))
    write(bank/'geometry.json', geoms)
    write(bank/'metadata.json', dict(meta, ids=[row['id'] for row in rows], cases=n,
                                    positive_demonstrations=0, reference_is_reward_only=True))
    write(out/'candidates.json', dict(rows=rows, rejected=rejected, source_traces=inputs))
    write(out/'reservation.json', dict(files={str(p): sha(p) for p in bank.iterdir()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--screen-only', action='store_true')
    approach = parser.add_mutually_exclusive_group()
    approach.add_argument('--counterbalance', action='store_true')
    approach.add_argument('--forward-lean', action='store_true')
    parser.add_argument('--transfer-clean',type=Path)
    args = parser.parse_args()
    destination = args.out.resolve()
    if not args.screen_only:
        prepare(destination, args.counterbalance, args.forward_lean,
                args.transfer_clean.resolve() if args.transfer_clean else None)
    if not args.prepare_only:
        for path, digest in read(destination/'candidates.json')['source_traces'].items():
            assert sha(path) == digest, path
        screen(destination)
