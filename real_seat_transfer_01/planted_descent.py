"""Fit a saved descent to two verified, stationary soles."""
import json

import numpy as np
from scipy.spatial.transform import Rotation

from rescan_sitting import retarget, shoe_frame, smooth


def seat_state(env, row):
    data = env.robot.data
    weight = float(data.body_mass.torch[row].sum()) * 9.81
    seat = env.raw.scene['pelvis_seat'].data.normal_force_matrix_w.torch[row].sum((0, 1))[2].item() / weight
    foot = min(env.raw.scene['rest_' + side + '_floor'].data.normal_force_matrix_w.torch[row]
               .sum((0, 1))[2].item() / weight for side in ('left', 'right'))
    speed = data.root_lin_vel_w.torch[row].norm().item()
    angular_speed = data.root_ang_vel_b.torch[row].norm().item()
    return seat, foot, speed, angular_speed


def press_seat(out, row, tick, env, state, options):
    from stance23_planner import chains, legs, solve

    seat, foot, speed, gyro = seat_state(env, row)
    qualifies = seat > .024 and foot > .20 and speed < .028 and gyro < .10
    state['settled'] = state['settled'] + 1 if qualifies else 0
    if state['started'] is None and state['settled'] >= 8:
        state['started'] = tick
    if state['started'] is None:
        return
    age = tick - state['started']
    weight = float(smooth(age / 126.))
    limit = options['press_depth_m']
    if age > 28:
        update = float(np.clip(.00155 * (.12 - seat), -.00007, .00007))
        state['drop'] = float(np.clip(state['drop'] + update, 0., limit))
    data = env.robot.data
    q = data.joint_pos.torch[row].cpu().numpy()
    up = data.projected_gravity_b.torch[row].cpu().numpy()
    feet, _, basis = shoe_frame(env.model, q, up)
    rotate = Rotation.from_euler('xy', weight * np.array([.015, -.018])).as_matrix()
    pose = basis @ rotate @ basis.T
    pivot = np.array([.0024, .026, -.1492])
    shift = pivot - pose @ pivot - basis[:, 2] * state['drop']
    for side, (chain, ids) in enumerate(zip(chains, legs)):
        goal = np.eye(4)
        goal[:3, :3] = pose.T @ feet[side, :3, :3]
        goal[:3, 3] = pose.T @ (feet[side, :3, 3] - shift)
        ang, error = solve(chain, q[ids].copy(), goal, ids)
        if error < .001:
            out[row, ids] += out.new_tensor(np.clip(ang - q[ids], -.09, .09))


def hold_loaded_pose(out, row, env, state, minimum):
    seat, foot, speed, gyro = seat_state(env, row)
    options = state['options']
    qualifies = seat >= minimum and foot >= .22 and speed <= .022 and gyro <= options.get('hold_gyro', .13)
    if 'hold_tilt' in options:
        gravity_z = env.robot.data.projected_gravity_b.torch[row, 2].item()
        qualifies &= -gravity_z >= np.cos(options['hold_tilt'])
    state['settled'] = state['settled'] + 1 if qualifies else 0
    if state['hold'] is None and state['settled'] >= options.get('hold_samples', 5):
        state['hold'] = env.previous[row, :23].clone()
    if state['hold'] is not None:
        out[row] = state['hold']


def attach(config, previous_attach):
    from chair_contact_motion import ChairContactMotion

    options = config.get('planted_descent')
    if options is None:
        return previous_attach()
    with np.load(options['library'], allow_pickle=False) as bank:
        library = {name: bank[name] for name in bank.files}
    previous = ChairContactMotion.apply

    def apply(motion, learner, obs, tick, nominal):
        out = previous(motion, learner, obs, tick, nominal)
        env = learner.env
        if tick == 0:
            motion.planted_paths = {}
            motion.planted_actions = {}
            evidence = []
            with np.load(env.out / 'single_step.npz', allow_pickle=False) as step:
                planted = (step['options'] == 0) & step['verified_floor'][-50:].all(axis=(0, 2))
                planted &= ~step['slip'][-50:].any(axis=(0, 2)) & ~step['flags'].any(axis=(0, 2))
                planted &= (step['floor_load'][-50:] > .15).all(axis=(0, 2))
                planted &= (step['flat_speed'][-50:] < .02).all(axis=0)
            scan_tick, depth, intrinsics, joints, gravity, _, _, _, mount = env.standing_samples[-1]
            assert scan_tick == 0
            for row in options['rows']:
                asset = config['templates_by_row'][row]
                if not planted[row]:
                    evidence.append(dict(row=row, asset=asset, admitted=False,
                                         reason='standing contacts did not qualify'))
                    continue
                prefix = asset + '_'
                demo = {name[len(prefix):]: value for name, value in library.items() if name.startswith(prefix)}
                settings = {**options['assets'].get(asset, {}), **options['treatments'].get(str(row), {})}
                if settings.get('lateral_m'):
                    # Reduce the pelvis travel across the fixed feet during the
                    # descent. The donor's foot, arm and torso traces are unmodified.
                    demo['body'] = demo['body'].copy()
                    mix = smooth((np.arange(1000) - 65.) / 290.)
                    demo['body'][:, 0, 1] += mix * settings['lateral_m']
                fit, proof = retarget(
                    demo, env.model, (depth[row], intrinsics[row], joints[row], gravity[row], mount),
                    env.previous[row, :23].cpu().numpy(), shift=settings.get('setback', 0.),
                    clearance_pitch=settings.get('pitch', 0.), clearance_roll=settings.get('roll', 0.))
                evidence.append(dict(row=row, asset=asset, **proof))
                if fit is None:
                    continue
                motion.planted_actions[row] = dict(options=settings, hold=None, started=None,
                                                   drop=0., settled=0)
                if settings.get('recenter'):
                    from copy import deepcopy
                    # Use the same post-contact rim recentering as the already
                    # passing fore-aft start, after the two current feet qualify.
                    source = next(i for i in motion.rescan_feedback
                                  if config['templates_by_row'][i] == asset)
                    motion.rescan_paths[row] = fit
                    motion.rescan_feedback[row] = deepcopy(motion.rescan_feedback[source])
                else:
                    motion.planted_paths[row] = fit['latent']
            (env.out / 'planted_descent.json').write_text(json.dumps(evidence, indent=2) + '\n')
        for row, path in motion.planted_paths.items():
            out[row] = out.new_tensor(path[tick])
        for row, state in motion.planted_actions.items():
            settings = state['options']
            if 'brake_start' in settings:
                # The ordinary trajectory clock advances one tick. Take away
                # that advance gradually to avoid asking the loaded ankles to
                # stop correcting balance in one control update.
                brake = smooth((tick - settings['brake_start']) / settings['brake_ticks'])
                motion.rescan_feedback[row]['clock'] -= float(brake)
            if tick >= settings.get('press_start', 1000):
                press_seat(out, row, tick, env, state, settings)
            if tick >= settings.get('hold_start', 1000):
                hold_loaded_pose(out, row, env, state, settings['hold_load'])
        return out

    ChairContactMotion.apply = apply
    return previous_attach()
