"""Replay one saved standing-start trial and measure a paired CPU standing reference."""

import argparse
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from import_cpu_g1 import run
from isolated_runtime import start


DT = .005
SUBSTEPS = 4
WARMUP = 25
ACTIVE = 1000
STANDING = 300
HOLD = 250
CONTACT_BODIES = (
    'pelvis', 'left_hip_yaw_link', 'right_hip_yaw_link', 'torso_link', 'logo_link',
    'left_ankle_roll_link', 'right_ankle_roll_link',
)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read_scorer(source):
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location('unchanged_physical_audit', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.physical


class ContactReadout:
    def __init__(self, view, links):
        self.indices = np.array([links.index(name) for name in CONTACT_BODIES])
        self.unexpected = np.setdiff1d(np.arange(len(links)), self.indices)
        self.columns = np.array([1, 1, 1, 1, 1, 0, 0])
        paths = ['/Robot/' + name for name in links]
        self.view = view.create_rigid_contact_view(paths, [['/RestFloor', '/Support'] for _ in paths], 4096)
        if self.view.sensor_count != 44:
            raise ValueError('Unexpected rigid contact body count')

    def history_row(self):
        matrix = self.view.get_contact_force_matrix(DT)
        net = self.view.get_net_contact_forces(DT)
        filtered = matrix[self.indices, self.columns].copy()
        return filtered, net.copy(), float(np.linalg.norm(matrix[:, 1], axis=-1).sum())

    def vectors_and_points(self, filtered, centers):
        # PhysX reuses counts/offsets between the normal and friction getters.
        # Aggregate the normal contact points before reading the friction data.
        _, points, _, _, counts, offsets = self.view.get_contact_data(DT)
        if counts.sum() >= len(points):
            raise ValueError('Contact point buffer saturated')
        means = centers[self.indices].copy()
        for k, (i, j) in enumerate(zip(self.indices, self.columns)):
            n, begin = int(counts[i, j]), int(offsets[i, j])
            if n:
                if not 0 <= begin <= len(points) - n:
                    raise ValueError('Invalid contact point span')
                means[k] = points[begin:begin+n].mean(axis=0)
        forces, _, counts, offsets = self.view.get_friction_data(DT)
        if counts.sum() >= len(forces):
            raise ValueError('Contact friction buffer saturated')
        force = filtered.copy()
        for k, (i, j) in enumerate(zip(self.indices, self.columns)):
            n, begin = int(counts[i, j]), int(offsets[i, j])
            if n:
                if not 0 <= begin <= len(forces) - n:
                    raise ValueError('Invalid friction span')
                force[k] += forces[begin:begin+n].sum(axis=0)
        return force, means


def frame(robot, indices, contact, history, properties, support, applied_torque):
    lower, upper, max_force, local_com, weight = properties
    q = robot.get_dof_positions()[0, indices].copy()
    qd = robot.get_dof_velocities()[0, indices].copy()
    pose = robot.get_link_transforms()[0].copy()
    velocity = robot.get_link_velocities()[0].copy()
    rotations = Rotation.from_quat(pose[:, 3:]).as_matrix()
    centers = pose[:, :3] + np.einsum('bij,bj->bi', rotations, local_com)
    filtered, net, _ = history[-1]
    force, points = contact.vectors_and_points(filtered, centers)
    j = contact.indices
    point_vel = velocity[j, :3] + np.cross(velocity[j, 3:], points - centers[j])
    rotation = Rotation.from_quat(support['quaternion_xyzw']).as_matrix()
    normal = np.repeat(rotation[None, :, 2], 7, axis=0)
    normal[5:] = (0, 0, 1)
    local = (points - support['top_center']) @ rotation
    inside = np.ones(7, dtype=bool)
    inside[:5] = ((abs(local[:5, 0]) < support['size'][0]/2-.01)
                  & (abs(local[:5, 1]) < support['size'][1]/2-.01)
                  & (abs(local[:5, 2]) < .015))
    torso = j[3]
    inside[3:5] &= ((points[3:5]-pose[torso, :3]) @ rotations[torso])[:, 0] < -.035
    tangent_vel = point_vel - (point_vel * normal).sum(-1, keepdims=True)*normal
    wrong_known = np.linalg.norm(net[j] - filtered, axis=-1).max()
    wrong_unknown = max(np.linalg.norm(h[1][contact.unexpected], axis=-1).max() for h in history)
    peak = np.linalg.norm(np.stack([h[0][:5] for h in history]), axis=-1).max(0).sum()/weight
    root_velocity = velocity[0, :3] - np.cross(velocity[0, 3:], centers[0]-pose[0, :3])
    return dict(
        q=q, qd=qd, pose=pose, contact_points=points, total_support_normal_N=history[-1][2],
        native_effort=robot.get_dof_projected_joint_forces()[0, indices].copy(),
        root_height=pose[0, 2], wrong_force=max(wrong_known, wrong_unknown), peak_bw=peak,
        physical_force_w=force, physical_normal_w=normal, physical_friction=np.full(7, .6),
        physical_permitted=np.ones(7, dtype=bool), physical_rest_patch=np.arange(7) < 5,
        physical_inside_patch=inside, physical_slip_m_s=np.linalg.norm(tangent_vel, axis=-1),
        physical_bodyweight_N=weight, physical_speed_m_s=np.linalg.norm(root_velocity),
        physical_angular_speed_rad_s=np.linalg.norm(velocity[0, 3:]),
        physical_trunk_tilt_rad=np.arccos(np.clip(rotations[0, 2, 2], -1, 1)),
        physical_joint_margin_ok=((q[:23] > lower+.005) & (q[:23] < upper-.005)).all(),
        physical_torque=applied_torque, physical_torque_scale=max_force,
    )


def stack(records):
    return {key: np.asarray([row[key] for row in records])[:, None] for key in records[0]}


def capture(path, physical, *, standing):
    def probe(stage, sim, view, robot, names, indices, contract, out):
        with np.load(path, allow_pickle=False) as archive:
            motion = {key: archive[key] for key in archive.files}
        if motion['desired'].shape != (ACTIVE, 23):
            raise ValueError('Expected 1000 saved commands')
        links = list(robot.shared_metatype.link_names)
        if links[0] != 'pelvis':
            raise ValueError('Unexpected root link')
        contact = ContactReadout(view, links)
        targets = robot.get_dof_position_targets().copy()
        velocity_targets = robot.get_dof_velocity_targets().copy()
        effort_targets = robot.get_dof_actuation_forces().copy()
        native_props = lambda key: getattr(robot, 'get_dof_'+key)()[0, indices].copy()
        kp, kd, max_force = (native_props(key) for key in ('stiffnesses', 'dampings', 'max_forces'))
        native_limits = native_props('limits')[:23]
        np.testing.assert_array_equal(native_limits, np.column_stack((motion['lower'], motion['upper'])))
        masses = robot.get_masses()[0].copy()
        com = robot.get_coms()[0, :, :3].copy()
        properties = (motion['lower'], motion['upper'], max_force, com, float(masses.sum())*9.81)
        previous = targets[0, indices].copy()
        np.testing.assert_array_equal(previous, motion['initial'])
        ids = np.array([0], dtype=np.uint32)
        records = []
        alive = True
        controls = STANDING if standing else WARMUP + ACTIVE
        try:
            for control in range(controls):
                if not standing and control >= WARMUP and alive:
                    k = control - WARMUP
                    desired = motion['desired'][k]
                    previous[:23] += np.clip(np.float32(.1)*(desired-previous[:23]),
                                             np.float32(-.003), np.float32(.003))
                    if abs(previous-motion['command'][k]).max() > 2e-7:
                        raise ValueError('Decoded target differs from frozen command')
                    targets[0, indices] = previous
                history = []
                for substep in range(SUBSTEPS):
                    if substep == SUBSTEPS-1:
                        # Match the implicit actuator telemetry used by the original scorer.
                        q = robot.get_dof_positions()[0, indices].copy()
                        qd = robot.get_dof_velocities()[0, indices].copy()
                        applied = np.clip(kp*(previous-q) + kd*(velocity_targets[0, indices]-qd)
                                          + effort_targets[0, indices], -max_force, max_force)
                    robot.set_dof_actuation_forces(effort_targets, ids)
                    robot.set_dof_position_targets(targets, ids)
                    robot.set_dof_velocity_targets(velocity_targets, ids)
                    sim.simulate(DT, (control*SUBSTEPS + substep)*DT)
                    sim.fetch_results()
                    history.append(contact.history_row())
                row = frame(robot, indices, contact, history, properties, contract['support'], applied)
                row['command'] = previous.copy()
                if not all(np.isfinite(value).all() for value in row.values()):
                    raise ValueError('Nonfinite CPU contact or state data')
                records.append(row)
                alive &= not bool(physical(stack([row]))['failure'][0, 0])
                if control % 250 == 249:
                    print(f'{out.name}: {control+1}/{controls} controls', flush=True)
        finally:
            if records:
                np.savez_compressed(out/'score_inputs.npz', **stack(records), joint_names=names, link_names=links)
            write(out/'progress.json', dict(controls=len(records), expected_controls=controls))
    return probe


def score(out, physical):
    with np.load(out/'standing/score_inputs.npz') as archive:
        standing = dict(archive)
    with np.load(out/'rollout/score_inputs.npz') as archive:
        rollout = dict(archive)
    if len(standing['q']) != STANDING or len(rollout['q']) != WARMUP + ACTIVE:
        raise ValueError('Incomplete CPU trial')
    trial = {key: value[WARMUP:] for key, value in rollout.items() if value.shape[0] == WARMUP+ACTIVE}
    flags = physical(rollout)
    verdict = physical(trial)
    warm_failure = flags['failure'][:WARMUP].any()
    good = verdict['good'][:, 0] & ~warm_failure
    bad = np.flatnonzero(~good)
    hold = (ACTIVE-(int(bad[-1])+1 if bad.size else 0))*.02
    valid_standing = bool(not physical(standing)['failure'].any()
                         and (standing['total_support_normal_N'][-HOLD:] < .01*standing['physical_bodyweight_N'][-HOLD:]).all()
                         and (abs(standing['qd'][-HOLD:, :, :23]) <= .05).all())
    effort = lambda data: float(np.square(data['native_effort'][-HOLD:, 0, :23].astype(float)).sum(-1).mean())
    base, final = effort(standing), effort(trial)
    force, normal = trial['physical_force_w'][-HOLD:, 0, :5], trial['physical_normal_w'][-HOLD:, 0, :5]
    weight = float(trial['physical_bodyweight_N'][0, 0])
    support_normal = (force*normal).sum(-1).clip(min=0).sum(-1)/weight
    support_up = force[..., 2].sum(-1)/weight
    failure = np.flatnonzero(verdict['failure'][:, 0])
    return dict(
        clean=bool(verdict['clean'][0] and not warm_failure), stable=bool(verdict['stable'][0] and not warm_failure),
        warmup_failure=bool(warm_failure), final_hold_s=hold,
        first_failure_s=(int(failure[0])+1)*.02 if failure.size else None,
        slip_ticks=int((~verdict['no_slip']).sum()), illegal_ticks=int((~verdict['legal']).sum()),
        standing_reference_valid=valid_standing, standing_tau_squared_Nm2=base, final_tau_squared_Nm2=final,
        final_over_standing_effort=final/base if valid_standing and base>0 else None,
        final_support_normal_mean_bw=float(support_normal.mean()), final_support_normal_min_bw=float(support_normal.min()),
        final_support_upward_mean_bw=float(support_up.mean()),
        final_foot_upward_mean_bw=float(trial['physical_force_w'][-HOLD:, 0, 5:, 2].sum(-1).mean()/weight),
        final_knee_flexion_deg=float(np.degrees(trial['q'][-HOLD:, 0][:, [list(rollout['joint_names']).index(s+'_knee_joint') for s in ('left','right')]]).mean()),
        physics_steps=(STANDING+WARMUP+ACTIVE)*SUBSTEPS, training_updates=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('contract', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    begun = time.monotonic()
    contract = json.loads(args.contract.read_text())
    args.output.mkdir(exist_ok=False)
    app = start()
    try:
        physical = read_scorer(Path(contract['scorer']))
        for label in ('standing', 'rollout'):
            out = args.output/label
            out.mkdir()
            run(args.contract, out, capture(Path(contract['motion']), physical, standing=label=='standing'))
        result = score(args.output, physical)
        result['case_id'] = contract['case_id']
        write(args.output/'result.json', result)
        print(json.dumps(result), flush=True)
    except Exception as error:
        write(args.output/'failure.json', dict(error=repr(error)))
        raise
    finally:
        import omni.physx
        omni.physx.get_physx_simulation_interface().detach_stage()
        app.shutdown()
        r = resource.getrusage(resource.RUSAGE_SELF)
        write(args.output/'resources.json', dict(wall_s=time.monotonic()-begun,
                                                 process_cpu_s=r.ru_utime+r.ru_stime,
                                                 peak_rss_KiB=r.ru_maxrss))


if __name__ == '__main__':
    main()
