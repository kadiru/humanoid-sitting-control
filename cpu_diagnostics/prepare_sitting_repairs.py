"""Retarget one clean steep-seat descent to the lower development supports."""

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'trainer'))
from audit import physical
from common import ROOT, read, sha, write
from humanoid_sitting.shared_frame import BodyModel


BASELINE = ROOT/'outputs/supported_hold_ppo_20261112_v1/sitting72'
DATASET = ROOT/'outputs/full_body_contact_imitation_v4/dataset/sitting'
MODEL = ROOT/'runs/g1_kinematics_geometry_04/model.json'
TARGETS = [4, 5, 6, 7, 28, 29, 52, 53]
DONOR = 14
FEET = ('left_ankle_roll_link', 'right_ankle_roll_link')
HOUSINGS = ('left_hip_pitch_link', 'right_hip_pitch_link')


def leg_fk(model, joint_names):
    paths = {}
    for foot in FEET:
        chain = []
        while foot != 'pelvis':
            joint = next(j for j in model.joints if j['body1'] == [foot])
            chain.append(joint)
            foot = joint['body0'][0]
        paths[chain[0]['body1'][0]] = [(joint_names.index(j['name']), *model.frames[j['name']],
                                         np.eye(3)['XYZ'.index(j['axis'])]) for j in reversed(chain)]

    def poses(q):
        out = []
        for chain in paths.values():
            pose = np.eye(4)
            for index, left, right, axis in chain:
                rotated = np.eye(4)
                rotated[:3, :3] = Rotation.from_rotvec(axis*q[index]).as_matrix()
                pose = pose @ left @ rotated @ right
            out.append(pose)
        return np.array(out)

    return poses, np.array(sorted(i for chain in paths.values() for i, *_ in chain))


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x**3*(10 + x*(-15 + x*6))


def retarget(target, donor, geometry, limits, model, fk, legs, refinement=0):
    g = geometry[target]
    center, normal = np.asarray(g['top_center']), np.asarray(g['normal'])
    axes = Rotation.from_quat(g['rotation_xyzw']).as_matrix()
    q, cmd, pose = (donor[k].astype(np.float64) for k in ('q', 'command', 'pelvis'))
    rotation = Rotation.from_quat(pose[:, 3:]).as_matrix()
    # The 20-degree plane intersects the housings on the unrotated transplant.
    # Rotate the pelvis back and shift the cap inside the front edge while
    # preserving both foot orientations.
    steep = g['pitch_deg'] == 20.
    tilt = np.deg2rad((30. if refinement >= 2 else 25. if refinement else -15.) if steep else 0.)
    dx = (-.02 if refinement >= 2 else -.035 if refinement else -.025) if steep else 0.
    onset = 540 if refinement and steep else 460
    duration = 190 if refinement and steep else 240
    amount = smoothstep((np.arange(1000)-onset)/duration)
    turn = Rotation.from_rotvec(np.c_[np.zeros(1000), tilt*amount, np.zeros(1000)]).as_matrix() @ rotation
    cap = model.vertices['pelvis'] @ turn[-1].T + pose[-1, :3] + [dx, 0., 0.]
    gap = (cap-center) @ normal
    lowest = gap.argmin()
    dz = gap[lowest]/normal[2]
    uv = (cap[lowest]-center-[0., 0., dz]) @ axes
    assert abs(uv[0]) < g['depth']/2-.01 and abs(uv[1]) < g['width']/2-.01
    # Lower the clean donor until its pelvis cap reaches the target plane.
    # Ramp the height change through the descent and preserve the donor feet.
    assert 0 < dz < .05
    ticks = np.arange(460, 821, 10)
    descent = amount.copy()
    if refinement == 3 and steep:
        # Reach the forward-clearance attitude before closing the last part
        # of the gap. At the original slew rate the hip was ~0.28 rad behind
        # the coordinated path when simultaneous descent hit the housing.
        descent = smoothstep((np.arange(1000)-580)/140)
        amount = smoothstep((np.arange(1000)-485)/255)
        turn = Rotation.from_rotvec(np.c_[np.zeros(1000), tilt*amount, np.zeros(1000)]).as_matrix() @ rotation
    shift = -dz*descent[ticks]
    changes = []
    max_residual = 0.
    for t, z in zip(ticks, shift):
        target_feet = fk(q[t])
        target_feet[:, :3, 3] = (target_feet[:, :3, 3] @ rotation[t].T - [dx*amount[t], 0., z]) @ turn[t]
        target_feet[:, :3, :3] = turn[t].T @ rotation[t] @ target_feet[:, :3, :3]

        def residual(angles):
            x = q[t].copy()
            x[legs] = angles
            feet = fk(x)
            rotation_error = Rotation.from_matrix(feet[:, :3, :3] @ target_feet[:, :3, :3].transpose(0, 2, 1)).as_rotvec()
            return np.c_[(feet[:, :3, 3]-target_feet[:, :3, 3]), .2*rotation_error].ravel()

        fit = least_squares(residual, q[t, legs], bounds=(limits[0][legs]+.03, limits[1][legs]-.03),
                            max_nfev=32, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        error = float(abs(fit.fun).max())
        if not fit.success or error >= 1e-6:
            raise ValueError(('FOOT_FIT', target, int(t), error, fit.message))
        max_residual = max(max_residual, error)
        changes.append(fit.x-q[t, legs])
    delta = PchipInterpolator(ticks, changes, axis=0)(np.clip(np.arange(1000), ticks[0], ticks[-1]))
    adjusted = cmd.copy()
    adjusted[:, legs] += delta
    if refinement and not steep:
        # A small ankle bias moves the pelvis rearward during the brief
        # forward shear transient observed after initial contact.
        tick = np.arange(1000)
        pulse = smoothstep((tick-618)/22)*(1-smoothstep((tick-672)/32))
        adjusted[:, [15, 16]] += (.012 if refinement >= 2 else -.008)*pulse[:, None]

    # Report relative hull height at the retargeted last pose. The pose model
    # selects a candidate. The simulator supplies the pass/fail evidence.
    estimate = q[-1].copy()
    estimate[legs] += delta[-1]
    final_fk = model.fk(estimate)
    clearance = {}
    for name in HOUSINGS:
        part = final_fk[name]
        vertices = model.vertices[name] @ part[:3, :3].T + part[:3, 3]
        world = vertices @ turn[-1].T + pose[-1, :3] + [dx, 0., -dz]
        clearance[name] = float(((world-center)@normal).min())
    if min(clearance.values()) <= 0:
        raise ValueError(('HOUSING_CLEARANCE', target, clearance))
    return adjusted, dict(plane_drop_m=float(dz), foot_fit_max_residual=max_residual,
                          housing_plane_clearance_m=clearance, donor_cap_uv_m=uv[:2].tolist())


def encode(target, initial, lower, upper):
    running = initial[:23].astype(np.float32).copy()
    low, high = lower.astype(np.float32)+np.float32(.03), upper.astype(np.float32)-np.float32(.03)
    desired, executed = [], []
    for row in target[:, :23]:
        latent = np.clip(running+(row-running)/.1, low, high).astype(np.float32)
        running = running + np.clip(np.float32(.1)*(latent-running), np.float32(-.003), np.float32(.003))
        assert (running >= low-2e-7).all() and (running <= high+2e-7).all()
        desired.append(latent)
        executed.append(running)
    return np.asarray(desired), np.asarray(executed)


def prepare(out, refinement=False):
    audit = read(BASELINE/'independent_audit.json')
    result = read(BASELINE/'results.json')
    rows = result['episodes']
    trace = BASELINE/'trajectory.npz'
    assert sha(trace) == audit['files']['trajectory.npz'] and audit['passed']
    assert [i for i,r in enumerate(rows) if not r['clean']] == TARGETS and rows[DONOR]['clean']
    geometry, meta = (read(DATASET/name) for name in ('geometry.json', 'metadata.json'))
    assert meta['joint_names'] == result['joint_names'] and meta['ids'] == [r['id'] for r in rows]
    lower, upper = (np.asarray(meta[k]) for k in ('lower', 'upper'))
    model = BodyModel(MODEL)
    assert model.names == meta['joint_names']
    fk, legs = leg_fk(model, meta['joint_names'])
    with np.load(trace, allow_pickle=False) as a:
        inputs = {k: a[k] for k in a.files if k.startswith('physical_') or k in ('root_height','wrong_force','peak_bw')}
        np.testing.assert_array_equal(physical(inputs)['clean'], audit['clean'])
        desired = a['desired'].copy()
        body = a['body'][:, DONOR, 0].copy()
        body[:, :3] -= np.asarray(result['origins'][DONOR])
        donor = dict(q=a['q'][:, DONOR], command=a['command'][:, DONOR], pelvis=body)
        signatures = a['obs'][0, :, 336:].copy()
        start = a['previous'][0].copy()
        diagnostics = []
        for i in TARGETS:
            invalid = ~a['no_slip'][:, i] | ~a['legal'][:, i] | a['failure'][:, i]
            tick = int(np.flatnonzero(invalid)[0])
            revised = refinement if i in (6, 7, 28, 29) else 0
            adjusted, geom = retarget(i, donor, geometry, (lower, upper), model, fk, legs, revised)
            latent, executed = encode(adjusted, start[i], lower, upper)
            assert np.isfinite(latent).all()
            assert abs(np.diff(np.vstack((start[i,:23], executed)),axis=0)).max() <= .0030002
            desired[:, i] = latent
            diagnostics.append(dict(case=rows[i]['id'], first_bad_s=(tick+1)*.02,
                                    initial_failure_flags=a['flags'][tick,i].tolist(),
                                    first_bad_slip_m_s=float(a['loaded_slip'][tick,i]),
                                    first_bad_wrong_force_N=float(a['wrong_force'][tick,i]),
                                    max_target_tracking_error_rad=float(abs(executed-adjusted[:,:23]).max()),
                                    **geom))
    out.mkdir(exist_ok=False)
    np.save(out/'candidate_paths.npy', desired)
    np.save(out/'signatures.npy', signatures)
    write(out/'construction.json', dict(baseline_sha256=sha(trace), model_sha256=sha(MODEL),
                                        source_sha256=sha(__file__), candidate_sha256=sha(out/'candidate_paths.npy'),
        donor=rows[DONOR]['id'], targets=TARGETS, diagnostics=diagnostics, refinement=refinement,
                                        action_limits_unchanged=True, success_criteria_unchanged=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--refine', type=int, choices=(1, 2, 3), default=0)
    args = parser.parse_args()
    prepare(args.output, args.refine)
