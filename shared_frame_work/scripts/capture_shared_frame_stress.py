"""Analytical depth-camera fixtures for observability failures, not physics trials.

Scene/root trajectories are used by this GENERATOR only. The runtime consumes
its sensor archive, with truth saved separately for the offline evaluator.
Frozen encoders plus translating root deliberately simulate common foot slip.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel, transform, apply


def depth_image(T_W_C, K, geometry, height=240, width=320):
    v, u = np.mgrid[:height, :width]
    rays = np.stack(((u-K[0, 2])/K[0, 0], (v-K[1, 2])/K[1, 1], np.ones_like(v)), axis=-1)
    rays = rays @ T_W_C[:3, :3].T
    origin = T_W_C[:3, 3]
    with np.errstate(divide='ignore', invalid='ignore'):
        floor = -origin[2]/rays[..., 2]
    floor = np.where((floor > .08) & (floor < 4.), floor, np.inf)
    R = Rotation.from_euler('z', geometry['yaw']).as_matrix()
    local_o = (origin-[geometry['x'], geometry['y'], 0.]) @ R
    local_r = rays @ R
    lo = np.array([-geometry['depth']/2, -geometry['width']/2, 0.])
    hi = np.array([geometry['depth']/2, geometry['width']/2, geometry['height']])
    with np.errstate(divide='ignore', invalid='ignore'):
        a = (lo-local_o)/local_r; b = (hi-local_o)/local_r
    enter = np.minimum(a, b).max(-1); leave = np.maximum(a, b).min(-1)
    box = np.where((enter <= leave) & (enter > .08) & (enter < 4.), enter, np.inf)
    return np.minimum(floor, box).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--holdout', action='store_true', help='Frozen second bank: new geometry, 16 cm blind slip, 180 degree turn, 2 mm depth noise')
    a = parser.parse_args()
    root = Path(__file__).resolve().parents[1]; out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    source = out/'source'; source.mkdir(); shutil.copy2(__file__, source/Path(__file__).name)
    model = BodyModel(root/'runs/g1_kinematics_geometry_04/model.json')
    with np.load(root/'runs/shared_frame_capture_01/sensors.npz') as archive:
        q = archive['q'][0, 3]; K0 = archive['K'][3]
    fk = model.fk(q); body_names = list(fk)
    sole_z = min(apply(fk[s+'_ankle_roll_link'], model.vertices[s+'_ankle_roll_link'])[:, 2].min()
                 for s in ('left', 'right'))
    initial_z = float(-sole_z+.003)
    forward = np.array([-1., 0., -1.7]); forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1]); right /= np.linalg.norm(right)
    T_P_C = transform(np.stack((right, np.cross(forward, right), forward), axis=1), [.05, .18, .55])
    definitions = [
        ('visible_slide', .56, .91, .27, -12.),
        ('featureless_along_edge_slide', .55, 8., .33, 0.),
        ('blackout_slide', .60, .94, .36, 8.),
        ('turn_away_return', .57, 1.04, .39, -7.),
        ('low_seat_diagonal_motion', .61, .76, .23, 22.),
        ('high_seat_diagonal_motion', .55, .96, .46, -18.),
    ]
    if a.holdout:
        definitions = [
            ('visible_slide', .59, 1.03, .285, 17.),
            ('featureless_along_edge_slide', .62, 8., .355, 0.),
            ('blackout_slide', .58, .89, .325, -16.),
            ('turn_away_return', .65, .97, .405, 11.),
            ('low_seat_diagonal_motion', .55, .82, .245, -23.),
            ('high_seat_diagonal_motion', .63, 1.08, .44, 24.),
        ]
    bank = [dict(family='box', shape='rect', depth=d, width=w, height=h,
                 x=-.43, y=.015, yaw=np.deg2rad(yaw)) for _, d, w, h, yaw in definitions]
    stamps = np.arange(81)*.1
    sensors = {k: [] for k in ('depth', 'rgb', 'q', 'gravity', 'delta_rotation')}
    truth = {k: [] for k in ('root_pose', 'body_pose', 'seat_pose')}
    previous_R = [np.eye(3) for _ in bank]
    rng = np.random.default_rng(92147)
    for t in stamps:
        sb = {k: [] for k in sensors}; tb = {k: [] for k in truth}
        progress = float(np.clip((t-2.)/2., 0, 1))
        for i, g in enumerate(bank):
            yaw = 0.; offset = np.zeros(3)
            if i in (0, 1, 2):
                offset[1] = ((.16 if a.holdout else .12) if i == 2 else (.10 if a.holdout else .08))*progress
            elif i == 3:
                yaw = np.deg2rad(180. if a.holdout else 100.)*np.sin(np.pi*t/8.)**2
            else:
                offset = np.array([-.05, .03, -.06])*np.sin(np.pi*t/8.)**2
                yaw = np.deg2rad(10.)*np.sin(np.pi*t/8.)**2
            R = Rotation.from_euler('z', yaw).as_matrix()
            T = transform(R, [0, 0, initial_z]+offset)
            depth = depth_image(T @ T_P_C, K0, g)
            if a.holdout:
                depth += rng.normal(0, .002, depth.shape).astype(np.float32)
            if i == 2 and 2. <= t < 4.5:
                depth[:] = np.nan
            gray = np.uint8(255*np.clip(np.nan_to_num(depth, nan=4., posinf=4.)/4., 0, 1))
            sb['depth'].append(depth); sb['rgb'].append(np.repeat(gray[..., None], 3, axis=-1))
            sb['q'].append(q); sb['gravity'].append(R.T @ [0, 0, -1.])
            sb['delta_rotation'].append(previous_R[i].T @ R); previous_R[i] = R
            tb['root_pose'].append(np.r_[T[:3, 3], Rotation.from_matrix(R).as_quat()])
            tb['body_pose'].append([np.r_[(T @ fk[n])[:3, 3], Rotation.from_matrix((T @ fk[n])[:3, :3]).as_quat()]
                                    for n in body_names])
            tb['seat_pose'].append(np.r_[[g['x'], g['y'], g['height']-.05],
                                        Rotation.from_euler('z', g['yaw']).as_quat()])
        for key in sensors:
            sensors[key].append(sb[key])
        for key in truth:
            truth[key].append(tb[key])
    np.savez_compressed(out/'sensors.npz', **{k: np.asarray(v) for k, v in sensors.items()},
                        stamp=stamps, K=np.tile(K0, (len(bank), 1, 1)))
    np.savez_compressed(out/'truth.npz', **{k: np.asarray(v) for k, v in truth.items()})
    manifest = dict(geometry_bank=bank, case_names=[d[0] for d in definitions], body_names=body_names,
        source_trajectory='Analytical pinhole ray casting; prescribed root motion with frozen encoders; no dynamics, body occlusion, lens distortion, or sensor bias. Not a simulator/hardware benchmark.',
        tests='10 cm visible/along-edge slip; 16 cm blackout slip; 180 degree turn; 2 mm depth noise' if a.holdout else
              '8 cm visible/along-edge slip; 12 cm blackout slip; 100 degree turn; new seat heights and yaws',
        holdout=a.holdout, frozen_tracker_sha256=hashlib.sha256((root/'sitting_task/src/humanoid_sitting/shared_frame.py').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('STRESS_CAPTURE_COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
