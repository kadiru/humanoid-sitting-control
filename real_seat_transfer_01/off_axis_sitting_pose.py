"""Recover the sitting policies' three alignment inputs from rear depth."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from rescan_sitting import match, shoe_frame
from seat_alignment import UNITS, correction, points


STUDY = Path(__file__).resolve().parent/'off_axis_sitting_pose_01'
STANDING_FRAME = 0
SAMPLE = ('depth', 'intrinsics', 'mount', 'q', 'gravity')


def extract(sample, model):
    depth, intrinsics, mount, joints, gravity = sample
    return points(depth, intrinsics, mount, joints, gravity, model)


def estimate(reference, sample, model):
    registration = match(reference, extract(sample, model))
    if registration is None:
        return np.zeros(3, dtype=np.float32), None, None
    pose, residual = registration
    return (correction(pose)/UNITS).astype(np.float32), pose, residual


def scan(path):
    with np.load(path, allow_pickle=False) as data:
        assert data['ticks'][STANDING_FRAME] == 0
        return {key: data[key][STANDING_FRAME] for key in (*SAMPLE, 'rotation', 'root')}


def frame(data, row):
    return tuple(data[key] if key == 'mount' else data[key][row] for key in SAMPLE)


def audit_pose(ref, ref_row, seen, row, model):
    def world(data, i):
        _, origin, axes = shoe_frame(model, data['q'][i], data['gravity'][i])
        root = data['rotation'][i]
        return data['root'][i]+root@origin, root@axes

    origin, axes = world(ref, ref_row)
    other, other_axes = world(seen, row)
    position = (other-origin)@axes
    yaw = Rotation.from_matrix(axes.T@other_axes).as_euler('zyx')[0]
    return np.r_[position[:2], yaw]


def replay(scale, study_rows, poses, models):
    from off_axis_sitting_study import STUDY as BASELINE, evaluate
    from run import sha

    outcomes = []
    for method, model in models.items():
        path = BASELINE/f'{method}_{scale}'/'evaluation'/'skill_decisions.npz'
        with np.load(path, allow_pickle=False) as data:
            use = data['enabled']
            rows, phases, states, actions = (data[key][use] for key in ('row','phase','features','action'))
        np.testing.assert_array_equal(evaluate(model, states, phases).argmax(1), actions)
        states[:,4:7] = poses[rows]
        other = next(network for name, network in models.items() if name != method)
        restored = evaluate(model, states, phases).argmax(1)
        outcomes.append(dict(method=method, archived_decisions=len(rows), input_sha256=sha(path),
                             own_choices_changed=int((restored!=actions).sum()),
                             actors_disagree=int((restored!=evaluate(other,states,phases).argmax(1)).sum()),
                             affected_cases=sorted({study_rows[i]['case'] for i in rows[restored!=actions]})))
    return outcomes


def prepare():
    from off_axis_sitting_study import STUDY as BASELINE, catalog, checkpoint
    from run import read, sha, write
    from stance23_planner import model

    protocol = read(BASELINE/'protocol.json')
    models = {name:checkpoint(path) for name,path in protocol['actor'].items()}
    STUDY.mkdir()
    receipts, frozen = {}, {}
    for scale in catalog.SIZES:
        nominal_path = catalog.STUDY/f'piecewise_{scale}'/'evaluation'/'standing_scan.npz'
        observed_path = BASELINE/f'piecewise_{scale}'/'evaluation'/'standing_scan.npz'
        nominal, observed = scan(nominal_path), scan(observed_path)
        reference = {asset:extract(frame(nominal, row), model) for row,asset in enumerate(protocol['roster'])}
        template = STUDY/f'seat_clouds_{scale}.npz'
        np.savez_compressed(template, **reference)
        frozen[str(template)] = sha(template)
        cases = protocol['banks'][str(scale)]['cases']
        poses, results = [], []
        for row, case in enumerate(cases):
            asset = case['asset']
            values, pose, residual = estimate(reference[asset], frame(observed,row), model)
            target = audit_pose(nominal, protocol['roster'].index(asset), observed, row, model)
            error = None if pose is None else pose-target
            results.append(dict(case, valid=pose is not None, registration_m=pose.tolist() if pose is not None else None,
                                target_m=target.tolist(), features=values.tolist(), residual_m=residual,
                                translation_error_m=None if error is None else float(np.linalg.norm(error[:2])),
                                yaw_error_deg=None if error is None else float(abs(np.rad2deg(error[2])))))
            poses.append(values)
        tested = [p for p in results if p['valid']]
        outcomes = replay(scale, cases, np.stack(poses), models)
        receipts[scale] = dict(cases=results, replay=outcomes,
                               source={str(path):sha(path) for path in (nominal_path,observed_path)})
        write(STUDY/f'camera_audit_{scale}.json',receipts[scale])
        accurate = sum(p['translation_error_m']<=.01 and p['yaw_error_deg']<=2.5 for p in tested)
        print(f'{scale}%: {len(tested)}/{len(results)} registered, {accurate} within 1 cm and 2.5 deg', flush=True)
        print([{k:v for k,v in row.items() if k not in ('affected_cases','input_sha256')} for row in outcomes],flush=True)
    write(STUDY/'camera_inputs.json', frozen)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    prepare()
