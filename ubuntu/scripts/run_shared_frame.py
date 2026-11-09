"""Replay the runtime estimator using ONLY sensors.npz and static calibration.

Capture and estimator are separate processes so simulator truth cannot leak
through an environment object. The same update() accepts live sensor packets.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel, SharedFrameTracker, SensorFrame, transform


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def sample(points, count):
    out = np.full((count, 3), np.nan, np.float32)
    if len(points):
        p = points[np.linspace(0, len(points)-1, min(count, len(points))).astype(int)]
        out[:len(p)] = p
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['clean', 'noisy', 'occlusion'], default='clean')
    parser.add_argument('--max-frames', type=int)
    a = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    capture = Path(a.capture); out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    source_dir = out/'source'; source_dir.mkdir()
    sources = [Path(__file__), root/'sitting_task/src/humanoid_sitting/shared_frame.py',
               root/'scripts/test_shared_frame.py', root/'scripts/evaluate_shared_frame.py']
    for path in sources:
        shutil.copy2(path, source_dir/path.name)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_dir.iterdir()}
    # Whitelist: runtime never loads capture manifest (which contains test geometry).
    with np.load(capture/'sensors.npz') as archive:
        # NpzFile decompresses a whole member on each access; load each once.
        data = {k: archive[k] for k in ('depth', 'q', 'gravity', 'delta_rotation', 'stamp', 'K')}
        calibration = archive['T_P_C'] if 'T_P_C' in archive.files else None
    mount = np.array([.05, .18, .55])
    f = np.array([-1., 0., -1.7]); f /= np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1]); right /= np.linalg.norm(right)
    T_P_C = transform(np.stack((right, np.cross(f, right), f), axis=1), mount)
    if calibration is not None:
        if (calibration.shape != (4, 4) or not np.isfinite(calibration).all() or
            not np.allclose(calibration[3], [0, 0, 0, 1]) or
            not np.allclose(calibration[:3, :3].T @ calibration[:3, :3], np.eye(3)) or
            not np.isclose(np.linalg.det(calibration[:3, :3]), 1.)):
            raise ValueError('T_P_C must be one static calibrated rigid transform')
        T_P_C = calibration
    model = BodyModel(root/'runs/g1_kinematics_geometry_04/model.json')
    ncase = data['q'].shape[1]
    trackers = [SharedFrameTracker(model, T_P_C) for _ in range(ncase)]
    rng = np.random.default_rng(41019)
    frames = len(data['stamp']) if a.max_frames is None else min(a.max_frames, len(data['stamp']))
    arrays = {k: [] for k in ('cloud', 'skeleton', 'proxy', 'T_O_P', 'T_O_S', 'boundary', 'cells', 'floor', 'junctions')}
    rows = []; timings = []
    for k in range(frames):
        stamp = float(data['stamp'][k]); batch = {x: [] for x in arrays}
        for i in range(ncase):
            depth = data['depth'][k, i].copy()
            if a.mode == 'noisy':
                depth += rng.normal(0, .005, depth.shape)
                depth[rng.random(depth.shape) < .10] = np.nan
            if a.mode == 'occlusion' and (6. <= stamp < 6.5 or 8. <= stamp < 10.):
                depth[:] = np.nan
            received = stamp + (.06 if a.mode != 'clean' else 0.)
            packet = SensorFrame(stamp, received, depth, data['K'][i], data['q'][k, i],
                data['gravity'][k, i], data['delta_rotation'][k, i], stamp, stamp)
            start = time.perf_counter()
            result = trackers[i].update(packet)
            timings.append(time.perf_counter()-start)
            row = {key: value for key, value in result.items()
                   if key not in ('cloud', 'skeleton', 'proxy_world', 'floor', 'seat')}
            row.update(case=i, frame=k)
            if result['seat'] is not None:
                row.update(dimensions=result['seat']['dimensions'], observed_area_m2=result['seat']['observed_area'],
                    axis_ambiguous=result['seat']['axis_ambiguous'], heading_modulo_pi=True,
                    axis_method=result['seat']['axis_method'], hull_fill=result['seat']['hull_fill'],
                    seat_center_O=result['seat']['T'][:3, 3], junction_evidence=result['seat'].get('junction_evidence', []))
            rows.append(serial(row))
            seat = result['seat']; floor = result['floor']
            batch['cloud'].append(sample(result['cloud'], 1600))
            batch['skeleton'].append(result['skeleton'])
            batch['proxy'].append(sample(result.get('proxy_world', []), 200))
            batch['T_O_P'].append(result['T_O_P'])
            batch['T_O_S'].append(seat['T'] if seat else np.full((4, 4), np.nan))
            boundary = np.c_[seat['boundary'], np.full(len(seat['boundary']), seat['T'][2, 3])] if seat else []
            cells = np.c_[seat['cells'], np.full(len(seat['cells']), seat['T'][2, 3])] if seat else []
            batch['boundary'].append(sample(boundary, 100)); batch['cells'].append(sample(cells, 900))
            batch['junctions'].append(sample(seat.get('junctions', []) if seat else [], 200))
            batch['floor'].append(np.r_[floor['normal'], floor['offset']] if floor else np.full(4, np.nan))
        for key in arrays:
            arrays[key].append(np.asarray(batch[key]))
        if k % 10 == 0:
            print('TRACK', a.mode, k, '/', frames, 'valid', sum(r['valid'] for r in rows[-ncase:]),
                  'last', rows[-1].get('clearance_m'), flush=True)
    np.savez_compressed(out/'tracks.npz', **{k: np.asarray(v) for k, v in arrays.items()})
    (out/'estimates.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    manifest = dict(arguments=vars(a), frames=frames, cases=ncase,
        source_sha256=source_hashes,
        camera_extrinsics_source='static T_P_C in sensor archive' if calibration is not None else 'legacy fixed mount calibration',
        T_P_C=T_P_C.tolist(),
        runtime_input_files=['sensors.npz', 'static joint/collision calibration'],
        runtime_does_not_read_truth_or_geometry_manifest=True,
        compute_seconds=dict(median=float(np.median(timings)), p95=float(np.quantile(timings, .95)), total=sum(timings)),
        input_sha256=hashlib.sha256((capture/'sensors.npz').read_bytes()).hexdigest(),
        output_sha256=hashlib.sha256((out/'estimates.json').read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('TRACK_COMPLETE', manifest['compute_seconds'], flush=True)


if __name__ == '__main__':
    main()
