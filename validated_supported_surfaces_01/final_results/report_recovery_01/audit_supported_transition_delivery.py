"""Additional offline integrity and sensor audits; never launches simulation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from humanoid_sitting.deployment import BODY_JOINT_NAMES

p = argparse.ArgumentParser()
p.add_argument('--study', required=True)
p.add_argument('--verified', required=True)
a = p.parse_args()
root, out = Path(a.study), Path(a.verified)
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
protocol = read(root / 'protocol.json')
checks = {'sources': {}, 'training': {}, 'evaluations': {}, 'sensor': {}}

for directory in sorted(root.iterdir()):
    if not (directory / 'manifest.json').is_file():
        continue
    manifest = read(directory / 'manifest.json')
    sources = {}
    for i, (path, expected) in enumerate(manifest['source_hashes'].items()):
        saved = directory / 'source' / f'{i:02d}_{Path(path).name}'
        assert sha(saved) == expected == protocol['hashes'][path]
        sources[path] = expected
    for name, expected in manifest['bank_hashes'].items():
        path = str(Path(manifest['args']['bank']) / name)
        assert expected == protocol['hashes'][path]
    checks['sources'][directory.name] = {'preserved_source_count': len(sources),
                                       'source_hashes': sources,
                                       'bank_hashes': manifest['bank_hashes']}

for seed in protocol['seeds']:
    d = root / f'train_{seed}'
    v = read(d / 'verification.json')
    progress = read(d / 'progress.json')
    for k in ['updates', 'transitions', 'active', 'warmup']:
        assert progress[k] == v[k]
    rows = read(d / 'results.json')['episodes']
    assert rows == [json.loads(x) for x in (d / 'episodes.jsonl').read_text().splitlines()]
    assert len(rows) == progress['episodes']
    assert all(row['steps'] * .02 == row['duration_s'] for row in rows)
    # Every completed episode triggers one 25-tick indexed reset. At most
    # 26 environments can have incomplete reset warmup when training stops.
    pending_warmup = len(rows) * 25 - v['warmup']
    unfinished_active = v['active'] - sum(x['steps'] for x in rows)
    assert 0 <= pending_warmup <= 26 * 25
    assert 0 <= unfinished_active < 26 * 1000
    final = torch.load(d / 'final.pt', map_location='cpu', weights_only=False)
    last = torch.load(d / 'model_223.pt', map_location='cpu', weights_only=False)
    assert final['iter'] == 223
    for role in ['actor_state_dict', 'critic_state_dict']:
        assert all(torch.isfinite(x).all() for x in final[role].values())
        assert all(torch.equal(x, last[role][k]) for k, x in final[role].items())
    checks['training'][str(seed)] = dict(passed=True, completed_episodes=len(rows),
        active_steps_in_completed_episodes=sum(x['steps'] for x in rows),
        unfinished_active_steps=unfinished_active, pending_warmup_steps=pending_warmup,
        final_equals_last_update=True, checkpoint_iteration_zero_based=223,
        initial_checkpoint_sha256=sha(d/'initial.pt'), final_checkpoint_sha256=sha(d/'final.pt'))

for bank in ['train', 'holdout']:
    for label in [f'fixed_{bank}'] + [f'eval_{s}_{bank}' for s in protocol['seeds']]:
        d = root / label
        results = read(d / 'results.json')
        rows = results['episodes']
        assert rows == [json.loads(x) for x in (d / 'episodes.jsonl').read_text().splitlines()]
        tr = np.load(d / 'trace.npz')
        startup = np.load(d / 'startup.npz')
        assert tr['t'].shape == (200,) and np.allclose(tr['t'], np.arange(1, 201)/10, atol=1e-12, rtol=0)
        assert all(x['steps'] == 1000 and x['duration_s'] == 20 and x['reset_index'] == 0 for x in rows)
        body = results['body_names'].index('pelvis')
        ids = [results['joint_names'].index(n) for n in BODY_JOINT_NAMES]
        other = [i for i in range(len(results['joint_names'])) if i not in ids]
        assert np.array_equal(tr['command'][:, :, other], np.broadcast_to(startup['command'][None, :, other], tr['command'][:, :, other].shape))
        geometry = read(d / 'geometry.json')
        failures = {x['case']: x for x in read(d / 'first_failures.json')}
        points, camera = [], []
        for case, g in enumerate(geometry):
            first = failures[case]['t'] + .02 if case in failures else float('inf')
            local = tr['body'][:, case, body, :3] - np.array(results['origins'][case])
            rotation = Rotation.from_quat(tr['body'][:, case, body, 3:]).as_matrix()
            normal = np.einsum('nji,j->ni', rotation, np.array(g['normal']))
            offset = (local - np.array(g['top_center'])) @ np.array(g['normal'])
            for j, t in enumerate(tr['t']):
                assert bool(tr['alive'][j, case]) == bool(t < first - 1.e-10)
                if t >= first - 1.e-10:
                    continue
                z = tr['geometry'][j, case]
                points.append(dict(case=case, t=float(t), valid=bool(z[11]), age=float(z[12]),
                    normal_error_deg=float(np.rad2deg(np.arccos(np.clip(z[2:5] @ normal[j], -1, 1)))) if z[11] else None,
                    offset_error_m=float(z[5] - offset[j]) if z[11] else None))
        for row in read(d / 'sensor_audit.json')['truth_validation']:
            first = failures[row['case']]['t'] + .02 if row['case'] in failures else float('inf')
            if row['t'] < first - 1.e-10:
                camera.append(row)
        def summarize(samples):
            valid = [x for x in samples if x['valid']]
            return dict(count=len(samples), valid=len(valid), lost=len(samples)-len(valid),
                normal_p95_deg=float(np.percentile([x['normal_error_deg'] for x in valid], 95)),
                offset_abs_p95_mm=float(1000*np.percentile([abs(x['offset_error_m']) for x in valid], 95)),
                offset_abs_max_mm=float(1000*max(abs(x['offset_error_m']) for x in valid)),
                maximum_age_s=max(x['age'] for x in valid))
        checks['sensor'][label] = dict(camera_samples_including_start=summarize(camera),
            recorded_10Hz_motion_samples=summarize(points),
            between_camera_samples=summarize([x for x in points if x['age'] > .001]),
            scope='Only before first physical failure. Recorded 10 Hz motion includes the 50 Hz kinematic bridge at alternate samples; it does not audit every control tick.')
        checks['evaluations'][label] = dict(full_duration=True, unactuated_commands_constant=True,
            first_failure_counts={k:sum(x['flags'][k] for x in failures.values()) for k in ['fall','forbidden','joint_limit','impact','torque']},
            physical_failures=sum(any(x['failures'].values()) for x in rows))

checks['passed'] = True
with (out / 'supplementary_audit.json').open('x') as f:
    json.dump(checks, f, indent=2)
print(json.dumps({'passed': True, 'training': checks['training'], 'sensor': checks['sensor']}, indent=2))
