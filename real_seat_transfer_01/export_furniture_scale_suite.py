"""Export portable USD objects, landing targets and prospective training splits."""
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from furniture_scale_suite import SIZES, SUITE
from run import read, sha, write


def assets():
    cases = read(SUITE / 'cases.json')
    with zipfile.ZipFile(SUITE / '32_furniture_four_scales.zip', 'x',
                         compression=zipfile.ZIP_DEFLATED, compresslevel=2, allowZip64=True) as archive:
        index = []
        for case in cases:
            obj = Path(case['mesh'])
            assert sha(obj) == case['mesh_sha256']
            rel = f"{case['scale_pct']}pct/{obj.name}"
            archive.write(obj, rel)
            geom = read(Path(case['bank']) / 'geometry.json')[case['trajectory_index']]
            assert geom['name'] == case['asset']
            geom['mesh_file'] = obj.name
            archive.writestr(str(Path(rel).with_suffix('.json')), json.dumps(geom, indent=2))
            index.append({k: v for k, v in case.items() if k not in ('bank','mesh','trajectory_index')} |
                         dict(mesh=rel))
        content = io.StringIO()
        w = csv.DictWriter(content, fieldnames=index[0])
        w.writeheader()
        w.writerows(index)
        archive.writestr('index.csv', content.getvalue())
        archive.writestr('README.txt', '''Each folder contains 32 standalone, meter-based, Z-up USD furniture collision meshes and the corresponding sitting target JSON. Paths in the JSON are relative to their folder. The robot faces +X, the seat is behind it and the floor is Z=0. Use the archived spawn translation and quaternion from the JSON. Floors, seat material, controller and action limits are common across scales. Scaling is uniform about the floor beneath the front of the registered seat target. These files represent rigid test surfaces with physics materials. The original decorative materials are not included.

index.csv assigns all four scales of eight objects to the object holdout. The other 24 objects supply training at 100%, 88% and 64%; 76% is reserved for interpolation evaluation. This split applies prospectively. All 32 original objects have previously been tested with the historical controller.
''')
    with zipfile.ZipFile(SUITE / '32_furniture_four_scales.zip') as archive:
        assert archive.testzip() is None
        assert len([k for k in archive.namelist() if k.endswith('.usdc')]) == 32 * len(SIZES)


def demos():
    # Export measured actions after all downstream filters, with the per-step observations
    # recorded by the same worker. The case split never depends on the trial outcome.
    with (SUITE / 'supported_training_rollouts.csv').open() as f:
        rows = list(csv.DictReader(f))
    by_trace = {}
    for r in rows:
        assert r['split'] == 'train' and r['supported_sit'] == 'True'
        by_trace.setdefault(r['trace'], []).append(r)
    obs, targets, previous, desired = [], [], [], []
    names, scales, strict, joint_order = [], [], [], None
    for trace, batch in by_trace.items():
        with np.load(trace) as z:
            columns = [int(r['trajectory_index']) for r in batch]
            assert sha(trace) == batch[0]['trace_sha256']
            assert np.all(z['learn'][:, columns]) and np.all(z['good'][-250:, columns])
            meta = read(Path(batch[0]['bank']) / 'metadata.json')
            assert list(meta['joint_ids']) == list(range(23))
            order = meta['joint_names'][:23]
            if joint_order is None:
                joint_order = order
            assert order == joint_order
            obs.extend(z['obs'][:, columns].transpose(1, 0, 2))
            targets.extend(z['command'][:, columns, :23].transpose(1, 0, 2))
            previous.extend(z['previous'][:, columns, :23].transpose(1, 0, 2))
            desired.extend(z['desired'][:, columns, :23].transpose(1, 0, 2))
            names.extend(r['case'] for r in batch)
            scales.extend(int(r['scale_pct']) for r in batch)
            strict.extend(r['strict_clean'] == 'True' for r in batch)
    obs, targets, previous, desired = (np.stack(v) for v in (obs, targets, previous, desired))
    assert obs.shape == (len(rows), 1000, 447) and targets.shape == (len(rows), 1000, 23)
    assert np.isfinite(obs).all() and np.isfinite(targets).all()
    assert np.max(np.abs(targets - previous)) < .0030003
    np.savez_compressed(SUITE / 'supported_training_demos.npz', case=np.array(names),
                        scale_pct=np.array(scales), strict_clean=np.array(strict),
                        episode_offsets=np.arange(len(rows) + 1) * 1000, joint_names=np.array(joint_order),
                        obs=obs.reshape(-1, 447), command_rad=targets.reshape(-1, 23),
                        previous_command_rad=previous.reshape(-1, 23), desired_rad=desired.reshape(-1, 23),
                        control_dt_s=np.array(.02))


def banks():
    """Give training and each evaluation their own simulator input directory."""
    cases = read(SUITE / 'cases.json')
    arrays = {}
    for path in {c['bank'] for c in cases}:
        with np.load(Path(path) / 'bank.npz', allow_pickle=False) as source:
            arrays[path] = {key: source[key] for key in source.files}
    root = SUITE / 'split_banks'
    root.mkdir()
    for role in ('train', 'held_out_scale', 'held_out_object'):
        group = [c for c in cases if c['split'] == role]
        target = root / role
        target.mkdir()
        ids = ['sit/' + c['case'] for c in group]
        geometry = []
        for index, c in enumerate(group):
            g = read(Path(c['bank']) / 'geometry.json')[c['trajectory_index']]
            assert sha(g['mesh_file']) == c['mesh_sha256']
            geometry.append(dict(g, case=index, surface_id=index, label=ids[index], split=role))
        write(target / 'geometry.json', geometry)
        meta = read(Path(group[0]['bank']) / 'metadata.json')
        write(target / 'metadata.json', dict(meta, cases=len(group), ids=ids, evaluation_split=role,
                                              fresh_holdout_used=False))
        bank = {}
        for key in arrays[group[0]['bank']]:
            if key == 'reference':
                bank[key] = np.arange(len(group))
                continue
            axis = 1 if key == 'q_post' else 0
            values = [np.take(arrays[c['bank']][key], c['trajectory_index'], axis=axis) for c in group]
            bank[key] = np.stack(values, axis=axis)
        np.savez_compressed(target / 'bank.npz', **bank)
        with np.load(target / 'bank.npz') as saved:
            for key in bank:
                np.testing.assert_array_equal(saved[key], bank[key])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('export', choices=('assets', 'demos', 'banks'))
    {'assets': assets, 'demos': demos, 'banks': banks}[parser.parse_args().export]()
