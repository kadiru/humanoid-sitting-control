"""Build and evaluate the four-size benchmark on the archived 32-object sitting starts."""
import argparse
import csv
import hashlib
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

from run import DISK_RESERVE, P, PREPARED, read, sha, stage, write
from surface import H, Seat

SUITE = P / 'furniture_scale_suite_01'
SIZES = (100, 88, 76, 64)
HALVES = (0, 16)


def original(half):
    return P / f'protected_hip_landing_{half}'


def run_folder(size, half):
    return original(half) if size == 100 else SUITE / f'size_{size}_{half}'


def split_objects(objects):
    # Hold out two objects per height quartile before reading the controller outcomes.
    ordered = sorted(objects, key=lambda g: (g['height'], g['name']))
    assert len({g['source_sha256'] for g in ordered}) == len(ordered) == 32
    held_out = set()
    for offset in range(0, 32, 8):
        quartile = ordered[offset:offset + 8]
        def key(g):
            return hashlib.sha256(('sitting-size-v1/' + g['name']).encode()).hexdigest()
        held_out.update(g['name'] for g in sorted(quartile, key=key)[:2])
    return held_out


def scaled_geometry(g, folder, size):
    factor = size / 100
    assert sha(g['mesh_file']) == g['mesh_sha256']
    assert g['mesh_root'] == [0, 0, 0] and g['mesh_root_xyzw'] == [0, 0, 0, 1]
    assert not g['parts']
    path = folder / Path(g['mesh_file']).name
    shutil.copyfile(g['mesh_file'], path)
    usd = Usd.Stage.Open(str(path))
    mesh = UsdGeom.Mesh.Get(usd, '/Furniture/Mesh')
    assert mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
    assert UsdGeom.GetStageMetersPerUnit(usd) == 1.
    center = np.array(g['top_center'])
    pivot = np.array([center[0] + g['depth'] / 2, center[1], 0.])
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    faces = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    new_points = (pivot + factor * (points - pivot)).astype('f')
    new_center = pivot + factor * (center - pivot)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(new_points))
    bounds = [Gf.Vec3f(*map(float, p)) for p in (new_points.min(0), new_points.max(0))]
    mesh.GetExtentAttr().Set(bounds)
    usd.GetRootLayer().Save()
    del mesh, usd
    usd = Usd.Stage.Open(str(path))
    mesh = UsdGeom.Mesh.Get(usd, '/Furniture/Mesh')
    assert UsdPhysics.CollisionAPI(mesh.GetPrim()).GetCollisionEnabledAttr().Get()
    np.testing.assert_array_equal(np.asarray(mesh.GetPointsAttr().Get()), new_points)
    np.testing.assert_array_equal(np.asarray(mesh.GetFaceVertexIndicesAttr().Get()), faces)
    return dict(g, x=new_center[0], y=new_center[1], height=new_center[2],
                top_center=new_center.tolist(), reference_height_cm=100 * new_center[2],
                depth=factor * g['depth'], width=factor * g['width'],
                size_m=(factor * np.array(g['size_m'])).tolist(),
                mesh_file=str(path), mesh_sha256=sha(path))


def height_at_target(g):
    seat = Seat(g)
    index = np.rint((np.array(g['top_center'][:2]) - seat.origin) / H).astype(int)
    h = float(seat.z[tuple(index)])
    assert seat.mask.any() and np.isfinite(h) and abs(h - g['height']) < .025, g['name']
    return round(100 * h, 3)


def prepare():
    SUITE.mkdir()
    source = {i: read(original(i) / 'bank/geometry.json') for i in HALVES}
    held_out = split_objects(source[0] + source[16])
    cases = []
    for size in SIZES:
        for half in HALVES:
            src = original(half)
            target = run_folder(size, half)
            bank = target / 'bank'
            if size == 100:
                geometry = source[half]
            else:
                bank.mkdir(parents=True)
                meshes = target / 'meshes'
                meshes.mkdir()
                geometry = [scaled_geometry(g, meshes, size) for g in source[half]]
                write(bank / 'geometry.json', geometry)
                for name in ('metadata.json', 'bank.npz'):
                    shutil.copyfile(src / 'bank' / name, bank / name)
                shutil.copyfile(src / 'landing_profile.json', target / 'landing_profile.json')
                write(target / 'reservation.json', dict(files={str(f): sha(f) for f in bank.iterdir()}))
            for idx, g in enumerate(geometry):
                assert g['name'] == source[half][idx]['name'] and sha(g['mesh_file']) == g['mesh_sha256']
                split = 'held_out_object' if g['name'] in held_out else ('held_out_scale' if size == 76 else 'train')
                cases.append(dict(case=f"{g['name']}@{size}", asset=g['name'], scale_pct=size, split=split,
                                  seat_surface_cm=height_at_target(g), bank=str(bank), trajectory_index=idx,
                                  mesh_sha256=g['mesh_sha256'], mesh=g['mesh_file']))
    assert Counter(c['split'] for c in cases) == {'train': 72, 'held_out_scale': 24, 'held_out_object': 32}
    write(SUITE / 'cases.json', cases)


def evaluate():
    cases = read(SUITE / 'cases.json')
    assert len(cases) == len({c['case'] for c in cases}) == 128
    for size in SIZES[1:]:
        for half in HALVES:
            src = original(half)
            target = run_folder(size, half)
            bank = target / 'bank'
            if (target / 'evaluation.score.json').exists():
                continue
            frozen = read(src / 'evaluation.freeze.json')
            for path, digest in frozen['inputs'].items():
                assert sha(path) == digest, path
            for name in ('bank/bank.npz', 'bank/metadata.json', 'landing_profile.json'):
                assert sha(src / name) == sha(target / name)
            rows = [c for c in cases if c['bank'] == str(bank)]
            assert len(rows) == 16 and all(sha(c['mesh']) == c['mesh_sha256'] for c in rows)
            command = frozen['command']
            stage(target, 'evaluation', bank, PREPARED / 'library.npz', worker=Path(command[2]),
                  extra=command[3:command.index('--mode')], disk_reserve=DISK_RESERVE,
                  inputs=[*map(Path, frozen['inputs']), Path(__file__), SUITE / 'cases.json',
                          target / 'landing_profile.json', *[Path(c['mesh']) for c in rows]])


def report():
    from publish_confirmed import assess
    cases = read(SUITE / 'cases.json')
    records = []
    for size in SIZES:
        for half in HALVES:
            folder = run_folder(size, half)
            audited, _ = assess(folder)
            camera = read(folder / 'evaluation/seen_seat.json')
            colliders = read(folder / 'evaluation/scene_assets_verified.json')
            assert len(audited) == len(colliders) == 16
            for case, result, collider in zip((c for c in cases if c['bank'] == str(folder / 'bank')),
                                               audited, colliders, strict=True):
                i = case['trajectory_index']
                assert case['asset'] == result['asset'] and collider['enabled_colliders'] == ['/Mesh']
                records.append(dict(**case, supported_sit=result['supported_sit'], strict_clean=result['strict_clean'],
                                    terminal_hold_s=result['terminal_hold_s'], outcome=result['reason'],
                                    contact=result['contact_detail'], matched_approach=result['camera_matched'],
                                    camera_seat_cm=round(camera['estimated_height_m'][i]*100, 2),
                                    trace=str(folder / 'evaluation/trajectory.npz'), trace_sha256=result['trajectory_sha256']))
    assert len(records) == 128
    def save(name, data):
        with (SUITE / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(data)
    save('case_results.csv', records)
    save('supported_training_rollouts.csv', [r for r in records if r['split'] == 'train' and r['supported_sit']])
    for size in SIZES:
        rows = [r for r in records if r['scale_pct'] == size]
        print(size, Counter(r['outcome'] for r in rows), 'matched', sum(r['matched_approach'] for r in rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'evaluate', 'report'])
    op = parser.parse_args().operation
    {'prepare': prepare, 'evaluate': evaluate, 'report': report}[op]()
