"""Preserve completed studies and reserve tests before new contact experiments."""
import copy
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path('/home/kadir/humanoid-sitting')
OLD = Path('/data/humanoid-sitting-perception/low_positive_transfer_01')
DATA = OLD.parent / 'low_seat_contact_01'

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2) + '\n')

def main():
    DATA.mkdir(exist_ok=False)
    protocol = read(OLD / 'ppo_01/protocol.json')
    prior = {**protocol['prior_hashes'], **protocol['hashes']}
    for p in OLD.rglob('*'):
        if p.is_file() and not p.is_symlink():
            prior[str(p)] = sha(p)
    mismatches = [p for p, digest in prior.items() if sha(p) != digest]
    assert not mismatches, mismatches
    write(DATA / 'prior_hashes.json', prior)
    source = OLD.parent / 'two_surface_curriculum_01/selected_path_01'
    template = read(source / 'geometry.json')[0]
    grids = {
        'probe': [(.36,15),(.36,20),(.37,11.25),(.37,16.25),
                  (.365,12.5),(.365,17.5),(.40,20),(.36,5),
                  (.385,7.5),(.425,12.5),(.44,15),(.40,10)],
        'holdout': [(h,s) for h in [.3675,.3775,.395,.415]
                    for s in [8.75,13.75,18.75]],
    }
    for name, grid in grids.items():
        bank = DATA / 'banks' / name
        bank.mkdir(parents=True)
        rows = []
        for height, slope in grid:
            row = copy.deepcopy(template)
            rotation = Rotation.from_euler('y', slope, degrees=True)
            row.update(case=len(rows),surface_id=len(rows),height=height,pitch_deg=slope,
                top_center=[template['top_center'][0],template['top_center'][1],height],
                rotation_xyzw=rotation.as_quat().tolist(),normal=rotation.as_matrix()[:,2].tolist(),
                label=f'contact_{name}_{height:.4f}_{slope:.2f}',split=name,
                known_anchor=False,reset_scope='standing only')
            rows.append(row)
        write(bank / 'geometry.json', rows)
        for filename in ['candidates.npz','reset_candidates.npz']:
            with np.load(source / filename) as arrays:
                np.savez_compressed(bank / filename,
                    **{k:np.repeat(arrays[k][:1],len(rows),axis=0) for k in arrays.files})
        write(bank / 'selection.json',dict(source=str(source),indices=[0]*len(rows),standing_only=True))
    for name, original in [('known_old','development_old'),('known_transfer','development_transfer'),
                           ('known_recent','holdout')]:
        shutil.copytree(OLD/'banks'/original,DATA/'banks'/name)
    write(DATA/'reservation.json',dict(timestamp=time.time(),fresh_holdout=grids['holdout'],
        holdout_simulations=0,all_previous_holdouts_are_now_development=True,
        fresh_holdout_excluded_from_repair_calibration_training_selection=True,
        hashes={str(p):sha(p) for p in (DATA/'banks').rglob('*') if p.is_file()}))
    write(DATA/'status.json',dict(phase='contact_diagnosis',training_started=False,
        fresh_holdout_started=False,protected_files=len(prior),physical_criteria_unchanged=True,
        next_gate='Validate repair, including current successful conditions, before PPO launch'))
    shutil.copy2(__file__,DATA/'prepare_source.py')
    print('PRESERVED',len(prior),'RESERVED',len(grids['holdout'])*2,'fresh trials')

if __name__ == '__main__':
    main()
