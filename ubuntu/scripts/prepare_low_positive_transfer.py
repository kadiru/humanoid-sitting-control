"""Reserve fresh tests and preserve completed evidence before new physics."""
import copy, hashlib, json, time, shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path('/home/kadir/humanoid-sitting')
OLD=Path('/data/humanoid-sitting-perception/positive_slope_01')
DATA=OLD.parent/'low_positive_transfer_01'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2))
assert not DATA.exists();DATA.mkdir()
prior={**read(OLD/'ppo_01/protocol.json')['prior_hashes'],**read(OLD/'ppo_01/protocol.json')['hashes']}
for p in OLD.rglob('*'):
    if p.is_file() and not p.is_symlink():prior[str(p)]=sha(p)
assert all(sha(p)==h for p,h in prior.items())
write(DATA/'prior_hashes.json',prior)
source=OLD.parent/'two_surface_curriculum_01/selected_path_01';g=read(source/'geometry.json')[0]
grids={'probe':[(.36,15),(.36,20),(.37,11.25),(.37,16.25),(.40,20),(.44,15)],
       'holdout':[(h,s) for h in [.365,.385,.405,.425] for s in [7.5,12.5,17.5]]}
for name,grid in grids.items():
    bank=DATA/'banks'/name;bank.mkdir(parents=True);rows=[]
    for h,s in grid:
        x=copy.deepcopy(g);R=Rotation.from_euler('y',s,degrees=True)
        x.update(case=len(rows),surface_id=len(rows),height=h,pitch_deg=s,top_center=[g['top_center'][0],g['top_center'][1],h],rotation_xyzw=R.as_quat().tolist(),normal=R.as_matrix()[:,2].tolist(),label=f'low_positive_{name}_{h:.3f}_{s:.2f}',split=name,known_anchor=False,reset_scope='standing only')
        rows.append(x)
    write(bank/'geometry.json',rows)
    for filename in ['candidates.npz','reset_candidates.npz']:
        a=np.load(source/filename);np.savez_compressed(bank/filename,**{k:np.repeat(a[k][:1],len(rows),axis=0) for k in a.files})
    write(bank/'selection.json',dict(source=str(source),indices=[0]*len(rows),standing_only=True))
for name,old in [('development_old','development'),('development_transfer','holdout')]:
    shutil.copytree(OLD/'banks'/old,DATA/'banks'/name)
write(DATA/'reservation.json',dict(timestamp=time.time(),holdout=grids['holdout'],holdout_triples=12,holdout_trials=24,prior_holdout_now_development=True,holdout_excluded_from_tuning_calibration_training_selection=True,hashes={str(p):sha(p) for p in (DATA/'banks').rglob('*') if p.is_file()}))
for name,cfg in [('phase_only',dict(low_target_x=-.18,low_target_pitch=.25,low_gap=.001)),('phase_deeper',dict(low_target_x=-.30,low_target_pitch=.10,low_gap=.004))]:
    write(DATA/(name+'.json'),dict(**cfg,travel_end=.45,landing_start=.55,landing_end=1.,load_start=.60,load_end=1.))
write(DATA/'status.json',dict(phase='physical_repair_probes',main_training_started=False,holdout_started=False,prior_hashes=len(prior)))
shutil.copy2(__file__,DATA/'prepare_source.py');print('Reserved and preserved',len(prior))
