"""Reserve positive-only holdout and preserve all earlier study evidence."""
import copy,time,shutil
import numpy as np
from scipy.spatial.transform import Rotation
from positive_slope_common import *

assert not DATA.exists();DATA.mkdir()
p=read(OLD/'ppo_01/protocol.json');prior={**p['prior_hashes'],**p['hashes']}
for f in OLD.rglob('*'):
    if f.is_file():prior[str(f)]=sha(f)
check(prior);write(DATA/'prior_hashes.json',prior)
source=DATA.parent/'two_surface_curriculum_01/selected_path_01';g=read(source/'geometry.json')[0]
grids={'development':[(h,s) for h in [.36,.40,.44] for s in [5.,10.,15.,20.]],
       'holdout':[(h,s) for h in [.37,.39,.41,.43] for s in [6.25,11.25,16.25]]}
for name,grid in grids.items():
    bank=DATA/'banks'/name;bank.mkdir(parents=True);items=[]
    for height,slope in grid:
        x=copy.deepcopy(g);rot=Rotation.from_euler('y',slope,degrees=True)
        x.update(case=len(items),surface_id=len(items),height=height,pitch_deg=slope,
                 top_center=[g['top_center'][0],g['top_center'][1],height],rotation_xyzw=rot.as_quat().tolist(),
                 normal=rot.as_matrix()[:,2].tolist(),label=f'positive_{name}_h{height:.3f}_s{slope:+.2f}',
                 split=name,known_anchor=False,reset_scope='Original standing reset only')
        items.append(x)
    write(bank/'geometry.json',items)
    for filename in ['candidates.npz','reset_candidates.npz']:
        b=np.load(source/filename);np.savez_compressed(bank/filename,**{k:np.repeat(b[k][0:1],len(items),axis=0) for k in b.files})
    write(bank/'selection.json',dict(source=str(source),indices=[0]*len(items),standing_only=True))
write(DATA/'reservation.json',dict(timestamp=time.time(),grids=grids,geometries_per_bank=12,trials_per_bank=24,
    slope_convention='Positive pitch descends from seat center toward initial robot feet (+x). Negative slopes excluded by user.',
    holdout_excluded_from_tuning_learning_calibration_selection=True,
    hashes={str(f):sha(f) for f in (DATA/'banks').rglob('*') if f.is_file()}))
write(DATA/'status.json',dict(phase='positive_slope_assessment',prior_hashes=len(prior),main_training_started=False))
shutil.copy2(__file__,DATA/'prepare_source.py')
print('Protected',len(prior),'files; reserved12 new positive holdout geometries before simulation.')
