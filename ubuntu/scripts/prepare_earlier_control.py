"""Protect prior studies and reserve wider geometry before new simulations."""
import copy,time,shutil
import numpy as np
from scipy.spatial.transform import Rotation
from earlier_control_common import *
assert not DATA.exists();DATA.mkdir()
p=read(OLD/'ppo_01/protocol.json');prior={**p['prior_hashes'],**p['hashes']}
for f in OLD.rglob('*'):
    if f.is_file():prior[str(f)]=sha(f)
check(prior);write(DATA/'prior_hashes.json',prior)
source=DATA.parent/'two_surface_curriculum_01/selected_path_01';g=read(source/'geometry.json')[0]
grid=[(h,s) for h in [.36,.38,.40,.42,.44] for s in [-10.,10.]]+[(h,s) for h in [.37,.39,.43] for s in [-7.5,7.5]]
bank=DATA/'wide_bank_01';bank.mkdir();items=[]
for height,slope in grid:
    x=copy.deepcopy(g);rot=Rotation.from_euler('y',slope,degrees=True)
    x.update(case=len(items),surface_id=len(items),height=height,pitch_deg=slope,top_center=[g['top_center'][0],g['top_center'][1],height],rotation_xyzw=rot.as_quat().tolist(),normal=rot.as_matrix()[:,2].tolist(),label=f'earlier_wide_h{height:.3f}_s{slope:+.1f}',split='reserved_wide_holdout',known_anchor=False,reset_scope='Original standing reset only')
    items.append(x)
write(bank/'geometry.json',items)
for name in ['candidates.npz','reset_candidates.npz']:
    b=np.load(source/name);np.savez_compressed(bank/name,**{k:np.repeat(b[k][0:1],len(items),axis=0) for k in b.files})
write(bank/'selection.json',dict(source=str(source),indices=[0]*len(items),standing_only=True))
write(DATA/'reservation.json',dict(timestamp=time.time(),grid=grid,geometries=16,trials=32,withheld_from_tuning_learning_calibration_selection=True,hashes={str(f):sha(f) for f in bank.iterdir()}))
write(DATA/'status.json',dict(phase='preflight',prior_hashes=len(prior)))
shutil.copy2(__file__,DATA/'prepare_source.py')
print('Protected',len(prior),'prior files; reserved16 wider surfaces/32 placements')
