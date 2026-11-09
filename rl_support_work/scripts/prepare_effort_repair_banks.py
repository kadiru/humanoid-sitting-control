"""Reserve a fresh holdout before the new small repair/effort comparison."""
import copy,hashlib,json,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

out=Path('/data/humanoid-sitting-perception/contact_effort_repair_01/banks')
out.mkdir(exist_ok=False)
previous=Path('/data/humanoid-sitting-perception/transfer_effort_01/banks/development')
shutil.copytree(previous,out/'development')
source=Path('/data/humanoid-sitting-perception/two_surface_curriculum_01/selected_path_01')
base=json.loads((source/'geometry.json').read_text())
grid=[(h,s) for h in (-.015,.015) for s in (-3.75,3.75)]+[(-.025,0.),(.025,0.),(0.,-6.25),(0.,6.25)]
folder=out/'holdout';folder.mkdir();items=[];indices=[]
for i,g in enumerate(base):
    for h,s in grid:
        x=copy.deepcopy(g);rotation=Rotation.from_euler('y',g['pitch_deg']+s,degrees=True)
        x.update(case=len(items),surface_id=len(items),source_motion_id=i,label=f"{g['kind']}_dh{h:+.3f}_ds{s:+.2f}",height=g['height']+h,pitch_deg=g['pitch_deg']+s,rotation_xyzw=rotation.as_quat().tolist(),normal=rotation.as_matrix()[:,2].tolist(),top_center=[g['top_center'][0],g['top_center'][1],g['top_center'][2]+h],height_delta_m=h,slope_delta_deg=s,split='holdout',known_anchor=False,reset_scope='Original standing only; no intermediate resets admitted on changed surfaces')
        for key in ('initial_forbidden_penetrations','nearest_bodies','source_validation','source_case_index'):x.pop(key,None)
        items.append(x);indices.append(i)
(folder/'geometry.json').write_text(json.dumps(items,indent=2))
for filename in ('candidates.npz','reset_candidates.npz'):
    data=np.load(source/filename)
    np.savez_compressed(folder/filename,**{k:data[k][indices] for k in data.files})
(folder/'selection.json').write_text(json.dumps(dict(source=str(source),indices=indices,scope='Fresh pre-training reserved heights/slopes; same standing states and reference library'),indent=2))
shutil.copy2(__file__,out/'prepare_source.py')
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
(out/'protocol.json').write_text(json.dumps(dict(holdout_grid=grid,withheld_from_training_calibration_selection=True,development_is_previous_training_grid=True,hashes=hashes),indent=2))
print('Reserved eight fresh geometries per task; no evaluation performed')
