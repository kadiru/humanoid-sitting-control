"""Reserve geometry before optimizer smoke, training and policy selection."""
import copy,json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

r=Path('/data/humanoid-sitting-perception/crouch_rest_bridge_01');out=r/'reserved_bank_01';out.mkdir(exist_ok=False)
source=Path('/data/humanoid-sitting-perception/two_surface_curriculum_01/selected_path_01');geoms=json.loads((source/'geometry.json').read_text());index=next(i for i,g in enumerate(geoms) if g['kind']=='sit');base=geoms[index]
grid=[(.3875,-2.),(.3875,2.),(.4025,-2.),(.4025,2.),(.4175,-2.),(.4175,2.),(.4075,-4.),(.4075,4.)]
rows=[]
for i,(height,slope) in enumerate(grid):
    g=copy.deepcopy(base);R=Rotation.from_euler('y',slope,degrees=True)
    g.update(case=i,surface_id=i,label=f'bridge_hold_h{height:.4f}_s{slope:+.1f}',height=height,pitch_deg=slope,top_center=[-.4,0.,height],rotation_xyzw=R.as_quat().tolist(),normal=R.as_matrix()[:,2].tolist(),height_delta_m=height-base['height'],slope_delta_deg=slope-base['pitch_deg'],split='reserved_holdout',known_anchor=False,reset_scope='Original standing reset only; intermediate/end source states are reference commands, not admitted resets')
    for k in ['initial_forbidden_penetrations','nearest_bodies','source_validation','source_case_index']:g.pop(k,None)
    rows.append(g)
(out/'geometry.json').write_text(json.dumps(rows,indent=2))
for name in ['candidates.npz','reset_candidates.npz']:
    a=np.load(source/name);np.savez_compressed(out/name,**{k:a[k][[index]*len(rows)] for k in a.files})
(out/'selection.json').write_text(json.dumps(dict(source=str(source),indices=[index]*len(rows),standing_only=True),indent=2))
paths=[p for p in out.iterdir() if p.is_file()]
(r/'holdout_reservation_01.json').write_text(json.dumps(dict(timestamp=time.time(),grid=grid,geometries=8,trials=16,withheld_from_controller_tuning_learning_calibration_and_selection=True,scope='Withheld from this new pilot; not a claim of never being used anywhere in project history',source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()},bank_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2))
print(out)
