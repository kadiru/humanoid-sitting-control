"""Select physically validated waypoints before any learning."""
from pathlib import Path
import json,numpy as np,shutil
r=Path('/data/humanoid-sitting-perception/two_surface_curriculum_01')
selection=[[(1,0),(2,11),(1,15),(2,32),(1,38),(1,48),(1,56)],
           [(1,57),(2,60),(1,70),(1,81),(2,78),(1,103),(2,100)]]
out=r/'selected_path_01';out.mkdir(exist_ok=False);poseout=r/'selected_pose_bank_01';poseout.mkdir(exist_ok=False)
rows=[];states=[];groups=[]
for surface,items in enumerate(selection):
 group=[]
 for stage,(version,index) in enumerate(items):
  p=r/f'candidate_validation_{version:02d}';result=json.loads((p/'results.json').read_text())['rows'][index]
  assert result['valid']
  if stage==6:assert result['supported']
  bank=np.load(p/'candidates.npz');state={k:bank[k][index] for k in bank.files};group.append(state)
  rows.append(dict(result['geometry'],case=len(rows),surface_id=surface,stage=stage,source_validation=str(p),source_case_index=index))
  states.append(state)
 groups.append(group)
keys=list(states[0]);geometry=[dict(rows[i*7+6],case=i) for i in range(2)]
(out/'geometry.json').write_text(json.dumps(geometry,indent=2))
np.savez_compressed(out/'candidates.npz',**{k:np.array([g[-1][k] for g in groups]) for k in keys})
np.savez_compressed(out/'reset_candidates.npz',**{k:np.array([[s[k] for s in g] for g in groups]) for k in keys},validated=np.ones((2,7),bool))
(out/'selection.json').write_text(json.dumps(dict(selection=selection,stage_count=7,scope='static20s validation before training; transition feasibility not implied',source_rows=rows),indent=2))
(poseout/'geometry.json').write_text(json.dumps(rows,indent=2))
np.savez_compressed(poseout/'candidates.npz',**{k:np.array([s[k] for s in states]) for k in keys})
shutil.copy2(__file__,out/'selection_source.py')
print('Selected14 validated waypoints; both endpoints supported')
