"""Offline candidate construction; physics validation determines admissibility."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from humanoid_sitting.shared_frame import BodyModel,apply

p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--refine',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=Path(a.output);out.mkdir(exist_ok=False)
source=Path('/data/humanoid-sitting-perception/supported_surfaces_01/transition_banks_03/train')
old=np.load(source/'reset_candidates.npz');goals=np.load(source/'candidates.npz');geometry=json.loads((source/'geometry.json').read_text())
model=BodyModel(root/'runs/g1_kinematics_geometry_04/model.json')
phases=np.array([0.,.20,.40,.60,.80,.90,1.]);items=[];states=[]
for surface,index in enumerate([3,7]):
 start={k:old[k][index,0].copy() for k in ['q','qd','command','root_pose','root_velocity']}
 end={k:goals[k][index].copy() for k in start}
 R0=Rotation.from_quat(start['root_pose'][3:]).as_matrix();fk=model.fk(start['q'])
 foot_target=np.mean([R0@fk[s+'_ankle_roll_link'][:3,3]+start['root_pose'][:3] for s in ['left','right']],axis=0)
 for stage,phase in enumerate(phases):
  corrections=[0.] if stage in [0,6] else [-.10,-.05,0.,.05,.10,.15,.20,.25,.30,.35,.40]
  if a.refine:
   ranges={(0,1):(0.,.05),(0,3):(.045,.095),(1,1):(-.13,.04),(1,4):(-.09,-.02),(1,6):(-.10,-.025)}
   if (surface,stage) not in ranges:continue
   corrections=np.linspace(*ranges[(surface,stage)],21).tolist()
  for correction in corrections:
   q=(1-phase)*start['q']+phase*end['q']
   R=Slerp([0,1],Rotation.from_quat([start['root_pose'][3:],end['root_pose'][3:]]))(phase).as_matrix()
   R=Rotation.from_euler('y',correction).as_matrix()@R
   command_offset=(1-phase)*(start['command']-start['q'])+phase*(end['command']-end['q'])
   if stage in [0,6] and correction==0:state={k:v.copy() for k,v in (start if stage==0 else end).items()}
   else:
    fk=model.fk(q)
    for side in ['left','right']:
     pitch=Rotation.from_matrix(R@fk[side+'_ankle_roll_link'][:3,:3]).as_euler('xyz')[1]
     q[model.names.index(side+'_ankle_pitch_joint')]-=pitch
    fk=model.fk(q);feet=np.mean([R@fk[s+'_ankle_roll_link'][:3,3] for s in ['left','right']],axis=0)
    soles=np.concatenate([apply(fk[s+'_ankle_roll_link'],model.vertices[s+'_ankle_roll_link'])@R.T for s in ['left','right']])
    position=np.r_[foot_target[:2]-feet[:2],.001-soles[:,2].min()]
    state=dict(q=q,qd=np.zeros(37),command=q+command_offset,root_pose=np.r_[position,Rotation.from_matrix(R).as_quat()],root_velocity=np.zeros(6))
   items.append(dict(geometry[index],case=len(items),surface_id=surface,source_case=index,stage=stage,phase=float(phase),pitch_correction_rad=correction))
   states.append(state)
np.savez_compressed(out/'candidates.npz',**{k:np.array([s[k] for s in states],np.float32) for k in states[0]})
(out/'geometry.json').write_text(json.dumps(items,indent=2));shutil.copy2(__file__,out/'prepare_source.py')
print(json.dumps(dict(candidates=len(items),phases=phases.tolist(),surfaces=[geometry[i]['label'] for i in [3,7]])))
