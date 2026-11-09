"""Freeze goal selection and foot-aligned reset curriculum before transition PPO."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from humanoid_sitting.shared_frame import BodyModel,apply
from humanoid_sitting.supported_surface import surface_frame
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(exist_ok=False)
m=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json');meta=json.loads((ROOT/'runs/g1_kinematics_geometry_04/model.json').read_text())
r=Path('/data/humanoid-sitting-perception/supported_surfaces_01');entries=[]
def append(path,index,geometry=None,split='train'):
 b=np.load(path/'candidates.npz');g=geometry or json.loads((path/'geometry.json').read_text())[index]
 entries.append((dict(g,source=str(path),source_index=index,split=split),{k:b[k][index].copy() for k in ['q','qd','command','root_pose','root_velocity']}))
old=Path('/data/humanoid-sitting-perception/resting_seated_capture_dev02')
for i,h in enumerate([.25,.30,.35,.40]):
 g=dict(case=i,label=f'level_{h}',kind='sit',family='slab',height=h,x=-.4,y=0.,yaw=0.,pitch_deg=0.,depth=.5,width=.9,top_center=[-.4,0.,h],normal=[0.,0.,1.],rotation_xyzw=[0.,0.,0.,1.])
 append(old,i*4,g)
for i in [18,28,34]:append(r/'validation_01',i)
for i in [8,9,20,21,32,33]:append(r/'validation_04',i)
for i in [22,26,32,36]:append(r/'validation_02',i,split='holdout')
for i in range(3):
 g0,b0=entries[i];g1,b1=entries[i+1];g=dict(g0);g['height']=(g0['height']+g1['height'])/2;g['top_center']=[-.4,0.,g['height']]
 g.update(split='holdout',label=f'unseen_level_{g["height"]}',source='interpolation of adjacent captured level goals')
 b={k:(b0[k]+b1[k])/2 for k in b0};b['root_pose'][3:]=Slerp([0,1],Rotation.from_quat([b0['root_pose'][3:],b1['root_pose'][3:]]))(.5).as_quat();entries.append((g,b))
for source in [0,1,2,8,10,12]:
 g,b=entries[source];g=dict(g);g.update(split='holdout',shape='chamfered' if source%2 else 'ellipse',width=.75,depth=g['depth']+.04,label=g['label']+'_unseen_outline',source_goal=source)
 entries.append((g,{k:v.copy() for k,v in b.items()}))
for i in [4,8,16]:append(r/'steep_validation_06',i,split='holdout')
standing=np.load(r/'standing_preload_validation_03/candidates.npz');standing_index=7
stand={k:standing[k][standing_index].copy() for k in ['q','qd','command','root_pose','root_velocity']}
for split in ['train','holdout']:
 sub=out/split;sub.mkdir();items=[(g,b) for g,b in entries if g['split']==split];geometry=[];bank={k:[] for k in ['q','qd','command','root_pose','root_velocity']};resets=[]
 for index,(g,b) in enumerate(items):
  g=dict(g,case=index);geometry.append(g)
  for k in bank:bank[k].append(b[k])
  endR=Rotation.from_quat(b['root_pose'][3:]).as_matrix();fk=m.fk(b['q'])
  endfeet=np.array([endR@fk[s+'_ankle_roll_link'][:3,3]+b['root_pose'][:3] for s in ['left','right']]).mean(0)
  nominal=stand['q'].copy()
  states=[]
  for blend in [0.,.25,.5,.75,.9,1.]:
   if blend==1.:states.append({k:v.copy() for k,v in b.items()});continue
   q=(1-blend)*nominal+blend*b['q'];R=Slerp([0,1],Rotation.from_quat([stand['root_pose'][3:],b['root_pose'][3:]]))(blend).as_matrix()
   if blend==0.:
    state={k:v.copy() for k,v in stand.items()};fk=m.fk(q);feet=np.array([R@fk[side+'_ankle_roll_link'][:3,3]+state['root_pose'][:3] for side in ['left','right']]).mean(0)
    state['root_pose'][:2]+=endfeet[:2]-feet[:2];states.append(state);continue
   # Flat feet and fixed average foot location are reset construction only.
   fk=m.fk(q)
   for side in ['left','right']:
    angle=Rotation.from_matrix(R@fk[side+'_ankle_roll_link'][:3,:3]).as_euler('xyz')[1]
    q[m.names.index(side+'_ankle_pitch_joint')]-=angle
   fk=m.fk(q);feet=np.array([R@fk[s+'_ankle_roll_link'][:3,3] for s in ['left','right']]).mean(0)
   soles=np.concatenate([apply(fk[s+'_ankle_roll_link'],m.vertices[s+'_ankle_roll_link'])@R.T for s in ['left','right']])
   pos=np.array([endfeet[0]-feet[0],endfeet[1]-feet[1],.002-soles[:,2].min()])
   states.append(dict(q=q,qd=np.zeros(37),command=q+(1-blend)*(stand['command']-stand['q'])+blend*(b['command']-b['q']),root_pose=np.r_[pos,Rotation.from_matrix(R).as_quat()],root_velocity=np.zeros(6)))
  resets.append(states)
 np.savez_compressed(sub/'candidates.npz',**{k:np.array(v,np.float32) for k,v in bank.items()})
 np.savez_compressed(sub/'reset_candidates.npz',**{k:np.array([[state[k] for state in row] for row in resets],np.float32) for k in bank})
 (sub/'geometry.json').write_text(json.dumps(geometry,indent=2))
 print(split,len(items))
shutil.copy2(__file__,out/'prepare_source.py')
(out/'selection.json').write_text(json.dumps(dict(training_count=sum(g['split']=='train' for g,b in entries),heldout_count=sum(g['split']=='holdout' for g,b in entries),reset_blends=[0,.25,.5,.75,.9,1],selection_before_training=True,notes='Rigid finite surfaces. Elliptical/chamfered outlines only in heldout. Physical validation still required.'),indent=2))
