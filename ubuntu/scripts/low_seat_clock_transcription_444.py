"""New direct clock-collocation mechanics, not an execution or optimizer.

The rejected440 records are the explicit research input. Each clock, half-clock
and original vertex has rootxy, descent f, orientation p, twelve leg angles and
eight PERSISTENT local-corner loads. Fixed fingers/arm rule are unchanged. The
fifteen local equations impose both declared foot frames, total vertical force
and two horizontal moments. No inverse gap root determines f: monotonic f and
actual gap-speed inequalities belong to the complete transcription together.

Unlike440, all states can move jointly when a future bounded solver is designed.
This file only prepares a differentiable local mechanics oracle. It does not
assemble/admit a complete NLP, run any optimizer, solve IK or repeat finite-work
audits. Fixed shared times retain the existing exact integer event allocation;
the progress of f and p against that clock is a decision, not fixed retiming of
440 commands. The prescribed feet still contain the inherited330 approach drift.
Canonical world points are fixed by those declared foot frames. Off-manifold
values are a numerical extension, never a physically admitted state.
"""
from pathlib import Path
from fractions import Fraction
import json, hashlib
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel

D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
FLOOR=16*float(np.finfo(np.float32).eps)
BUFFER=2.**-23
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()

def load_inputs():
 """Saved-array/input loading only; no FK, root or physical evaluation."""
 z=dict(np.load(D/'contact_lift_440/states.npz'))
 records=read(D/'contact_lift_440/records.json')
 m=BodyModel(D/'cooked_model_50/model.json')
 ine=read(D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json')
 ini=dict(np.load(D/'lattice_path_430/initial.npz'))
 old434=dict(np.load(D/'clock_lift_v2_434/lift.npz'))
 prep=read(D/'contact_lift_preparation_438/preparation.json')
 sides=['left','right'];legs=[m.names.index(s+'_'+j+'_joint') for s in sides for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']]
 arms=[m.names.index(s+'_'+j+'_joint') for s in sides for j in ['shoulder_pitch','elbow_pitch']]
 moving=legs+arms;fixed=[j for j in range(37) if j not in moving]
 q0=z['q'][0].copy();c0=z['command'][0].copy();tau0=z['torque'][0].copy()
 root0=z['root'][0,:3].copy();end_record=1351+105
 dz=z['root'][end_record,2]-root0[2]
 f=np.array([r['phase'] for r in records]);p=np.array([r['orientation_phase'] for r in records])
 R=Rotation.from_quat(z['root'][:,3:]).as_matrix()
 # This restores nominal orientation-phase freedom. Its reference is exactly
 # the saved440 rotation at each time, with one common physical rotation axis.
 rotvec=Rotation.from_matrix(R[0].T@R[end_record]).as_rotvec()
 baseR=R@Rotation.from_rotvec(-p[:,None]*rotvec).as_matrix()
 X=np.c_[z['root'][:,:2],f,p,z['q'][:,legs],z['weights']]
 assert X.shape==(1467,24)
 byname={j['name']:j for j in m.joints};parent={j['body1'][0]:j['body0'][0] for j in m.joints if j['body0']}
 def below(b,a):
  while True:
   if b==a:return True
   if b not in parent:return False
   b=parent[b]
 sub={name:{b for b in set(ine['body_names'])|set(m.vertices) if below(b,byname[name]['body1'][0])} for name in m.names}
 pending=m.joints.copy();known={'pelvis'};ordered=[]
 while pending:
  done=[j for j in pending if j['body0'] and j['body0'][0] in known]
  assert done
  for j in done:ordered.append(j);known.add(j['body1'][0])
  pending=[j for j in pending if j not in done]
 mass=np.array(ine['mass']);weight=mass.sum()*9.81
 kp=np.array([200,150,150,200,160,160]*2+[40]*4,dtype=float)
 n=old434['plane_normal'];plane_d=float(old434['plane_d'])
 limits={j['name']:np.deg2rad(j['limits']) for j in m.joints if j['type']=='PhysicsRevoluteJoint'}
 return locals()

class Mechanics:
 """Batch-independent 24-coordinate -> 51-output differentiable map.

 Outputs: feet12, momentxy2, totalnormal1, command16, torque16, gap1, COM3.
 The reference seat point is held in the same off-manifold extension as old371;
 its normal is zero at every nonfixed state. Fixed load states are not optimized.
 """
 def __init__(self,s,indices):
  torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
  self.s=s;self.indices=np.asarray(indices,dtype=int);self.N=len(indices)
  self.T=lambda v:torch.as_tensor(np.asarray(v),dtype=torch.float64)
  self.eye=torch.eye(3);self.baseR=self.T(s['baseR'][indices]);self.P=self.T(s['z']['points'][indices]);self.F=self.T(s['z']['foot_frames'][indices]);self.normal=self.T(s['z']['normal'][indices]);self.n=self.T(s['n'])
  rv=s['rotvec'];self.angle=float(np.linalg.norm(rv));a=rv/self.angle
  self.K=self.T([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
  self.frames={j['name']:(self.T(s['m'].frames[j['name']][0]),self.T(s['m'].frames[j['name']][1])) for j in s['ordered']}
  self.cross={}
  for j in s['ordered']:
   a=np.eye(3)['XYZ'.index(j.get('axis') or 'X')]
   self.cross[j['name']]=self.T([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
  # Recover the already-computed seat moment directly from saved equilibrium.
  # It is fixed because all nonzero seat-load records have fixed poses.
  sv=s['z']['normal'][indices]/s['n'][2]
  footmoment=np.sum(s['z']['weights'][indices,:,None]*s['z']['points'][indices,:,:2],axis=1)
  self.seatmoment=self.T(s['z']['equilibrium'][indices,1:]-footmoment+s['z']['com'][indices,:2])
  assert np.max(abs(self.seatmoment.numpy()[sv==0]))<1e-13
  self.sv=self.T(sv)
 def calc(self,x):
  s=self.s;N=self.N;T=self.T;f=x[:,2];p=x[:,3];w=x[:,16:24]
  q=T(s['q0']).repeat(N,1)
  q=q.index_copy(1,torch.as_tensor(s['legs']),x[:,4:16])
  q=q.index_copy(1,torch.as_tensor(s['arms']),T(s['q0'][s['arms']])[None,:]-.35*f[:,None])
  a=p*self.angle;R=self.baseR@(self.eye+torch.sin(a)[:,None,None]*self.K+(1-torch.cos(a))[:,None,None]*(self.K@self.K))
  root=torch.cat([x[:,:2],(s['root0'][2]+s['dz']*f)[:,None]],1)
  def homogeneous(rot,pos):
   top=torch.cat([rot,pos[:,:,None]],2);bottom=T([0.,0.,0.,1.]).expand(N,1,4)
   return torch.cat([top,bottom],1)
  poses={'pelvis':homogeneous(R,root)};Jworld={}
  for j in s['ordered']:
   name=j['name'];A,B=self.frames[name];JW=poses[j['body0'][0]]@A;Jworld[name]=JW
   theta=q[:,s['m'].names.index(name)] if name in s['m'].names else torch.zeros(N)
   K=self.cross[name];rot=self.eye+torch.sin(theta)[:,None,None]*K+(1-torch.cos(theta))[:,None,None]*(K@K)
   poses[j['body1'][0]]=JW@homogeneous(rot,torch.zeros((N,3)))@B
  pts={b:(poses[b][:,:3,:3]@T(cc)[:,None]).squeeze(-1)+poses[b][:,:3,3] for b,cc in zip(s['ine']['body_names'],s['ine']['local_com'])}
  com=sum(mm*pts[b] for b,mm in zip(s['ine']['body_names'],s['mass']))/s['mass'].sum()
  feet=[]
  for k,side in enumerate(s['sides']):
   F=poses[side+'_ankle_roll_link'];A=self.F[:,k,:3,:3].transpose(1,2)@F[:,:3,:3]
   feet.extend([F[:,:3,3]-self.F[:,k,:3,3],.1*torch.stack([A[:,2,1]-A[:,1,2],A[:,0,2]-A[:,2,0],A[:,1,0]-A[:,0,1]],1)])
  moment=(w[:,:,None]*self.P[:,:,:2]).sum(1)+self.seatmoment-com[:,:2]
  total=w.sum(1)+self.sv-1
  tau=[]
  for j in s['moving']:
   name=s['m'].names[j];JW=Jworld[name];axis=JW[:,:3,'XYZ'.index(s['byname'][name]['axis'])];o=JW[:,:3,3];M=torch.zeros((N,3))
   for b,mm in zip(s['ine']['body_names'],s['mass']):
    if b in s['sub'][name]:M=M+torch.linalg.cross(pts[b]-o,T([0,0,-mm*9.81]).expand(N,3))
   mask=T([float(side+'_ankle_roll_link' in s['sub'][name]) for side in ['left']*4+['right']*4])
   vertical=s['weight']*w*mask;force=torch.stack([torch.zeros_like(vertical),torch.zeros_like(vertical),vertical],2)
   M=M+torch.linalg.cross(self.P-o[:,None,:],force).sum(1)
   tau.append(-(axis*M).sum(1))
  tau=torch.stack(tau,1)
  cmd=T(s['c0'][s['moving']])+q[:,s['moving']]-T(s['q0'][s['moving']])+(tau-T(s['tau0']))/T(s['kp'])
  pelvis=(R@T(s['m'].vertices['pelvis']).T).transpose(1,2)+root[:,None,:]
  gap=(pelvis@self.n+s['plane_d']).min(1).values
  return torch.cat(feet+[moment,total[:,None],cmd,tau,gap[:,None],com],1)
 def values(self,x):return self.calc(self.T(x)).detach().numpy()
 def jacobian(self,x):
  x=self.T(x);columns=[]
  for j in range(24):
   d=torch.zeros_like(x);d[:,j]=1
   _,col=torch.func.jvp(self.calc,(x,),(d,));columns.append(col.detach().numpy())
  return np.stack(columns,2)

def graph(s):
 """Exact time labels stay intact. Equal times and the synthetic hold share DOF.

 This is a new transcription identification, not a claim that independent440
 Newton results were bitwise identical. Differences are explicitly reported.
 """
 records=s['records'];times=[Fraction(int(r['exact_time'][0]),int(r['exact_time'][1])) for r in records]
 hover=times[1351+99];release=times[1351+100]
 unique=sorted(set(times));lookup={t:i for i,t in enumerate(unique)}
 label_to_time=np.array([lookup[t] for t in times]);clock_time=label_to_time[:676]
 canonical=[];label_to_state=[];keys={}
 hover_rep=1351+99
 for i,t in enumerate(times):
  key=hover if hover<=t<=release else t
  if key not in keys:keys[key]=len(keys);canonical.append(hover_rep if key==hover else i)
  label_to_state.append(keys[key])
 label_to_state=np.array(label_to_state);canonical=np.array(canonical)
 expanded=s['X'][canonical][label_to_state]
 fixed=np.array([times[k]==0 or times[k]>=times[1351+105] for k in canonical])
 inactive=s['z']['weights'][canonical]==0
 return dict(times=times,unique=unique,label_to_time=label_to_time,clock_time=clock_time,canonical=canonical,label_to_state=label_to_state,fixed=fixed,inactive=inactive,alias_max_coordinate_difference=float(abs(expanded-s['X']).max()),hover=hover,release=release)
