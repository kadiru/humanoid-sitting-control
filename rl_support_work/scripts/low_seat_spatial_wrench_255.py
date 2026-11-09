"""OFFLINE six-joint quasi-static foot-wrench observability; never installed.

Inputs: measured encoders, IMU gravity, approximate torque telemetry and frozen
kinematic/inertial model. Output is not a measured contact wrench. Dynamics,
other leg contacts, model error and implicit-PD telemetry bias remain unmodeled.
"""
import numpy as np
from humanoid_sitting.shared_frame import level_rotation,apply
class SpatialWrench:
 def __init__(self,model,inertial):
  self.model=model;self.weight=sum(inertial['mass'])*9.81
  self.mass={b:(float(m),np.asarray(c)) for b,m,c in zip(inertial['body_names'],inertial['mass'],inertial['local_com'])}
  self.joints={j['name']:j for j in model.joints};self.parents={j['body1'][0]:j['body0'][0] for j in model.joints if j['body0']}
  self.names={s:[s+'_'+j+'_joint' for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']] for s in ['left','right']}
  self.desc={}
  for names in self.names.values():
   for name in names:
    child=self.joints[name]['body1'][0];bs=[]
    for b in self.mass:
     cur=b
     while cur!='pelvis':
      if cur==child:bs.append(b);break
      cur=self.parents[cur]
    self.desc[name]=bs
 def matrices(self,q,gravity):
  fk=self.model.fk(q);R=level_rotation(gravity);out=[]
  for side in ['left','right']:
   foot=R@fk[side+'_ankle_roll_link'][:3,3];A=[];g=[]
   for name in self.names[side]:
    j=self.joints[name];frame=fk[j['body0'][0]]@self.model.frames[name][0]
    origin=R@frame[:3,3];axis=R@frame[:3,'XYZ'.index(j['axis'])]
    A.append(np.r_[np.cross(axis,foot-origin),axis])
    g.append(sum(self.mass[b][0]*9.81*np.cross(axis,R@apply(fk[b],self.mass[b][1])-origin)[2] for b in self.desc[name]))
   out.append((np.array(A),np.array(g)))
  return out
 def estimate(self,q,torque,gravity):
  out=[]
  for side,(A,g) in zip(['left','right'],self.matrices(q,gravity)):
   ids=[self.model.names.index(n) for n in self.names[side]];rhs=g-np.asarray(torque)[ids]
   # No regularization or silent rank fallback. Moment scaling is reported,
   # not a new physical or controller gate. Rank threshold is NumPy default.
   rank=int(np.linalg.matrix_rank(A));inv=np.linalg.inv(A) if rank==6 else None
   out.append(dict(A=A,gravity_torque=g,rhs=rhs,rank=rank,condition=np.linalg.cond(A),condition_0p1m=np.linalg.cond(A@np.diag([1,1,1,.1,.1,.1])),inverse=inv,wrench=None if inv is None else np.linalg.solve(A,rhs)))
  return out
