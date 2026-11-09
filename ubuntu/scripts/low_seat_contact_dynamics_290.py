"""Offline floating-base inertia and frozen bilateral-contact drive sensitivity.

No runtime/controller, physics step, fitted force or velocity substitution.
"""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,level_rotation,apply

def skew(v):
 x,y,z=v;return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])

class ContactDynamics:
 def __init__(self,geometry,inertia,actuator):
  self.m=BodyModel(geometry);self.inertia=inertia;self.actuator=actuator;self.n=43
  parents={j['body1'][0]:j for j in self.m.joints};self.ancestors={}
  for name in inertia['body_names']:
   chain=[];cur=name
   while cur!='pelvis':
    j=parents[cur]
    if j['name'] in self.m.names:chain.append(self.m.names.index(j['name']))
    cur=j['body0'][0]
   self.ancestors[name]=chain
  self.joints={j['name']:j for j in self.m.joints}
  self.K=np.diag(np.r_[np.zeros(6),actuator['stiffness'][0]])
  self.B=np.diag(np.r_[np.zeros(6),actuator['damping'][0]])
  self.arm=np.r_[np.zeros(6),actuator['armature'][0]]
  self.roll=[self.m.names.index(s+'_ankle_roll_joint') for s in ['left','right']]
 def matrices(self,q,g):
  R=level_rotation(g);fk=self.m.fk(q);axes=[];origins=[]
  for name in self.m.names:
   j=self.joints[name];T=fk[j['body0'][0]]@self.m.frames[name][0]
   axes.append(R@T[:3,'XYZ'.index(j['axis'])]);origins.append(R@T[:3,3])
  axes=np.array(axes);origins=np.array(origins)
  def jac(name,p):
   Jv=np.zeros((3,43));Jw=np.zeros((3,43));Jv[:,:3]=np.eye(3);Jv[:,3:6]=-skew(p);Jw[:,3:6]=np.eye(3)
   for j in self.ancestors[name]:Jv[:,j+6]=np.cross(axes[j],p-origins[j]);Jw[:,j+6]=axes[j]
   return Jv,Jw
  M=np.diag(self.arm);body=[]
  for b,name in enumerate(self.inertia['body_names']):
   p=R@apply(fk[name],self.inertia['local_com'][b]);Rb=R@fk[name][:3,:3];I=Rb@np.array(self.inertia['inertia_body'][b])@Rb.T;Jv,Jw=jac(name,p)
   M+=self.inertia['mass'][b]*(Jv.T@Jv)+Jw.T@I@Jw;body.append((Jv,Jw,I))
  C=[]
  for s in ['left','right']:
   name=s+'_ankle_roll_link';v=self.m.vertices[name];sole=v[abs(v[:,2]-v[:,2].min())<1e-10].mean(0);p=R@apply(fk[name],sole);Jv,Jw=jac(name,p);C.extend([Jv,Jw])
  return M,np.concatenate(C),body
 def response(self,M,C):
  h=.005;E=M+h*self.B+h*h*self.K;inv=np.linalg.solve(E,np.eye(43));W=inv@C.T;G=C@W;L=inv-W@np.linalg.solve(G,C@inv)
  U=np.zeros((43,2));U[np.array(self.roll)+6,np.arange(2)]=1
  x=np.zeros((43,2));v=x.copy();xf=x.copy();vf=x.copy()
  for _ in range(4):
   v=L@(M@v+h*self.K@(U-x));x=x+h*v
   vf=inv@(M@vf+h*self.K@(U-xf));xf=xf+h*vf
  return x,v,xf,vf,L,G
