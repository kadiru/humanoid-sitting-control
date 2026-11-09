"""Quasi-static sagittal foot wrench from encoders, torque, IMU and robot model.

No simulation state/contact APIs. The floating-base translation cancels in the
Jacobian; gravity alignment is supplied by the IMU. Dynamics and other contacts
on a leg can bias the estimate and must be checked separately.
"""
import numpy as np
from humanoid_sitting.shared_frame import level_rotation,apply

class FootWrenchObserver:
    def __init__(self,model,inertial):
        self.model=model;self.weight=sum(inertial['mass'])*9.81
        self.mass={b:(m,np.asarray(c)) for b,m,c in zip(inertial['body_names'],inertial['mass'],inertial['local_com'])}
        self.joints={j['name']:j for j in model.joints};self.parents={j['body1'][0]:j['body0'][0] for j in model.joints if j['body0']}
        self.descendants={}
        for name in [s+'_'+j+'_joint' for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']]:
            child=self.joints[name]['body1'][0];bodies=[]
            for b in self.mass:
                cur=b
                while cur!='pelvis':
                    if cur==child:bodies.append(b);break
                    cur=self.parents[cur]
            self.descendants[name]=bodies

    def estimate(self,q,torque,gravity,fk=None,return_jacobian=False):
        model=self.model;fk=model.fk(q) if fk is None else fk;R=level_rotation(gravity);result=[];conditions=[];jacobians=[]
        for side in ['left','right']:
            foot=R@fk[side+'_ankle_roll_link'][:3,3];J=[];rhs=[]
            for joint in ['hip_pitch','knee','ankle_pitch']:
                name=side+'_'+joint+'_joint';j=self.joints[name];frame=fk[j['body0'][0]]@model.frames[name][0]
                origin=R@frame[:3,3];axis=R@frame[:3,'XYZ'.index(j['axis'])];jac=np.cross(axis,foot-origin)
                g=sum(m*9.81*np.cross(axis,R@apply(fk[b],com)-origin)[2] for b,(m,com) in self.mass.items() if b in self.descendants[name])
                J.append([jac[0],jac[2],axis[1]]);rhs.append(g-torque[model.names.index(name)])
            J=np.asarray(J);jacobians.append(J);conditions.append(float(np.linalg.cond(J)));result.append(np.linalg.solve(J,np.asarray(rhs)) if conditions[-1]<200 else np.zeros(3))
        values=(np.asarray(result),np.asarray(conditions))
        return (*values,np.asarray(jacobians)) if return_jacobian else values
