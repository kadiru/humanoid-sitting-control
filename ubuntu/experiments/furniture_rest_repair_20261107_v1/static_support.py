"""Full-joint gravity feedforward with balanced bilateral floor wrenches.
Uses static robot geometry/inertia and nominal pose only. No measured contact
claim, controller gain change, simulator truth, or velocity replacement.
"""
import numpy as np
from humanoid_sitting.shared_frame import apply

class StaticSupport:
    def __init__(self,model,inertia,mesh):
        self.model,self.inertia,self.mesh=model,inertia,mesh
        self.joints={j['name']:j for j in model.joints}
        self.desc={}
        for name in model.names[:23]:
            j=self.joints[name];bodies={j['body1'][0]}
            while True:
                new={k['body1'][0] for k in model.joints if k['body0'] and k['body0'][0] in bodies}
                if new<=bodies:break
                bodies|=new
            self.desc[name]=bodies
        self.k=np.array([160 if 'ankle' in n else 150 if 'hip_roll' in n or 'hip_yaw' in n else 200 if 'hip_pitch' in n or 'knee' in n or n=='torso_joint' else 40 for n in model.names[:23]],float)

    def evaluate(self,q,root):
        fk=self.model.fk(q);names=self.inertia['body_names'];mass=np.array(self.inertia['mass']);W=mass.sum()*9.81
        centers={b:apply(root@fk[b],c) for b,c in zip(names,self.inertia['local_com'])}
        com=np.average(list(centers.values()),weights=mass,axis=0)
        feet=[s+'_ankle_roll_link' for s in ('left','right')];sole=[]
        for body in feet:
            v=apply(root@fk[body],self.mesh[body]);sole.append(v[v[:,2]<v[:,2].min()+.004])
        y=np.array([v[:,1].mean() for v in sole]);f=W*np.array([(com[1]-y[1])/(y[0]-y[1]),(y[0]-com[1])/(y[0]-y[1])])
        points=np.array([[com[0],yi,float(v[:,2].min())] for yi,v in zip(y,sole)])
        margins=[min(points[i,0]-v[:,0].min(),v[:,0].max()-points[i,0],points[i,1]-v[:,1].min(),v[:,1].max()-points[i,1]) for i,v in enumerate(sole)]
        if min(f)<0 or min(margins)<.005:raise ValueError(('STATIC_FLOOR_SUPPORT_REJECTED',f.tolist(),margins))
        tau=[]
        for name in self.model.names[:23]:
            j=self.joints[name];frame=root@fk[j['body0'][0]]@self.model.frames[name][0]
            axis=frame[:3,'XYZ'.index(j['axis'])];origin=frame[:3,3];desc=self.desc[name]
            gravity=sum(m*9.81*np.cross(axis,centers[b]-origin)[2] for b,m in zip(names,mass) if b in desc)
            support=sum(fi*np.cross(axis,pt-origin)[2] for b,fi,pt in zip(feet,f,points) if b in desc)
            tau.append(gravity-support)
        torque=np.array(tau)
        balance=np.r_[np.array([0.,0.,f.sum()-W]),np.cross(points-com,np.c_[np.zeros(2),np.zeros(2),f]).sum(0)]
        return torque/self.k,dict(torque_Nm=torque.tolist(),com=com.tolist(),foot_normal_N=f.tolist(),foot_cop=points.tolist(),minimum_foot_inset_m=min(margins),base_wrench_residual=balance.tolist(),measured_force=False,measured_effort=False)

    def audit_virtual_work(self,q,root,report):
        # Separate finite-difference potential/work implementation; not actor input.
        fk=self.model.fk(q);feet=[s+'_ankle_roll_link' for s in ('left','right')]
        local=[apply(np.linalg.inv(root@fk[b]),p) for b,p in zip(feet,report['foot_cop'])]
        values=[]
        for i in range(23):
            e=[]
            for sign in (-1,1):
                trial=q.copy();trial[i]+=sign*1e-5;t=self.model.fk(trial)
                U=sum(m*9.81*apply(root@t[b],c)[2] for b,m,c in zip(self.inertia['body_names'],self.inertia['mass'],self.inertia['local_com']))
                work=sum(f*apply(root@t[b],p)[2] for b,f,p in zip(feet,report['foot_normal_N'],local))
                e.append(U-work)
            values.append((e[1]-e[0])/2e-5)
        err=float(np.max(abs(np.array(values)-report['torque_Nm'])))
        if err>2e-6 or np.max(np.abs(report['base_wrench_residual']))>1e-8:raise ValueError(('STATIC_INDEPENDENT_AUDIT',err,report['base_wrench_residual']))
        return dict(passed=True,maximum_analytic_vs_virtual_work_Nm=err,physics_success=False)
