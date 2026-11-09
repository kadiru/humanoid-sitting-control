"""23-joint, floating-base quasi-static rest construction.

Fixed feet + one material support point; exact force/moment allocation; full
23-joint torque proxy. No torque-capacity weighting masquerades as total effort.
Current sensors/static models are construction inputs. Actual mesh is audit-only.
"""
import time
from collections import OrderedDict
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from humanoid_sitting.shared_frame import apply,transform,level_rotation

class FullRest:
    def __init__(self,model,inertia,shapes,q,gravity,previous,normal,offset,patch,body,marker,target,task):
        self.model=model;self.inertia=inertia;self.q0=np.asarray(q,float);self.previous=np.asarray(previous,float)
        self.R0=level_rotation(np.asarray(gravity,float).copy());self.root0=transform(self.R0)
        self.n=np.asarray(normal,float);self.n/=np.linalg.norm(self.n);self.d=float(offset)
        self.body=body;self.marker=np.asarray(marker,float);self.target=np.asarray(target,float);self.task=task
        self.mesh={b:np.concatenate([s['vertices'] for s in shapes if s['body']==b]) for b in {s['body'] for s in shapes}}
        self.names=inertia['body_names'];self.mass=np.asarray(inertia['mass']);self.W=self.mass.sum()*9.81
        self.joints={j['name']:j for j in model.joints};self.desc={}
        for name in model.names[:23]:
            b={self.joints[name]['body1'][0]}
            while True:
                new={j['body1'][0] for j in model.joints if j['body0'] and j['body0'][0] in b}
                if new<=b:break
                b|=new
            self.desc[name]=b
        self.mass_membership=np.array([[b in self.desc[n] for b in self.names] for n in model.names[:23]])
        self.K=np.array([160 if 'ankle' in n else 150 if 'hip_roll' in n or 'hip_yaw' in n else 200 if 'hip_pitch' in n or 'knee' in n or n=='torso_joint' else 40 for n in model.names[:23]],float)
        limits=np.array([np.deg2rad(self.joints[n]['limits']) for n in model.names[:23]])
        self.lower=limits[:,0];self.upper=limits[:,1]
        fk=model.fk(self.q0);self.feet=[self.root0@fk[s+'_ankle_roll_link'] for s in ('left','right')]
        self.foot_bodies=['left_ankle_roll_link','right_ankle_roll_link']
        soles=[apply(t,self.mesh[b]) for t,b in zip(self.feet,self.foot_bodies)]
        self.floor=min(v[:,2].min() for v in soles)
        soles=[v[v[:,2]<v[:,2].min()+.004] for v in soles]
        self.foot_y=np.array([v[:,1].mean() for v in soles])
        self.foot_polygons=[ConvexHull(v[:,:2]).equations for v in soles]
        self.e=np.array([1.,0,0])-self.n*self.n[0]
        if np.linalg.norm(self.e)<.1:self.e=np.array([0.,1.,0.])-self.n*self.n[1]
        self.e/=np.linalg.norm(self.e);self.b=np.cross(self.n,self.e)
        self.patch=ConvexHull(np.c_[patch@self.e,patch@self.b]).equations
        self.x0=np.r_[self.q0[:23],np.zeros(6)];self.key=None;self.cache=None;self.states=OrderedDict();self.calls=0
        s=self.state(self.x0,0.);self.initial_bias=self.previous[:23]-self.q0[:23]-s['tau']/self.K
        self.standing_effort=max(float(s['tau']@s['tau']),1.)
        self.initial_point=s['point'].copy();self.initial_com=s['com'].copy()

    def pose(self,x):
        q=self.q0.copy();q[:23]=x[:23]
        root=transform(self.R0@Rotation.from_rotvec(x[26:29]).as_matrix(),x[23:26])
        return q,root,self.model.fk(q)

    def state(self,x,load):
        key=(np.asarray(x).tobytes(),float(load))
        if key in self.states:return self.states[key]
        q,root,fk=self.pose(x);point=apply(root@fk[self.body],self.marker)
        centers={b:apply(root@fk[b],c) for b,c in zip(self.names,self.inertia['local_com'])}
        com=np.average(list(centers.values()),weights=self.mass,axis=0)
        rest=self.W*load*self.n;ground=np.array([0.,0.,self.W])-rest
        moment=np.cross(com,np.array([0.,0.,self.W]))-np.cross(point,rest)
        cop=np.array([(self.floor*ground[0]-moment[1])/ground[2],(moment[0]+self.floor*ground[1])/ground[2],self.floor])
        yl,yr=self.foot_y
        fzl=(cop[1]-yr)*ground[2]/(yl-yr)
        fxl=(cop[0]*ground[1]-moment[2]-yr*ground[0])/(yl-yr)
        left=np.array([fxl,ground[1]*fzl/ground[2],fzl]);right=ground-left
        feet_force=[left,right];points=np.array([[cop[0],y,self.floor] for y in self.foot_y])
        contacts=[(b,p,F) for b,p,F in zip(self.foot_bodies,points,feet_force)]+[(self.body,point,rest)]
        axes=[];origins=[];membership=[]
        for name in self.model.names[:23]:
            j=self.joints[name];f=root@fk[j['body0'][0]]@self.model.frames[name][0];axis=f[:3,'XYZ'.index(j['axis'])];origin=f[:3,3];desc=self.desc[name]
            axes.append(axis);origins.append(origin);membership.append([b in desc for b,p,F in contacts])
        axes=np.asarray(axes);origins=np.asarray(origins);positions=np.array([centers[b] for b in self.names])
        gravity=np.sum(np.cross(axes[:,None,:],positions[None,:,:]-origins[:,None,:])[...,2]*self.mass[None,:]*9.81*self.mass_membership,axis=1)
        contact_positions=np.array([p for b,p,F in contacts]);contact_forces=np.array([F for b,p,F in contacts])
        external=np.sum(np.sum(np.cross(axes[:,None,:],contact_positions[None,:,:]-origins[:,None,:])*contact_forces[None,:,:],axis=2)*np.asarray(membership),axis=1)
        tau=gravity-external
        foot_error=[]
        for b,anchor in zip(self.foot_bodies,self.feet):
            f=root@fk[b];foot_error.extend(f[:3,3]-anchor[:3,3]);foot_error.extend(Rotation.from_matrix(anchor[:3,:3].T@f[:3,:3]).as_rotvec())
        ground_wrench=np.r_[left+right+rest-np.array([0.,0.,self.W]),sum(np.cross(p,F) for _,p,F in contacts)-np.cross(com,[0.,0.,self.W])]
        self.calls+=1
        value=dict(q=q,root=root,fk=fk,point=point,com=com,tau=np.array(tau),cop=cop,foot_forces=feet_force,foot_points=points,contacts=contacts,foot_error=np.asarray(foot_error),base_residual=ground_wrench)
        self.states[key]=value
        if len(self.states)>128:self.states.popitem(last=False)
        self.key=key;self.cache=value;return value

    def independent_work_audit(self,x,load):
        s=self.state(x,load);q=s['q'];root=s['root'];fk=s['fk']
        local=[(b,apply(np.linalg.inv(root@fk[b]),p),F) for b,p,F in s['contacts']]
        numeric=[]
        for j in range(23):
            terms=[]
            for sign in (-1,1):
                trial=q.copy();trial[j]+=sign*1e-5;f=self.model.fk(trial)
                U=sum(m*9.81*apply(root@f[b],c)[2] for b,m,c in zip(self.names,self.mass,self.inertia['local_com']))
                work=sum(F@apply(root@f[b],p) for b,p,F in local)
                terms.append(U-work)
            numeric.append((terms[1]-terms[0])/2e-5)
        error=float(np.max(abs(np.asarray(numeric)-s['tau'])))
        if error>2e-6 or max(abs(s['base_residual']))>1e-8:raise ValueError(('FULL_BODY_WORK_AUDIT',error,s['base_residual'].tolist()))
        return dict(passed=True,maximum_joint_torque_error_Nm=error,maximum_base_wrench_residual=float(max(abs(s['base_residual']))))

    def constraints(self,x,load,preload,previous_command,spent,phase):
        s=self.state(x,load);command=x[:23]+s['tau']/self.K+preload
        uv=np.array([s['point']@self.e,s['point']@self.b]);insets=-(self.patch[:,:2]@uv+self.patch[:,2])
        margins=[]
        for eq,pt,F in zip(self.foot_polygons,s['foot_points'],s['foot_forces']):
            margins.extend(-(eq[:,:2]@pt[:2]+eq[:,2])-.01)
            margins.extend([F[2]/self.W-.05,(.5*F[2]-np.linalg.norm(F[:2]))/self.W])
        up=s['root'][:3,2];margins.append(up[2]-np.cos(np.deg2rad(38)))
        # Geometry guide only. Exact expanded-mesh audit remains mandatory.
        clearance=[]
        allowed={'pelvis','left_hip_yaw_link','right_hip_yaw_link'} if self.task=='sitting' else {'torso_link','logo_link'}
        for body in ['pelvis','left_hip_pitch_link','right_hip_pitch_link','left_knee_link','right_knee_link','left_elbow_pitch_link','right_elbow_pitch_link']:
            if body in allowed or body not in self.mesh:continue
            p=apply(s['root']@s['fk'][body],self.mesh[body]);xy=np.c_[p@self.e,p@self.b]
            inside=np.all(xy@self.patch[:,:2].T+self.patch[:,2]<=0.,axis=1)
            clearance.append(float(np.min(p[inside]@self.n+self.d)-.006) if inside.any() else 1.)
        # Charge command arc and the mandatory per-phase time as constructed.
        # 12mm at5.5mm/s stays within the unchanged6mm/s ceiling.
        delta=float(np.max(abs(command-previous_command)))
        minimum=.003/.0055 if phase=='normal' else .25 if phase=='load' else .04
        cost=max(delta/.125,minimum)+getattr(self,'extra_quiet',0.)
        budget=9.7 if phase=='approach' else 13.4
        contact_insets=insets-.022 if phase!='approach' else np.ones(len(insets))
        return np.r_[margins,contact_insets,clearance,command-self.lower-.055,self.upper-.055-command,budget-spent-cost]

    def solve(self):
        deadline=time.monotonic()+45.;self.work_checks=[self.independent_work_audit(self.x0,0.)]
        lo=np.r_[self.lower+.031,[-.5,-.2,-.6],[-.65,-.65,-.35]];hi=np.r_[self.upper-.031,[.15,.2,.15],[.65,.65,.35]]
        for side in ('left','right'):lo[self.model.names.index(side+'_knee_joint')]=max(lo[self.model.names.index(side+'_knee_joint')],.04)
        x=self.x0.copy();nodes=[];spent=0.;prior=self.previous[:23].copy()
        hover=self.target+.012*self.n
        for phase,steps in [('approach',8),('normal',4),('load',4)]:
            for i in range(steps+1):
                if nodes and i==0:continue
                u=i/steps;load=(.20*u if self.task=='leaning' else .50*u) if phase=='load' else 0.
                self.extra_quiet=.5 if phase=='normal' and i==1 else 0.
                target=(1-u)*self.initial_point+u*hover if phase=='approach' else self.target+(1-u)*.012*self.n if phase=='normal' else self.target
                preload=(1-u)*self.initial_bias if phase=='approach' else np.zeros(23)
                def equal(y):
                    s=self.state(y,load);return np.r_[s['foot_error'],s['point']-target]
                def cost(y):
                    if time.monotonic()>deadline:raise TimeoutError('FULL_BODY_CONSTRUCTION_DEADLINE')
                    s=self.state(y,load);cmd=y[:23]+s['tau']/self.K+preload
                    ready=np.max(abs(cmd-self.previous[:23]))/(.15*14.4)
                    return float(s['tau']@s['tau']/self.standing_effort+ready*ready+.05*np.sum((y[:23]-self.q0[:23])**2))
                if not nodes:
                    accepted=x.copy()
                else:
                    fit=minimize(cost,x.copy(),method='SLSQP',bounds=list(zip(lo,hi)),constraints=[dict(type='eq',fun=equal),dict(type='ineq',fun=lambda y:self.constraints(y,load,preload,prior,spent,phase))],options={'maxiter':60,'ftol':1e-9})
                    if not fit.success or max(abs(equal(fit.x)))>.002 or min(self.constraints(fit.x,load,preload,prior,spent,phase))<-1e-6:
                        raise ValueError(dict(reason='FULL_BODY_NODE_REJECTED',phase=phase,node=i,message=str(fit.message),maximum_equality=float(max(abs(equal(fit.x)))),minimum_inequality=float(min(self.constraints(fit.x,load,preload,prior,spent,phase)))))
                    accepted=fit.x
                x=accepted;s=self.state(x,load);command=x[:23]+s['tau']/self.K+preload
                if nodes:
                    dt=max(float(np.max(abs(command-prior))/.125),.003/.0055 if phase=='normal' else .25 if phase=='load' else .04)+self.extra_quiet
                    spent+=dt
                full=self.previous.copy();full[:23]=command
                nodes.append(dict(q=s['q'].tolist(),root=s['root'].tolist(),command=full.tolist(),phase=phase,seconds=spent,gap=float(s['point']@self.n+self.d),point=s['point'].tolist(),cop=s['cop'].tolist(),load=load,model_torque_Nm=s['tau'].tolist(),model_effort_ratio=float(s['tau']@s['tau']/self.standing_effort)))
                prior=command
                self.provisional=dict(nodes=nodes,work_audits=self.work_checks,state_evaluations=self.calls)
        self.work_checks.append(self.independent_work_audit(x,load))
        if spent+1.>=14.4:raise ValueError(('FULL_BODY_HOLD_RESERVE',spent+1.))
        return dict(nodes=nodes,seconds=spent,settled_seconds=spent+1.,work_audits=self.work_checks,standing_torque_proxy_Nm2=self.standing_effort,return_model='reverse force allocation and contact path, then unloaded approach; not physically validated',physical_success=False)
