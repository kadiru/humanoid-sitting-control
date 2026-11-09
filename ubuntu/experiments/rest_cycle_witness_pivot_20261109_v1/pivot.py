"""Retarget one accepted contact witness; unload by pivoting before lifting.

The contact point is one fixed pelvis collider vertex. No failed endpoint or
time-indexed command trajectory is imported.12leg IK preserves full foot poses.
"""
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from scipy.optimize import least_squares
from humanoid_sitting.shared_frame import apply,transform
from static_model import FullRest

class Mechanics(FullRest):
    support_bodies=('pelvis','pelvis')
    def __init__(self,*a):
        self.support_normals=np.tile(np.asarray(a[6]),(2,1))
        super().__init__(*a)
    def support_points(self,root,fk):return np.tile(apply(root,self.marker),(2,1))

def schedule(nodes,lower,upper):
    """Exact original inverse filter, including endpoint directional headroom."""
    current=np.asarray(nodes[0]['command']).copy();commands=[];desired=[];phase=[]
    for a,b in zip(nodes,nodes[1:]):
        target=np.asarray(b['command']);delta=target-current
        head=np.where(delta[:23]>=0,upper-.03-target[:23],target[:23]-lower-.03)
        if np.min(head)<0:raise ValueError('ENDPOINT_COMMAND_OUTSIDE_ORIGINAL_MARGIN')
        # For a linear segment, desired_last=target+9*delta/N. Check all ticks
        # independently afterwards. This retains the original.03rad margin.
        n=max(1,int(np.ceil(np.max(abs(delta))/.0025)),int(np.ceil(np.max(9*abs(delta[:23])/np.maximum(head,1e-12)))))
        if a.get('normal') or b.get('normal'):
            n=max(n,int(np.ceil(abs(b['gap']-a['gap'])/.0055/.02)))
        if n>1000 or len(commands)+n>1000:raise ValueError('FINITE20S_COMMAND_RETURN_CAP')
        for j in range(1,n+1):
            cmd=current+delta*j/n;previous=current+delta*(j-1)/n
            desired.append((previous[:23]+10*(cmd[:23]-previous[:23])).tolist())
            commands.append(cmd.tolist());phase.append(b['phase'])
        current=target
    return dict(commands=commands,desired=desired,phase=phase,seconds=len(commands)*.02)

class Pivot:
    def __init__(self,model,inertia,shapes,frame,previous,source,front):
        from humanoid_sitting.shared_frame import level_rotation
        self.model=model;self.q0=frame['q'].astype(float);self.previous=np.asarray(previous,float)
        self.R0=level_rotation(frame['gravity'].astype(float).copy());g=frame['geometry']
        self.n=self.R0@g[2:5];self.n/=np.linalg.norm(self.n);self.d=float(g[5]);self.cloud=frame['selected_points']@self.R0.T
        assert g[11] and g[12]<=.21 and self.n[2]>.97
        vertices=np.concatenate([s['vertices'] for s in shapes if s['body']=='pelvis'])
        sr=np.asarray(source['root']);sn=np.asarray(source['normal'])
        heights=apply(sr,vertices)@sn
        self.marker=vertices[int(np.argmin(heights))].copy()
        se=np.array([1.,0,0])-sn*sn[0];se/=np.linalg.norm(se)
        source_point=apply(sr,self.marker)
        source_inset=float(source['front_projection']-source_point@se)
        self.inset=max(.022,source_inset)
        e=np.array([1.,0,0])-self.n*self.n[0];e/=np.linalg.norm(e)
        # Sourceplane->currentplane rigid orientation registration. Same physical
        # material marker and front-edge inset, not a new arbitrary contact ray.
        align=Rotation.align_vectors(self.n[None],sn[None])[0].as_matrix()
        self.endR=align@sr[:3,:3]
        self.target=np.array([float(front)-self.inset,0.,0.]);self.target[2]=-(self.d+self.n[0]*self.target[0])/self.n[2]
        self.restq=np.asarray(source['q'],float).copy();self.restq[23:]=self.q0[23:]
        self.mechanics=Mechanics(model,inertia,shapes,self.q0,frame['gravity'],self.previous,self.n,self.d,self.cloud,'pelvis',self.marker,self.target,'sitting')
        self.legs=[model.names.index(s+'_'+j+'_joint') for s in ('left','right') for j in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
        self.lo=self.mechanics.lower[self.legs]+.031;self.hi=self.mechanics.upper[self.legs]-.031
        self.copx=float(np.mean([self.mechanics.feet[i][0,3] for i in range(2)]))
        self.initial=self.mechanics.state(self.mechanics.x0,0.)
        self.evidence=dict(association=dict(body='pelvis',local_vertex=self.marker.tolist(),source_inset_m=source_inset,target_inset_m=self.inset,target_point=self.target.tolist()),stages=[])
    def x(self,q,root):
        x=np.zeros(30);x[:23]=q[:23];x[23:26]=root[:3,3];x[26:29]=Rotation.from_matrix(self.R0.T@root[:3,:3]).as_rotvec();return x
    def state(self,q,root,load=0.):return self.mechanics.state(self.x(q,root),load)
    def feet_residual(self,q,root):
        fk=self.model.fk(q);rows=[]
        for body,anchor in zip(self.mechanics.foot_bodies,self.mechanics.feet):
            f=root@fk[body];rows.extend(f[:3,3]-anchor[:3,3]);rows.extend(.2*Rotation.from_matrix(anchor[:3,:3].T@f[:3,:3]).as_rotvec())
        return np.asarray(rows)
    def fit_feet(self,q,root,label):
        seed=q[self.legs].copy()
        def fun(legs):
            z=q.copy();z[self.legs]=legs;return self.feet_residual(z,root)
        fit=least_squares(fun,seed,bounds=(self.lo,self.hi),max_nfev=60,xtol=1e-9,ftol=1e-9,gtol=1e-9)
        err=float(abs(fit.fun).max());self.evidence['stages'].append(dict(stage=label,success=bool(fit.success),nfev=int(fit.nfev),maximum_scaled_foot_error=err))
        if not fit.success or err>2e-6:raise ValueError(('WITNESS_FOOT_REGISTRATION_FAILED',label,err))
        q=q.copy();q[self.legs]=fit.x;return q
    def contact_root(self,R):return transform(R,self.target-R@self.marker)
    def load_for_cop(self,q,root):
        s=self.state(q,root);p=s['point'];c=s['com'];m=self.mechanics
        denominator=(p[2]-m.floor)*self.n[0]+(self.copx-p[0])*self.n[2]
        if abs(denominator)<1e-6:raise ValueError('SUPPORT_LOAD_LEVER_ARM_DEGENERATE')
        return float((self.copx-c[0])/denominator)
    def node(self,q,root,load,phase,normal=False):
        m=self.mechanics;s=self.state(q,root,load);p=apply(root,m.mesh['pelvis']);gap=float((p@self.n+self.d).min())
        marker_gap=float(s['point']@self.n+self.d)
        ratio=float(s['tau']@s['tau']/m.standing_effort)
        command=self.previous.copy();standing_gap=float((apply(m.root0,m.mesh['pelvis'])@self.n+self.d).min())
        blend=float(np.clip((gap-.012)/max(standing_gap-.012,.001),0,1))
        command[:23]=q[:23]+s['tau']/m.K+blend*m.initial_bias
        return dict(q=q.tolist(),root=root.tolist(),command=command.tolist(),phase=phase,normal=normal,gap=gap,marker_gap=marker_gap,
            load=load,model_cop=s['cop'].tolist(),model_effort_ratio=ratio,model_tau=s['tau'].tolist())
    def construct(self):
        root=self.contact_root(self.endR)
        restq=self.fit_feet(self.restq,root,'registered_accepted_witness')
        rest_load=self.load_for_cop(restq,root)
        self.evidence['registered_rest']=dict(load=rest_load,root=root.tolist(),q=restq.tolist())
        if not .05<=rest_load<=.8:raise ValueError(('REGISTERED_SUPPORT_LOAD_OUTSIDE_RANGE',rest_load))
        # Solve one fixed-contact forward pivot so BOTH feet carry the load before
        # liftoff. Roottranslation is eliminated by the material-point equation.
        angles=Rotation.from_matrix(self.endR).as_euler('xyz')
        def pose(y):
            a=angles.copy();a[1]=y[-1];rt=self.contact_root(Rotation.from_euler('xyz',a).as_matrix())
            q=restq.copy();q[self.legs]=y[:-1];return q,rt
        def residual(y):
            q,rt=pose(y);return np.r_[self.feet_residual(q,rt),self.state(q,rt)['com'][0]-self.copx]
        seed=np.r_[restq[self.legs],angles[1]]
        fit=least_squares(residual,seed,bounds=(np.r_[self.lo,-.15],np.r_[self.hi,np.deg2rad(38)]),max_nfev=80,xtol=1e-9,ftol=1e-9,gtol=1e-9)
        err=float(abs(fit.fun).max());self.evidence['balance_pivot']=dict(success=bool(fit.success),nfev=int(fit.nfev),max_residual=err,pitch=float(fit.x[-1]))
        if not fit.success or err>2e-6:raise ValueError('CONTACT_PIVOT_BALANCE_FAILED')
        balanced,br=pose(fit.x);pivot=[];q=restq.copy()
        interpolation=Slerp([0,1],Rotation.from_matrix(np.stack([self.endR,br[:3,:3]])))
        for k,u in enumerate(np.linspace(0,1,9)):
            rt=self.contact_root(interpolation(u).as_matrix());q=self.fit_feet(q,rt,'unload_pivot_'+str(k))
            load=self.load_for_cop(q,rt)
            if abs(load)<1e-5:load=0.
            if not 0<=load<=.8:raise ValueError(('PIVOT_LOAD_OUTSIDE_RANGE',k,load))
            pivot.append(self.node(q,rt,load,'landing'))
        # Purely normal initial12mm retraction; never drag a loaded hip sideways.
        normal=[]
        for k,distance in enumerate(np.linspace(0,.012,5)):
            rt=br.copy();rt[:3,3]+=distance*self.n
            q=self.fit_feet(q,rt,'normal_retraction_'+str(k));normal.append(self.node(q,rt,0.,'landing',True))
        hover=rt.copy();hq=q.copy();lift=[normal[-1]]
        interpolation=Slerp([0,1],Rotation.from_matrix(np.stack([hover[:3,:3],self.R0])))
        for k,u in enumerate(np.linspace(0,1,17)[1:],1):
            rt=transform(interpolation(u).as_matrix(),(1-u)*hover[:3,3]);qr=(1-u)*hq+u*self.q0
            target_com=(1-u)*self.copx+u*self.initial['com'][0]
            def pose_lift(y):
                qi=qr.copy();qi[self.legs]=y[:-1];rr=rt.copy();rr[0,3]=y[-1];return qi,rr
            def residual_lift(y):
                qi,rr=pose_lift(y);return np.r_[self.feet_residual(qi,rr),self.state(qi,rr)['com'][0]-target_com]
            fit=least_squares(residual_lift,np.r_[q[self.legs],rt[0,3]],bounds=(np.r_[self.lo,-.5],np.r_[self.hi,.15]),max_nfev=60,xtol=1e-9,ftol=1e-9,gtol=1e-9)
            err=float(abs(fit.fun).max());self.evidence['stages'].append(dict(stage='balanced_lift_'+str(k),success=bool(fit.success),nfev=int(fit.nfev),max_residual=err))
            if not fit.success or err>2e-6:raise ValueError(('BALANCED_LIFT_FAILED',k,err))
            q,rt=pose_lift(fit.x);lift.append(self.node(q,rt,0.,'approach'))
        last=lift[-1]
        if np.max(abs(np.asarray(last['q'])-self.q0))>2e-5 or np.max(abs(np.asarray(last['root'])-self.mechanics.root0))>2e-5:raise ValueError('RETURN_NOT_ORIGINAL_STANDING')
        last.update(q=self.q0.tolist(),root=self.mechanics.root0.tolist(),command=self.previous.tolist())
        # Stand->hover->normal contact->supportedpivot, then exactly inverse
        # geometry/load ordering. Runtime sensor qualification is still separate.
        approach=list(reversed(lift))+list(reversed(normal))[1:]+list(reversed(pivot))[1:]
        return_path=pivot+normal[1:]+lift[1:]
        forward=schedule(approach,self.mechanics.lower,self.mechanics.upper)
        backward=schedule(return_path,self.mechanics.lower,self.mechanics.upper)
        result=dict(nodes=approach,return_nodes=return_path,forward=forward,backward=backward,
            nominal_rest_settle_s=forward['seconds']+.5+1.,nominal_return_settle_s=backward['seconds']+.5+2.,
            rest_load=rest_load,rest_model_effort_ratio=approach[-1]['model_effort_ratio'],association=self.evidence['association'],
            runtime_load_observer_admitted=False,physical_success=False)
        return result
