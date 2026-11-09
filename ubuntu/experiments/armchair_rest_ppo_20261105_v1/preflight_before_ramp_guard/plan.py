"""A separate high-surface pelvis-rest construction; original rear-back retained."""
import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.spatial.transform import Rotation
from original_plan import LeanPlan as OriginalPlan, ROOT
from humanoid_sitting.shared_frame import apply

class LeanPlan(OriginalPlan):
    def __init__(self,*args,profile='rear_back',**kwargs):
        super().__init__(*args,**kwargs);self.profile=profile
        if profile=='rear_pelvis':
            self.trunk={'pelvis':self.mesh['pelvis']}
            # Straight-legged resting is not knee hyperextension. This is a
            # stricter teacher-posture constraint, not a physical gate change.
            self.lower[1]=max(self.lower[1],.04)
    def solve(self):
        if self.profile=='rear_back':return super().solve()
        def constraints(x):
            gap,pt,com,pitch=self.geometry(x)
            cop=(com[0]+.20*((pt[2]-self.floor)*self.n[0]-pt[0]*self.n[2]))/(1-.20*self.n[2])
            cmd=x+self.feedforward(x,.20)
            return np.r_[pt@self.up-self.band[0]-.04,self.band[1]-.04-pt@self.up,
                cop-self.foot_bounds[0]-.01,self.foot_bounds[1]-.01-cop,
                .35-pitch,.35+pitch,cmd-self.lower,self.upper-cmd]
        def cost(x):
            tau=self.feedforward(x,.20)*np.array([200.,200.,160.])
            return float(np.sum(tau*tau)/1000+.10*(x[1]-.08)**2)
        fit=minimize(cost,self.start,method='SLSQP',bounds=list(zip(self.lower,self.upper)),
            constraints=[{'type':'eq','fun':lambda x:self.geometry(x)[0]}, {'type':'ineq','fun':constraints}],
            options={'maxiter':100,'ftol':1e-9})
        if not fit.success or abs(self.geometry(fit.x)[0])>1e-5 or constraints(fit.x).min()<-1e-6:
            raise ValueError(('PELVIS_PLAN_INFEASIBLE',fit.message,fit.x.tolist(),constraints(fit.x).min()))
        final=fit.x;_,tip,_,pitch=self.geometry(final)
        root,f=self.pose(final)
        vertices=apply(f['pelvis'],self.mesh['pelvis']);marker=vertices[(apply(root,vertices)@self.n).argmin()]
        initial_root,_=self.pose(self.start);initial=apply(initial_root,marker)
        _,_,_,p0=self.geometry(self.start);height=tip@self.up
        def residual(x,g,h,p):
            rt,_=self.pose(x);pt=apply(rt,marker)
            return np.r_[pt@self.n+self.off-g,pt@self.up-h,.35*(Rotation.from_matrix(rt[:3,:3]).as_euler('xyz')[1]-p)]
        bias=self.prev-self.start-self.feedforward(self.start,0);nodes=[];seed=self.start.copy()
        for phase,steps in [('approach',48),('load',12)]:
            for i in range(steps+1):
                if nodes and i==0:continue
                u=i/steps
                g=(1-u)*(initial@self.n+self.off) if phase=='approach' else 0.
                h=(1-u)*(initial@self.up)+u*height if phase=='approach' else height
                p=(1-u)*p0+u*pitch if phase=='approach' else pitch
                fit=least_squares(residual,seed,bounds=(self.lower,self.upper),args=(g,h,p),max_nfev=60)
                if np.max(np.abs(fit.fun))>.002:raise ValueError(('PELVIS_PATH_INFEASIBLE',fit.fun.tolist()))
                seed=fit.x;gap,pt,com,ap=self.geometry(seed);load=.20*u if phase=='load' else 0.
                cmd=seed+self.feedforward(seed,load)+(bias*(1-u) if phase=='approach' else 0.)
                cop=(com[0]+load*((pt[2]-self.floor)*self.n[0]-pt[0]*self.n[2]))/(1-load*self.n[2])
                nodes.append(dict(q=seed.tolist(),cmd=cmd.tolist(),gap=gap,height=float(pt@self.up),pitch=ap,cop=float(cop),load=load,phase=phase))
        commands=np.array([n['cmd'] for n in nodes])
        if (commands<self.lower).any() or (commands>self.upper).any():raise ValueError('PELVIS_COMMAND_LIMIT')
        diff=np.max(abs(np.diff(commands,axis=0)),axis=1)
        times=np.r_[0,np.cumsum(np.maximum(diff/.125,[.19 if n['phase']=='load' else .14 for n in nodes[1:]]))]
        if times[-1]>=14.4:raise ValueError(('PATH_TOO_LONG',times[-1]))
        return nodes,times
