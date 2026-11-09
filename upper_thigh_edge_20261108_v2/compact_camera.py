"""Exact feasibility compression of v1's fixed material-sample constraints.

min(samples for body) >= 0 iff every original sample residual >= 0.
Whole bodies outside one camera frustum halfspace have only unknown rows (=1)
and are culled using their local AABB. No physical clearance is reduced.
The minimum is piecewise differentiable; optimizer success remains mandatory.
"""
import numpy as np
from itertools import product
from camera_clearance import CameraClearance

class CompactCamera(CameraClearance):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.boxes={b:np.array(list(product(*zip(v.min(0),v.max(0))))) for b,v in self.samples.items()}
        self.last_active_samples=0

    def rows(self,root,fk,required_gap):
        result=[];active=0
        # The exact five linear inequalities of the original pixel domain.
        K=self.K
        for body,samples in self.samples.items():
            pose=root@fk[body]
            M=self.T[:3,:3].T@self.R0.T@pose[:3,:3]
            t=self.T[:3,:3].T@(self.R0.T@pose[:3,3]-self.T[:3,3])
            box=self.boxes[body]@M.T+t
            x,y,z=box.T
            halfspaces=np.array([z-.08,K[0,0]*x+(K[0,2]-.5)*z,(self.w-.5-K[0,2])*z-K[0,0]*x,
                                 K[1,1]*y+(K[1,2]-.5)*z,(self.h-.5-K[1,2])*z-K[1,1]*y])
            if np.any(halfspaces.max(1)<-1e-12):
                result.append(1.);continue
            C=samples@M.T+t;z=C[:,2];safe=np.where(z>.08,z,1.)
            u=C[:,0]/safe*K[0,0]+K[0,2]-.5;v=C[:,1]/safe*K[1,1]+K[1,2]-.5
            col=np.floor(u).astype(int);row=np.floor(v).astype(int)
            inside=(z>.08)&(row>=0)&(row<self.h-1)&(col>=0)&(col<self.w-1)
            ids=np.flatnonzero(inside)
            if not len(ids):result.append(1.);continue
            rr=row[ids];cc=col[ids];side=((u[ids]-cc)+(v[ids]-rr)>1).astype(int)
            known=self.valid[rr,cc,side]
            if body in ('left_ankle_roll_link','right_ankle_roll_link'):known &= ~self.ground[rr,cc,side]
            ids=ids[known];rr=rr[known];cc=cc[known];side=side[known]
            if not len(ids):result.append(1.);continue
            points=samples[ids]@pose[:3,:3].T+pose[:3,3];active+=len(ids)
            target=(body in getattr(self,'support_bodies',(self.body,)))&self.target[rr,cc,side]
            noncontact=self.sample_clearance[body][ids] if hasattr(self,'sample_clearance') else .006
            margin=np.where(target,max(0.,required_gap),noncontact)
            residual=np.sum(points*self.N[rr,cc,side],axis=1)+self.D[rr,cc,side]-margin
            # Original unknown samples contribute +1 and must be retained when
            # comparing the numerical minimum, not just its feasible sign.
            result.append(min(float(residual.min()),1.) if len(ids)<len(samples) else float(residual.min()))
        self.last_active_samples=active
        return np.asarray(result)
