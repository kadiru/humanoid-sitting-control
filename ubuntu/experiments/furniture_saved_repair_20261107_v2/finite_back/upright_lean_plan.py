"""Choose supported leaning height instead of imposing a squat waypoint.
The objective compares dimensionless knee bend, height loss and model torque.
It does not claim measured effort/energy savings or change physical acceptance.
"""
import time
import numpy as np
from scipy.optimize import minimize,least_squares
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from finite_geometry import LeanGeometry as OriginalPlan
from humanoid_sitting.shared_frame import apply

class LeanPlan(OriginalPlan):
    def solve(self):
        deadline=time.monotonic()+45
        initial_root,_=self.pose(self.start);initial_height=float(initial_root[2,3])
        leg_length=initial_height-self.floor
        lateral=np.cross(self.n,self.up)
        facet=ConvexHull(np.c_[self.cloud@self.up,self.cloud@lateral]).equations
        def pelvis_clearance(x):
            root,fk=self.pose(x);points=apply(root@fk['pelvis'],self.mesh['pelvis'])
            uv=np.c_[points@self.up,points@lateral]
            on_observed_facet=np.all(uv@facet[:,:2].T+facet[:,2]<=0.,axis=1)
            # The prior rejected path hit the pelvis on this measured back facet.
            # Do not extend the plane beyond its observed finite footprint.
            return float(np.min(points[on_observed_facet]@self.n+self.off)-.006) if on_observed_facet.any() else 1.
        def constraint(x):
            gap,point,com,pitch=self.geometry(x)
            cop=(com[0]+.20*((point[2]-self.floor)*self.n[0]-point[0]*self.n[2]))/(1-.20*self.n[2])
            cmd=np.stack([x+self.feedforward(x,f) for f in [0.,.20]])
            return np.r_[point@self.up-self.band[0]-.04,self.band[1]-.04-point@self.up,
                cop-self.foot_bounds[0]-.01,self.foot_bounds[1]-.01-cop,
                .35-pitch,.35+pitch,pelvis_clearance(x),(cmd-self.lower-.001).ravel(),(self.upper-cmd-.001).ravel()]
        def cost(x):
            if time.monotonic()>deadline:raise TimeoutError('45_SECOND_TOTAL_CONSTRUCTION_CAP')
            rt,_=self.pose(x);tau=self.feedforward(x,.20)*np.array([200.,200.,160.])
            return float(2*np.sum((tau/np.array([300.,300.,20.]))**2)+(max(0.,initial_height-rt[2,3])/leg_length)**2+(x[1]/np.pi)**2)
        # Starts from current standing encoders, no old failed endpoint or hidden pose seed.
        fit=minimize(cost,self.start.copy(),method='SLSQP',bounds=list(zip(self.lower,self.upper)),
                     constraints=[dict(type='eq',fun=lambda x:self.geometry(x)[0]),dict(type='ineq',fun=constraint)],options={'maxiter':100,'ftol':1e-9})
        if not fit.success or abs(self.geometry(fit.x)[0])>1e-5 or constraint(fit.x).min()<-1e-6:
            raise ValueError(('UPRIGHT_LEAN_ENDPOINT_REJECTED',fit.message,fit.x.tolist(),float(constraint(fit.x).min())))
        goal=fit.x;root,fk=self.pose(goal);_,point,_,pitch=self.geometry(goal)
        marker=(point-root[:3,3])@root[:3,:3]
        start=apply(initial_root,marker);initial_pitch=self.geometry(self.start)[3];height=float(point@self.up)
        bias=self.prev-self.start-self.feedforward(self.start,0.);seed=self.start.copy();nodes=[]
        def residual(x,g,h,p):
            rt,_=self.pose(x);pt=apply(rt,marker)
            return np.r_[pt@self.n+self.off-g,pt@self.up-h,.35*(Rotation.from_matrix(rt[:3,:3]).as_euler('xyz')[1]-p)]
        for phase,steps in [('approach',48),('load',12)]:
            for i in range(steps+1):
                if nodes and i==0:continue
                if time.monotonic()>deadline:raise TimeoutError('45_SECOND_TOTAL_CONSTRUCTION_CAP')
                u=i/steps
                g=(1-u)*(start@self.n+self.off) if phase=='approach' else 0.
                h=(1-u)*(start@self.up)+u*height if phase=='approach' else height
                p=(1-u)*initial_pitch+u*pitch if phase=='approach' else pitch
                match=least_squares(residual,seed,bounds=(self.lower,self.upper),args=(g,h,p),max_nfev=60)
                if np.max(abs(match.fun))>.002:raise ValueError(('UPRIGHT_LEAN_PATH_REJECTED',match.fun.tolist()))
                seed=match.x;gap,pt,com,ap=self.geometry(seed);load=.2*u if phase=='load' else 0.
                command=seed+self.feedforward(seed,load)+(bias*(1-u) if phase=='approach' else 0.)
                cop=(com[0]+load*((pt[2]-self.floor)*self.n[0]-pt[0]*self.n[2]))/(1-load*self.n[2])
                nodes.append(dict(q=seed.tolist(),cmd=command.tolist(),gap=gap,height=float(pt@self.up),pitch=ap,cop=float(cop),load=load,phase=phase))
        command=np.array([n['cmd'] for n in nodes])
        if (command<self.lower).any() or (command>self.upper).any():raise ValueError('UPRIGHT_LEAN_COMMAND_LIMIT')
        delta=np.max(abs(np.diff(command,axis=0)),axis=1)
        times=np.r_[0,np.cumsum(np.maximum(delta/.125,[.19 if n['phase']=='load' else .14 for n in nodes[1:]]))]
        if times[-1]>=14.4:raise ValueError(('UPRIGHT_LEAN_TIME_LIMIT',float(times[-1])))
        self.posture_selection=dict(initial_root_height=initial_height,final_root_height=float(root[2,3]),
            final_knee_rad=float(goal[1]),final_pitch_rad=float(pitch),objective=float(fit.fun),physical_effort_saving_unproven=True,
            changed='Contact witness lies on original rear-torso collision triangles clipped to observed finite back footprint; unchanged40mm height and20mm lateral reserves. No compulsory upper-torso contact above object.')
        return nodes,times
