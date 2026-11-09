"""Contact location is a decision, with finite-front housing clearance inside the solve.

Same measured initialization, joint boxes, feet, patch reserve and action limits.
Unlike the rejected ray plan, root pose defines the continuous approach; a
changing lowest-vertex identity is never interpolated as a material coordinate.
"""
import time
import numpy as np
from scipy.optimize import minimize,least_squares
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import apply
from seat_plan import SeatPlan
from static_model import FullRest

class PelvisMechanics(FullRest):
    support_bodies=('pelvis','pelvis')
    def __init__(self,*args):
        self.support_normals=np.tile(np.asarray(args[6]),(2,1))
        super().__init__(*args)
    def support_points(self,root,fk):
        p=apply(root,self.mesh['pelvis']);heights=p@self.n+self.d
        # Evaluation at one pose; the subsequent path uses root coordinates,
        # not interpolation between changing vertices.
        point=p[int(np.argmin(heights))]
        return np.tile(point,(2,1))

class PartialPlan(SeatPlan):
    def __init__(self,*args,front_upper):
        super().__init__(*args)
        self.arm_reach=0.
        self.front=float(front_upper)+.006
        self.mechanics=PelvisMechanics(*args[:3],args[3],args[4],args[7],self.n,self.d,self.cloud,
                                      'pelvis',np.zeros((2,3)),self.cloud.mean(0),'sitting')
        self._cache={}
        self.initial_tau=self.mechanical(self.z0,0.)['tau']
        self.initial_preload=self.previous[:23]-self.q0[:23]-self.initial_tau/self.mechanics.K
        self.reference_effort=max(1.,float(self.initial_tau@self.initial_tau))
    def mechanical(self,z,load):
        q,root,*_=self.state(z)
        x=np.zeros(30);x[:23]=q[:23];x[23:26]=root[:3,3]
        x[26:29]=Rotation.from_matrix(self.R0.T@root[:3,:3]).as_rotvec()
        return self.mechanics.state(x,load)
    def foot_eq(self,z):
        s=self.mechanical(z,0.);out=[]
        for b,anchor in zip(self.mechanics.foot_bodies,self.feet):
            t=s['root']@s['fk'][b]
            out.extend((t[:3,3]-anchor[:3,3])[[0,2]])
            out.append(.2*Rotation.from_matrix(anchor[:3,:3].T@t[:3,:3]).as_rotvec()[1])
        return np.asarray(out)
    def equality(self,z):return np.r_[self.foot_eq(z),self.state(z)[4].min()-.012]
    def housing(self,z):
        q,root,fk,*_=self.state(z);rows=[]
        for body,vertices in self.mesh.items():
            if body=='pelvis':continue
            p=apply(root@fk[body],vertices)
            # A separating plane for an entire body, never per-vertex OR.
            rows.append(max(float((p@self.n+self.d).min()),float(p[:,0].min()-self.front))-.006)
        return np.asarray(rows)
    def rows(self,z):
        q,root,fk,p,h,tip,com=self.state(z)
        uv=np.c_[p@self.e,p@self.b]
        inset=-(uv@self.hull.equations[:,:2].T+self.hull.equations[:,2])
        # Fixed-size conservative extension of the unchanged20mm bottom-patch
        # test: vertices within2mm of lowest point get no vertical slack.
        patch=float(np.min(inset-.022+10*np.maximum(0,h-h.min()-.002)[:,None]))
        rows=[patch,*self.housing(z)]
        for load in [0.,.10]:
            s=self.mechanical(z,load);cmd=q[:23]+s['tau']/self.mechanics.K
            for eq,pt,F in zip(self.mechanics.foot_polygons,s['foot_points'],s['foot_forces']):
                rows.extend(-(eq[:,:2]@pt[:2]+eq[:,2])-.01)
                rows.extend([F[2]/self.mechanics.W-.05,(.5*F[2]-np.linalg.norm(F[:2]))/self.mechanics.W])
            rows.extend(cmd-self.mechanics.lower-.05501)
            rows.extend(self.mechanics.upper-.05501-cmd)
            rows.append(10.4-(1.+np.max(abs(cmd-self.previous[:23]))/.1499985))
        return np.asarray(rows)
    def objective(self,z):
        s=self.mechanical(z,.1)
        return float(s['tau']@s['tau']/self.reference_effort+.1*np.sum((z-self.z0)**2))
    def command(self,z,progress,load=0.):
        q,root,*_=self.state(z);s=self.mechanical(z,load)
        cmd=self.previous.copy();cmd[:23]=q[:23]+s['tau']/self.mechanics.K+(1-progress)*self.initial_preload
        return cmd
    def node(self,z,progress,phase,travel=0.):
        q,root,fk,p,h,tip,com=self.state(z);s=self.mechanical(z,0.)
        return dict(q=q.tolist(),root=root.tolist(),command=self.command(z,progress).tolist(),
            phase=phase,gap=float(h.min()),travel_m=float(travel),contact_tangent=float(tip@self.e),
            support_point=s['point'].tolist(),balance=float(com[0]),static_tau_zero=s['tau'].tolist())
    def solve_partial(self):
        self.deadline=time.monotonic()+45.
        lo=np.r_[self.lower,-.5,-.6,-.15,self.arm_lower]
        hi=np.r_[self.upper,.1,.1,np.deg2rad(40)-.05,self.arm_upper]
        fit=minimize(self.objective,self.z0.copy(),method='SLSQP',bounds=list(zip(lo,hi)),
            constraints=[dict(type='eq',fun=self.equality),dict(type='ineq',fun=self.rows)],
            options=dict(maxiter=80,ftol=1e-10,eps=1e-7,disp=False))
        self.stage_evidence=dict(solver_success=bool(fit.success),status=int(fit.status),message=str(fit.message),
            iterations=int(fit.nit),max_equality=float(abs(self.equality(fit.x)).max()),minimum_inequality=float(self.rows(fit.x).min()),
            initialization='new current measured q0 only',endpoint_variables=fit.x.tolist(),global_optimality=False)
        if not fit.success or abs(self.equality(fit.x)).max()>2e-6 or self.rows(fit.x).min()<-1e-8:
            raise ValueError('PARTIAL_SUPPORT_ENDPOINT_NOT_ADMITTED')
        end=fit.x;seed=self.z0[:6].copy();nodes=[]
        # Cartesian/root interpolation avoids the previous changing-minimum-vertex discontinuity.
        for u in np.linspace(0,1,33):
            z=(1-u)*self.z0+u*end
            if u:
                def fun(legs):return self.foot_eq(np.r_[legs,z[6:]])
                fit=least_squares(fun,seed,bounds=(self.lower,self.upper),max_nfev=60,xtol=1e-9,ftol=1e-9,gtol=1e-9)
                if abs(fit.fun).max()>.002:raise ValueError('FOOT_ANCHORED_PARTIAL_APPROACH_FAILED')
                seed=fit.x;z[:6]=seed
            if self.housing(z).min()<-1e-8:raise ValueError('PARTIAL_APPROACH_HOUSING_CLEARANCE_FAILED')
            nodes.append(self.node(z,float(u),'approach'))
        nodes[0]['command']=self.previous.tolist()
        land=[];seed=end[:6].copy()
        for travel in np.linspace(0,.015,16):
            z=end.copy();z[6]-=travel*self.n[0];z[7]-=travel*self.n[2]
            fit=least_squares(lambda legs:self.foot_eq(np.r_[legs,z[6:]]),seed,bounds=(self.lower,self.upper),max_nfev=60,xtol=1e-9,ftol=1e-9,gtol=1e-9)
            if abs(fit.fun).max()>.002:raise ValueError('PARTIAL_NORMAL_IK_FAILED')
            seed=fit.x;z[:6]=seed
            if self.housing(z).min()<-1e-8:raise ValueError('PARTIAL_LANDING_HOUSING_CLEARANCE_FAILED')
            land.append(self.node(z,1.,'landing',travel))
        commands=np.asarray([n['command'] for n in nodes]);arc=np.r_[0,np.cumsum(abs(np.diff(commands,axis=0)).max(1))]
        from complete_seat_path import arc_clock,interpolate
        duration=arc_clock(0,arc[-1])[1];count=int(np.ceil(duration/.02));scheduled=[];desired=[];previous=self.previous.copy()
        for tick in range(count):
            distance,_=arc_clock(min((tick+1)*.02,duration),arc[-1]);target=interpolate(commands,arc,distance)
            desired.append((previous+(target-previous)/.1).tolist());scheduled.append(target.tolist());previous=target
        settled=count*.02+.5+2.5+1
        if settled>=14.4:raise ValueError('PARTIAL_CONTACT_TIME_RESERVE_FAILED')
        mechanics=self.mechanical(end,.1)
        report=dict(nodes=nodes,landing=land,command_arc=arc.tolist(),approach_commands=scheduled,approach_desired=desired,
            approach_seconds=count*.02,nominal_latest_settled_s=settled,endpoint=self.stage_evidence,
            model_effort_ratio_at10percent_load=float(mechanics['tau']@mechanics['tau']/self.reference_effort),
            modeled_rest_fraction=.1,physical_success=False,return_primitive_admitted=False)
        return report
