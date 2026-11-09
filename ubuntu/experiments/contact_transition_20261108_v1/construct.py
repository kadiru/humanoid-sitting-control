"""Collision constraints participate in full-body pose construction.

The contact's two tangent coordinates are free inside the observed support
until touch, then anchored during load transfer. This removes the artificial
fixed-height restriction on pelvis leaning while retaining no-slip contact.
"""
import time
import numpy as np
from scipy.optimize import minimize
from static_model import FullRest as Base
from humanoid_sitting.shared_frame import apply

class FullRest(Base):
    def bind_camera(self, camera):
        self.camera=camera

    def constraints(self,x,load,preload,previous_command,spent,phase):
        s=self.state(x,load);command=x[:23]+s['tau']/self.K+preload
        uv=np.array([s['point']@self.e,s['point']@self.b]);insets=-(self.patch[:,:2]@uv+self.patch[:,2])
        margins=[]
        for eq,pt,F in zip(self.foot_polygons,s['foot_points'],s['foot_forces']):
            margins.extend(-(eq[:,:2]@pt[:2]+eq[:,2])-.01)
            margins.extend([F[2]/self.W-.05,(.5*F[2]-np.linalg.norm(F[:2]))/self.W])
        margins.append(s['root'][2,2]-np.cos(np.deg2rad(38)))
        # A body must stay clear of the surface, not remain parallel to it at
        # every approach pose. At touch the selected material point is on the
        # surface and every other vertex must still be nonpenetrating.
        required=min(.00601,max(0.,self.required_gap))
        vertices=apply(s['root']@s['fk'][self.body],self.mesh[self.body])
        support_clearance=vertices@self.n+self.d-required
        camera_clearance=self.camera.rows(s['root'],s['fk'],required)
        delta=command-previous_command
        minimum=.003/.0055 if phase=='normal' else .25 if phase=='load' else .04
        remaining=self.node_deadline-spent-self.extra_quiet
        ready=(command-self.previous[:23])/(.15*14.4)
        contact_insets=insets-.022 if phase!='approach' or self.required_gap<=.04 else np.ones(len(insets))
        normal_joint_speed=np.r_[.05*remaining-(x[:23]-self.prior_model_q),.05*remaining+(x[:23]-self.prior_model_q)] if phase=='normal' else np.empty(0)
        return np.r_[margins,contact_insets,support_clearance,camera_clearance,normal_joint_speed,
            command-self.lower-.05501,self.upper-.05501-command,
            .125*remaining-delta,.125*remaining+delta,remaining-minimum,x[29]-ready,x[29]+ready]

    def solve(self):
        start=time.monotonic();deadline=start+45.;self.work_checks=[self.independent_work_audit(self.x0,0.)]
        lo=np.r_[self.lower+.031,[-.5,-.2,-.6],[-.65,-.65,-.35],0.]
        hi=np.r_[self.upper-.031,[.15,.2,.15],[.65,.65,.35],np.inf]
        for side in ('left','right'):lo[self.model.names.index(side+'_knee_joint')]=max(lo[self.model.names.index(side+'_knee_joint')],.04)
        x=self.x0.copy();nodes=[];spent=0.;prior=self.previous[:23].copy();initial_gap=float(self.initial_point@self.n+self.d)
        self.optimizations=[];hover_point=contact_point=None
        initial_camera=self.camera.rows(self.root0,self.model.fk(self.q0),.00601)
        if initial_camera.min()<-1e-6:raise ValueError(('INITIAL_CAMERA_COLLISION',float(initial_camera.min())))
        for phase,steps in [('approach',8),('normal',4),('load',4)]:
            phase_start=spent;phase_end=9.66 if phase=='approach' else 12.36 if phase=='normal' else 13.36
            for i in range(steps+1):
                if nodes and i==0:continue
                u=i/steps;load=(.20*u if self.task=='leaning' else .50*u) if phase=='load' else 0.
                self.extra_quiet=.5 if phase=='normal' and i==1 else 0.
                self.node_deadline=phase_start+(.5 if i else 0.)+u*(phase_end-phase_start-.5) if phase=='normal' else phase_start+u*(phase_end-phase_start)
                self.required_gap=(1-u)*initial_gap+u*.012 if phase=='approach' else (1-u)*.012 if phase=='normal' else 0.
                preload=(1-u)*self.initial_bias if phase=='approach' else np.zeros(23)
                self.prior_model_q=np.asarray(nodes[-1]['q'][:23]) if nodes else self.q0[:23].copy()
                def equal(y):
                    s=self.state(y,load)
                    if phase in ('approach','normal'):contact=np.array([s['point']@self.n+self.d-self.required_gap])
                    else:
                        anchor=contact_point
                        contact=s['point']-(anchor-self.n*(float(anchor@self.n+self.d)-self.required_gap))
                    return np.r_[s['foot_error'],contact]
                def cost(y):
                    if time.monotonic()>deadline:raise TimeoutError('FULL_BODY_CONSTRUCTION_DEADLINE')
                    s=self.state(y,load);tangent=self.preference_point(s)-self.target; tangent-=self.n*(tangent@self.n)
                    lower=max(0.,self.root0[2,3]-s['root'][2,3])/max(.2,self.root0[2,3]-self.floor)
                    return float(s['tau']@s['tau']/self.standing_effort+y[29]**2+.05*np.sum((y[:23]-self.q0[:23])**2)+.05*(tangent@tangent)/.15**2+lower**2)
                if nodes:
                    fit=minimize(cost,x.copy(),method='SLSQP',bounds=list(zip(lo,hi)),constraints=[dict(type='eq',fun=equal),dict(type='ineq',fun=lambda y:self.constraints(y,load,preload,prior,spent,phase))],options={'maxiter':60,'ftol':1e-9})
                    eq=float(np.max(abs(equal(fit.x))));ineq=float(np.min(self.constraints(fit.x,load,preload,prior,spent,phase)))
                    sfit=self.state(fit.x,load)
                    diagnostic=dict(phase=phase,node=i,success=bool(fit.success),iterations=int(fit.nit),equality=eq,inequality=ineq,message=str(fit.message),
                        foot_pose_residual=float(np.max(abs(sfit['foot_error']))),
                        gap_m=float(sfit['point']@self.n+self.d),required_gap_m=float(self.required_gap),
                        camera_minimum_m=float(self.camera.rows(sfit['root'],sfit['fk'],min(.00601,max(0.,self.required_gap))).min()),
                        model_contact_point_local=self.contact_local(sfit).tolist(),
                        point_frozen=bool(self.contact_frozen))
                    self.optimizations.append(diagnostic)
                    if not fit.success or eq>.002 or ineq<-1e-6:raise ValueError({'reason':'COLLISION_AWARE_NODE_REJECTED',**self.optimizations[-1]})
                    x=fit.x
                s=self.state(x,load);command=x[:23]+s['tau']/self.K+preload
                if nodes:
                    active_duration=max(float(np.max(abs(command-prior))/.125),float(np.max(abs(s['q'][:23]-self.prior_model_q))/.05) if phase=='normal' else 0.,.003/.0055 if phase=='normal' else .25 if phase=='load' else .04)
                    if self.extra_quiet:
                        # Explicit zero-motion reservation in finite encoding;
                        # runtime still needs25 CURRENT qualifying samples.
                        spent+=self.extra_quiet
                        nodes.append(dict(nodes[-1],phase='hover',seconds=spent,quiet_reserve_only=True))
                    spent+=active_duration
                full=self.previous.copy();full[:23]=command
                nodes.append(dict(q=s['q'].tolist(),root=s['root'].tolist(),command=full.tolist(),phase=phase,seconds=spent,gap=float(s['point']@self.n+self.d),point=s['point'].tolist(),cop=s['cop'].tolist(),load=load,model_torque_Nm=s['tau'].tolist(),model_effort_ratio=float(s['tau']@s['tau']/self.standing_effort),camera_minimum_residual_m=float(self.camera.rows(s['root'],s['fk'],min(.00601,max(0.,self.required_gap))).min())))
                prior=command
                if phase=='approach' and i==steps:
                    hover_point=s['point'].copy()

                if phase=='normal' and i==steps:
                    contact_point=s['point'].copy();self.target=contact_point.copy()
                    self.freeze_contact(s)
                self.provisional=dict(nodes=nodes,work_audits=self.work_checks,state_evaluations=self.calls,optimizations=self.optimizations,camera_samples=self.camera.count)
        self.work_checks.append(self.independent_work_audit(x,load))
        if spent+1.>=14.4:raise ValueError(('FULL_BODY_HOLD_RESERVE',spent+1.))
        return dict(nodes=nodes,seconds=spent,settled_seconds=spent+1.,work_audits=self.work_checks,standing_torque_proxy_Nm2=self.standing_effort,return_model='Reverse model contact/load path; physical stand-up not validated',physical_success=False,camera_samples=self.camera.count,optimizations=self.optimizations)
