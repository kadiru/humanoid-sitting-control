"""Find a load-transfer endpoint, then connect it backward to measured standing.

All 23 joints remain optimization variables. Only original measurements seed
the construction. Failed solver iterates are diagnostic output, never motion.
"""
import time
from collections import OrderedDict
import numpy as np
from scipy.optimize import minimize
from static_model import FullRest as Base
from humanoid_sitting.shared_frame import apply
from seed import construct as geometric_seed


class NumericalPlan(Base):
    def __init__(self,*args,**kwargs):
        self.contact_frozen=False
        self.pose_cache=OrderedDict()
        super().__init__(*args,**kwargs)
        self.x0=np.r_[self.x0,0.]
        self.preference_marker=self.marker.copy()
        self.optimizations=[]

    def pose(self,x):
        key=np.asarray(x[:29]).tobytes()
        if key not in self.pose_cache:
            self.pose_cache[key]=super().pose(x)
            if len(self.pose_cache)>64:self.pose_cache.popitem(last=False)
        return self.pose_cache[key]

    def exposed_local(self,root,fk):
        pose=root@fk[self.body];direction=pose[:3,:3].T@self.n
        vertices=self.mesh[self.body];projection=vertices@direction
        return vertices[projection<=projection.min()+1e-10].mean(0)

    def material_point(self,root,fk,load):
        # Endpoint search compares loads at the SAME candidate pose. This
        # selects one material point; it never permits loaded physical motion.
        marker=self.marker if self.contact_frozen else self.exposed_local(root,fk)
        return apply(root@fk[self.body],marker)

    def bind_camera(self,camera):
        self.camera=camera;self.camera_cache=OrderedDict()

    def camera_rows(self,x,gap):
        key=(np.asarray(x[:29]).tobytes(),float(gap))
        if key not in self.camera_cache:
            _,root,fk=self.pose(x)
            self.camera_cache[key]=self.camera.rows(root,fk,min(.00601,max(0.,gap)))
            if len(self.camera_cache)>64:self.camera_cache.popitem(last=False)
        return self.camera_cache[key]

    def command(self,x,load,bias=None):
        return x[:23]+self.state(x,load)['tau']/self.K+(0. if bias is None else bias)

    def mechanical_rows(self,x,load,bias=None):
        s=self.state(x,load);cmd=self.command(x,load,bias)
        rows=[]
        for eq,pt,F in zip(self.foot_polygons,s['foot_points'],s['foot_forces']):
            rows.extend(-(eq[:,:2]@pt[:2]+eq[:,2])-.01)
            rows.extend([F[2]/self.W-.05,(.5*F[2]-np.linalg.norm(F[:2]))/self.W])
        ready=(cmd-self.previous[:23])/(.15*14.4)
        return np.r_[rows,cmd-self.lower-.05501,self.upper-.05501-cmd,x[29]-ready,x[29]+ready]

    def geometry_rows(self,x,gap):
        s=self.state(x,0.);uv=np.array([s['point']@self.e,s['point']@self.b])
        inset=-(self.patch[:,:2]@uv+self.patch[:,2])-.022
        vertices=apply(s['root']@s['fk'][self.body],self.mesh[self.body])
        return np.r_[s['root'][2,2]-np.cos(np.deg2rad(38)),inset,
                     vertices@self.n+self.d-min(.00601,max(0.,gap)),self.camera_rows(x,gap),
                     (self.contact_edges[:,:3]@s['point']+self.contact_edges[:,3]) if gap<=.01200001 else np.empty(0),
                     self.housing_rows(x)]

    def housing_rows(self,x):
        _,root,fk=self.pose(x)
        return np.asarray([float(np.min(apply(root@fk[b],self.mesh[b])@self.n+self.d))-.006
                           for b in ('left_hip_pitch_link','right_hip_pitch_link')])

    def housing_jac(self,x):
        j=self.jets(x,0.);result=[]
        for b in ('left_hip_pitch_link','right_hip_pitch_link'):
            pose=j['world'][b];J=j['Jworld'][b];V=self.mesh[b]
            k=int(np.argmin((V@pose[:3,:3].T+pose[:3,3])@self.n))
            result.append(self.n@(np.einsum('ijk,j->ik',J[:3,:3],V[k])+J[:3,3]))
        return np.asarray(result)

    def equal(self,x,gap):
        s=self.state(x,0.)
        return np.r_[s['foot_error'],s['point']@self.n+self.d-gap]

    def posture_cost(self,x,load):
        if time.monotonic()>self.deadline:raise TimeoutError('FULL_CONSTRUCTION_DEADLINE')
        s=self.state(x,load)
        p=apply(s['root']@s['fk'][self.body],self.preference_marker)
        tangent=p-self.target;tangent-=self.n*(tangent@self.n)
        lowering=max(0.,-s['root'][2,3])/max(.2,-self.floor)
        return float(s['tau']@s['tau']/self.standing_effort+x[29]**2+
            .05*np.sum((x[:23]-self.q0[:23])**2)+.05*(tangent@tangent)/.15**2+lowering**2)

    def load_seconds(self,x):
        cmds=[self.command(x,float(l)) for l in self.loads]
        return sum(max(float(np.max(abs(b-a)))/.125,.25) for a,b in zip(cmds,cmds[1:]))

    def endpoint_row_labels(self):
        labels=['tilt_reserve']+['patch_inset_'+str(i) for i in range(len(self.patch))]
        labels+=['pelvis_plane_vertex_'+str(i) for i in range(len(self.mesh[self.body]))]
        labels+=['camera_'+b for b in self.camera.samples]+['contact_facet_edge_'+str(i) for i in range(3)]+['left_hip_pitch_housing_clearance','right_hip_pitch_housing_clearance']
        for load in self.loads:
            prefix='load_'+str(float(load))+':'
            for b,poly in zip(self.foot_bodies,self.foot_polygons):
                labels += [prefix+b+':cop_edge_'+str(i) for i in range(len(poly))]+[prefix+b+':normal_load',prefix+b+':friction']
            for group in ['command_lower','command_upper','readiness_positive','readiness_negative']:
                labels += [prefix+group+':'+n for n in self.model.names[:23]]
        return labels+['total_time_reserve']

    def endpoint_rows(self,x):
        # A necessary time bound, followed by exact time accounting on the
        # constructed path. Parallel joint/normal motion is charged by max.
        travel=max(float(np.max(abs(self.command(x,0.)-self.previous[:23])))/.125,
                   .012/.0055+.5)
        return np.r_[self.geometry_rows(x,0.),*[self.mechanical_rows(x,float(l)) for l in self.loads],
                     14.399-1.-self.load_seconds(x)-travel]

    def optimize(self,x,gap,rows,cost,bounds,label):
        fit=minimize(cost,x.copy(),method='SLSQP',bounds=bounds,
            constraints=[dict(type='eq',fun=lambda y:self.equal(y,gap)),dict(type='ineq',fun=rows)],
            options={'maxiter':60,'ftol':1e-9})
        eq=float(np.max(abs(self.equal(fit.x,gap))));ineq=float(np.min(rows(fit.x)))
        s=self.state(fit.x,0.)
        diagnostic=dict(stage=label,success=bool(fit.success),iterations=int(fit.nit),
            equality=eq,inequality=ineq,message=str(fit.message),
            foot_residual=float(np.max(abs(s['foot_error']))),gap_m=float(s['point']@self.n+self.d),
            camera_minimum_m=float(self.camera_rows(fit.x,gap).min()),
            objective=float(fit.fun),solver_status=int(fit.status))
        self.optimizations.append(diagnostic)
        if not fit.success or eq>.002 or ineq<-1e-6:
            raise ValueError(dict(reason='ENDPOINT_OR_CONNECTION_REJECTED',**diagnostic))
        return fit.x.copy()

    def node(self,x,load,bias,phase,seconds):
        s=self.state(x,load);cmd=self.previous.copy();cmd[:23]=self.command(x,load,bias)
        return dict(q=s['q'].tolist(),root=s['root'].tolist(),command=cmd.tolist(),phase=phase,
            seconds=float(seconds),gap=float(s['point']@self.n+self.d),point=s['point'].tolist(),
            cop=s['cop'].tolist(),load=float(load),model_torque_Nm=s['tau'].tolist(),
            model_effort_ratio=float(s['tau']@s['tau']/self.standing_effort))

    def solve(self):
        self.deadline=time.monotonic()+45.;self.loads=np.linspace(0.,.2,5)
        self.work_checks=[self.independent_work_audit(self.x0,0.)]
        lo=np.r_[self.lower+.031,[-.5,-.2,-.6],[-.65,-.65,-.35],0.,0.]
        hi=np.r_[self.upper-.031,[.15,.2,.15],[.65,.65,.35],np.inf,14.4]
        for side in ('left','right'):
            j=self.model.names.index(side+'_knee_joint');lo[j]=max(lo[j],.04)
        endpoint_hi=hi.copy();endpoint_hi[30]=0.
        initial_gap=float(self.initial_point@self.n+self.d)
        if initial_gap<=.012:raise ValueError('ORIGINAL_START_HAS_NO_APPROACH')
        if np.min(self.camera_rows(self.x0,initial_gap))<-1e-6:raise ValueError('INITIAL_CAMERA_COLLISION')
        seed,seed_report=geometric_seed(self)
        # Fix one original-geometry contact feature before endpoint optimization.
        # All hull nonpenetration rows remain: a wrong feature is rejected.
        _,seed_root,seed_fk=self.pose(seed)
        self.marker=self.exposed_local(seed_root,seed_fk).copy()
        self.contact_frozen=True;self.states.clear()
        seed_report['endpoint_material_local']=self.marker.tolist()
        seed_report['feature_selection']='Original analytic seed only; no failed candidate'
        seed[29]=max(float(np.max(abs(self.command(seed,float(load))-self.previous[:23])))/(.15*14.4) for load in self.loads)
        seed_report['readiness_initialization']='Exact maximum across ALL five constrained load levels, not only unloaded command'
        seed_report['readiness_epigraph']=float(seed[29])
        seed_report['initial_full_inequality_minimum']=float(self.endpoint_rows(seed).min())
        self.provisional=dict(seed=seed_report,accepted_endpoint=False)
        self.record('seed',seed_report)
        endpoint=self.optimize(seed,0.,self.endpoint_rows,lambda x:self.posture_cost(x,.2),
                               list(zip(lo,endpoint_hi)),'load_transfer_endpoint')
        # Independent dense load samples test what lies between the five
        # constrained loads; no optimizer callback or endpoint repair follows.
        dense_load_min=min(float(self.mechanical_rows(endpoint,float(l)).min()) for l in np.linspace(0.,.2,17))
        if dense_load_min<-1e-6:raise ValueError(('BETWEEN_LOAD_MECHANICS',dense_load_min))
        self.work_checks.append(self.independent_work_audit(endpoint,.2))
        endpoint_node=self.node(endpoint,0.,None,'normal',0.)
        mesh=self.endpoint_audit(endpoint_node)
        if not mesh['passed']:raise ValueError('ENDPOINT_FULL_MESH_REJECTED')
        endpoint_state=self.state(endpoint,0.)
        endpoint_marker=apply(np.linalg.inv(endpoint_state['root']@endpoint_state['fk'][self.body]),endpoint_state['point'])
        self.provisional.update(accepted_endpoint=True,endpoint=endpoint_node,endpoint_mesh=mesh,
                                dense_load_minimum=dense_load_min,work_audits=self.work_checks)
        self.record('accepted_endpoint_model_only',self.provisional)
        loading_time=self.load_seconds(endpoint)
        # Zero-load backward approach may change exposed feature; only the
        # loaded endpoint uses the fixed material chosen before its solve.
        self.contact_frozen=False;self.states.clear()
        # Reverse points: touch -> 12mm hover -> original standing. Recorded
        # interval duration belongs to the edge from the new point to prior.
        reverse=[dict(x=endpoint.copy(),gap=0.,bias=np.zeros(23),edge_seconds=0.)]
        spent=0.;x=endpoint.copy()
        gaps=list(np.linspace(.003,.012,4))+list(np.linspace(.012,initial_gap,9)[1:])
        for i,gap in enumerate(gaps):
            normal=i<4
            bias=np.zeros(23) if normal else self.initial_bias*(gap-.012)/(initial_gap-.012)
            prior=x.copy();prior_cmd=self.command(prior,0.,reverse[-1]['bias'])
            def connection_rows(y):
                dt=y[30];cmd=self.command(y,0.,bias);delta=cmd-prior_cmd
                # Reserve the unbuilt path from standing, the remaining normal
                # part, the current-sample quiet interval, load, and settle.
                remaining=max(float(np.max(abs(cmd-self.previous[:23])))/.125,
                              max(0.,.012-gap)/.0055)
                joint=np.r_[.05*dt-(y[:23]-prior[:23]),.05*dt+(y[:23]-prior[:23])] if normal else np.empty(0)
                minimum=.003/.0055 if normal else .04
                return np.r_[self.geometry_rows(y,gap),self.mechanical_rows(y,0.,bias),
                    .125*dt-delta,.125*dt+delta,joint,dt-minimum,
                    14.399-1.-.5-loading_time-spent-dt-remaining]
            fraction=(initial_gap-gap)/initial_gap
            reference=fraction*endpoint[:29]+(1-fraction)*self.x0[:29]
            self.connection_context=dict(prior=prior.copy(),prior_cmd=prior_cmd.copy(),gap=float(gap),bias=bias.copy(),normal=normal,loading_time=loading_time,spent=spent,reference=reference.copy())
            def cost(y):
                return self.posture_cost(y,0.)+.1*np.sum((y[:29]-reference)**2)+.01*y[30]
            if i==len(gaps)-1:
                x=self.x0.copy()
                x[29]=float(np.max(abs(self.command(x,0.,bias)-self.previous[:23])))/(.15*14.4)
                x[30]=max(float(np.max(abs(self.command(x,0.,bias)-prior_cmd)))/.125,.04)
                if np.max(abs(self.equal(x,gap)))>.002 or np.min(connection_rows(x))<-1e-6:
                    raise ValueError('FIXED_STANDING_CONNECTION_REJECTED')
            else:
                x[30]=.003/.0055 if normal else .5
                lower=lo.copy();lower[30]=.003/.0055 if normal else .04
                x=self.optimize(x,gap,connection_rows,cost,list(zip(lower,hi)),
                                ('reverse_normal_' if normal else 'reverse_approach_')+str(i))
            dt=float(x[30]);spent+=dt
            reverse.append(dict(x=x.copy(),gap=float(gap),bias=bias.copy(),edge_seconds=dt))
            self.provisional.update(reverse_points=len(reverse),reverse_seconds=spent,
                accepted_endpoint=True,optimizations=self.optimizations,state_evaluations=self.calls)
        nodes=[];elapsed=0.
        chronological=list(reversed(reverse))
        for j,item in enumerate(chronological):
            if j:elapsed+=chronological[j-1]['edge_seconds']
            phase='approach' if item['gap']>=.012 else 'normal'
            nodes.append(self.node(item['x'],0.,item['bias'],phase,elapsed))
            if abs(item['gap']-.012)<1e-10:
                elapsed+=.5
                nodes.append(dict(nodes[-1],phase='hover',seconds=elapsed,quiet_reserve_only=True))
        self.marker=endpoint_marker.copy();self.target=endpoint_state['point'].copy()
        self.contact_frozen=True;self.states.clear()
        prior_cmd=self.command(endpoint,0.)
        for load in self.loads[1:]:
            cmd=self.command(endpoint,float(load));elapsed+=max(float(np.max(abs(cmd-prior_cmd)))/.125,.25)
            nodes.append(self.node(endpoint,float(load),None,'load',elapsed));prior_cmd=cmd
        if elapsed+1.>=14.4:raise ValueError(('FULL_HOLD_RESERVE',elapsed+1.))
        self.provisional.update(nodes=nodes)
        return dict(nodes=nodes,seconds=elapsed,settled_seconds=elapsed+1.,seed=seed_report,
            work_audits=self.work_checks,optimizations=self.optimizations,camera_samples=self.camera.count,
            dense_load_minimum=dense_load_min,endpoint_mesh=mesh,
            standing_torque_proxy_Nm2=self.standing_effort,physical_success=False,
            return_model='Same-pose unloading then reversed approach; model only, stand-up unvalidated')

from analytic_model import AnalyticMixin
from readiness_graph import ReadinessGraphMixin
class FullRest(ReadinessGraphMixin,AnalyticMixin,NumericalPlan):
    pass
