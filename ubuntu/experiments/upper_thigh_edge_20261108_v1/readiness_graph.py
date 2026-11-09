"""Exact elimination of a monotone auxiliary epigraph, not a pose repair."""
from collections import OrderedDict
import numpy as np
from scipy.optimize import minimize

class ReadinessGraphMixin:
    def optimize(self,x,gap,rows,cost,bounds,label):
        endpoint=label=='load_transfer_endpoint'
        loads=tuple(map(float,self.loads)) if endpoint else (0.,)
        bias=None if endpoint else self.connection_context['bias']
        assert bounds[29][0]==0. and np.isposinf(bounds[29][1])
        cache=OrderedDict();scale=.15*14.4
        def lift(y):
            key=np.asarray(y).tobytes()
            if key not in cache:
                full=np.insert(y,29,0.)
                ready=np.asarray([(self.command(full,l,bias)-self.previous[:23])/scale for l in loads])
                if not np.isfinite(ready).all():raise ValueError('NONFINITE_READINESS_GRAPH')
                index=np.unravel_index(int(np.argmax(abs(ready))),ready.shape)
                full[29]=abs(ready[index])
                cache[key]=(full,loads[index[0]],int(index[1]),float(np.sign(ready[index])))
                if len(cache)>64:cache.popitem(last=False)
            return cache[key]
        def graph_grad(y):
            full,load,joint,sign=lift(y)
            return sign*self.jets(full,load)['command'][joint]/scale
        y0=np.delete(x,29);full0=lift(y0)[0]
        if endpoint:self.audit_derivatives(full0)
        geom=len(self.geometry_rows(full0,gap));mechanical=len(self.mechanical_rows(full0,loads[0]))
        total=len(rows(full0));keep=np.ones(total,bool)
        for i in range(len(loads)):
            start=geom+i*mechanical+mechanical-46
            keep[start:start+46]=False
        readiness_rows=~keep.copy()
        facet_rows=[]
        if endpoint and getattr(self,'patch_active',False):
            labels=self.endpoint_row_labels()
            facet_rows=[i for i,n in enumerate(labels) if n.startswith('contact_facet_edge_')]
            assert len(facet_rows)==6
            keep[facet_rows]=False
        original_jac=self.endpoint_jac if endpoint else self.connection_jac
        original_cost_jac=(lambda z:self.posture_jac(z,.2)) if endpoint else self.connection_cost_jac
        def objective(y):return cost(lift(y)[0])
        def objective_jac(y):
            full=lift(y)[0];g=original_cost_jac(full)
            return np.delete(g+g[29]*graph_grad(y),29)
        def ineq(y):return rows(lift(y)[0])[keep]
        def ineq_jac(y):
            J=original_jac(lift(y)[0])[keep]
            if np.any(J[:,29]!=0.):raise ValueError('READINESS_COUPLED_TO_RETAINED_ROW')
            return np.delete(J,29,axis=1)
        def eq(y):return self.equal(lift(y)[0],gap)
        def eq_jac(y):
            J=self.equal_jac(lift(y)[0],gap)
            if np.any(J[:,29]!=0.):raise ValueError('READINESS_COUPLED_TO_EQUALITY')
            return np.delete(J,29,axis=1)
        if endpoint:
            h=1e-6;numeric=np.zeros(len(y0))
            for k in range(len(y0)):
                a=y0.copy();b=y0.copy();a[k]-=h;b[k]+=h
                numeric[k]=(objective(b)-objective(a))/(2*h)
            error=float(np.max(abs(numeric-objective_jac(y0))))
            original_ready=rows(full0)[readiness_rows]
            report=dict(passed=bool(error<2e-5 and np.min(original_ready)>=0.),
                graph_objective_gradient_error=error,original_readiness_minimum=float(np.min(original_ready)),
                optimizer_coordinates=len(y0),original_coordinates=len(full0),removed_rows=int((~keep).sum()),finite_facet_rows_enforced_by_query_and_contact_equality=facet_rows,
                source='Original input seed; no rejected solver vector',physical_success=False)
            self.record('readiness_graph_audit',report)
            if not report['passed']:raise ValueError(('READINESS_GRAPH_AUDIT_FAILED',report))
            ineq_jac(y0);eq_jac(y0)
        fit=minimize(objective,y0.copy(),jac=objective_jac,method='SLSQP',
            bounds=[b for i,b in enumerate(bounds) if i!=29],
            constraints=[dict(type='eq',fun=eq,jac=eq_jac),dict(type='ineq',fun=ineq,jac=ineq_jac)],
            options={'maxiter':60,'ftol':1e-9})
        full=lift(fit.x)[0].copy()
        if not np.array_equal(np.delete(full,29).view(np.uint64),np.asarray(fit.x).view(np.uint64)):
            raise ValueError('RETAINED_COORDINATE_CHANGED')
        allrows=rows(full);equality=float(np.max(abs(self.equal(full,gap))));s=self.state(full,0.)
        original_camera=float(self.camera.original_rows(s['root'],s['fk'],min(.00601,max(0.,gap))).min())
        diagnostic=dict(stage=label,success=bool(fit.success),iterations=int(fit.nit),
            equality=equality,inequality=float(np.min(allrows)),message=str(fit.message),
            foot_residual=float(np.max(abs(s['foot_error']))),gap_m=float(np.mean(np.einsum('bi,bi->b',s['support_points'],self.support_normals)+self.support_offsets)),support_gaps_m=(np.einsum('bi,bi->b',s['support_points'],self.support_normals)+self.support_offsets).tolist(),
            camera_minimum_m=original_camera,objective=float(fit.fun),solver_status=int(fit.status),
            readiness_graph=float(full[29]),original_readiness_minimum=float(np.min(allrows[readiness_rows])),
            retained_coordinates_bitwise_unchanged=True,all_original_rows_checked=len(allrows))
        if endpoint:
            labels=self.endpoint_row_labels();assert len(labels)==len(allrows)
            diagnostic['negative_rows']=[dict(row=i,name=labels[i],residual=float(v)) for i,v in enumerate(allrows) if v<-1e-6]
        self.optimizations.append(diagnostic)
        if not fit.success or equality>.002 or np.min(allrows)<-1e-6 or original_camera<-1e-6:
            raise ValueError(dict(reason='GRAPH_FULL_ORIGINAL_REJECTED',**diagnostic))
        return full
