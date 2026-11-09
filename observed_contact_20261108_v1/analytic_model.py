"""Forward derivatives of the existing full-body model; no changed equations.

Original NumPy/SciPy values and independent finite differences remain the
validator. Camera derivatives follow the actual selected material/facet row.
"""
import time
from collections import OrderedDict
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

D=31
def skew(v):
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])

def right_jacobian(v,inverse=False):
    t=float(np.linalg.norm(v));K=skew(v)
    if inverse:
        a=1/12+t*t/720 if t<1e-4 else 1/t**2-(1+np.cos(t))/(2*t*np.sin(t))
        return np.eye(3)+.5*K+a*K@K
    a=.5-t*t/24 if t<1e-4 else (1-np.cos(t))/t**2
    b=1/6-t*t/120 if t<1e-4 else (t-np.sin(t))/t**3
    return np.eye(3)-a*K+b*K@K

def vee(M):return np.array([M[2,1]-M[1,2],M[0,2]-M[2,0],M[1,0]-M[0,1]])*.5
def point(T,J,p):return T[:3,:3]@p+T[:3,3],np.einsum('ijk,j->ik',J[:3,:3],p)+J[:3,3]
def cross_derivative(a,da,b,db):
    return np.cross(da.T,b).T+np.cross(a,db.T).T

class AnalyticMixin:
    def __init__(self,*a,**kw):
        self.derivative_cache=OrderedDict()
        super().__init__(*a,**kw)

    def forward_jets(self,x):
        q,root,fk=self.pose(x)
        Jroot=np.zeros((4,4,D));Jroot[:3,3,23:26]=np.eye(3)
        Jr=right_jacobian(x[26:29]);R=root[:3,:3]
        for i in range(3):Jroot[:3,:3,26+i]=R@skew(Jr[:,i])
        Jfk={'pelvis':np.zeros((4,4,D))};pending=self.model.joints.copy()
        while pending:
            remaining=[]
            for j in pending:
                if not j['body0'] or j['body0'][0] not in Jfk:
                    remaining.append(j);continue
                parent=j['body0'][0];body=j['body1'][0];A,B=self.model.frames[j['name']]
                index=self.model.names.index(j['name']) if j['name'] in self.model.names else None
                axis=np.eye(3)['XYZ'.index(j['axis'])] if index is not None else np.zeros(3)
                angle=q[index] if index is not None else 0.
                Rq=np.eye(4);Rq[:3,:3]=Rotation.from_rotvec(axis*angle).as_matrix()
                J=np.einsum('ijk,jl->ilk',Jfk[parent],A@Rq@B)
                if index is not None and index<23:
                    dRq=np.zeros((4,4));dRq[:3,:3]=Rq[:3,:3]@skew(axis)
                    J[:,:,index]+=fk[parent]@A@dRq@B
                Jfk[body]=J
            if len(remaining)==len(pending):raise ValueError('DISCONNECTED_DERIVATIVE_TREE')
            pending=remaining
        worlds={b:root@T for b,T in fk.items()}
        Jworlds={b:np.einsum('ijk,jl->ilk',Jroot,T)+np.einsum('ij,jlk->ilk',root,Jfk[b]) for b,T in fk.items()}
        return worlds,Jworlds,Jroot

    def jets(self,x,load):
        key=(np.asarray(x).tobytes(),float(load),bool(self.contact_frozen),self.marker.tobytes())
        if key in self.derivative_cache:return self.derivative_cache[key]
        s=self.state(x,load);world,Jworld,Jroot=self.forward_jets(x)
        marker=self.marker if self.contact_frozen else self.exposed_local(s['root'],s['fk'])
        p,dp=point(world[self.body],Jworld[self.body],marker)
        centers=[];dcenters=[]
        for b,c in zip(self.names,self.inertia['local_com']):
            pc,jc=point(world[b],Jworld[b],np.asarray(c));centers.append(pc);dcenters.append(jc)
        centers=np.asarray(centers);dcenters=np.asarray(dcenters)
        dc=np.einsum('b,bij->ij',self.mass,dcenters)/self.mass.sum()
        rest=self.W*load*self.n;ground=np.array([0.,0.,self.W])-rest
        dm=np.cross(dc.T,[0.,0.,self.W]).T-np.cross(dp.T,rest).T
        dcop=np.zeros((3,D));dcop[0]=-dm[1]/ground[2];dcop[1]=dm[0]/ground[2]
        yl,yr=self.foot_y;dfz=dcop[1]*ground[2]/(yl-yr)
        dfx=(dcop[0]*ground[1]-dm[2])/(yl-yr)
        dleft=np.array([dfx,ground[1]/ground[2]*dfz,dfz]);dforces=[dleft,-dleft,np.zeros((3,D))]
        dfoot=np.zeros((3,D));dfoot[0]=dcop[0]
        dpoints=[dfoot,dfoot,dp]
        dtau=[]
        for name in self.model.names[:23]:
            j=self.joints[name];parent=j['body0'][0];A=self.model.frames[name][0]
            frame=world[parent]@A;Jframe=np.einsum('ijk,jl->ilk',Jworld[parent],A)
            k='XYZ'.index(j['axis']);axis=frame[:3,k];daxis=Jframe[:3,k];origin=frame[:3,3];dorigin=Jframe[:3,3]
            total=np.zeros(D)
            for b,m,c,dc_i in zip(self.names,self.mass,centers,dcenters):
                if b in self.desc[name]:total+=m*9.81*cross_derivative(axis,daxis,c-origin,dc_i-dorigin)[2]
            for (b,cp,F),dcp,dF in zip(s['contacts'],dpoints,dforces):
                if b in self.desc[name]:
                    moment=np.cross(axis,cp-origin)
                    total-=F@cross_derivative(axis,daxis,cp-origin,dcp-dorigin)+moment@dF
            dtau.append(total)
        dtau=np.asarray(dtau);dcmd=dtau/self.K[:,None];dcmd[:,:23]+=np.eye(23)
        dfeet=[]
        for b,anchor in zip(self.foot_bodies,self.feet):
            dfeet.extend(Jworld[b][:3,3])
            rel=anchor[:3,:3].T@world[b][:3,:3]
            v=Rotation.from_matrix(rel).as_rotvec();Jinv=right_jacobian(v,inverse=True)
            Jrel=np.einsum('ij,jlk->ilk',anchor[:3,:3].T,Jworld[b][:3,:3])
            omega=np.array([vee(rel.T@Jrel[:,:,i]) for i in range(D)]).T
            dfeet.extend(Jinv@omega)
        value=dict(world=world,Jworld=Jworld,Jroot=Jroot,point=dp,com=dc,cop=dcop,
                   forces=dforces[:2],foot_points=[dfoot,dfoot],tau=dtau,command=dcmd,feet=np.asarray(dfeet))
        self.derivative_cache[key]=value
        if len(self.derivative_cache)>128:self.derivative_cache.popitem(last=False)
        return value

    def mechanical_jac(self,x,load,bias=None):
        s=self.state(x,load);j=self.jets(x,load);rows=[]
        for eq,F,dpt,dF in zip(self.foot_polygons,s['foot_forces'],j['foot_points'],j['forces']):
            rows.extend(-eq[:,:2]@dpt[:2]);rows.append(dF[2]/self.W)
            norm=np.linalg.norm(F[:2]);normgrad=F[:2]@dF[:2]/norm if norm>0 else np.zeros(D)
            rows.append((.5*dF[2]-normgrad)/self.W)
        ready=j['command']/(.15*14.4);epi=np.eye(D)[29]
        return np.vstack([rows,j['command'],-j['command'],epi-ready,epi+ready])

    def camera_jac(self,x,gap):
        """Exact derivative of each original body's current minimum row."""
        c=self.camera;s=self.state(x,0.);j=self.jets(x,0.);out=[];values=[]
        if hasattr(c,'jacobian'):return c.jacobian(j,min(.00601,max(0.,gap)))
        required=min(.00601,max(0.,gap))
        for body,samples in c.samples.items():
            pose=j['world'][body]
            M=c.T[:3,:3].T@c.R0.T@pose[:3,:3]
            t=c.T[:3,:3].T@(c.R0.T@pose[:3,3]-c.T[:3,3])
            C=samples@M.T+t;z=C[:,2];safe=np.where(z>.08,z,1.)
            u=C[:,0]/safe*c.K[0,0]+c.K[0,2]-.5;v=C[:,1]/safe*c.K[1,1]+c.K[1,2]-.5
            col=np.floor(u).astype(int);row=np.floor(v).astype(int)
            inside=(z>.08)&(row>=0)&(row<c.h-1)&(col>=0)&(col<c.w-1)
            ids=np.flatnonzero(inside)
            if len(ids):
                rr=row[ids];cc=col[ids];side=((u[ids]-cc)+(v[ids]-rr)>1).astype(int)
                known=c.valid[rr,cc,side].copy()
                if body in self.foot_bodies:known &= ~c.ground[rr,cc,side]
                ids=ids[known];rr=rr[known];cc=cc[known];side=side[known]
            if not len(ids):values.append(1.);out.append(np.zeros(D));continue
            normals=c.N[rr,cc,side];offsets=c.D[rr,cc,side]
            target=(body==self.body)&c.target[rr,cc,side]
            margin=np.where(target,required,.006)
            points=samples[ids]@pose[:3,:3].T+pose[:3,3]
            residual=np.sum(points*normals,axis=1)+offsets-margin;k=int(np.argmin(residual))
            if len(ids)<len(samples) and residual[k]>=1.:
                values.append(1.);out.append(np.zeros(D));continue
            _,dp=point(pose,j['Jworld'][body],samples[ids[k]])
            values.append(float(residual[k]));out.append(normals[k]@dp)
        if np.max(abs(np.asarray(values)-self.camera_rows(x,gap)))>1e-10:raise ValueError('CAMERA_JAC_ROW_VALUE_MISMATCH')
        return np.asarray(out)

    def geometry_jac(self,x,gap):
        j=self.jets(x,0.);dp=j['point'];uv=np.array([self.e@dp,self.b@dp])
        v=self.mesh[self.body];J=j['Jworld'][self.body]
        dv=np.einsum('ijk,vj->vik',J[:3,:3],v)+J[:3,3]
        return np.vstack([j['Jroot'][2,2],-self.patch[:,:2]@uv,
                          np.einsum('i,vik->vk',self.n,dv),self.camera_jac(x,gap),
                          (self.contact_edges[:,:3]@dp) if gap<=.01200001 else np.empty((0,D))])

    def equal_jac(self,x,gap):
        j=self.jets(x,0.)
        return np.vstack([j['feet'],self.n@j['point']])

    def posture_jac(self,x,load):
        s=self.state(x,load);j=self.jets(x,load)
        p,dp=point(j['world'][self.body],j['Jworld'][self.body],self.preference_marker)
        tangent=p-self.target;tangent-=self.n*(tangent@self.n)
        grad=2*s['tau']@j['tau']/self.standing_effort
        grad[29]+=2*x[29];grad[:23]+=.1*(x[:23]-self.q0[:23])
        grad+=.1*tangent@dp/.15**2
        if s['root'][2,3]<0:grad[25]+=2*s['root'][2,3]/max(.2,-self.floor)**2
        return grad

    def load_seconds_jac(self,x):
        result=np.zeros(D)
        for a,b in zip(self.loads,self.loads[1:]):
            delta=self.command(x,float(b))-self.command(x,float(a));k=int(np.argmax(abs(delta)))
            if abs(delta[k])/.125>.25:
                result+=np.sign(delta[k])*(self.jets(x,float(b))['command'][k]-self.jets(x,float(a))['command'][k])/.125
        return result

    def endpoint_jac(self,x):
        delta=self.command(x,0.)-self.previous[:23];k=int(np.argmax(abs(delta)))
        travel=np.zeros(D)
        if abs(delta[k])/.125>.012/.0055+.5:travel=np.sign(delta[k])*self.jets(x,0.)['command'][k]/.125
        return np.vstack([self.geometry_jac(x,0.),*[self.mechanical_jac(x,float(l)) for l in self.loads],
                          -self.load_seconds_jac(x)-travel])

    def connection_jac(self,x):
        c=self.connection_context;j=self.jets(x,0.);time_row=np.eye(D)[30]
        rows=[self.geometry_jac(x,c['gap']),self.mechanical_jac(x,0.,c['bias']),
              .125*time_row-j['command'],.125*time_row+j['command']]
        if c['normal']:
            eye=np.eye(D)[:23];rows.extend([.05*time_row-eye,.05*time_row+eye])
        delta=self.command(x,0.,c['bias'])-self.previous[:23];k=int(np.argmax(abs(delta)))
        remaining=np.zeros(D)
        if abs(delta[k])/.125>max(0.,.012-c['gap'])/.0055:remaining=np.sign(delta[k])*j['command'][k]/.125
        rows.extend([time_row,-time_row-remaining])
        return np.vstack(rows)

    def connection_cost_jac(self,x):
        g=self.posture_jac(x,0.);g[:29]+=.2*(x[:29]-self.connection_context['reference']);g[30]+=.01
        return g

    def audit_derivatives(self,x):
        # Original implementation is the independent numerical evaluator.
        # Audit force/kinematic primitives and smooth objective at ORIGINAL
        # analytic seed. Camera's selected-row values are checked separately.
        h=1e-6;analytic=self.jets(x,.2)
        pieces=[]
        for k in range(29):
            a=x.copy();b=x.copy();a[k]-=h;b[k]+=h
            sa=self.state(a,.2);sb=self.state(b,.2)
            pieces.append(np.r_[(sb['tau']-sa['tau'])/(2*h),(sb['foot_error']-sa['foot_error'])/(2*h),
                               (sb['point']-sa['point'])/(2*h)])
        numeric=np.asarray(pieces).T
        expected=np.vstack([analytic['tau'],analytic['feet'],analytic['point']])[:,:29]
        error=float(np.max(abs(numeric-expected)));scaled=float(np.max(abs(numeric-expected)/(1+abs(numeric))))
        grad=self.posture_jac(x,.2);numerical_cost=np.zeros(D)
        for k in range(D):
            a=x.copy();b=x.copy();a[k]-=h;b[k]+=h
            numerical_cost[k]=(self.posture_cost(b,.2)-self.posture_cost(a,.2))/(2*h)
        cost_error=float(np.max(abs(grad-numerical_cost)))
        camera_shape=self.camera_jac(x,0.).shape
        result=dict(passed=bool(error<2e-5 and scaled<2e-6 and cost_error<2e-5),
            maximum_primitive_derivative_error=error,maximum_scaled_error=scaled,
            maximum_objective_derivative_error=cost_error,camera_rows=camera_shape[0],
            source='Original analytic seed only; independent central differences of original model',physics_steps=0)
        self.record('derivative_audit',result)
        if not result['passed']:raise ValueError(('ANALYTIC_DERIVATIVE_AUDIT_FAILED',result))

    def optimize(self,x,gap,rows,cost,bounds,label):
        endpoint=label=='load_transfer_endpoint'
        if endpoint:self.audit_derivatives(x)
        jac=self.endpoint_jac if endpoint else self.connection_jac
        costjac=(lambda y:self.posture_jac(y,.2)) if endpoint else self.connection_cost_jac
        fit=minimize(cost,x.copy(),jac=costjac,method='SLSQP',bounds=bounds,
            constraints=[dict(type='eq',fun=lambda y:self.equal(y,gap),jac=lambda y:self.equal_jac(y,gap)),
                         dict(type='ineq',fun=rows,jac=jac)],options={'maxiter':60,'ftol':1e-9})
        eq=float(np.max(abs(self.equal(fit.x,gap))));ineq=float(np.min(rows(fit.x)));s=self.state(fit.x,0.)
        diagnostic=dict(stage=label,success=bool(fit.success),iterations=int(fit.nit),equality=eq,inequality=ineq,
            message=str(fit.message),foot_residual=float(np.max(abs(s['foot_error']))),
            gap_m=float(s['point']@self.n+self.d),camera_minimum_m=float(self.camera.original_rows(s['root'],s['fk'],min(.00601,max(0.,gap))).min()),
            construction_clearance_minimum_m=float(self.camera_rows(fit.x,gap).min()),
            objective=float(fit.fun),solver_status=int(fit.status),derivatives='analytic full model, piecewise exact active camera row')
        self.optimizations.append(diagnostic)
        if not fit.success or eq>.002 or ineq<-1e-6 or diagnostic['camera_minimum_m']<-1e-6:raise ValueError(dict(reason='ANALYTIC_NODE_REJECTED',**diagnostic))
        return fit.x.copy()
