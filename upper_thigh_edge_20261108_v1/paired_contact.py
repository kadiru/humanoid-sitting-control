"""Two legal proximal-thigh contacts with independent observed normals."""
from collections import OrderedDict
import numpy as np
from scipy.spatial import ConvexHull,HalfspaceIntersection
from humanoid_sitting.shared_frame import apply

BODIES=('left_hip_yaw_link','right_hip_yaw_link')
GUARDS=('left_hip_pitch_link','right_hip_pitch_link')

def upper_vertices(model,shapes,body):
    side=body.split('_')[0]
    knee=next(j for j in model.joints if j['name']==side+'_knee_joint')
    assert knee['body0'][0]==body
    axis=np.asarray(knee['p0'],float)
    V=np.concatenate([s['vertices'] for s in shapes if s['body']==body])
    fraction=V@axis/(axis@axis)
    result=V[(fraction>=0.)&(fraction<=.5)]
    if len(result)<3:raise ValueError(('NO_PROXIMAL_THIGH_HULL',body))
    return result

class PairedContactMixin:
    patch_active=False
    support_bodies=BODIES

    def __init__(self,*a,**kw):
        self.upper_mesh={b:upper_vertices(a[0],a[2],b) for b in BODIES}
        self.support_normals=np.tile(np.asarray(a[6],float),(2,1))
        self.support_normals/=np.linalg.norm(self.support_normals,axis=1)[:,None]
        self.support_offsets=np.repeat(float(a[7]),2)
        self.contact_edges=np.zeros((2,3,4))
        super().__init__(*a,**kw)
        self.housing_mesh={}
        for body in GUARDS:
            pieces=[]
            for shape in a[2]:
                if shape['body']!=body:continue
                V=np.asarray(shape['vertices']);hull=ConvexHull(V);equations=hull.equations.copy();equations[:,3]-=.006
                pieces.append(HalfspaceIntersection(equations,V[hull.vertices].mean(0)).intersections)
            if not pieces:raise ValueError(('MISSING_BUFFERED_GUARD',body))
            self.housing_mesh[body]=np.concatenate(pieces)
        self.reset_standing_gaps()

    def exposed_local(self,root,fk):
        result=[]
        for body,N in zip(BODIES,self.support_normals):
            pose=root@fk[body];V=self.upper_mesh[body];projection=V@(pose[:3,:3].T@N)
            result.append(V[projection<=projection.min()+1e-10].mean(0))
        return np.asarray(result)

    def support_points(self,root,fk):
        markers=self.marker if self.contact_frozen else self.exposed_local(root,fk)
        return np.asarray([apply(root@fk[b],m) for b,m in zip(BODIES,markers)])

    def material_point(self,root,fk,load):return self.support_points(root,fk).mean(0)

    def reset_standing_gaps(self):
        _,root,fk=self.pose(self.x0);markers=self.exposed_local(root,fk)
        points=np.asarray([apply(root@fk[b],m) for b,m in zip(BODIES,markers)])
        self.initial_support_gaps=np.einsum('bi,bi->b',points,self.support_normals)+self.support_offsets

    def desired_gaps(self,gap):
        if gap<=.012:return np.repeat(gap,2)
        denominator=float(self.initial_support_gaps.mean())-.012
        if denominator<=0:raise ValueError('PAIRED_START_NO_APPROACH')
        return .012+(gap-.012)/denominator*(self.initial_support_gaps-.012)

    def activate_contact(self,x):
        if not self.patch_active:return
        key=np.asarray(x[:29]).tobytes()
        if key not in self.patch_cache:
            _,root,fk=self.pose(x)
            self.patch_cache[key]=[self.observed_patch.query(apply(root@fk[b],m)) for b,m in zip(BODIES,self.marker)]
            if len(self.patch_cache)>128:self.patch_cache.popitem(last=False)
        facets=self.patch_cache[key]
        self.support_normals=np.array([z[0] for z in facets]);self.support_offsets=np.array([z[1] for z in facets])
        self.contact_edges=np.array([z[2] for z in facets]);self.current_facets=[z[3] for z in facets]

    def state(self,x,load):
        self.activate_contact(x)
        return super().state(x,load)

    def jets(self,x,load):
        self.activate_contact(x)
        return super().jets(x,load)

    def housing_rows(self,x):return np.empty(0)
    def guard_rows_jac(self,x):return np.empty((0,31))

    def geometry_rows(self,x,gap):
        s=self.state(x,0.);P=s['support_points'];gaps=self.desired_gaps(gap)
        insets=np.concatenate([-(self.patch[:,:2]@np.array([p@self.e,p@self.b])+self.patch[:,2])-.022 for p in P])
        edges=np.concatenate([e[:,:3]@p+e[:,3] for e,p in zip(self.contact_edges,P)]) if gap<=.01200001 else np.empty(0)
        return np.r_[s['root'][2,2]-np.cos(np.deg2rad(38)),insets,self.camera_rows(x,gap),edges,self.housing_rows(x)]

    def geometry_jac(self,x,gap):
        j=self.jets(x,0.);insets=[];planes=[]
        for b,dp,N in zip(BODIES,j['support_points'],self.support_normals):
            insets.append(-self.patch[:,:2]@np.array([self.e@dp,self.b@dp]))

        edges=np.vstack([e[:,:3]@dp for e,dp in zip(self.contact_edges,j['support_points'])]) if gap<=.01200001 else np.empty((0,31))
        return np.vstack([j['Jroot'][2,2],*insets,self.camera_jac(x,gap),edges,self.guard_rows_jac(x)])

    def equal(self,x,gap):
        s=self.state(x,0.)
        return np.r_[s['foot_error'],np.einsum('bi,bi->b',s['support_points'],self.support_normals)+self.support_offsets-self.desired_gaps(gap)]

    def equal_jac(self,x,gap):
        j=self.jets(x,0.)
        return np.vstack([j['feet'],*[N@dp for N,dp in zip(self.support_normals,j['support_points'])]])

    def endpoint_row_labels(self):
        labels=['tilt_reserve']+[b+':patch_inset_'+str(i) for b in BODIES for i in range(len(self.patch))]
        labels += ['camera_'+b for b in self.camera.samples]+['contact_facet_edge_'+str(i) for i in range(6)]
        for load in self.loads:
            prefix='load_'+str(float(load))+':'
            for b,poly in zip(self.foot_bodies,self.foot_polygons):
                labels += [prefix+b+':cop_edge_'+str(i) for i in range(len(poly))]+[prefix+b+':normal_load',prefix+b+':friction']
            for group in ['command_lower','command_upper','readiness_positive','readiness_negative']:
                labels += [prefix+group+':'+n for n in self.model.names[:23]]
        return labels+['total_time_reserve']

    def node(self,x,load,bias,phase,seconds):
        result=super().node(x,load,bias,phase,seconds);s=self.state(x,load)
        result.update(support_points=s['support_points'].tolist(),support_bodies=list(BODIES),
                      support_forces=s['support_forces'].tolist())
        return result

    def optimize(self,x,gap,rows,cost,bounds,label):
        if label!='load_transfer_endpoint':return super().optimize(x,gap,rows,cost,bounds,label)
        self.patch_active=True;self.patch_cache=OrderedDict();self.states.clear();self.derivative_cache.clear()
        try:
            result=super().optimize(x,gap,rows,cost,bounds,label)
            self.activate_contact(result)
            self.record('accepted_contact_facets',dict(facets=self.current_facets,patch=self.observed_patch.report()))
            return result
        finally:
            self.patch_active=False;self.states.clear();self.derivative_cache.clear()
            self.record('patch_query_receipt',self.observed_patch.report())

    def audit_derivatives(self,x):
        super().audit_derivatives(x)
        expected=self.equal_jac(x,0.).copy();numeric=[];h=1e-6
        for k in range(29):
            a=x.copy();b=x.copy();a[k]-=h;b[k]+=h
            numeric.append((self.equal(b,0.)-self.equal(a,0.))/(2*h))
        error=float(np.max(abs(np.asarray(numeric).T-expected[:,:29])))
        result=dict(passed=error<2e-5,paired_foot_and_contact_jacobian_error=error,physical_success=False)
        self.record('paired_derivative_audit',result)
        if not result['passed']:raise ValueError(('PAIRED_DERIVATIVE_REJECTED',result))
