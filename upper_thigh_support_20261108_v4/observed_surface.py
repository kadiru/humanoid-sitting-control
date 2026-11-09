"""Contact association on one connected, actually observed target surface."""
from collections import deque, OrderedDict
import numpy as np
from perception import cloud
from contact_facet import edges_for_triangle
from humanoid_sitting.shared_frame import apply


def connected(mask, start):
    """Pixel-triangle edge adjacency, not diagonal or nearest-neighbour fill."""
    seen=set();todo=deque([tuple(start)])
    while todo:
        r,c,s=todo.popleft()
        if (r,c,s) in seen or not (0<=r<mask.shape[0] and 0<=c<mask.shape[1]):continue
        if not mask[r,c,s]:continue
        seen.add((r,c,s))
        todo.extend([(r,c,1),(r-1,c,1),(r,c-1,1)] if s==0
                    else [(r,c,0),(r+1,c,0),(r,c+1,0)])
    return seen


class ObservedPatch:
    def __init__(self,camera,depth,start):
        self.camera=camera
        self.component=connected(camera.target & camera.valid & ~camera.ground,start)
        if tuple(start) not in self.component:raise ValueError('ORIGINAL_CONTACT_COMPONENT_MISSING')
        self.start=tuple(start)
        P,_=cloud(depth,camera.K,camera.T);self.P=P@camera.R0.T
        self.facets={};self.queries=0;self.visited=set()

    def facet(self,key):
        if key not in self.component:raise ValueError(('CONTACT_OUTSIDE_OBSERVED_COMPONENT',key))
        if key not in self.facets:
            r,c,s=key
            ij=[(r,c),(r,c+1),(r+1,c)] if s==0 else [(r,c+1),(r+1,c+1),(r+1,c)]
            V=np.array([self.P[a,b] for a,b in ij]);N=self.camera.N[key].copy();d=float(self.camera.D[key])
            if np.max(abs(V@N+d))>1e-9:raise ValueError('PATCH_PLANE_VERTEX_MISMATCH')
            self.facets[key]=(N,d,edges_for_triangle(V,N),V)
        return self.facets[key]

    def query(self,point):
        c=self.camera;C=(point@c.R0-c.T[:3,3])@c.T[:3,:3]
        if C[2]<=.08 or not np.isfinite(C).all():raise ValueError('PATCH_CONTACT_NOT_VISIBLE')
        uv=C[:2]/C[2]*[c.K[0,0],c.K[1,1]]+[c.K[0,2]-.5,c.K[1,2]-.5]
        col,row=np.floor(uv).astype(int);key=(int(row),int(col),int(sum(uv-[col,row])>1.))
        N,d,edges,V=self.facet(key)
        center=c.R0@c.T[:3,3];ray=point-center;slope=float(N@ray)
        if slope>=0:raise ValueError('PATCH_CONTACT_RAY_REVERSED')
        t=-(N@center+d)/slope;hit=center+t*ray
        if t<=0 or abs(N@hit+d)>1e-9 or np.min(edges[:,:3]@hit+edges[:,3])<-1e-9:
            raise ValueError('PATCH_CONTACT_RAY_MEMBERSHIP')
        self.queries+=1;self.visited.add(key)
        return N,d,edges,dict(pixel=list(key),normal=N.tolist(),offset=d,
                              ray_contact=hit.tolist(),vertices=V.tolist())

    def report(self):
        V=np.concatenate([self.facet(k)[3] for k in sorted(self.component)])
        initial=self.facet(self.start)[3]
        return dict(connected_triangles=len(self.component),initial_triangle=list(self.start),
                    patch_extent_m=(V.max(0)-V.min(0)).tolist(),
                    initial_triangle_extent_m=(initial.max(0)-initial.min(0)).tolist(),
                    queries=self.queries,visited_triangles=len(self.visited),
                    original_target_mask_unchanged=True,physical_success=False)


class PatchContactMixin:
    patch_active=False

    def activate_contact(self,x):
        if not self.patch_active:return
        key=np.asarray(x[:29]).tobytes()
        if key not in self.patch_cache:
            _,root,fk=self.pose(x);point=apply(root@fk[self.body],self.marker)
            self.patch_cache[key]=self.observed_patch.query(point)
            if len(self.patch_cache)>128:self.patch_cache.popitem(last=False)
        self.n,self.d,self.contact_edges,self.current_facet=self.patch_cache[key]

    def state(self,x,load):
        self.activate_contact(x)
        return super().state(x,load)

    def jets(self,x,load):
        self.activate_contact(x)
        return super().jets(x,load)

    def optimize(self,x,gap,rows,cost,bounds,label):
        if label!='load_transfer_endpoint':return super().optimize(x,gap,rows,cost,bounds,label)
        if not self.contact_frozen:raise ValueError('PATCH_REQUIRES_FIXED_MATERIAL')
        self.patch_active=True;self.patch_cache=OrderedDict()
        self.states.clear();self.derivative_cache.clear()
        try:
            result=super().optimize(x,gap,rows,cost,bounds,label)
            self.activate_contact(result)
            self.record('accepted_contact_facet',dict(self.current_facet,association=self.observed_patch.report()))
            return result
        finally:
            self.patch_active=False
            self.states.clear();self.derivative_cache.clear()
            self.record('patch_query_receipt',self.observed_patch.report())
