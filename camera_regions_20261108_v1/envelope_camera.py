"""Conservative continuous clearance for noncontact bodies, from current depth.

Each observed triangle is moved 6mm along its plane normal while preserving
its camera-ray footprint. The front support plane of these shifted triangles
is a sufficient condition for every original per-pixel plane-clearance row.
Contact pelvis retains the original detailed per-facet rule. No mesh truth.
"""
import numpy as np
from compact_camera import CompactCamera
from perception import cloud


def front_envelope(vertices,normals,offsets,mask,camera,axis,margin=.006):
    V=vertices[mask];N=normals[mask];d=offsets[mask]
    if not len(V):raise ValueError('EMPTY_NAMED_SURFACE_ENVELOPE')
    alpha=N@camera+d
    if np.min(alpha)<=margin:raise ValueError('CAMERA_INSIDE_CLEARANCE_OFFSET')
    cn=float(camera@axis)
    if np.max(V@axis)>=cn-1e-10:raise ValueError('SURFACE_NOT_BEHIND_ENVELOPE_CAMERA')
    # For every triangle, alpha is constant across its three vertices. This
    # homothety gives the exact intersection of its ray cone with its offset
    # plane, not a normal-translated triangle with a changed image footprint.
    shifted=V+(margin/alpha)[:,None,None]*(camera-V)
    bound=float(np.max(shifted@axis))
    if bound>=cn-1e-10:raise ValueError('ENVELOPE_CROSSES_CAMERA')
    return bound,dict(triangles=len(V),minimum_camera_plane_distance_m=float(alpha.min()),
                     maximum_surface_projection_m=float(np.max(V@axis)),
                     offset_envelope_projection_m=bound,margin_m=margin)


class EnvelopeCamera(CompactCamera):
    def __init__(self,model,shapes,depth,K,T,q,R0,normal,offset,body):
        super().__init__(model,shapes,depth,K,T,q,R0,normal,offset,body)
        P,_=cloud(depth,K,T);points=P@R0.T;h,w=depth.shape
        ids=np.arange(h*w).reshape(h,w);a=ids[:-1,:-1];b=ids[:-1,1:];c=ids[1:,:-1];d=ids[1:,1:]
        triangles=np.stack([np.stack([a,b,c],-1),np.stack([b,d,c],-1)],-2)
        V=points.reshape(-1,3)[triangles];self.camera_position=R0@T[:3,3]
        self.axis=np.asarray(normal,float);self.axis/=np.linalg.norm(self.axis)
        self.back_offset,back=front_envelope(V,self.N,self.D,self.valid&~self.ground,self.camera_position,self.axis)
        self.floor_offset=None;floor=None
        if np.any(self.valid&self.ground):
            self.floor_offset,floor=front_envelope(V,self.N,self.D,self.valid&self.ground,self.camera_position,np.array([0.,0.,1.]))
        self.vertices={b:np.concatenate([s['vertices'] for s in shapes if s['body']==b]) for b in self.samples}
        self.back_bodies=[b for b in self.samples if b!=self.body]
        self.floor_bodies=[b for b in self.samples if b not in ('left_ankle_roll_link','right_ankle_roll_link')] if self.floor_offset is not None else []
        self.report=dict(mechanism='Camera-ray offset-triangle front envelope, exact original6mm plane margin',
            current_camera_source=True,actual_furniture_mesh_used=False,back=back,floor=floor,
            back_body_associations=self.back_bodies,floor_body_associations=self.floor_bodies,
            contact_body_original_detailed_rule=self.body,
            optimizer_rows=len(self.back_bodies)+len(self.floor_bodies)+1,
            conservative_not_equivalent=True,physical_success=False)

    def original_rows(self,root,fk,required_gap):
        return super().rows(root,fk,required_gap)

    def pelvis_row(self,pose,required_gap):
        samples=self.samples[self.body]
        M=self.T[:3,:3].T@self.R0.T@pose[:3,:3]
        t=self.T[:3,:3].T@(self.R0.T@pose[:3,3]-self.T[:3,3])
        C=samples@M.T+t;z=C[:,2];safe=np.where(z>.08,z,1.)
        u=C[:,0]/safe*self.K[0,0]+self.K[0,2]-.5;v=C[:,1]/safe*self.K[1,1]+self.K[1,2]-.5
        col=np.floor(u).astype(int);row=np.floor(v).astype(int)
        ids=np.flatnonzero((z>.08)&(row>=0)&(row<self.h-1)&(col>=0)&(col<self.w-1))
        if not len(ids):return 1.,None
        rr=row[ids];cc=col[ids];side=((u[ids]-cc)+(v[ids]-rr)>1).astype(int)
        known=self.valid[rr,cc,side];ids=ids[known];rr=rr[known];cc=cc[known];side=side[known]
        if not len(ids):return 1.,None
        points=samples[ids]@pose[:3,:3].T+pose[:3,3]
        normals=self.N[rr,cc,side];offsets=self.D[rr,cc,side]
        margins=np.where(self.target[rr,cc,side],max(0.,required_gap),.006)
        value=np.sum(points*normals,axis=1)+offsets-margins;k=int(np.argmin(value))
        if len(ids)<len(samples) and value[k]>=1.:return 1.,None
        return float(value[k]),(samples[ids[k]],normals[k])

    def rows(self,root,fk,required_gap):
        values=[]
        for b in self.back_bodies:
            pose=root@fk[b];vertices=self.vertices[b]@pose[:3,:3].T+pose[:3,3]
            values.append(float(np.min(vertices@self.axis)-self.back_offset))
        for b in self.floor_bodies:
            pose=root@fk[b];vertices=self.vertices[b]@pose[:3,:3].T+pose[:3,3]
            values.append(float(np.min(vertices[:,2])-self.floor_offset))
        values.append(self.pelvis_row(root@fk[self.body],required_gap)[0])
        return np.asarray(values)

    def jacobian(self,jets,required_gap):
        from analytic_model import point
        rows=[]
        for bodies,axis in ((self.back_bodies,self.axis),(self.floor_bodies,np.array([0.,0.,1.]))):
            for b in bodies:
                pose=jets['world'][b];v=self.vertices[b]
                k=int(np.argmin((v@pose[:3,:3].T+pose[:3,3])@axis))
                _,derivative=point(pose,jets['Jworld'][b],v[k]);rows.append(axis@derivative)
        _,witness=self.pelvis_row(jets['world'][self.body],required_gap)
        if witness is None:rows.append(np.zeros(31))
        else:
            material,normal=witness;_,derivative=point(jets['world'][self.body],jets['Jworld'][self.body],material)
            rows.append(normal@derivative)
        return np.asarray(rows)
