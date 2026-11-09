"""Differentiable-within-facet camera clearance rows for the path optimizer.

Every row has a fixed robot material sample. Camera facets change only when its
projection crosses an observed pixel triangle. No mesh truth, contact truth or
root world pose enters this calculation. Unknown pixels do not certify space;
the independent full mesh gate is still mandatory.
"""
import numpy as np
from scipy.spatial import ConvexHull
from humanoid_sitting.shared_frame import apply
from perception import cloud

class CameraClearance:
    def __init__(self,model,shapes,depth,K,T,q,R0,normal,offset,body):
        if not np.isfinite(K).all() or not np.isfinite(T).all():raise ValueError("NONFINITE_CAMERA_CALIBRATION")
        P,valid=cloud(depth,K,T);h,w=valid.shape
        valid &= model.self_mask(P.reshape(-1,3),model.fk(q),margin=.009).reshape(h,w)
        self.K=np.asarray(K);self.T=np.asarray(T);self.R0=R0;self.h=h;self.w=w
        # All observed neighbouring triangles; no plane-only torso exemption.
        pts=P@R0.T;ids=np.arange(h*w).reshape(h,w)
        a=ids[:-1,:-1];b=ids[:-1,1:];c=ids[1:,:-1];d=ids[1:,1:]
        triangles=np.stack([np.stack([a,b,c],-1),np.stack([b,d,c],-1)],-2)
        V=pts.reshape(-1,3)[triangles];ok=valid.ravel()[triangles].all(-1)
        for i,j in [(0,1),(1,2),(2,0)]:ok &= np.linalg.norm(V[...,i,:]-V[...,j,:],axis=-1)<=.05
        normals=np.cross(V[...,1,:]-V[...,0,:],V[...,2,:]-V[...,0,:]);length=np.linalg.norm(normals,axis=-1)
        ok &= length>1e-10;normals/=np.maximum(length[...,None],1e-30)
        camera=R0@T[:3,3];normals=np.where((np.sum(normals*(camera-V[...,0,:]),axis=-1)<0)[...,None],-normals,normals)
        if np.count_nonzero(ok)<30:raise ValueError("INSUFFICIENT_OBSERVED_TRIANGLES")
        self.N=normals;self.D=-np.sum(normals*V[...,0,:],axis=-1);self.valid=ok
        fk=model.fk(q)
        floor=min((apply(fk[side+'_ankle_roll_link'],model.vertices[side+'_ankle_roll_link'])@R0.T)[:,2].min() for side in ('left','right'))
        self.ground=ok&(np.max(abs(V[...,2]-floor),axis=-1)<=.01)&(normals[...,2]>.97)
        self.target=(np.max(abs(V@normal+offset),axis=-1)<=.002)&(np.sum(normals*normal,axis=-1)>.97)&ok
        self.samples={};self.body=body
        for shape in shapes:
            v=np.asarray(shape['vertices']);f=ConvexHull(v);tri=v[f.simplices]
            # Vertices, face centres and edge midpoints. Full exact hull/mesh
            # collision audit remains independent of these construction rows.
            samples=np.concatenate([v[f.vertices],tri.mean(1),((tri+np.roll(tri,-1,axis=1))*.5).reshape(-1,3)])
            self.samples.setdefault(shape['body'],[]).append(samples)
        self.samples={b:np.unique(np.concatenate(v),axis=0) for b,v in self.samples.items()}
        self.count=sum(len(v) for v in self.samples.values())

    def rows(self,root,fk,required_gap):
        points=np.concatenate([apply(root@fk[b],v) for b,v in self.samples.items()])
        is_support=np.concatenate([np.full(len(v),b in getattr(self,'support_bodies',(self.body,)),bool) for b,v in self.samples.items()])
        feet=np.concatenate([np.full(len(v),b in ('left_ankle_roll_link','right_ankle_roll_link'),bool) for b,v in self.samples.items()])
        # O -> startup pelvis -> fixed ROS camera.
        C=(points@self.R0-self.T[:3,3])@self.T[:3,:3]
        z=C[:,2];safe=np.where(z>.08,z,1.)
        u=C[:,0]/safe*self.K[0,0]+self.K[0,2]-.5;v=C[:,1]/safe*self.K[1,1]+self.K[1,2]-.5
        col=np.floor(u).astype(int);row=np.floor(v).astype(int)
        inside=(z>.08)&(row>=0)&(row<self.h-1)&(col>=0)&(col<self.w-1)
        rr=np.clip(row,0,self.h-2);cc=np.clip(col,0,self.w-2);side=((u-col)+(v-row)>1).astype(int)
        known=inside&self.valid[rr,cc,side]&~(feet&self.ground[rr,cc,side]);plane=self.N[rr,cc,side];offset=self.D[rr,cc,side]
        gap=np.sum(points*plane,axis=1)+offset
        target=is_support&self.target[rr,cc,side]
        clearance=np.where(target,max(0.,required_gap),.006)
        return np.where(known,gap-clearance,1.)

def rasterize(vertices,faces,K,T,R0,h,w):
    """Rigid saved-point augmentation only; fresh real depth required at reset."""
    C=(vertices@R0-T[:3,3])@T[:3,:3];z=C[:,2]
    uv=C[:,:2]/np.maximum(z[:,None],1e-12)*[K[0,0],K[1,1]]+[K[0,2]-.5,K[1,2]-.5]
    image=np.full((h,w),np.inf)
    for face in faces:
        if min(z[face])<=.08:continue
        p=uv[face];lo=np.maximum(np.ceil(p.min(0)).astype(int),0);hi=np.minimum(np.floor(p.max(0)).astype(int),[w-1,h-1])
        if np.any(lo>hi):continue
        yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1];xy=np.c_[xx.ravel(),yy.ravel()]
        A=np.c_[p[1]-p[0],p[2]-p[0]];det=np.linalg.det(A)
        if abs(det)<1e-10:continue
        weights=(xy-p[0])@np.linalg.inv(A).T;a,b=weights.T
        valid=(a>=-1e-10)&(b>=-1e-10)&(a+b<=1+1e-10)
        inverse=(1-a-b)/z[face[0]]+a/z[face[1]]+b/z[face[2]];depth=1/np.maximum(inverse,1e-30)
        old=image[yy.ravel(),xx.ravel()];image[yy.ravel(),xx.ravel()]=np.where(valid,np.minimum(old,depth),old)
    image[~np.isfinite(image)]=np.nan
    return image
