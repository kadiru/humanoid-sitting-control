"""One contact-driven XY placement from an admitted pose and measured depth.

The object is translated, never scaled, rotated or raised. A labelled z-buffer
reprojects only measured triangles into the original fixed rear camera. Missing
surfaces remain missing. A fresh physical reset/image is still mandatory.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import apply,transform
from perception import cloud
from paired_contact import BODIES,upper_vertices


def render(vertices,faces,labels,K,T,R,h,w):
    C=(vertices@R-T[:3,3])@T[:3,:3];z=C[:,2]
    uv=C[:,:2]/np.maximum(z[:,None],1e-12)*[K[0,0],K[1,1]]+[K[0,2]-.5,K[1,2]-.5]
    image=np.full((h,w),np.inf);tag=np.full((h,w),-1,int)
    for fi,face in enumerate(faces):
        if min(z[face])<=.08:continue
        p=uv[face];lo=np.maximum(np.ceil(p.min(0)).astype(int),0);hi=np.minimum(np.floor(p.max(0)).astype(int),[w-1,h-1])
        if np.any(lo>hi):continue
        A=np.c_[p[1]-p[0],p[2]-p[0]]
        if abs(np.linalg.det(A))<1e-10:continue
        yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1];xy=np.c_[xx.ravel(),yy.ravel()]
        a,b=((xy-p[0])@np.linalg.inv(A).T).T
        inside=(a>=-1e-10)&(b>=-1e-10)&(a+b<=1+1e-10)
        depth=1/np.maximum((1-a-b)/z[face[0]]+a/z[face[1]]+b/z[face[2]],1e-30)
        old=image[yy.ravel(),xx.ravel()];replace=inside&(depth<old)
        image[yy.ravel()[replace],xx.ravel()[replace]]=depth[replace]
        tag[yy.ravel()[replace],xx.ravel()[replace]]=int(labels[fi])
    image[~np.isfinite(image)]=np.nan
    return image,tag


def construct(model,shapes,q,reference_q,R,depth,K,T,expanded_pixels,goal):
    feet=[transform(R)@model.fk(q)[s+'_ankle_roll_link'] for s in ('left','right')]
    fk=model.fk(reference_q);root=feet[0]@np.linalg.inv(fk['left_ankle_roll_link'])
    points=[]
    for b in BODIES:
        V=upper_vertices(model,shapes,b);W=apply(root@fk[b],V)
        points.append(W[W[:,2]<=W[:,2].min()+1e-10].mean(0))
    points=np.asarray(points)
    shift=np.r_[points.mean(0)[:2]-goal[:2],0.]
    if np.linalg.norm(shift[:2])>.35:raise ValueError('PLACEMENT_EXCEEDS_FROZEN35CM_DOMAIN')
    P,valid=cloud(depth,K,T);h,w=valid.shape
    valid &= model.self_mask(P.reshape(-1,3),model.fk(q),margin=.009).reshape(h,w)
    V=P.reshape(-1,3)@R.T;ids=np.arange(h*w).reshape(h,w)
    faces=np.concatenate([np.stack([ids[:-1,:-1],ids[:-1,1:],ids[1:,:-1]],-1).reshape(-1,3),
                          np.stack([ids[:-1,1:],ids[1:,1:],ids[1:,:-1]],-1).reshape(-1,3)])
    top=np.zeros((h,w),bool);top[expanded_pixels[:,0],expanded_pixels[:,1]]=True
    tri=V[faces];normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(normals,axis=1)
    floor=min(apply(f,model.vertices[s+'_ankle_roll_link'])[:,2].min() for f,s in zip(feet,('left','right')))
    ok=valid.ravel()[faces].all(1)&(length>1e-10)&(tri[:,:,2].min(1)>floor+.055)
    for i,j in [(0,1),(1,2),(2,0)]:ok &= np.linalg.norm(tri[:,i]-tri[:,j],axis=1)<=.05
    labels=(top.ravel()[faces].all(1)&(abs(normals[:,2])/np.maximum(length,1e-30)>.97)).astype(int)
    faces=faces[ok];labels=labels[ok]
    shifted=V+shift
    image,tag=render(shifted,faces,labels,K,T,R,h,w)
    report=dict(shift_gravity_m=shift.tolist(),original_goal=goal.tolist(),reference_support_points=points.tolist(),
        observed_triangles=len(faces),observed_top_triangles=int(labels.sum()),
        rendered_pixels=int(np.isfinite(image).sum()),rendered_top_pixels=int((tag==1).sum()),
        translation_only=True,height_unchanged=True,truth_input=False,original_camera_extrinsics=True,
        current_physical_camera_validation=False,physical_success=False)
    return shift,image,tag,report


def bind_target(camera,tag):
    a=tag[:-1,:-1];b=tag[:-1,1:];c=tag[1:,:-1];d=tag[1:,1:]
    target=np.stack([(a==1)&(b==1)&(c==1),(b==1)&(d==1)&(c==1)],axis=-1)
    # A curved observed top is a union of real local facets, not a plane band.
    camera.target=target&camera.valid&~camera.ground&(camera.N[:,:, :,2]>.97)
    if camera.target.sum()<30:raise ValueError('INSUFFICIENT_TRANSLATED_OBSERVED_TOP')

