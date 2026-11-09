"""Five sufficient halfspaces for avoiding a finite camera-ray region.

For p=z*[u,v,1], each original facet margin is z*N.ray+alpha-delta.
The front-depth cutoff is the minimum positive intersection over all facet
planes and all rays in this pixel rectangle. No averaging of depth or normals.
"""
import numpy as np

def certificate_planes(bounds,normals,alpha,margin=.006):
    left,right,top,bottom=map(float,bounds)
    if not left<right or not top<bottom:raise ValueError('EMPTY_PIXEL_RECTANGLE')
    N=np.asarray(normals,float);alpha=np.asarray(alpha,float)
    if not len(N) or not np.isfinite(N).all() or not np.isfinite(alpha).all():
        raise ValueError('INVALID_FACET_PLANES')
    if np.any(alpha<=margin):raise ValueError('CAMERA_WITHIN_OR_BEHIND_OFFSET_PLANE')
    rays=np.array([[x,y,1.] for x in (left,right) for y in (top,bottom)])
    denominator=np.max(-N@rays.T,axis=1)
    active=denominator>0
    if not np.any(active):raise ValueError('NO_FORWARD_FACET_INTERSECTION')
    depths=(alpha[active]-margin)/denominator[active]
    index=int(np.argmin(depths));cut=float(depths[index])
    if not np.isfinite(cut) or cut<=0:raise ValueError('NONPOSITIVE_FRONT_CUTOFF')
    # Each row separately implies safety for the entire selected tile.
    A=np.array([[-1.,0.,left],[1.,0.,-right],[0.,-1.,top],[0.,1.,-bottom],[0.,0.,-1.]])
    b=np.array([0.,0.,0.,0.,cut])
    length=np.linalg.norm(A,axis=1)
    A=A/length[:,None];b=b/length
    # Pixel cells have included edges: equality on an angular side is NOT an
    # outside certificate. A positive inward reserve enforces strict exclusion.
    b[:4]-=.00001
    return A,b,dict(front_depth_m=cut,limiting_facet_in_input=int(np.flatnonzero(active)[index]))
