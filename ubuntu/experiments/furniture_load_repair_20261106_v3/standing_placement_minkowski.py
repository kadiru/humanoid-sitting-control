"""First contact by ray/convex-Minkowski intersection, independently checked.

Preserves the6mm expanded body polytope. Replaces thousands of cold3-variable
LPs with hull intersections and one LP for the earliest-contact witness.
"""
import numpy as np
from scipy.spatial import ConvexHull,HalfspaceIntersection
from scipy.optimize import linprog


def closer_stance(model,shapes,furniture,q,root_rotation,root_position,basis):
    triangles=furniture['vertices'][furniture['faces']]
    lo,hi=triangles.min(1),triangles.max(1);fk=model.fk(q);direction=-basis[:,0]
    first=.15;witness=None;certificate=None;hulls=0
    for shape in shapes:
        vertices=np.asarray(shape['vertices']);body_hull=ConvexHull(vertices)
        expanded=body_hull.equations.copy();expanded[:,3]-=.006
        body_vertices=HalfspaceIntersection(expanded,vertices.mean(0)).intersections
        body=fk[shape['body']];R=root_rotation@body[:3,:3];p=root_position+root_rotation@body[:3,3]
        world=body_vertices@R.T+p;end=world+.15*direction
        lower=np.minimum(world.min(0),end.min(0));upper=np.maximum(world.max(0),end.max(0))
        candidates=np.flatnonzero(np.all(hi>=lower,axis=1)&np.all(lo<=upper,axis=1));d=R.T@direction
        for index in candidates:
            hulls+=1
            if hulls>10000:raise RuntimeError('MINKOWSKI_HULL_COUNT_CAP')
            tri=(triangles[index]-p)@R
            points=(tri[:,None,:]-body_vertices[None,:,:]).reshape(-1,3)
            hull=ConvexHull(points);a=hull.equations[:,:3]@d;b=hull.equations[:,3]
            parallel=abs(a)<1e-12
            if np.any(b[parallel]>1e-10):continue
            lower_t=max(0.,float(np.max(-b[a < -1e-12]/a[a < -1e-12],initial=-np.inf)))
            upper_t=min(.15,float(np.min(-b[a > 1e-12]/a[a > 1e-12],initial=np.inf)))
            if lower_t>upper_t+1e-10 or lower_t>=first:continue
            first=lower_t;witness=dict(body=shape['body'],face=int(index))
            certificate=(body_hull.equations.copy(),tri.copy(),d.copy())
    if certificate is not None:
        eq,tri,d=certificate;A=eq[:,:3];b=eq[:,3]
        M=np.c_[A@(tri[1]-tri[0]),A@(tri[2]-tri[0]),-A@d]
        fit=linprog([0.,0.,1.],A_ub=np.vstack([M,[1.,1.,0.]]),
                    b_ub=np.r_[.006-b-A@tri[0],1.],bounds=[(0,1),(0,1),(0,.15)],
                    method='highs',options={'time_limit':1.})
        assert fit.success and abs(fit.x[2]-first)<1e-8,'INDEPENDENT_FIRST_CONTACT_LP_DISAGREES'
        witness['LP_translation_m']=float(fit.x[2])
    shift=min(.14,first-.010)
    if shift<=.055:raise ValueError(('NO_MEANINGFUL_COLLISION_FREE_CLOSER_STANCE',shift))
    return shift,dict(closer_m=shift,first_contact_translation_m=first,
        first_contact_witness=witness,minkowski_hulls=hulls,independent_geometry_LPs=int(certificate is not None),
        existing_clearance_m=.006,additional_separation_m=.010,
        actual_rollout_initial_mesh_check_required=True)
