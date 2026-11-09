"""Exact triangle/convex first-contact translation for one static standing pose.

Used only to choose an offline scene placement. It is not an actor observation,
does not move feet during a rollout, and retains the existing6mm clearance hull.
"""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from humanoid_sitting.shared_frame import apply


def closer_stance(model, shapes, furniture, q, root_rotation, root_position, basis):
    triangles=furniture['vertices'][furniture['faces']]
    lo,hi=triangles.min(1),triangles.max(1)
    fk=model.fk(q);direction=-basis[:,0]
    first=.15;witness=None;solves=0
    for shape in shapes:
        vertices=np.asarray(shape['vertices']);hull=ConvexHull(vertices)
        body=fk[shape['body']]
        R=root_rotation@body[:3,:3]
        p=root_position+root_rotation@body[:3,3]
        world=vertices@R.T+p
        end=world+.15*direction
        lower=np.minimum(world.min(0),end.min(0))-.006
        upper=np.maximum(world.max(0),end.max(0))+.006
        candidates=np.flatnonzero(np.all(hi>=lower,axis=1)&np.all(lo<=upper,axis=1))
        d=R.T@direction;A=hull.equations[:,:3];b=hull.equations[:,3]
        for index in candidates:
            solves+=1
            if solves>2000:raise RuntimeError('PLACEMENT_LP_COUNT_CAP')
            tri=(triangles[index]-p)@R
            matrix=np.c_[A@(tri[1]-tri[0]),A@(tri[2]-tri[0]),-A@d]
            rhs=.006-b-A@tri[0]
            fit=linprog([0.,0.,1.],A_ub=np.vstack([matrix,[1.,1.,0.]]),
                        b_ub=np.r_[rhs,1.],bounds=[(0,1),(0,1),(0,.15)],
                        method='highs',options={'time_limit':1.})
            if fit.status==2:continue
            if not fit.success:raise RuntimeError(('PLACEMENT_LP_NOT_CERTIFIED',fit.message))
            if fit.x[2]<first:
                first=float(fit.x[2]);witness=dict(body=shape['body'],face=int(index),
                    barycentric=[float(1-fit.x[0]-fit.x[1]),float(fit.x[0]),float(fit.x[1])])
    shift=min(.14,first-.010)
    if shift<=.055:raise ValueError(('NO_MEANINGFUL_COLLISION_FREE_CLOSER_STANCE',shift))
    return shift,dict(closer_m=shift,first_contact_translation_m=first,
                      additional_separation_m=.010,existing_clearance_m=.006,
                      first_contact_witness=witness,three_variable_geometry_LPs=solves,
                      actual_rollout_initial_mesh_check_required=True)
