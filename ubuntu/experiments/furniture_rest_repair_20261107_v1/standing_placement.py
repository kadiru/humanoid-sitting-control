"""First-contact placement with conservative swept-AABB lower bounds.
No geometry, expansion, LP budget, or physical criterion is relaxed.
"""
import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog
from audit_mesh_clearance_fast import clip_triangle


def entry_times(lo, hi, body_lo, body_hi, direction, maximum):
    enter=np.zeros(len(lo));leave=np.full(len(lo),maximum)
    for k,d in enumerate(direction):
        if abs(d)<1e-14:
            leave[(hi[:,k]<body_lo[k])|(lo[:,k]>body_hi[k])]=-1
        else:
            a=(lo[:,k]-body_hi[k])/d;b=(hi[:,k]-body_lo[k])/d
            enter=np.maximum(enter,np.minimum(a,b));leave=np.minimum(leave,np.maximum(a,b))
    return enter,leave


def closer_stance(model,shapes,furniture,q,root_rotation,root_position,basis,maximum_translation):
    triangles=furniture['vertices'][furniture['faces']]
    lo,hi=triangles.min(1),triangles.max(1);fk=model.fk(q);direction=-basis[:,0]
    prepared=[];queue=[];candidates_total=0
    for si,shape in enumerate(shapes):
        v=np.asarray(shape['vertices']);h=ConvexHull(v);eq=h.equations.copy()
        expanded=eq.copy();expanded[:,3]-=.006
        ev=HalfspaceIntersection(expanded,v[h.vertices].mean(0)).intersections
        body=fk[shape['body']];R=root_rotation@body[:3,:3];p=root_position+root_rotation@body[:3,3]
        world=ev@R.T+p
        # Conservative outward rounding; offset planes can extend farther than6mm along AABB axes.
        enter,leave=entry_times(lo,hi,world.min(0)-1e-9,world.max(0)+1e-9,direction,maximum_translation)
        ids=np.flatnonzero(enter<=leave+1e-9);candidates_total+=len(ids)
        queue.extend((float(enter[i]),si,int(i)) for i in ids)
        prepared.append((shape['body'],R,p,eq))
    queue.sort();first=float(maximum_translation);witness=None;solves=0;pruned=0
    for lower,si,index in queue:
        if lower>first+1e-9:
            pruned+=1;continue
        if solves>=2000:raise RuntimeError('PLACEMENT_LP_COUNT_CAP')
        body,R,p,eq=prepared[si];A=eq[:,:3];b=eq[:,3];d=R.T@direction
        tri=(triangles[index]-p)@R
        matrix=np.c_[A@(tri[1]-tri[0]),A@(tri[2]-tri[0]),-A@d]
        rhs=.006-b-A@tri[0];solves+=1
        fit=linprog([0.,0.,1.],A_ub=np.vstack([matrix,[1.,1.,0.]]),b_ub=np.r_[rhs,1.],
                    bounds=[(0,1),(0,1),(0,min(maximum_translation,first+1e-9))],method='highs',options={'time_limit':1.})
        if fit.status==2:continue
        if not fit.success:raise RuntimeError(('PLACEMENT_LP_NOT_CERTIFIED',fit.message))
        if fit.x[2]<=first:
            first=float(fit.x[2]);witness=dict(body=body,shape=si,face=index,
                barycentric=[float(1-fit.x[0]-fit.x[1]),float(fit.x[0]),float(fit.x[1])])
    if witness is None:raise ValueError('NO_PHYSICAL_PLACEMENT_BOUND_WITNESS')
    shift=first-.010
    if shift<=.055:raise ValueError(('NO_MEANINGFUL_COLLISION_FREE_CLOSER_STANCE',shift))
    # Independent admission uses every triangle and polygon clipping, not LP status or swept AABBs.
    collisions=[];clips=0
    for si,(body,R,p,eq) in enumerate(prepared):
        for start in range(0,len(triangles),256):
            tr=(triangles[start:start+256]-(p+shift*direction))@R
            signed=np.einsum('ntj,fj->ntf',tr,eq[:,:3])+eq[:,3]-.006
            possible=np.flatnonzero(~np.any(np.min(signed,axis=1)>1e-10,axis=1))
            for j in possible:
                clips+=1
                if len(clip_triangle(tr[j],eq,.006)):collisions.append((body,start+int(j)))
    si=witness['shape'];body,R,p,eq=prepared[si]
    point=np.asarray(witness['barycentric'])@triangles[witness['face']]
    residual=(point-p-first*direction)@R@eq[:,:3].T+eq[:,3]-.006
    if collisions or residual.max()>1e-7 or min(witness['barycentric']) < -1e-7:
        raise RuntimeError(('INDEPENDENT_PLACEMENT_AUDIT',collisions[:4],float(residual.max())))
    return shift,dict(closer_m=shift,first_contact_translation_m=first,geometric_translation_endpoint_m=maximum_translation,
        additional_separation_m=.010,existing_clearance_m=.006,first_contact_witness=witness,
        three_variable_geometry_LPs=solves,swept_candidates=candidates_total,bound_pruned=pruned,
        independent_all_triangle_clip_audit=dict(passed=True,triangles=len(triangles),bodies=len(prepared),clips=clips,witness_max_residual_m=float(residual.max())),
        actual_rollout_initial_mesh_check_required=True)
