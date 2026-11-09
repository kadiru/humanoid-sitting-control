"""Refine the visible support at the actual contact lane; sensor inputs only."""
import numpy as np
from scipy.spatial import ConvexHull
from humanoid_sitting.shared_frame import level_rotation,apply


def refine(model,q,gravity,geometry,points):
    geometry=np.asarray(geometry,float).copy();points=np.asarray(points,float)
    assert geometry[11]>.5 and 0<=geometry[12]<=.21
    assert len(points)>=30 and np.isfinite(points).all()
    R=level_rotation(np.asarray(gravity,float).copy())
    n=R@geometry[2:5];n/=np.linalg.norm(n)
    e=np.array([1.,0,0])-n*n[0];e/=np.linalg.norm(e);b=np.cross(n,e)
    cloud=points@R.T;uv=np.c_[cloud@e,cloud@b]
    pelvis=np.asarray(model.vertices['pelvis'])@R.T
    lane=float(pelvis[np.argmin(pelvis@n)]@b)
    eq=ConvexHull(uv).equations;front=eq[:,0]>1e-5
    edge=float(np.min((-eq[front,2]-eq[front,1]*lane)/eq[front,0]))
    selected=np.flatnonzero((uv[:,0]>=edge-.14)&(uv[:,0]<=edge+1e-6)&(abs(uv[:,1]-lane)<=.075))
    if len(selected)<30:raise ValueError(('LOCAL_CONTACT_PATCH_TOO_FEW_POINTS',len(selected)))
    local=points[selected];center=local.mean(0)
    _,_,axes=np.linalg.svd(local-center,full_matrices=False)
    normal=axes[-1]
    if normal@(-np.asarray(gravity))<0:normal=-normal
    offset=float(-center@normal);residual=local@normal+offset
    if (R@normal)[2]<=.97:raise ValueError('LOCAL_PATCH_NOT_HORIZONTAL_SUPPORT')
    maximum=float(abs(residual).max())
    if maximum>.002:raise ValueError(('LOCAL_PATCH_NOT_PLANAR_TO_2MM',maximum,len(selected)))
    poses=model.fk(q)
    rear=np.concatenate([apply(poses[body],model.vertices[body]) for body in ['torso_link','logo_link']])
    span=np.ptp((local-center)@axes[:2].T,axis=0)
    value=np.array([(np.asarray(model.vertices['pelvis'])@normal+offset).min(),
                    (rear@normal+offset).min(),*normal,offset,*center,*span,1.,geometry[12]])
    return value,local,dict(selected_source_indices=selected.tolist(),source_points=len(points),local_points=len(local),
        maximum_plane_residual_m=maximum,rms_plane_residual_m=float(np.sqrt(np.mean(residual**2))),
        source_normal=geometry[2:5].tolist(),source_offset=float(geometry[5]),
        refined_normal=normal.tolist(),refined_offset=offset,normal_offset_change_m=offset-float(geometry[5]),
        actor_truth_input=False,selection_rule='front140mm_by150mm_at_measured_pelvis_lane')
