"""Contact-specific camera triangle and spatial housing heights, no truth mesh."""
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from humanoid_sitting.shared_frame import level_rotation,apply
from local_patch import refine


def associate(model,q,gravity,geometry,points,intent):
    R=level_rotation(np.asarray(gravity,float).copy());points=np.asarray(points,float)
    cloud=points@R.T
    triangulation=Delaunay(cloud[:,:2])
    height=LinearNDInterpolator(triangulation,cloud[:,2])
    if intent=='seat':
        value,local,report=refine(model,q,gravity,geometry,points)
        target=None
    else:
        # A dished stool is represented by its observed height field. The
        # tangent is associated with the central supported region, not its rim.
        target_xy=cloud[:,:2].mean(0)
        simplex=int(triangulation.find_simplex(target_xy));assert simplex>=0
        ids=triangulation.simplices[simplex];tri=cloud[ids]
        longest=max(np.linalg.norm(tri[(i+1)%3]-tri[i]) for i in range(3))
        if longest>.05:raise ValueError('CONTACT_TRIANGLE_CROSSES_UNOBSERVED_GAP')
        n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);n/=np.linalg.norm(n)
        if n[2]<0:n=-n
        if n[2]<=.97:raise ValueError(('CENTRAL_CONTACT_TRIANGLE_TOO_STEEP',n.tolist()))
        d=float(-n@tri[0]);normal=R.T@n;target=np.r_[target_xy,float(height(target_xy)[0])]
        body_target=R.T@target;poses=model.fk(q)
        rear=np.concatenate([apply(poses[b],model.vertices[b]) for b in ['torso_link','logo_link']])
        value=np.array([(np.asarray(model.vertices['pelvis'])@normal+d).min(),(rear@normal+d).min(),
            *normal,d,*body_target,*np.ptp(cloud[:,:2],axis=0),1.,geometry[12]])
        local=points
        report=dict(actor_truth_input=False,mode='observed_piecewise_planar_height_field',
            contact_source_indices=ids.tolist(),contact_triangle_gravity=tri.tolist(),
            contact_target_gravity=target.tolist(),longest_contact_triangle_edge_m=float(longest),
            maximum_plane_residual_m=0.,whole_cloud_claimed_planar=False)
    return value,local,report,cloud,height,target
