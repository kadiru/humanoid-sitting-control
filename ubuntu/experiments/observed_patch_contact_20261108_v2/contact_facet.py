"""Associate contact to the single observed triangle on the original goal ray.

The original fitted plane still defines which camera triangles are eligible for
contact. Only the mechanics contact plane becomes the selected observed facet.
No failed candidate, mesh truth, nearest-facet search or plane shifting is used.
"""
import numpy as np
from perception import cloud

def edges_for_triangle(vertices,normal):
    V=np.asarray(vertices,float);center=V.mean(0);rows=[]
    for i in range(3):
        a=V[i];direction=V[(i+1)%3]-a
        inward=np.cross(normal,direction);inward/=np.linalg.norm(inward)
        if inward@(center-a)<0:inward=-inward
        rows.append(np.r_[inward,-inward@a])
    return np.asarray(rows)

def select(camera,depth,goal):
    C=(goal@camera.R0-camera.T[:3,3])@camera.T[:3,:3]
    if C[2]<=.08:raise ValueError('CONTACT_GOAL_NOT_VISIBLE')
    uv=C[:2]/C[2]*[camera.K[0,0],camera.K[1,1]]+[camera.K[0,2]-.5,camera.K[1,2]-.5]
    col,row=np.floor(uv).astype(int);side=int(sum(uv-[col,row])>1.)
    if not (0<=row<camera.h-1 and 0<=col<camera.w-1):raise ValueError('CONTACT_GOAL_OUTSIDE_IMAGE')
    if not camera.target[row,col,side] or camera.ground[row,col,side]:raise ValueError('CONTACT_RAY_NOT_ORIGINAL_TARGET')
    N=camera.N[row,col,side].copy();d=float(camera.D[row,col,side])
    P,_=cloud(depth,camera.K,camera.T);P=P@camera.R0.T
    corners=[(row,col),(row,col+1),(row+1,col)] if side==0 else [(row,col+1),(row+1,col+1),(row+1,col)]
    V=np.array([P[r,c] for r,c in corners]);edges=edges_for_triangle(V,N)
    A=camera.R0@camera.T[:3,:3];t=-camera.T[:3,:3].T@camera.T[:3,3]
    center=np.linalg.solve(A.T,-t);ray=goal-center
    slope=float(N@ray)
    if slope>=0:raise ValueError('CONTACT_RAY_NOT_TOWARDS_FACET')
    parameter=-(N@center+d)/slope;point=center+parameter*ray
    signed=V@N+d;inset=edges[:,:3]@point+edges[:,3]
    if parameter<=0 or np.max(abs(signed))>1e-9 or abs(N@point+d)>1e-9 or np.min(inset)<-1e-9:
        raise ValueError('OBSERVED_FACET_INDEPENDENT_ASSOCIATION_FAILED')
    report=dict(original_goal=goal.tolist(),camera_pixel=[int(row),int(col),side],
        observed_vertices=V.tolist(),observed_normal=N.tolist(),observed_offset=d,
        ray_contact_point=point.tolist(),original_goal_signed_facet_distance_m=float(N@goal+d),
        plane_vertex_residual_m=float(np.max(abs(signed))),triangle_edge_insets_m=inset.tolist(),
        edges=edges.tolist(),actual_triangle_contact_required=True,original_camera_target_mask_unchanged=True,
        original_fitted_patch_inset_retained=True,physical_success=False,
        source='Original standing-pelvis support goal ray in original current depth; exactly one facet, no alternate search')
    return N,d,point,edges,report
