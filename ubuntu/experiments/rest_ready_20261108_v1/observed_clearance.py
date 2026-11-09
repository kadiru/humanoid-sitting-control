"""Camera-only surrounding-surface clearance, including the contact body.

Retains all valid neighbouring depth triangles, rather than only the chosen
support plane. Missing/discontinuous pixels are not bridged. This is a veto,
not a certificate that unobserved space is empty or a measured contact force.
"""
import numpy as np
from scipy.spatial import ConvexHull
from perception import cloud
from humanoid_sitting.shared_frame import apply, level_rotation
from audit_full_mesh import audit

def observed_mesh(model, depth, K, T, q, gravity, geometry):
    if not np.isfinite(geometry).all() or geometry[11]<=.5 or geometry[12]>.21:
        raise ValueError('CURRENT_GEOMETRY_REQUIRED')
    points, valid=cloud(depth,K,T);h,w=valid.shape
    valid &= model.self_mask(points.reshape(-1,3),model.fk(q),margin=.009).reshape(h,w)
    R=level_rotation(np.asarray(gravity,float).copy());v=points.reshape(-1,3)@R.T
    feet=np.concatenate([apply(model.fk(q)[s+'_ankle_roll_link'],model.vertices[s+'_ankle_roll_link']) for s in ['left','right']])@R.T
    floor=float(feet[:,2].min())
    ids=np.arange(h*w).reshape(h,w);a=ids[:-1,:-1].ravel();b=ids[:-1,1:].ravel();c=ids[1:,:-1].ravel();d=ids[1:,1:].ravel()
    faces=np.concatenate([np.stack([a,b,c],1),np.stack([b,d,c],1)])
    tri=v[faces];good=valid.ravel()[faces].all(1)&(tri[:,:,2].min(1)>floor+.055)
    for i,j in [(0,1),(1,2),(2,0)]:good &= np.linalg.norm(tri[:,i]-tri[:,j],axis=1)<=.05
    good &= np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)>1e-10
    faces=faces[good];tri=v[faces]
    if len(faces)<30:raise ValueError('INSUFFICIENT_CURRENT_OBSTACLE_TRIANGLES')
    n=R@np.asarray(geometry[2:5]);offset=float(geometry[5])
    # Only triangles of the observed target plane are eligible for resting.
    # Other orientations (including a chair's top edge) remain obstacles.
    normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normals/=np.linalg.norm(normals,axis=1)[:,None]
    eligible=(np.max(abs(tri@n+offset),axis=1)<=.002)&(abs(normals@n)>.97)
    return dict(vertices=v,faces=faces),np.flatnonzero(eligible).tolist(),dict(triangles=len(faces),target_plane_triangles=int(eligible.sum()),camera_only=True,unobserved_space_certified=False,force_sensor=False)

def check(model,shapes,mesh,target_faces,nodes,task):
    # The chosen contact body is intentionally included during approach.
    import audit_full_mesh
    audit_full_mesh.REST_BODIES={'torso_link','logo_link'} if task=='leaning' else {'pelvis','left_hip_yaw_link','right_hip_yaw_link'}
    return audit(model,shapes,mesh,target_faces,nodes,np.eye(3),np.zeros(3))
