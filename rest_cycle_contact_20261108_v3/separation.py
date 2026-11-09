"""Current depth top/front separation witness; shadow use until physical audit.

Each entire legal rest body must lie above the top OR in front of the table.
This is a per-body separating-plane test, not a per-vertex union that might
miss an edge crossing. The actual complete fixture must fit both negative
halfspaces in an independent evaluation-only domain check.
"""
import numpy as np
from humanoid_sitting.shared_frame import apply
from perception import cloud
REST_BODIES=('pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link')

def witness(model,q,g,points,K,depth_range):
    g=np.asarray(g,float);points=np.asarray(points,float)
    if g.shape!=(13,) or not np.isfinite(g).all() or not(g[11]>.5 and 0<=g[12]<=.21):
        return dict(candidate=False,reason='no_current_plane',control_admitted=False)
    if len(points)<30 or not np.isfinite(points).all():return dict(candidate=False,reason='missing_observed_top',control_admitted=False)
    n=g[2:5];n=n/np.linalg.norm(n);d=g[5];center=g[6:9]
    # A horizontal direction toward the robot, constructed from the measured top.
    front=np.array([1.,0.,0.]);front-=n*(front@n);front/=np.linalg.norm(front)
    # Expand furniture to cover the unsampled pixel boundary. This enlarges the
    # obstacle; it never relaxes the robot clearance or contact score.
    edge_cover=2*float(depth_range)/min(float(K[0,0]),float(K[1,1]))
    front_offset=-float(np.max(points@front))-edge_cover
    fk=model.fk(q);lower={};winner={}
    for b in REST_BODIES:
        p=apply(fk[b],model.vertices[b]);reserve=.003+.006+.03*np.linalg.norm(p-center,axis=1)
        top=float(np.min(p@n+d-reserve));ahead=float(np.min(p@front+front_offset-reserve))
        lower[b]=max(top,ahead);winner[b]='top' if top>=ahead else 'front'
    return dict(candidate=min(lower.values())>0,minimum_reserved_gap_m=min(lower.values()),
                body_lower_gaps_m=lower,separating_face=winner,edge_cover_m=edge_cover,
                envelope_normals=[n.tolist(),front.tolist()],envelope_offsets=[float(d),front_offset],
                plane_age_s=float(g[12]),control_admitted=False,load_inference=None,
                reason='conditional_on_complete_fixture_top_front_envelope_and_live_shapes')

class SeparationObserver:
    def __init__(self,model,T):self.model=model;self.T=np.asarray(T).copy();self.cache=None;self.timestamp=None
    def read(self,raw,q,gravity,g):
        # Only consume the camera image at its actual refresh. Repeated old
        # pixels are never made fresh by a newer encoder or control timestamp.
        camera=raw.scene['rear_camera'];depth=camera.data.output['distance_to_image_plane'].torch[0].cpu().numpy().squeeze()
        K=camera.data.intrinsic_matrices.torch[0].cpu().numpy()
        if float(g[12])>0:
            return dict(candidate=False,reason='not_a_new_depth_frame',control_admitted=False)
        return from_depth(self.model,q,g,depth,K,self.T)


def from_depth(model,q,g,depth,K,T):
    P,valid=cloud(depth,K,T);p=P[valid]
    keep=model.self_mask(p,model.fk(q),margin=.009)
    p=p[keep];n=np.asarray(g[2:5],float);d=float(g[5])
    p=p[abs(p@n+d)<=.002]
    if len(p)<30:return dict(candidate=False,reason='missing_observed_top',control_admitted=False)
    optical=(p-T[:3,3])@T[:3,:3]
    return witness(model,q,g,p,K,float(optical[:,2].max()))
