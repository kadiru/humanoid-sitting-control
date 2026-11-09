"""Shadow camera/encoder separation witness; NOT admitted for control.

For the selected flat-top table only, an independent full-mesh check must prove
all furniture is behind the observed top plane. Then every legal rest body
being ahead of that plane implies separation from this fixture. A local plane
alone cannot justify this premise for an armchair or a curved/compound object.
The numerical margins below are existing model/perception guards, not a live
collision-shape or hardware calibration certificate.
"""
import numpy as np
from humanoid_sitting.shared_frame import apply

REST_BODIES=('pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link')

def shadow_separation(model,q,geometry):
    g=np.asarray(geometry,float)
    if g.shape!=(13,) or not np.isfinite(g).all() or not(g[11]>.5 and 0<=g[12]<=.21):
        return dict(candidate=False,reason='no_current_plane',control_admitted=False)
    n=g[2:5];d=g[5];center=g[6:9]
    if not .99<=np.linalg.norm(n)<=1.01:return dict(candidate=False,reason='invalid_normal',control_admitted=False)
    fk=model.fk(q);mins={};raw={}
    for b in REST_BODIES:
        vertices=apply(fk[b],model.vertices[b])
        gap=vertices@n+d
        # Per-point normal error, existing3mm offset and6mm shape envelope.
        reserve=.003+.006+.03*np.linalg.norm(vertices-center,axis=1)
        mins[b]=float(np.min(gap-reserve));raw[b]=float(np.min(gap))
    return dict(candidate=min(mins.values())>0,minimum_reserved_gap_m=min(mins.values()),
                body_lower_gaps_m=mins,raw_body_gaps_m=raw,plane_age_s=float(g[12]),
                control_admitted=False,load_inference=None,
                reason='conditional_on_complete_fixture_halfspace_and_live_shape_envelope')

