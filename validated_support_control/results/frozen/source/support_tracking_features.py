"""Preserve the frozen actor's ten-channel measured-patch observation contract.

The landing tracker can infer distance when a patch is hidden. The old actor's
valid bit also promises a measured patch centre and extent, so those distance-
only outputs cannot be substituted as complete observations. No simulator data.
"""
import numpy as np
from .support_landing_geometry import extract_scene


def measured_tracking_features(tracker, result, depth, q, gravity):
    values=np.zeros(10,dtype=np.float32);values[-1]=1.
    if not result['valid'] or result['source'] in ('kinematic_prediction','seat_from_observed_floor','unavailable'):
        return values,dict(usable=False,reason='no_current_observed_patch')
    scene=extract_scene(tracker.model,tracker.K,tracker.T_P_C,depth,q,gravity)
    normal=np.asarray(result['normal_P']);clearance=result['clearance_m']
    matches=[]
    for plane in scene['planes']:
        n=scene['R'].T@plane['normal']
        c=float((tracker.model.vertices['pelvis']@n+plane['offset']).min())
        if np.linalg.norm(n-normal)<1e-6 and abs(c-clearance)<1e-6:matches.append(plane)
    if len(matches)!=1:return values,dict(usable=False,reason='ambiguous_patch_identity')
    points=matches[0]['points']@scene['R']
    values[:]=[clearance,*normal,*points[:,:2].mean(0),*np.ptp(points[:,:2],axis=0),1.,0.]
    return values,dict(usable=True,reason='current_measured_patch',points=len(points))
