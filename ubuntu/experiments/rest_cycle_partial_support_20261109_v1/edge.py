"""A depth drop-off is an edge; a nearer occluder is not an edge.

Only current depth, selected plane pixels, calibrated camera extrinsics and
IMU orientation enter this module. Scene meshes/poses never enter detection.
Each accepted boundary supplies a pixel-width interval, not a point-valued
guess about a hidden edge. Unknown occluded spans remain unknown.
"""
import numpy as np
from scipy.ndimage import label


def classify_depth(actual, expected, plane_tolerance=.003, discontinuity=.015):
    actual, expected = np.asarray(actual), np.asarray(expected)
    valid = np.isfinite(expected) & (expected > .08) & (expected < 3.)
    measured = np.isfinite(actual) & (actual > .08)
    on = valid & measured & (abs(actual-expected) <= plane_tolerance)
    occlusion = valid & measured & (actual < expected-discontinuity)
    beyond = valid & ((measured & (actual > expected+discontinuity)) | np.isposinf(actual))
    return on, occlusion, beyond


def detect(depth, K, camera_in_body, normal, offset, level, selected_pixels):
    depth = np.asarray(depth); h,w = depth.shape
    v,u = np.mgrid[:h,:w]
    ray = np.stack([(u+.5-K[0,2])/K[0,0],(v+.5-K[1,2])/K[1,1],np.ones_like(u)],-1)
    T = np.asarray(camera_in_body); n=np.asarray(normal)
    directions = ray@T[:3,:3].T
    denominator = directions@n
    expected = np.divide(-(T[:3,3]@n+offset), denominator,
                         out=np.full((h,w),np.inf), where=abs(denominator)>1e-9)
    on,occluded,beyond = classify_depth(depth,expected)
    seed=np.zeros((h,w),bool);ij=np.asarray(selected_pixels,int)
    seed[ij[:,0],ij[:,1]]=True
    components,count=label(on)
    sizes=np.bincount(components[seed],minlength=count+1);sizes[0]=0
    if not sizes.any():return dict(passed=False,reason='selected_plane_has_no_connected_depth_support'),{}
    plane=components==sizes.argmax()
    surface=(T[:3,3]+directions*expected[...,None])@np.asarray(level).T
    rows=[];rejected_occlusion=0
    for dv,du in [(1,0),(-1,0),(0,1),(0,-1)]:
        # A front-boundary ray moves towards +body-forward on this horizontal plane.
        a0,a1=max(1,1-dv),min(h-1,h-1-dv)
        b0,b1=max(1,1-du),min(w-1,w-1-du)
        s=(slice(a0,a1),slice(b0,b1));t=(slice(a0+dv,a1+dv),slice(b0+du,b1+du))
        p,q=surface[s],surface[t];delta=q-p
        outward=(delta[...,0]>1e-5)&(delta[...,0]>2*abs(delta[...,1]))
        boundary=plane[s]&~plane[t]&outward
        rejected_occlusion+=int((boundary&occluded[t]).sum())
        take=boundary&beyond[t]
        rr,cc=np.nonzero(take)
        for r,c in zip(rr,cc):
            rows.append([*(p[r,c]),*(q[r,c]),r+a0,c+b0,r+a0+dv,c+b0+du])
    x=np.asarray(rows,float).reshape(-1,10)
    report=dict(candidate_dropoff_pixels=len(x),rejected_foreground_occlusion_pixels=rejected_occlusion,
                selected_plane_pixels=int(plane.sum()),actor_truth_input=False,
                center_directly_observed=False,source='current calibrated depth ray discontinuities')
    arrays=dict(edge_segments=x,plane_mask=plane,occlusion_mask=occluded,background_mask=beyond)
    if len(x)<20:return dict(report,passed=False,reason='insufficient_true_front_dropoff'),arrays
    # Exclude lateral corners from a straight-front hypothesis. This is an
    # observed contact-lane association, not a lookup of the furniture width.
    lane=(abs(x[:,1])>=.15)&(abs(x[:,1])<=.45)
    left=lane&(x[:,1]<0);right=lane&(x[:,1]>0)
    report.update(left_count=int(left.sum()),right_count=int(right.sum()))
    if min(left.sum(),right.sum())<10:
        return dict(report,passed=False,reason='true_edge_not_visible_on_both_flanks'),arrays
    # For an IMU-aligned frontal support, each ray segment bounds the x edge.
    # Keep the original3mm independent-side agreement and6mm full-mesh check.
    intervals=[]
    for side in [left,right]:
        lo=float(x[side,0].max());hi=float(x[side,3].min())
        intervals.append([lo,hi])
    lo=max(a[0] for a in intervals);hi=min(a[1] for a in intervals)
    mid=[.5*sum(a) for a in intervals]
    checks=dict(nonempty_each_side=all(a<=b for a,b in intervals),
        common_pixel_interval=lo<=hi,original_side_agreement=abs(mid[0]-mid[1])<=.003,
        finite_edge_uncertainty=0<=hi-lo<=.006)
    return dict(report,passed=all(checks.values()),tests=checks,side_intervals_m=intervals,
                front_x_interval_m=[lo,hi],side_center_difference_m=abs(mid[0]-mid[1]),
                assumption='Straight frontal edge across the occluded contact lane; independent fixture audit still required.'),arrays
