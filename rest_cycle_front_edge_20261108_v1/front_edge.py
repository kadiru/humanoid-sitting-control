"""Current-depth bilateral edge association; no scene/world-truth inputs."""
import numpy as np
from humanoid_sitting.shared_frame import level_rotation

def qualify(points,pixels,gravity,image_shape):
    p=np.asarray(points)@level_rotation(np.asarray(gravity,float).copy()).T
    ij=np.asarray(pixels);h,w=image_shape
    if len(p)<20 or ij.shape!=(len(p),2):return dict(passed=False,reason='invalid_current_point_pixels')
    margin=np.minimum.reduce([ij[:,0],h-1-ij[:,0],ij[:,1],w-1-ij[:,1]])
    # The contact lane is y=0. Both flanks must independently constrain the
    # same front edge; filling across them is interpolation, not extrapolation.
    band=(p[:,0]>=p[:,0].max()-.006)&(margin>=2)&(abs(p[:,1])>=.15)&(abs(p[:,1])<=.45)
    left=band&(p[:,1]<0);right=band&(p[:,1]>0)
    result=dict(left_points=int(left.sum()),right_points=int(right.sum()),
        current_points_only=True,scene_id_input=False,center_directly_observed=False)
    if min(left.sum(),right.sum())<10:return dict(result,passed=False,reason='missing_bilateral_edge')
    line=np.linalg.lstsq(np.c_[p[band,1],np.ones(band.sum())],p[band,0],rcond=None)[0]
    residual=p[band,0]-np.c_[p[band,1],np.ones(band.sum())]@line
    errors=[]
    for mask in [left,right]:
        local=np.linalg.lstsq(np.c_[p[mask,1],np.ones(mask.sum())],p[mask,0],rcond=None)[0]
        errors.append(float(local[1]))
    checks=dict(collinear=float(abs(residual).max())<=.003,
        agreement_at_contact_lane=abs(errors[0]-errors[1])<=.003,
        front_orientation=abs(float(line[0]))<=.03,
        supported_interpolation=bool(p[left,1].max()<0<p[right,1].min()))
    return dict(result,passed=all(checks.values()),tests=checks,edge_x_at_center_m=float(line[1]),
        edge_slope=float(line[0]),max_line_residual_m=float(abs(residual).max()),
        left_right_center_predictions_m=errors,minimum_image_margin_px=int(margin[band].min()),
        interpretation='A common line supported on both visible flanks; occluded center is inferred. Independent fixture clearance remains mandatory.')
