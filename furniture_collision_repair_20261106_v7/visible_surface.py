"""Recover the connected curved support containing the selected camera patch."""
import numpy as np
from scipy.ndimage import label
from perception import cloud


def expand(model,K,T,depth,q,gravity,selected_pixels):
    P,valid=cloud(depth,K,T);h,w=valid.shape;fk=model.fk(q)
    valid &= model.self_mask(P.reshape(-1,3),fk,margin=.009).reshape(h,w)
    dx=P[1:-1,2:]-P[1:-1,:-2];dy=P[2:,1:-1]-P[:-2,1:-1]
    n=np.cross(dx,dy);length=np.linalg.norm(n,axis=-1)
    ok=valid[1:-1,1:-1]&valid[1:-1,2:]&valid[1:-1,:-2]&valid[2:,1:-1]&valid[:-2,1:-1]
    ok&=(length>1e-8)&(np.linalg.norm(dx,axis=-1)<.15)&(np.linalg.norm(dy,axis=-1)<.15)
    n/=np.maximum(length[...,None],1e-30);points=P[1:-1,1:-1]
    n[(n*(points-T[:3,3])).sum(-1)>0]*=-1
    up=-np.asarray(gravity,float);up/=np.linalg.norm(up)
    components,count=label(ok&(n@up>.75))
    pixels=np.asarray(selected_pixels,int)-1
    assert ((pixels>=0)&(pixels<np.array([h-2,w-2]))).all()
    ids=components[pixels[:,0],pixels[:,1]];counts=np.bincount(ids,minlength=count+1);counts[0]=0
    if counts.max()<20:raise ValueError('SELECTED_SUPPORT_COMPONENT_NOT_RECOVERED')
    selected=components==counts.argmax();result=points[selected];result_pixels=np.argwhere(selected)+1
    if len(result)<30 or not np.isfinite(result).all():raise ValueError('VISIBLE_SUPPORT_INVALID')
    return result,result_pixels,dict(source='actual_aligned_depth_connected_to_selected_patch',
        selected_patch_pixels=len(pixels),expanded_support_points=len(result),
        self_mask_margin_m=.009,normal_up_threshold=.75,actor_truth_input=False,
        flat_plane_residual_not_used_to_discard_curved_top=True)
