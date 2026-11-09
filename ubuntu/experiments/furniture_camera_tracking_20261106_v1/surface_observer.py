"""Track the chosen camera patch among all current depth points, not one winner."""
import importlib.util
from pathlib import Path
import numpy as np
from humanoid_sitting.shared_frame import level_rotation
source=Path(__file__).resolve().parent.parent/'furniture_camera_coverage_20261106_v1/surface_observer.py'
spec=importlib.util.spec_from_file_location('anchored_patch_base',source);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

class Observer(base.Observer):
    def __call__(self,model,K,T,depth,q,gravity,prior=None):
        if self.anchors is None or self.intent!='seat':
            return super().__call__(model,K,T,depth,q,gravity,prior)
        from perception import cloud
        P,valid=cloud(depth,K,T);h,w=valid.shape
        valid &= model.self_mask(P.reshape(-1,3),model.fk(q),margin=.009).reshape(h,w)
        dx=P[1:-1,2:]-P[1:-1,:-2];dy=P[2:,1:-1]-P[:-2,1:-1]
        normal=np.cross(dx,dy);length=np.linalg.norm(normal,axis=-1)
        ok=valid[1:-1,1:-1]&valid[1:-1,2:]&valid[1:-1,:-2]&valid[2:,1:-1]&valid[:-2,1:-1]
        ok &= (length>1e-8)&(np.linalg.norm(dx,axis=-1)<.15)&(np.linalg.norm(dy,axis=-1)<.15)
        normal/=np.maximum(length[...,None],1e-30);points=P[1:-1,1:-1]
        normal[(normal*(points-T[:3,3])).sum(-1)>0]*=-1
        up=-np.asarray(gravity,float);up/=np.linalg.norm(up);ok &= normal@up>.75
        points=points[ok];pixels=np.argwhere(ok)+1
        info=dict(intent=self.intent,actor_truth_input=False,association='fixed chosen patch within all current valid depth pixels',candidate_current_points=len(points))
        try:
            value,local,report=self.refine(model,q,gravity,None,points)
            lookup={tuple(p):i for i,p in enumerate(points)};ids=[lookup[tuple(p)] for p in local]
            info.update(surface_anchor=report,selected_points=local.tolist(),selected_pixels=pixels[ids].tolist())
        except (ValueError,AssertionError) as e:
            value=None;info['anchored_patch_rejected']=repr(e)
        self.last=(value,info);self.raw_last=self.last;return self.last
