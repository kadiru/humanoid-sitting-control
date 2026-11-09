"""Persistent measured geometry with explicit localization uncertainty.

Stored surfaces may be outside the current image. They are usable only with a
CURRENT independently qualified localization bound. This class does not claim
to produce that bound from fixed-foot consistency or to detect common slip.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Localization:
    current_from_map: np.ndarray
    timestamp: float
    translation_bound_m: float | None
    rotation_bound_rad: float | None
    qualified: bool


class SurfaceMemory:
    def __init__(self,points,normal,offset,observed_at):
        self.points=np.asarray(points,float).copy();self.normal=np.asarray(normal,float).copy()
        self.offset=float(offset);self.observed_at=float(observed_at);self.invalid=False
        assert self.points.ndim==2 and self.points.shape[1]==3 and len(self.points)>=3
        assert np.isfinite(self.points).all() and np.isfinite(self.normal).all() and np.isfinite(self.offset)
        assert abs(np.linalg.norm(self.normal)-1.)<1e-6

    def estimate(self,now,localization: Localization):
        l=localization;T=np.asarray(l.current_from_map,float)
        reason=None
        if self.invalid:reason='map_invalidated'
        elif T.shape!=(4,4) or not np.isfinite(T).all():reason='invalid_transform'
        elif (not np.allclose(T[3],[0,0,0,1],rtol=0,atol=1e-9)
              or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),rtol=0,atol=1e-6)
              or abs(np.linalg.det(T[:3,:3])-1)>1e-6):reason='nonrigid_transform'
        elif not l.qualified or l.translation_bound_m is None or l.rotation_bound_rad is None:reason='localization_unqualified'
        elif not 0<=now-l.timestamp<=.21:reason='localization_stale'
        elif not np.isfinite([l.translation_bound_m,l.rotation_bound_rad]).all() or min(l.translation_bound_m,l.rotation_bound_rad)<0:reason='invalid_uncertainty'
        if reason:return dict(valid=False,reason=reason,visual_age_s=now-self.observed_at)
        radius=float(np.linalg.norm(self.points,axis=1).max())
        error=float(l.translation_bound_m+2*radius*np.sin(min(l.rotation_bound_rad,np.pi)/2))
        if error>.003:return dict(valid=False,reason='surface_uncertainty_exceeds3mm',error_bound_m=error)
        R=T[:3,:3];t=T[:3,3];n=R@self.normal
        return dict(valid=True,points=self.points@R.T+t,normal=n,offset=self.offset-n@t,
                    error_bound_m=error,visual_age_s=now-self.observed_at,
                    localization_age_s=now-l.timestamp,current_image_visibility_required=False,
                    original_depth_timestamp_preserved=True)

    def verify_observation(self,mapped_points):
        # Only externally established correspondences may call this method.
        p=np.asarray(mapped_points,float)
        if p.shape!=self.points.shape or not np.isfinite(p).all() or np.max(np.linalg.norm(p-self.points,axis=1))>.003:
            self.invalid=True
        return not self.invalid
