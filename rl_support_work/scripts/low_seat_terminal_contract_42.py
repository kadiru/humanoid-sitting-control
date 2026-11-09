"""Necessary model checks for future candidates, not a physical admission gate."""
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.resting_objective import RestingObjectiveCfg

def terminal_posture_check(root, research_tilt_reserve_deg=2.):
    root=np.asarray(root,dtype=float)
    if root.shape!=(7,) or not np.isfinite(root).all():
        return dict(passed=False,reason='Nonfinite or malformed root pose')
    cfg=RestingObjectiveCfg()
    tilt=float(np.rad2deg(np.arccos(np.clip(Rotation.from_quat(root[3:]).as_matrix()[2,2],-1,1))))
    limit=cfg.maximum_trunk_tilt_deg
    return dict(passed=bool(tilt<=limit-research_tilt_reserve_deg),model_tilt_deg=tilt,
        unchanged_physical_support_limit_deg=limit,research_tilt_reserve_deg=research_tilt_reserve_deg,
        meets_physical_posture_requirement=bool(tilt<=limit),
        scope='Necessary terminal posture screen only. Does not establish legal support, contact clearance, dynamics, or physical success.')
