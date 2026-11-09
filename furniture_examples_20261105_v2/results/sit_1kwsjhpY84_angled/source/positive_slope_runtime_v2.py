"""Install a separately selected sensor-only positive-slope reference.

Selection and configuration must be saved before simulation. No test labels
are inspected here; the runtime uses only kinematics, IMU, measured plane and
the explicit reference motion. Physical labels remain diagnostic outputs.
"""
import json
from pathlib import Path
import numpy as np
from positive_slope_planner_v2 import plan

def install(harness,path):
    selected=json.loads(Path(path).read_text());config=selected['planner'];bridge=harness.traction.bridge
    harness.traction.progressive.rolling_plan=lambda *a,**kw:plan(*a,**kw,inertial=bridge.inertial,**config)
    duration=selected.get('duration_s',5.)
    if duration!=5.:
        import positive_timed_reference as timing
        timing.DURATION=duration
        bridge.BridgeEnv._positive_original_split_reference=lambda self:bridge.progressive.original.SplitEnv.reference(self)
        bridge.BridgeEnv.reference=timing.timed_reference
    return selected

def parent_decision_steps(env):
    """Use planned transfer completion, never simulator contact/pose labels."""
    import torch
    result=torch.full((env.num_envs,),110,dtype=torch.long,device=env.device)
    for audit in env.plan_audit:
        if audit.get('positive_candidate_applied',False):
            # The measured-plane planner begins at10s and lasts5s.
            result[audit['case']]=150
    return result

def material_audit(env):
    """Read-only simulator metadata for reporting; not an actor/controller input."""
    result={'legacy_ratio_coefficient':.6,'legacy_warning_threshold':.9}
    for name in ['robot','seat','rest_floor']:
        try:
            item=env.raw.scene[name]
            view=getattr(item,'root_physx_view',None)
            if view is None:view=item.root_physx_view
            values=view.get_material_properties()
            values=values.cpu().numpy() if hasattr(values,'cpu') else values
            result[name]={'shape':list(values.shape),'values':np.unique(np.asarray(values).reshape(-1,3),axis=0).tolist()}
        except Exception as exc:result[name]={'unavailable':str(exc)}
    return result
