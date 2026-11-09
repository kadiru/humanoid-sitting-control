"""Preserve already-clear upright motion; repair only the remaining steep plans."""
from traction_clearance_planner_v2 import plan as original
from positive_slope_planner import plan as repaired
import numpy as np

def plan(*args,inertial,**config):
    commands,qs,a=original(*args,inertial=inertial)
    slope=float(np.arctan2(a['normal'][0],a['normal'][2]))
    preserve=slope<=np.deg2rad(10.5) or (a['endpoint_screen_passed'] and a['chosen_pitch']<=.1000001)
    if preserve:
        a.update(positive_candidate_applied=False,positive_preserved_reason='mild slope or already-clear upright original')
        return commands,qs,a
    return repaired(*args,inertial=inertial,**config)
