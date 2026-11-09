"""A bounded effort cost calibrated from the fixed-posture baseline, before PPO."""
from dataclasses import replace
import math
from .resting_objective import RestingObjectiveCfg, resting_objective


def calibrated_rest_reward(metrics, alive, reference, weight=1.):
    if not math.isfinite(reference) or reference<=0 or not 0<=weight<=1:
        raise ValueError('A positive fixed effort reference and weight in [0,1] are required')
    cfg=replace(RestingObjectiveCfg(),effort_weight=0.)
    reward,good,detail=resting_objective(metrics,alive,cfg)
    effort=detail['normalized_torque_squared']
    # Strictly monotonic for every finite positive effort, including the high
    # seat. A hard clip would erase the learning signal on its costly posture.
    cost=-weight*effort/(effort+reference)*good
    detail['effort_cost']=cost
    return reward+cfg.control_dt*cost,good,detail
