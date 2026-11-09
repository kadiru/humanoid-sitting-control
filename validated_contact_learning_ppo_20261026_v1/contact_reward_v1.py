"""Reward labels only; no simulator state is added to policy observations."""
import numpy as np
import torch
from feedback_slip_reward import per_tick_cost

def tick_reward(record, elapsed):
    normal_load=np.maximum((record['force']*record['normal']).sum(-1),0)
    allowed=record['permitted'] & record['inside'] & record['rest_patch']
    support=(normal_load*allowed).sum(-1)/record['weight']
    legal=record['legal'] & (record['wrong']<=10.) & ~record['failure'] & record['alive']
    # Saturating support target: no reward for pressing harder beyond 0.2 BW.
    load=np.clip(support/.2,0,1)
    still=np.exp(-(record['root_speed']/.10)**2-(record['gyro']/.25)**2)
    bonus=(2*load*still+2*record['good'])*legal*(elapsed>=8.)
    return .02*bonus-per_tick_cost(record)

def reward_to_go(integrals, baseline, penalty, decision_steps, dense):
    per_step=-.25*(integrals-baseline)/1350.+dense
    assert torch.isfinite(per_step).all() and torch.isfinite(penalty).all()
    return per_step.flip(0).cumsum(0).flip(0)[decision_steps]-penalty[None,:]
