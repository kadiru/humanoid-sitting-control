"""Original success-first reward; effort averages the entire final clean hold.

Applied-torque telemetry is a proxy, not measured electrical energy.
"""
import torch
from common import DT
GAMMA=.999

def phase_from_clock(seconds,contact_time,settle_time):
    return (seconds>=contact_time).long()+(seconds>=settle_time).long()+(seconds>=15).long()

def potential(x):
    safe=x['alive'].bool()&x['legal'].bool();phase=x['phase'];good=safe&x['good'].bool()
    tracking=torch.exp(-x['q_error'].clamp_min(0))
    balance=torch.exp(-(x['speed']/.1).square()-(x['gyro']/.25).square())
    load=(x['support_bw']/.2).clamp(0,1)
    slip=(x['loaded_slip']/.02).square().clamp(0,1)
    wrong=(x['wrong_force']/10).square().clamp(0,1)
    impact=(x['peak_bw']-1).clamp(0,1).square()
    smooth=x['normalized_command_delta'].square().mean(-1).clamp(0,1)
    phi=2*tracking*safe+2*(phase>=1)*safe*load*balance+4*good*(x['held_ticks']/250).clamp(0,1)-slip-wrong-impact-.1*smooth
    return torch.where(x['first_failure']|x['horizon']|~x['alive'],0.,phi)

def staged_reward(x,imitation_weight=1.):
    assert imitation_weight==1.
    phi=potential(x)
    completion=400*x['clean_completion'].float()+20*x['stable_completion'].float()
    effort=.2*x['clean_completion'].float()/(1+x['steady_torque_squared']/1500.)
    base=torch.full_like(phi,DT)-20*x['first_failure'].float()+completion+effort
    shape=GAMMA*phi-x['previous_potential']
    reward=base+shape
    assert torch.isfinite(reward).all()
    return reward,dict(potential=phi,base=base,shaping=shape)
