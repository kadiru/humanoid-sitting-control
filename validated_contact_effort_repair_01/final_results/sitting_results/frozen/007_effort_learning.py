"""Slow coordinated exploration and support-constrained torque objective."""
import torch


def coordinated_actions(parent, child, ramp, span, names):
    knee=.02*torch.tanh(child[:,0])*ramp
    ankle=.012*torch.tanh(child[:,1])*ramp
    extra=torch.zeros_like(parent)
    for side in ('left','right'):
        for name,value in [('knee',knee),('hip_pitch',-.5*knee),('ankle_pitch',-.5*knee+ankle)]:
            extra[:,names.index(side+'_'+name+'_joint')]=value
    mixed=torch.atanh((torch.tanh(parent)+extra/span).clamp(-.999,.999))
    return torch.where(extra!=0,mixed,parent)


def resting_reward(total_squared_torque, reference, good, alive, learning_time, first_failure, dt=.02):
    cost=(total_squared_torque/reference).clamp(0,2)
    return dt*learning_time*(2*good-cost*good-5*(~good&alive))-5*first_failure


def mask_learning_advantages(advantages, masks):
    valid=masks[...,None].expand_as(advantages)
    values=advantages[valid]
    if values.numel():
        return torch.where(valid,(advantages-values.mean())/(values.std(unbiased=False)+1e-8),torch.zeros_like(advantages))
    return torch.zeros_like(advantages)
