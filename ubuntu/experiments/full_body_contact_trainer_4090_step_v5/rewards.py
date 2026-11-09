"""Staged rewards. Physical success remains the inherited separate predicate.

No phase-transition bonus: oscillating or skipping phases cannot farm a bonus.
Slip cost is increasing beyond the2cm/s threshold, rather than saturating there.
"""
import torch
from common import DT


def staged_reward(x, imitation_weight=1.):
    safe = x['alive'].bool() & x['legal'].bool()
    good = safe & x['good'].bool()
    phase = x['phase'].long()
    tracking = torch.exp(-x['q_error'].clamp_min(0))
    balance = torch.exp(-(x['speed']/.1).square()-(x['gyro']/.25).square())
    # Independent of the phase scheduler: every phase pays for contact mistakes.
    slip_cost = 2*(x['loaded_slip']/.02).square()
    wrong_cost = 2*(x['wrong_force']/10).square()
    impact_cost = 2*(x['peak_bw']-1).clamp_min(0).square()
    smooth_cost = .02*x['normalized_command_delta'].square().mean(-1)
    imitation = imitation_weight*torch.where(phase == 0, 2., .5)*tracking*safe
    contact = (phase >= 1)*safe*x['no_slip'].bool()*balance*x['support_bw'].clamp(0, .2)/.2
    settling = (phase >= 2)*3*good
    hold = (phase == 3)*2*good
    # Effort only after an actually measured continuous5s good hold.
    effort = .2*(phase == 3)*good*(x['held_ticks'] >= 250)/(1+x['torque_squared']/1500.)
    terms = dict(imitation=imitation, contact=contact, settling=settling, hold=hold,
                 effort=effort, slip_cost=slip_cost, wrong_cost=wrong_cost,
                 impact_cost=impact_cost, smooth_cost=smooth_cost)
    dense = DT*(imitation+contact+settling+hold+effort-slip_cost-wrong_cost-impact_cost-smooth_cost)
    # One-off failure and completion terms, not repeated per physics sample.
    terminal = 400*x['clean_completion'].float()+20*x['stable_completion'].float()-20*x['first_failure'].float()
    reward = dense+terminal
    assert torch.isfinite(reward).all()
    return reward, terms


def phase_from_clock(seconds, contact_time, settle_time):
    # Exogenous goal phase; no privileged contact signal is fed to the actor.
    return (seconds >= contact_time).long()+(seconds >= settle_time).long()+(seconds >= 15).long()
