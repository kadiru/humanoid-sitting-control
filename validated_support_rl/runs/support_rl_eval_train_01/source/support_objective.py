"""PPO sitting objective. Pure tensors; input values are TRAINING LABELS only.

Force/pose labels must never be packed into the policy observation. Effort
economy is conditional on actual, stable pelvis/thigh support, not a substitute
for it. Feet may retain useful load; 75% seat load is a soft preference.
"""
import torch


def support_objective(m, issued_delta, residual, alive, dt=.02):
    upright = m['gravity_z'] < -.90
    stable = (m['speed'] < .20) & (m['angular_speed'] < .35)
    supported = (m['support'] > .25) & m['inside'] & m['height_ok']
    good = supported & upright & stable & ~m['failure'] & alive
    gate = good.float()
    terms = {
        'approach': 1.5 * torch.exp(-m['goal_error2']/.025),
        'support': 4. * m['support'].clamp(0, .8)/.8 * m['inside'] * upright,
        'hold': 6. * gate,
        'seat_load_share': 2. * torch.exp(-((m['support']-.75)/.25).square()) * gate,
        'upright': 1.5 * torch.exp(-m['tilt2']/.03) * gate,
        'quiet': torch.exp(-m['speed'].square()/.02-m['angular_speed'].square()/.04) * gate,
        'leg_economy': 1.5 * torch.exp(-m['leg_effort2']) * gate,
        'impact': -3. * (m['peak_force_ratio']-1.).clamp_min(0).square(),
        'descent': -2. * ((-m['vz']-.2).clamp_min(0)/.3).square(),
        'effort': -.05 * m['effort2'],
        'command_slew': -.10 * (issued_delta/.02).square().mean(-1),
        'correction_size': -.02 * residual.square().mean(-1),
    }
    reward = sum(terms.values()) * dt
    newly_failed = m['failure'] & alive
    reward = torch.where(alive, reward, torch.zeros_like(reward)) - 5.*newly_failed
    return reward, good, terms
