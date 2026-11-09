"""Sensor-side reference control and direct all-joint effort objective."""
import torch


def posture_span(names, device='cpu'):
    return torch.tensor([
        .04 if ('hip_roll' in n or 'ankle_roll' in n) else
        .08 if 'ankle_pitch' in n else
        .30 if ('shoulder' in n or 'elbow' in n) else .15
        for n in names], device=device)


def reference_command(reference, previous, action, lower, upper, span):
    """Memory of the initial command is outside all trainable network layers.

    Targets are absolute offsets, not an integrator. Zero returns to reference.
    No privileged contact, pose, or failure signal controls this function.
    """
    if not torch.isfinite(action).all():
        raise ValueError('Nonfinite policy action')
    lo = torch.maximum(reference - span, lower + .03)
    hi = torch.minimum(reference + span, upper - .03)
    if not ((reference >= lo) & (reference <= hi)).all():
        raise ValueError('Reference command lacks physical joint margin')
    desired = (reference + span * torch.tanh(action)).clamp(min=lo, max=hi)
    return (previous + (.1 * (desired - previous)).clamp(-.002, .002)).clamp(min=lo, max=hi)


def effort_reward(torque, reference_squared_torque, good, first_failure, command_delta, dt=.02):
    """Minimize total squared Nm over ALL joints, under physical support gates.

    The positive reward is strictly decreasing for every finite torque cost;
    failure never earns a saving. Baseline calibration only rescales reward.
    """
    if not torch.isfinite(torque).all() or not torch.isfinite(reference_squared_torque).all():
        raise ValueError('Nonfinite effort input')
    if not (reference_squared_torque > 0).all():
        raise ValueError('Positive measured baseline is required')
    total = torque.square().sum(-1)
    efficiency = 1 / (1 + total / reference_squared_torque)
    smooth_cost = .001 * (command_delta / .002).square().mean(-1)
    # Nonnegative per-tick credit; no low-torque reward when unsupported.
    reward = dt * (efficiency - smooth_cost).clamp_min(0) * good - 5. * first_failure
    return reward, total
