"""Prototype reward for maintaining a supported resting posture.

NOT connected to a simulator or any existing training run. All inputs are
training/evaluation labels, never actor observations. Contact arrays describe
patches (feet/floor plus allowed pelvis/thigh/back/surface patches). Forces must
include normal AND friction forces on the robot, in the same world frame as
outward surface normals. The existing normal-force-only adapter is insufficient.

This instantaneous gate is not a stability proof. Full-duration hold, slip,
perturbation and reset-feasibility tests are required before training/promotion.
"""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class RestingObjectiveCfg:
    control_dt: float = .02
    stable_reward_per_s: float = 2.
    effort_weight: float = 1.
    failure_cost: float = 5.
    minimum_rest_normal_load_bw: float = .05
    contact_active_load_bw: float = .01
    maximum_speed_m_s: float = .10
    maximum_angular_speed_rad_s: float = .25
    maximum_slip_m_s: float = .02
    maximum_trunk_tilt_deg: float = 40.
    maximum_friction_utilization: float = .90


def resting_objective(m, alive, cfg=RestingObjectiveCfg()):
    """Return reward, qualifying mask and diagnostics for a batch.

    m: torque [B,J], torque_scale [J] or [B,J], joint_velocity [B,J];
    force_w/normal_w [B,K,3], friction [B,K], permitted/rest_patch/inside_patch
    [B,K] bool, slip_m_s [B,K]; bodyweight_N/speed_m_s/angular_speed_rad_s/
    trunk_tilt_rad [B]; joint_margin_ok/failure [B] bool. torque_scale is a
    declared positive reference (continuous ratings if available; otherwise a
    named limit-normalized proxy), NOT an inferred electrical-energy model.
    alive [B] is the pre-step latched alive state. Caller latches returned failures.
    """
    if not (0 < cfg.control_dt <= .1 and 0 <= cfg.effort_weight <= 1):
        raise ValueError('Invalid time step or initial bounded effort coefficient')
    force, normal = m['force_w'], m['normal_w']
    if force.ndim != 3 or force.shape[-1] != 3 or normal.shape != force.shape:
        raise ValueError('Contact forces/normals must be [batch, patch, 3]')
    for key in ('force_w', 'normal_w', 'friction', 'slip_m_s', 'torque',
                'torque_scale', 'joint_velocity', 'bodyweight_N', 'speed_m_s',
                'angular_speed_rad_s', 'trunk_tilt_rad'):
        if not torch.isfinite(m[key]).all():
            raise ValueError(f'Non-finite label: {key}')
    if (m['torque_scale'] <= 0).any() or (m['bodyweight_N'] <= 0).any():
        raise ValueError('Torque scales and body weights must be positive')
    if (m['friction'] < 0).any() or (m['slip_m_s'] < 0).any():
        raise ValueError('Friction and slip speed must be nonnegative')
    if not torch.allclose(normal.norm(dim=-1), torch.ones_like(m['friction']), atol=1e-5):
        raise ValueError('Normals must be unit vectors pointing out of the surface')
    fn = (force * normal).sum(-1)
    ft = (force - fn[..., None] * normal).norm(dim=-1)
    weight = m['bodyweight_N'][:, None]
    active = force.norm(dim=-1) > cfg.contact_active_load_bw * weight
    capacity = m['friction'] * fn.clamp_min(0)
    utilization = torch.where(ft <= 1e-7, torch.zeros_like(ft),
                              ft / capacity.clamp_min(1e-7))
    legal = (~active | (m['permitted'] & m['inside_patch'] & (fn >= 0))).all(-1)
    traction = (~active | (utilization <= cfg.maximum_friction_utilization)).all(-1)
    no_slip = (~active | (m['slip_m_s'] <= cfg.maximum_slip_m_s)).all(-1)
    rest_load = (fn.clamp_min(0) * m['rest_patch'] * m['permitted']).sum(-1) / weight[:, 0]
    supported = rest_load >= cfg.minimum_rest_normal_load_bw
    stable = ((m['speed_m_s'] <= cfg.maximum_speed_m_s) &
              (m['angular_speed_rad_s'] <= cfg.maximum_angular_speed_rad_s) &
              (m['trunk_tilt_rad'].abs() <= math.radians(cfg.maximum_trunk_tilt_deg)))
    normalized_torque = m['torque'] / m['torque_scale']
    effort = normalized_torque.square().mean(-1)
    # Hardware torque-limit violation remains a separate latched failure label;
    # torque_scale may be a continuous rating rather than a peak limit.
    good = (alive & ~m['failure'] & m['joint_margin_ok'] & legal & traction &
            no_slip & supported & stable)
    hold = cfg.stable_reward_per_s * good
    # Bound only the reward cost. Preserve the unbounded metric for diagnosis.
    effort_cost = -cfg.effort_weight * effort.clamp(max=1.) * good
    failure_cost = -cfg.failure_cost * (m['failure'] & alive)
    reward = (hold + effort_cost) * cfg.control_dt + failure_cost
    diagnostics = dict(stable_hold=hold, effort_cost=effort_cost,
        failure_cost=failure_cost, normalized_torque_squared=effort,
        all_joint_rms_Nm=m['torque'].square().mean(-1).sqrt(),
        absolute_mechanical_power_W=(m['torque'] * m['joint_velocity']).abs().sum(-1),
        rest_normal_load_bw=rest_load,
        rest_vertical_load_bw=(force[..., 2] * m['rest_patch'] * m['permitted']).sum(-1) / weight[:, 0],
        maximum_friction_utilization=torch.where(active, utilization, 0.).max(-1).values,
        contacts_legal=legal, traction_ok=traction, no_slip=no_slip)
    return reward, good, diagnostics
